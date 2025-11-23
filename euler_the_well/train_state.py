import os
import random
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from dataset.euler_coarse import EulerPeriodicDataset
from model.egnn_state import EGNNStateModel
from rollout import rollout_one_simulation
from utils import teacher_forcing_schedule


def train_one_epoch(model, dataloader, optimizer, device,
                    loss_weights: Optional[Dict[str, float]] = None,
                    clip_grad: Optional[float] = None,
                    teacher_forcing_prob: float = 1.0,
                    unroll_steps: int = 1):
    """
    train_one_epoch with simple K-step unrolled autoregressive loss.
    - teacher_forcing_prob: probability to use GT when appending inputs during the unroll.
    - unroll_steps: number of autoregressive steps to unroll (K).
    """
    model.train()
    model.to(device)

    if loss_weights is None:
        loss_weights = {"density": 1.0, "energy": 1.0, "pressure": 1.0, "momentum": 1.0}

    # accumulators for epoch
    total_loss_sum = 0.0
    total_mse_density = 0.0
    total_mse_energy = 0.0
    total_mse_pressure = 0.0
    total_mse_momentum = 0.0
    total_nodes = 0

    batch_losses = []

    # derive base dataset (works with ConcatDataset or single Dataset)
    ds_wrap = dataloader.dataset
    if hasattr(ds_wrap, "datasets"):  # ConcatDataset
        base_ds = ds_wrap.datasets[0]
    else:
        base_ds = ds_wrap

    Hc, Wc = int(base_ds.Hc), int(base_ds.Wc)
    target_type = str(base_ds.target)
    train_t_w = int(base_ds.time_window)                # expanded training window length
    model_t_w = train_t_w - (unroll_steps - 1)          # actual model time window

    # channel counts
    C_total = (train_t_w - 1) * 5       # channels in batch.x
    C_model_in = (model_t_w - 1) * 5    # channels model expects as input

    # training-channel slice indices (for x_np_total)
    train_d_count = train_e_count = train_p_count = train_t_w - 1

    train_d_start = 0
    train_d_end = train_d_start + train_d_count
    train_e_start = train_d_end
    train_e_end = train_e_start + train_e_count
    train_p_start = train_e_end
    train_p_end = train_p_start + train_p_count
    train_m_start = train_p_end

    # dummies to call _arrays_to_graph (won't use Data.y_* for loss, instead take
    # from expanded window GT)
    y_density_dummy = np.zeros((Hc, Wc), dtype=np.float32)
    y_energy_dummy = np.zeros((Hc, Wc), dtype=np.float32)
    y_pressure_dummy = np.zeros((Hc, Wc), dtype=np.float32)
    y_momentum_dummy = np.zeros((Hc, Wc, 2), dtype=np.float32)

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        # prepare total-channel numpy array from batch.x
        x_nodes = batch.x.detach().cpu().numpy()                # (N, C_total)
        x_np_total = x_nodes.T.reshape(C_total, Hc, Wc)         # (C_total, Hc, Wc)

        # initial model input (first C_model_in channels -> frames 0..model_t_w-2)
        x_current_np = x_np_total[0:C_model_in, :, :].copy()    # copy so can modify

        # accumulators for this batch's per-step MSEs
        sum_mse_d = 0.0
        sum_mse_e = 0.0
        sum_mse_p = 0.0
        sum_mse_m = 0.0

        # total loss tensor to backprop through
        total_loss_t = torch.tensor(0.0, device=device)

        # unroll loop: s = 0 .. unroll_steps-1
        for s in range(unroll_steps):
            # get current time-step for global features
            time_scalar = float(batch.u[0,1].detach().cpu().item()) if (getattr(batch,'u',None) is not None and batch.u.shape[1] > 1) else 0.0
            # build Data object for current input window (x_current_np is numpy array)
            data_current = base_ds._arrays_to_graph(
                x_current_np,
                y_density_dummy,
                y_energy_dummy,
                y_pressure_dummy,
                y_momentum_dummy,
                time_step=time_scalar
            )
            data_current = data_current.to(device)

            # forward (with gradients) -> get predictions for the "last" timestep of current window
            out = model(data_current)

            p_density  = out["density"]        # (N,)
            p_energy   = out["energy"]         # (N,)
            p_pressure = out["pressure"]       # (N,)
            p_momentum = out["momentum"]       # (N,2)

            # flatten predictions
            p_density_flat  = p_density.view(-1)                # (N,)
            p_energy_flat   = p_energy.view(-1)
            p_pressure_flat = p_pressure.view(-1)
            p_momentum_flat = p_momentum.view(-1, 2)            # (N,2)

            # determine GT frame index inside x_np_total for this prediction
            gt_frame_idx = (model_t_w - 1) + s

            # if gt_frame_idx is less than the last input index, read from x_np_total
            if gt_frame_idx < (train_t_w - 1):
                gt_d_np = x_np_total[train_d_start + gt_frame_idx, :, :]
                gt_e_np = x_np_total[train_e_start + gt_frame_idx, :, :]
                gt_p_np = x_np_total[train_p_start + gt_frame_idx, :, :]
                gt_mx_np = x_np_total[train_m_start + 2 * gt_frame_idx, :, :]
                gt_my_np = x_np_total[train_m_start + 2 * gt_frame_idx + 1, :, :]
                gt_m_np = np.stack([gt_mx_np, gt_my_np], axis=-1)   # (Hc,Wc,2)
            else:
                # if gt_frame_idx == train_t_w - 1 (the final target frame), read from batch.y_*
                # bc final target frame is stored there (flattened). batch is on device.
                #NOTE: this assumes batch_size==1, if batch_size>1 must
                # index per graph in the batch
                gt_d_np = batch.y_density.detach().cpu().numpy().reshape(Hc, Wc)
                gt_e_np = batch.y_energy.detach().cpu().numpy().reshape(Hc, Wc)
                gt_p_np = batch.y_pressure.detach().cpu().numpy().reshape(Hc, Wc)
                gt_m_np = batch.y_momentum.detach().cpu().numpy().reshape(Hc, Wc, 2)
                gt_mx_np = gt_m_np[..., 0]
                gt_my_np = gt_m_np[..., 1]

            if target_type == "delta":
                # compute "first" frame of the current model input window (normalized space)
                d_count = e_count = p_count = model_t_w - 1
                m_count = 2 * (model_t_w - 1)

                # relative slice indices inside x_current_np (which holds the current model input window)
                d_s = 0
                d_e = d_s + d_count
                e_s = d_e
                e_e = e_s + e_count
                p_s = e_e
                p_e = p_s + p_count
                m_s = p_e

                # gather first-frame arrays
                first_d = x_current_np[d_s].astype(np.float32)   # (Hc,Wc)
                first_e = x_current_np[e_s].astype(np.float32)
                first_p = x_current_np[p_s].astype(np.float32)
                first_mx = x_current_np[m_s].astype(np.float32)
                first_my = x_current_np[m_s + 1].astype(np.float32)
                first_m = np.stack([first_mx, first_my], axis=-1)  # (Hc,Wc,2)

                # convert absolute GT frame to target-space (delta = last - first)
                gt_d_np = (gt_d_np.astype(np.float32) - first_d).astype(np.float32)
                gt_e_np = (gt_e_np.astype(np.float32) - first_e).astype(np.float32)
                gt_p_np = (gt_p_np.astype(np.float32) - first_p).astype(np.float32)
                gt_m_np = (gt_m_np.astype(np.float32) - first_m).astype(np.float32)

            # convert GT to torch on device (flattened to match preds)
            gt_d = torch.tensor(gt_d_np.reshape(-1), dtype=p_density_flat.dtype, device=device)
            gt_e = torch.tensor(gt_e_np.reshape(-1), dtype=p_energy_flat.dtype, device=device)
            gt_pr = torch.tensor(gt_p_np.reshape(-1), dtype=p_pressure_flat.dtype, device=device)
            gt_m = torch.tensor(gt_m_np.reshape(-1, 2), dtype=p_momentum_flat.dtype, device=device)

            # compute per-step MSEs (torch tensors)
            mse_d_t = F.mse_loss(p_density_flat, gt_d, reduction="mean")
            mse_e_t = F.mse_loss(p_energy_flat, gt_e, reduction="mean")
            mse_pr_t= F.mse_loss(p_pressure_flat, gt_pr, reduction="mean")
            mse_m_t = F.mse_loss(p_momentum_flat, gt_m, reduction="mean")

            # accumulate to scalar total loss (use loss_weights)
            step_loss = (
                loss_weights.get("density", 1.0) * mse_d_t
                + loss_weights.get("energy", 1.0) * mse_e_t
                + loss_weights.get("pressure", 1.0) * mse_pr_t
                + loss_weights.get("momentum", 1.0) * mse_m_t
            )
            total_loss_t += step_loss

            # store floats for averaging
            sum_mse_d += float(mse_d_t.detach().cpu().item())
            sum_mse_e += float(mse_e_t.detach().cpu().item())
            sum_mse_p += float(mse_pr_t.detach().cpu().item())
            sum_mse_m += float(mse_m_t.detach().cpu().item())

            # advance the input window: shift left and append either GT or model-pred (detached)
            if random.random() < teacher_forcing_prob:
                # append ground truth frame (use arrays from x_np_total)
                append_d = gt_d_np
                append_e = gt_e_np
                append_p = gt_p_np
                append_mx = gt_mx_np
                append_my = gt_my_np
            else:
                # append detached model prediction (numpy)
                append_d = p_density_flat.detach().cpu().numpy().reshape(Hc, Wc)
                append_e = p_energy_flat.detach().cpu().numpy().reshape(Hc, Wc)
                append_p = p_pressure_flat.detach().cpu().numpy().reshape(Hc, Wc)
                append_m = p_momentum.detach().cpu().numpy().reshape(Hc, Wc, 2)
                append_mx = append_m[..., 0]
                append_my = append_m[..., 1]

            # perform shift-and-append on x_current_np (C_model_in, Hc, Wc)
            # channel layout in x_current_np: [d0..d_{model_t_w-2}, e0.., p0.., m0_ x,y ...]
            d_count = e_count = p_count = model_t_w - 1
            m_count = 2 * (model_t_w - 1)
            d_s = 0
            d_e = d_s + d_count
            e_s = d_e
            e_e = e_s + e_count
            p_s = e_e
            p_e = p_s + p_count
            m_s = p_e
            m_e = m_s + m_count

            # shift density/energy/pressure and append new frame
            x_current_np[d_s:d_e] = np.concatenate([x_current_np[d_s + 1:d_e], np.expand_dims(append_d, axis=0)], axis=0)
            x_current_np[e_s:e_e] = np.concatenate([x_current_np[e_s + 1:e_e], np.expand_dims(append_e, axis=0)], axis=0)
            x_current_np[p_s:p_e] = np.concatenate([x_current_np[p_s + 1:p_e], np.expand_dims(append_p, axis=0)], axis=0)

            # momentum channels: drop first pair, shift, append append_mx/append_my
            x_current_np[m_s:m_e] = np.concatenate(
                [x_current_np[m_s + 2:m_e], np.expand_dims(append_mx, axis=0), np.expand_dims(append_my, axis=0)],
                axis=0
            )

        # average the accumulated total loss over unroll length so scale remains comparable
        final_total_loss = total_loss_t / float(unroll_steps)

        # bckpropagate once per batch
        optimizer.zero_grad()
        final_total_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        # build batch-level metrics (averages across unrolled steps)
        avg_mse_d = sum_mse_d / float(unroll_steps)
        avg_mse_e = sum_mse_e / float(unroll_steps)
        avg_mse_p = sum_mse_p / float(unroll_steps)
        avg_m = sum_mse_m / float(unroll_steps)

        avg_loss_float = float(final_total_loss.detach().cpu().item())

        print(f"\nCurrent step: {step+1}/{len(dataloader)}")
        print(f"Batch Loss: {avg_loss_float:.6f} "
              f"(Density MSE: {avg_mse_d:.6f}, "
              f"Energy MSE: {avg_mse_e:.6f}, "
              f"Pressure MSE: {avg_mse_p:.6f}, "
              f"Momentum MSE: {avg_m:.6f})")

        batch_losses.append(avg_loss_float)

        # accumulate (weight by number of nodes so averaging is correct)
        n_nodes = Hc * Wc
        total_nodes += n_nodes
        total_loss_sum += avg_loss_float * n_nodes
        total_mse_density += avg_mse_d * n_nodes
        total_mse_energy += avg_mse_e * n_nodes
        total_mse_pressure += avg_mse_p * n_nodes
        total_mse_momentum += avg_m * n_nodes

    return {
        "loss": total_loss_sum / total_nodes,
        "mse_density": total_mse_density / total_nodes,
        "mse_energy": total_mse_energy / total_nodes,
        "mse_pressure": total_mse_pressure / total_nodes,
        "mse_momentum": total_mse_momentum / total_nodes,
        "num_nodes": total_nodes,
        "batch_losses": batch_losses
    }


def train(model, train_loader, valid_dataset=None, optimizer=None, device="cuda", epochs=10, 
          save_dir="./checkpoints", save_every=1, fname="model", floss="loss",
          mixed_train=True, teacher_forcing_start=1.0, unroll_steps=1):
    """
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        valid_dataset: optional Dataset for validation
        optimizer: PyTorch optimizer
        device: "cuda" or "cpu"
        epochs: number of epochs to train
        save_dir: directory to save checkpoints
        save_every: save model every N epochs
        fname: base model filename for saving checkpoints
        floss: base loss filename for saving losses
        mixed_train: whether to use scheduled sampling (teacher forcing)
        teacher_forcing_start: starting probability for teacher forcing
        unroll_steps: number of autoregressive unroll steps during training
    """
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        if mixed_train:
            # linear schedule for teacher forcing probability
            teacher_forcing_prob = teacher_forcing_schedule(epoch, epochs, start=teacher_forcing_start, end=0.0)
        else:
            teacher_forcing_prob = 1.0  # always teacher force
        # train for one epoch
        results = train_one_epoch(model, train_loader, optimizer, device,
                                teacher_forcing_prob=teacher_forcing_prob, unroll_steps=unroll_steps)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {results['loss']:.6f}")
        
        # save batch losses for this epoch
        loss_save_path = os.path.join(save_dir, f"{floss}_epoch_{epoch}.pt")
        torch.save(torch.tensor(results["batch_losses"]), loss_save_path)
        print(f"Saved losses: {loss_save_path}")

        # validation rollout
        if valid_dataset is not None:
            print("\n -----------------[Validation Rollout]------------------")
 
            num_val_sims = 2
            mid = lambda s: s // 2

            total_sims = int(valid_dataset.n_sims)
            true_val_sims = min(num_val_sims, max(0, total_sims))

            # randomly sample among true_val_sims
            sim_indices = random.sample(range(total_sims), true_val_sims)

            for sim_i in sim_indices:
                out = rollout_one_simulation(
                    model,
                    valid_dataset,
                    sim_idx=sim_i,
                    start_t=0,
                    rollout_steps=None, # full rollout
                    device=device,
                    return_denormalized=True,
                    save_path=None,
                    verbose=False
                )

                metrics = out["metrics"]
                n_steps = len(metrics["rmse_density"])
                mid_step = mid(n_steps)

                print(f"\nSim {sim_i}:")
                print(f"  Step 0   | dens={metrics['rmse_density'][0]:.4e} "
                    f"energy={metrics['rmse_energy'][0]:.4e} "
                    f"press={metrics['rmse_pressure'][0]:.4e} "
                    f"momentum_x={metrics['rmse_momentum_x'][0]:.4e} "
                    f"momentum_y={metrics['rmse_momentum_y'][0]:.4e}")
                print(f"  Step {mid_step} | dens={metrics['rmse_density'][mid_step]:.4e} "
                    f"energy={metrics['rmse_energy'][mid_step]:.4e} "
                    f"press={metrics['rmse_pressure'][mid_step]:.4e} "
                    f"momentum_x={metrics['rmse_momentum_x'][mid_step]:.4e} "
                    f"momentum_y={metrics['rmse_momentum_y'][mid_step]:.4e}")
                print(f"  Step {n_steps-1} | dens={metrics['rmse_density'][-1]:.4e} "
                    f"energy={metrics['rmse_energy'][-1]:.4e} "
                    f"press={metrics['rmse_pressure'][-1]:.4e} "
                    f"momentum_x={metrics['rmse_momentum_x'][-1]:.4e} "
                    f"momentum_y={metrics['rmse_momentum_y'][-1]:.4e}")

                avg_d = metrics["rmse_density"].mean()
                avg_e = metrics["rmse_energy"].mean()
                avg_p = metrics["rmse_pressure"].mean()
                avg_mx = metrics["rmse_momentum_x"].mean()
                avg_my = metrics["rmse_momentum_y"].mean()

                print(f"  Avg RMSE | dens={avg_d:.4e} energy={avg_e:.4e} press={avg_p:.4e} "
                      f"momentum_x={avg_mx:.4e} momentum_y={avg_my:.4e}\n\n")
        
        # save checkpoint
        if epoch % save_every == 0:
            # complete fname with epoch
            fname_epoch = f"{fname}_epoch_{epoch}.pt"
            checkpoint_path = os.path.join(save_dir, fname_epoch)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    # h5 files path
    h5_paths_train = ["/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5",
                    #   "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.33_H2O_20.hdf5",
                    #   "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.4_Dry_air_20.hdf5",
                      ]
    
    h5_path_valid = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/valid/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"

    # create dataset and dataloader
    time_window = 5
    coarsen = (2,2)
    target = "delta" # "delta" or "absolute"

    # training unroll size
    unroll_steps = 3
    # model_time_window is the window the model expects (history length)
    model_time_window = time_window
    # training window must include extra future frames for unrolling
    train_time_window = model_time_window + (unroll_steps - 1)

    if model_time_window < 2:
        # this is already checked in the datasets creation, but warn user here for clarity
        raise ValueError("model_time_window must be at least 2 to predict next step.")

    # create train datasets using the expanded train_time_window
    train_datasets = [
        EulerPeriodicDataset(h5_path=p, stats_path=stats_path,
                             time_window=train_time_window, 
                             coarsen=coarsen, target=target)
        for p in h5_paths_train
    ]
    train_dataset = ConcatDataset(train_datasets)

    # validation dataset should use the model's window (not the expanded one)
    valid_dataset = EulerPeriodicDataset(h5_path=h5_path_valid,
                                         stats_path=stats_path,
                                         time_window=model_time_window,
                                         coarsen=coarsen,
                                         target=target)

    print(f"Total train samples: {len(train_dataset)}")
    print(f"Current grid dimension: {len(train_datasets[0]._static_cache['x_coords_coarse'])}")
    print(f"Model time_window: {model_time_window}")
    print(f"Training time_window (expanded): {train_time_window}")
    print(f"Dataset coarsen: ({train_datasets[0].sh},{train_datasets[0].sw})")
    print(f"Dataset target: {target}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # get a sample from a dataset that uses the model_time_window
    # to extract input dims
    sample = valid_dataset[0]
    input_node_feats = sample.x.shape[1]        # (model_time_window-1) * 5
    global_feat_dim = sample.u.shape[1] if getattr(sample, "u", None) is not None else 0

    # create model (use input_node_feats from the sample above)
    model = EGNNStateModel(input_feat_dim=input_node_feats, global_feat_dim=global_feat_dim, use_separate_heads=True)

    # use AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # teacher forcing start value
    teacher_forcing_start = 0.5

    fname = f"model_n-datasets_{len(h5_paths_train)}_forcing_{teacher_forcing_start}_time-window_{time_window}_unroll-steps_{unroll_steps}_coarsen_{coarsen[0]}-{coarsen[1]}_target_{target}"
    floss = f"loss_n-datasets_{len(h5_paths_train)}_forcing_{teacher_forcing_start}_time-window_{time_window}_unroll-steps_{unroll_steps}_coarsen_{coarsen[0]}-{coarsen[1]}_target_{target}"

    # pass unroll_steps to train
    train(model, train_loader, valid_dataset=valid_dataset, optimizer=optimizer,
          epochs=10, fname=fname, floss=floss, mixed_train=True,
          teacher_forcing_start=teacher_forcing_start, unroll_steps=unroll_steps)

    print("Training complete.")
