import os
import random
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from dataset.euler_coarse import EulerPeriodicDataset
from model.gnn_flux import FluxGNN
from rollout import rollout_one_simulation
from utils import teacher_forcing_schedule, autoregressive_pred


def train_one_epoch(model, dataloader, optimizer, device,
                    loss_weights: Optional[Dict[str, float]] = None,
                    clip_grad: Optional[float] = None,
                    teacher_forcing_prob: float = 1.0):
    """
    Args:
        model: model returning dict with keys "density","energy","pressure","momentum".
        dataloader: PyG DataLoader yielding Batch objects with fields:
                    .x, .pos, .edge_index, .edge_attr (opt), .batch (opt),
                    and ground-truth: .y_density, .y_energy, .y_pressure, .y_momentum
        optimizer: optimizer to step.
        device: torch device.
        loss_weights: dict with keys "density","energy","pressure","momentum" -> float.
                      If None, equal weighting is used.
        clip_grad: optional max-norm for gradients.
        - teacher_forcing_prob in [0,1]: prob of using ground-truth inputs (teacher forcing).
    Returns:
        dict with averaged metrics: loss, mse_density, mse_energy, mse_pressure, mse_momentum, num_nodes
    """
    model.train()
    model.to(device)

    if loss_weights is None:
        loss_weights = {"density": 1.0, "energy": 1.0, "pressure": 1.0, "momentum": 1.0}

    total_loss = 0.0
    total_mse_density = 0.0
    total_mse_energy = 0.0
    total_mse_pressure = 0.0
    total_mse_momentum = 0.0
    total_nodes = 0

    # list to store batch losses
    batch_losses = []

    # derive a base dataset (works with ConcatDataset or single Dataset)
    ds_wrap = dataloader.dataset
    if hasattr(ds_wrap, "datasets"): # ConcatDataset
        base_ds = ds_wrap.datasets[0]
    else:
        base_ds = ds_wrap # single Dataset

    Hc, Wc = int(base_ds.Hc), int(base_ds.Wc)
    t_w = int(base_ds.time_window)
    target_type = str(base_ds.target)

    C_in = (t_w - 1) * 5  # expected input channels

    # channel layout in x: [density(0..t_w-2), energy(...), pressure(...), mom_ch(...)]
    d_count, e_count, p_count, mom_count = t_w - 1, t_w - 1, t_w - 1, 2*(t_w - 1)
    # slice indices
    d_start = 0
    d_end = d_start + d_count
    e_start = d_end
    e_end = e_start + e_count
    p_start = e_end
    p_end = p_start + p_count
    m_start = p_end
    m_end = m_start + mom_count

    # build start and end lists
    start = [d_start, e_start, p_start, m_start]
    end = [d_end, e_end, p_end, m_end]
    
    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        # ground truth
        y_density  = getattr(batch, "y_density", None).view(-1)
        y_energy   = getattr(batch, "y_energy", None).view(-1)
        y_pressure = getattr(batch, "y_pressure", None).view(-1)
        y_momentum = getattr(batch, "y_momentum", None).view(-1, 2)

        # decide whether to teacher-force this batch
        if random.random() < teacher_forcing_prob:
            # standard teacher forcing: direct forward and loss
            preds = model(batch)
        else:
            batch_mixed = autoregressive_pred(model, batch, Hc, Wc, C_in, target_type, start, end)
            # forward with gradients on mixed batch
            preds = model(batch_mixed)

        p_density  = preds["density"].view(-1)
        p_energy   = preds["energy"].view(-1)
        p_pressure = preds["pressure"].view(-1)
        p_momentum = preds["momentum"].view(-1, 2)

        mse_density  = F.mse_loss(p_density, y_density, reduction="mean")
        mse_energy   = F.mse_loss(p_energy, y_energy, reduction="mean")
        mse_pressure = F.mse_loss(p_pressure, y_pressure, reduction="mean")
        mse_momentum = F.mse_loss(p_momentum, y_momentum, reduction="mean")  # reduction="mean" -> averaged over all dims

        loss = (
            loss_weights.get("density") * mse_density
            + loss_weights.get("energy") * mse_energy
            + loss_weights.get("pressure") * mse_pressure
            + loss_weights.get("momentum") * mse_momentum
        )

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        print(f"\nCurrent step: {step+1}/{len(dataloader)}")
        print(f"Batch Loss: {loss.item():.6f} "
              f"(Density MSE: {mse_density.item():.6f}, "
              f"Energy MSE: {mse_energy.item():.6f}, "
              f"Pressure MSE: {mse_pressure.item():.6f}, "
              f"Momentum MSE: {mse_momentum.item():.6f})")
        
        batch_losses.append(loss.item())

        # accumulate (weight by number of nodes so averaging is correct)
        # (note that in the Euler case all sims have the same number of nodes)
        n_nodes = p_density.numel()
        total_nodes += n_nodes
        total_loss += float(loss.detach()) * n_nodes
        total_mse_density += float(mse_density.detach()) * n_nodes
        total_mse_energy  += float(mse_energy.detach()) * n_nodes
        total_mse_pressure+= float(mse_pressure.detach()) * n_nodes
        total_mse_momentum+= float(mse_momentum.detach()) * n_nodes

    return {
        "loss": total_loss / total_nodes,
        "mse_density": total_mse_density / total_nodes,
        "mse_energy": total_mse_energy / total_nodes,
        "mse_pressure": total_mse_pressure / total_nodes,
        "mse_momentum": total_mse_momentum / total_nodes,
        "num_nodes": total_nodes,
        "batch_losses": batch_losses
    }
 

def train(model, train_loader, valid_dataset=None, optimizer=None, device="cuda", epochs=10, 
          save_dir="./checkpoints", save_every=1, fname="model", floss="loss", mixed_train=True,
          teacher_forcing_start=1.0):
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
        teacher_forcing_start: starting probability for teacher forcing (if mixed_train is True)
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
                                teacher_forcing_prob=teacher_forcing_prob)
        
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
                      "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.4_Dry_air_20.hdf5"]
    
    h5_path_valid = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/valid/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"

    # create dataset and dataloader
    time_window = 5
    coarsen = (2,2)
    target = "delta" # "delta" or "absolute"
    
    # create three train datasets (use centroids conversion)
    train_datasets = [
        EulerPeriodicDataset(h5_path=p,
                            stats_path=stats_path,
                            time_window=time_window,
                            coarsen=coarsen,
                            target=target,
                            to_centroids=True)
        for p in h5_paths_train
    ]
    # concatenate into a single dataset (one epoch = all samples across all files)
    train_dataset = ConcatDataset(train_datasets)

    # validation dataset (also use centroids)
    valid_dataset = EulerPeriodicDataset(h5_path=h5_path_valid,
                                        stats_path=stats_path,
                                        time_window=time_window,
                                        coarsen=coarsen,
                                        target=target,
                                        to_centroids=True)

    print(f"Total train samples: {len(train_dataset)}")
    # grid dims from the first dataset
    print(f"Current grid dimension: {len(train_datasets[0]._static_cache['x_coords_coarse'])}")
    print(f"Dataset time_window: {time_window}")
    print(f"Dataset coarsen: ({train_datasets[0].sh},{train_datasets[0].sw})")
    print(f"Dataset target: {target}")
 
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # get a sample for input/global dims
    sample = train_dataset[0]
    input_node_feats = sample.x.shape[1]  # node input dim (C_in)

    # read gamma if provided in sample.u else default
    gamma_val = None
    if getattr(sample, "u", None) is not None:
        try:
            gamma_val = float(sample.u[0, 0].item())
        except Exception:
            gamma_val = None

    # instantiate FluxGNN
    model = FluxGNN(in_node_dim=input_node_feats,
                    node_hidden=(128,),
                    node_embed_dim=64,
                    n_layers=4,
                    dt_max=0.015,
                    gamma=(gamma_val if gamma_val is not None else 1.4),
                    use_residual=True)

    # use AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # teacher forcing start value
    teacher_forcing_start = 1.0

    fname = f"model_n-datasets_{len(h5_paths_train)}_forcing_{teacher_forcing_start}_time-window_{time_window}_coarsen_{coarsen[0]}-{coarsen[1]}_target_{target}"
    floss = f"loss_n-datasets_{len(h5_paths_train)}_forcing_{teacher_forcing_start}_time-window_{time_window}_coarsen_{coarsen[0]}-{coarsen[1]}_target_{target}"

    # train the model
    train(model, train_loader, valid_dataset=valid_dataset, optimizer=optimizer,
          epochs=10, fname=fname, floss=floss, mixed_train=True,
          teacher_forcing_start=teacher_forcing_start)

    print("Training complete.")