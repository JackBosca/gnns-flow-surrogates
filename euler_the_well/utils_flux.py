from typing import Optional, Dict
import torch
import torch.nn.functional as F

def train_one_epoch_unrolled(model, dataloader, stats, optimizer, device,
                             loss_weights: Optional[Dict[str, float]] = None,
                             clip_grad: Optional[float] = None,
                             noise_std: float = 0.01,
                             n_unroll_steps: int = 5):
    """
    Performs N-Step Unrolled Training (t0 -> t1 -> ... -> tN).
    Supports models with history (node_in_dim > 5).
    Requires dataset with time_window >= n_unroll_steps + model_history_steps.
    """
    model.train()
    model.to(device)

    if loss_weights is None:
        loss_weights = {"density": 1.0, "energy": 1.0, "pressure": 1.0, "momentum": 5.0}

    total_loss = 0.0
    total_mse_density = 0.0
    total_mse_energy = 0.0
    total_mse_pressure = 0.0
    total_mse_momentum = 0.0
    total_nodes = 0

    batch_losses = []

    ds_wrap = dataloader.dataset
    if hasattr(ds_wrap, "datasets"):
        base_ds = ds_wrap.datasets[0]
    else:
        base_ds = ds_wrap

    t_w = int(base_ds.time_window)
    dataset_history_steps = t_w - 1
    target_type = str(base_ds.target)

    # Determine model's expected history length from node_in_dim
    # C_in = steps * 5. So steps = node_in_dim // 5
    model_in_steps = model.node_in_dim // 5

    # Dataset must provide enough history for the model + future targets
    # need 'model_in_steps' to start, plus 'n_unroll_steps' targets
    if t_w < model_in_steps + n_unroll_steps:
        raise ValueError(f"Dataset time_window ({t_w}) too small. "
                         f"Model needs {model_in_steps} inputs + {n_unroll_steps} unroll targets. "
                         f"Increase dataset time_window to >= {model_in_steps + n_unroll_steps}.")

    def slide_window(window_flat, new_state_flat, n_steps):
        """
        Slides the grouped input window forward by 1 step.
        window_flat: (N, n_steps*5) - grouped [rho..., e..., p..., m...]
        new_state_flat: (N, 5) - [rho, e, p, mx, my]
        Returns: (N, n_steps*5)
        """
        # Slice out the components (dropping the oldest timestep, i.e., index 0)
        # rho: indices 1 to n_steps-1
        rho_hist = window_flat[:, 1 : n_steps] 
        # energy: indices n_steps+1 to 2*n_steps-1
        e_hist   = window_flat[:, n_steps + 1 : 2 * n_steps]
        # pressure: indices 2*n_steps+1 to 3*n_steps-1
        p_hist   = window_flat[:, 2 * n_steps + 1 : 3 * n_steps]
        # momentum: indices 3*n_steps+2 to 5*n_steps-1 (dropping 2 channels)
        mom_hist = window_flat[:, 3 * n_steps + 2 : 5 * n_steps]

        # Append new state
        rho_new = torch.cat([rho_hist, new_state_flat[:, 0:1]], dim=1)
        e_new   = torch.cat([e_hist,   new_state_flat[:, 1:2]], dim=1)
        p_new   = torch.cat([p_hist,   new_state_flat[:, 2:3]], dim=1)
        mom_new = torch.cat([mom_hist, new_state_flat[:, 3:5]], dim=1)

        return torch.cat([rho_new, e_new, p_new, mom_new], dim=1)

    # Helper to extract a (N, 5) state vector from the dataset X
    def get_state_at_step(x_tensor, step_idx):
        n_history = dataset_history_steps
        
        rho_start = 0
        e_start   = rho_start + n_history
        p_start   = e_start + n_history
        mom_start = p_start + n_history 

        rho = x_tensor[:, rho_start + step_idx : rho_start + step_idx + 1]
        e   = x_tensor[:, e_start   + step_idx : e_start   + step_idx + 1]
        p   = x_tensor[:, p_start   + step_idx : p_start   + step_idx + 1]
        
        m_idx = mom_start + (step_idx * 2)
        mom = x_tensor[:, m_idx : m_idx + 2]

        return torch.cat([rho, e, p, mom], dim=1)
    
    # Helper to extract the INITIAL grouped window for the model
    def get_initial_window(x_tensor):
        # We need the first 'model_in_steps' from the dataset X
        n_history = dataset_history_steps
        m_steps = model_in_steps

        # Offsets in dataset X
        rho_start = 0
        e_start   = rho_start + n_history
        p_start   = e_start + n_history
        mom_start = p_start + n_history 

        rho = x_tensor[:, rho_start : rho_start + m_steps]
        e   = x_tensor[:, e_start   : e_start   + m_steps]
        p   = x_tensor[:, p_start   : p_start   + m_steps]
        mom = x_tensor[:, mom_start : mom_start + 2*m_steps]

        return torch.cat([rho, e, p, mom], dim=1)


    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        # initialize slliding window with the first M steps from Batch X
        current_window = get_initial_window(batch.x)
        
        loss = 0.0
        
        # metric accumulators
        sum_d, sum_e, sum_p, sum_m = 0.0, 0.0, 0.0, 0.0
        
        #variable to store first step predictions for logging
        preds_1 = None 

        for k in range(1, n_unroll_steps + 1):
            # In general at iter k, predict step index: (model_in_steps + k - 1)
            target_step_idx = model_in_steps + k - 1

            if target_step_idx < dataset_history_steps:
                # Target is available in batch.x
                u_target = get_state_at_step(batch.x, target_step_idx)
            else:
                # Final target is in batch.y
                y_vec = torch.cat([
                    getattr(batch, "y_density", None).view(-1, 1),
                    getattr(batch, "y_energy", None).view(-1, 1),
                    getattr(batch, "y_pressure", None).view(-1, 1),
                    getattr(batch, "y_momentum", None).view(-1, 2)
                ], dim=1)

                if target_type == "delta":
                    # Reconstruction: Target = Previous_State + Delta
                    # Previous state is at index (target_step_idx - 1)
                    prev_idx = target_step_idx - 1
                    u_prev = get_state_at_step(batch.x, prev_idx)
                    u_target = u_prev + y_vec
                else:
                    u_target = y_vec

            # prepare input batch for model
            batch_k = batch.clone()
            batch_k.x = current_window # wliding window input

            # Noise injection (Only on first step input)
            if k == 1 and noise_std > 0.0:
                 noise = torch.randn_like(current_window) * noise_std
                 batch_k.x = batch_k.x + noise
            
            # forward Step
            preds_k = model(batch_k, stats)
            
            # Save step 1 for logging later
            if k == 1:
                preds_1 = preds_k

            # reconstruct prediction
            u_pred = torch.cat([
                preds_k["density"].unsqueeze(-1),
                preds_k["energy"].unsqueeze(-1),
                preds_k["pressure"].unsqueeze(-1),
                preds_k["momentum"]
            ], dim=1)

            # compute loss for step k
            l_rho = F.mse_loss(u_pred[:, 0], u_target[:, 0])
            l_e   = F.mse_loss(u_pred[:, 1], u_target[:, 1])
            l_p   = F.mse_loss(u_pred[:, 2], u_target[:, 2])
            l_mom = F.mse_loss(u_pred[:, 3:5], u_target[:, 3:5])

            step_loss = (loss_weights.get("density") * l_rho +
                         loss_weights.get("energy") * l_e +
                         loss_weights.get("pressure") * l_p +
                         loss_weights.get("momentum") * l_mom)
            
            # weighting: heavily penalize the final step
            step_weight = 10.0 if k == n_unroll_steps else 1.0
            loss += step_weight * step_loss

            # Update metrics
            sum_d += l_rho
            sum_e += l_e
            sum_p += l_p
            sum_m += l_mom
            
            # update window for next iteration (autoregressive slide)
            # drop oldest history, append u_pred
            current_window = slide_window(current_window, u_pred, model_in_steps)

        # average metrics across steps for logging
        mse_density  = sum_d / n_unroll_steps
        mse_energy   = sum_e / n_unroll_steps
        mse_pressure = sum_p / n_unroll_steps
        mse_momentum = sum_m / n_unroll_steps

        if not torch.isfinite(loss):
            print(f"\n⚠️ Non-finite loss (Inf/NaN) at step {step+1}: {loss.item()}. Skipping batch.")
            optimizer.zero_grad() 
            continue 

        optimizer.zero_grad()
        loss.backward()

        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"\n⚠️ NaN gradients detected at step {step+1}. Skipping optimizer step.")
            optimizer.zero_grad()
            continue 

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
        optimizer.step()
        
        # check CFL on step 1
        dt_used = sum(preds_1["dt_layers"])
        dt_target = 0.015
        if dt_used < (dt_target * 0.99):
            print(f"⚠️ CLIPPING DETECTED: Model evolved {dt_used:.5f}s, target {dt_target:.5f}s")

        # monitor alpha
        current_alpha = preds_1.get("mean_alpha", torch.tensor(0.0)).item()

        print(f"\nCurrent step: {step+1}/{len(dataloader)}")
        print(f"Batch Loss: {loss.item():.6f} "
              f"(Density MSE: {mse_density.item():.6f}, "
              f"Energy MSE: {mse_energy.item():.6f}, "
              f"Pressure MSE: {mse_pressure.item():.6f}, "
              f"Momentum MSE: {mse_momentum.item():.6f})")
        print(f"Current Alpha: {current_alpha:.6f}")
        
        batch_losses.append(loss.item())

        n_nodes = current_window.shape[0]
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