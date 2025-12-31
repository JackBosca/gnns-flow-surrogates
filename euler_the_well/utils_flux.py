from typing import Optional, Dict
import torch
import torch.nn.functional as F

# def train_one_epoch_unrolled(model, dataloader, stats, optimizer, device,
#                              loss_weights: Optional[Dict[str, float]] = None,
#                              clip_grad: Optional[float] = None,
#                              noise_std: float = 0.01):
#     """
#     Performs 2-Step Unrolled Training (t0 -> t1 -> t2).
#     Requires dataset with time_window=3.
#     input: t0 -> pred: t1 (Loss vs GT t1) -> input: pred_t1 -> pred: t2 (Loss vs GT t2)
#     """
#     model.train()
#     model.to(device)

#     if loss_weights is None:
#         loss_weights = {"density": 1.0, "energy": 1.0, "pressure": 1.0, "momentum": 5.0}

#     total_loss = 0.0
#     total_mse_density = 0.0
#     total_mse_energy = 0.0
#     total_mse_pressure = 0.0
#     total_mse_momentum = 0.0
#     total_nodes = 0

#     batch_losses = []

#     ds_wrap = dataloader.dataset
#     if hasattr(ds_wrap, "datasets"):
#         base_ds = ds_wrap.datasets[0]
#     else:
#         base_ds = ds_wrap

#     # Validation check for Unrolling: We need t0, t1, t2 loaded
#     t_w = int(base_ds.time_window)
#     if t_w != 3:
#         raise ValueError(f"Unrolled training requires time_window=3, but got {t_w}")

#     target_type = str(base_ds.target)

#     # Helper to extract a (N, 5) state vector from the interleaved batch.x
#     # x layout for window=3: [rho_0, rho_1, e_0, e_1, p_0, p_1, mom_0, mom_1]
#     def get_state_at_step(x_tensor, step_idx):
#         n_steps = 2 # input x contains 2 steps
        
#         # Offsets based on dataset concatenation order
#         rho_start = 0
#         e_start   = rho_start + n_steps
#         p_start   = e_start + n_steps
#         mom_start = p_start + n_steps 

#         rho = x_tensor[:, rho_start + step_idx : rho_start + step_idx + 1]
#         e   = x_tensor[:, e_start   + step_idx : e_start   + step_idx + 1]
#         p   = x_tensor[:, p_start   + step_idx : p_start   + step_idx + 1]
        
#         # Momentum indices (2 channels per step)
#         m_idx = mom_start + (step_idx * 2)
#         mom = x_tensor[:, m_idx : m_idx + 2]

#         return torch.cat([rho, e, p, mom], dim=1)

#     for step, batch in enumerate(dataloader):
#         batch = batch.to(device)

#         # ---------------------------------------------------------------------
#         # 1. Prepare Ground Truths (t0, t1, t2)
#         # ---------------------------------------------------------------------
        
#         # Input State t0
#         u_0 = get_state_at_step(batch.x, 0)
        
#         # Target State t1 (Also Input for Step 2)
#         u_1 = get_state_at_step(batch.x, 1)
        
#         # Target State t2
#         # Reconstruct from batch.y
#         y_vec = torch.cat([
#             getattr(batch, "y_density", None).view(-1, 1),
#             getattr(batch, "y_energy", None).view(-1, 1),
#             getattr(batch, "y_pressure", None).view(-1, 1),
#             getattr(batch, "y_momentum", None).view(-1, 2)
#         ], dim=1)

#         if target_type == "delta":
#             # Dataset gives y = (t2 - t0). So t2 = u_0 + y
#             u_2 = u_0 + y_vec
#         else:
#             # Dataset gives y = t2
#             u_2 = y_vec

#         # ---------------------------------------------------------------------
#         # 2. STEP 1: Predict t0 -> t1
#         # ---------------------------------------------------------------------
        
#         # Prepare Input Batch
#         batch_1 = batch.clone()
#         batch_1.x = u_0 # Input is t0
        
#         # Noise Injection (Only on Step 1)
#         if noise_std > 0.0:
#             noise = torch.randn_like(u_0) * noise_std
#             batch_1.x = batch_1.x + noise

#         # Forward Step 1
#         preds_1 = model(batch_1, stats)
        
#         # Reconstruct Step 1 Prediction (Absolute State)
#         u_1_pred = torch.cat([
#             preds_1["density"].unsqueeze(-1),
#             preds_1["energy"].unsqueeze(-1),
#             preds_1["pressure"].unsqueeze(-1),
#             preds_1["momentum"]
#         ], dim=1)

#         # ---------------------------------------------------------------------
#         # 3. STEP 2: Predict t1_pred -> t2 (Autoregressive)
#         # ---------------------------------------------------------------------
        
#         # Input is the PREDICTION from Step 1 (Maintains Gradient Chain)
#         batch_2 = batch.clone()
#         batch_2.x = u_1_pred # Input is predicted t1
        
#         # Forward Step 2 (No Noise)
#         preds_2 = model(batch_2, stats)
        
#         # Reconstruct Step 2 Prediction (Absolute State)
#         u_2_pred = torch.cat([
#             preds_2["density"].unsqueeze(-1),
#             preds_2["energy"].unsqueeze(-1),
#             preds_2["pressure"].unsqueeze(-1),
#             preds_2["momentum"]
#         ], dim=1)

#         # ---------------------------------------------------------------------
#         # 4. Compute Loss (Sum of both steps)
#         # ---------------------------------------------------------------------
        
#         # Helper to compute weighted loss for a step
#         def compute_step_loss(pred_state, target_state):
#             l_rho = F.mse_loss(pred_state[:, 0], target_state[:, 0])
#             l_e   = F.mse_loss(pred_state[:, 1], target_state[:, 1])
#             l_p   = F.mse_loss(pred_state[:, 2], target_state[:, 2])
#             l_mom = F.mse_loss(pred_state[:, 3:5], target_state[:, 3:5])
            
#             w_loss = (loss_weights.get("density") * l_rho +
#                       loss_weights.get("energy") * l_e +
#                       loss_weights.get("pressure") * l_p +
#                       loss_weights.get("momentum") * l_mom)
#             return w_loss, l_rho, l_e, l_p, l_mom

#         # Loss 1 (t1)
#         loss_1, d1, e1, p1, m1 = compute_step_loss(u_1_pred, u_1)
        
#         # Loss 2 (t2)
#         loss_2, d2, e2, p2, m2 = compute_step_loss(u_2_pred, u_2)

#         # Total Loss
#         # loss = loss_1 + loss_2
#         loss = loss_1 + 10.0 * loss_2 # weight more loss 2 to try to prevent slowness

#         # Average metrics for logging
#         mse_density = (d1 + d2) / 2
#         mse_energy  = (e1 + e2) / 2
#         mse_pressure = (p1 + p2) / 2
#         mse_momentum = (m1 + m2) / 2

#         # ---------------------------------------------------------------------
#         # 5. Optimization & Checks
#         # ---------------------------------------------------------------------

#         if not torch.isfinite(loss):
#             print(f"\n⚠️ Non-finite loss (Inf/NaN) at step {step+1}: {loss.item()}. Skipping batch.")
#             optimizer.zero_grad() 
#             continue 

#         optimizer.zero_grad()
#         loss.backward()

#         has_nan_grad = False
#         for param in model.parameters():
#             if param.grad is not None and not torch.isfinite(param.grad).all():
#                 has_nan_grad = True
#                 break
        
#         if has_nan_grad:
#             print(f"\n⚠️ NaN gradients detected at step {step+1}. Skipping optimizer step.")
#             optimizer.zero_grad()
#             continue 

#         if clip_grad is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
#         optimizer.step()

#         # ---------------------------------------------------------------------
#         # 6. Logging
#         # ---------------------------------------------------------------------
        
#         # Check CFL on Step 1 (Diagnostic)
#         dt_used = sum(preds_1["dt_layers"])
#         dt_target = 0.015
#         if dt_used < (dt_target * 0.99):
#             print(f"⚠️ CLIPPING DETECTED: Model evolved {dt_used:.5f}s, target {dt_target:.5f}s")

#         # Monitor Alpha (from Step 1)
#         current_alpha = preds_1.get("mean_alpha", torch.tensor(0.0)).item()

#         print(f"\nCurrent step: {step+1}/{len(dataloader)}")
#         print(f"Batch Loss: {loss.item():.6f} "
#               f"(Density MSE: {mse_density.item():.6f}, "
#               f"Energy MSE: {mse_energy.item():.6f}, "
#               f"Pressure MSE: {mse_pressure.item():.6f}, "
#               f"Momentum MSE: {mse_momentum.item():.6f})")
#         print(f"Current Alpha: {current_alpha:.6f}")
        
#         batch_losses.append(loss.item())

#         n_nodes = u_0.shape[0]
#         total_nodes += n_nodes
#         total_loss += float(loss.detach()) * n_nodes
#         total_mse_density += float(mse_density.detach()) * n_nodes
#         total_mse_energy  += float(mse_energy.detach()) * n_nodes
#         total_mse_pressure+= float(mse_pressure.detach()) * n_nodes
#         total_mse_momentum+= float(mse_momentum.detach()) * n_nodes

#     return {
#         "loss": total_loss / total_nodes,
#         "mse_density": total_mse_density / total_nodes,
#         "mse_energy": total_mse_energy / total_nodes,
#         "mse_pressure": total_mse_pressure / total_nodes,
#         "mse_momentum": total_mse_momentum / total_nodes,
#         "num_nodes": total_nodes,
#         "batch_losses": batch_losses
#     }


# def train_one_epoch_unrolled(model, dataloader, stats, optimizer, device,
#                              loss_weights: Optional[Dict[str, float]] = None,
#                              clip_grad: Optional[float] = None,
#                              noise_std: float = 0.01,
#                              n_unroll_steps: int = 10):
#     """
#     Performs N-Step Unrolled Training (t0 -> t1 -> ... -> tN).
#     Requires dataset with time_window >= n_unroll_steps + 1.
#     """
#     model.train()
#     model.to(device)

#     if loss_weights is None:
#         loss_weights = {"density": 1.0, "energy": 1.0, "pressure": 1.0, "momentum": 5.0}

#     total_loss = 0.0
#     total_mse_density = 0.0
#     total_mse_energy = 0.0
#     total_mse_pressure = 0.0
#     total_mse_momentum = 0.0
#     total_nodes = 0

#     batch_losses = []

#     ds_wrap = dataloader.dataset
#     if hasattr(ds_wrap, "datasets"):
#         base_ds = ds_wrap.datasets[0]
#     else:
#         base_ds = ds_wrap

#     # Validation check for Unrolling: We need t0 + N targets
#     t_w = int(base_ds.time_window)
#     if t_w < n_unroll_steps + 1:
#         raise ValueError(f"Unrolled training requires time_window >= {n_unroll_steps+1}, but got {t_w}")

#     target_type = str(base_ds.target)

#     # Helper to extract a (N, 5) state vector from the interleaved batch.x
#     def get_state_at_step(x_tensor, step_idx):
#         # x contains history steps (time_window - 1)
#         n_history = t_w - 1 
        
#         # Offsets based on dataset concatenation order
#         rho_start = 0
#         e_start   = rho_start + n_history
#         p_start   = e_start + n_history
#         mom_start = p_start + n_history 

#         rho = x_tensor[:, rho_start + step_idx : rho_start + step_idx + 1]
#         e   = x_tensor[:, e_start   + step_idx : e_start   + step_idx + 1]
#         p   = x_tensor[:, p_start   + step_idx : p_start   + step_idx + 1]
        
#         # Momentum indices (2 channels per step)
#         m_idx = mom_start + (step_idx * 2)
#         mom = x_tensor[:, m_idx : m_idx + 2]

#         return torch.cat([rho, e, p, mom], dim=1)

#     for step, batch in enumerate(dataloader):
#         batch = batch.to(device)

#         # ---------------------------------------------------------------------
#         # 1. Unrolled Prediction Loop (t0 -> t1 -> ... -> tN)
#         # ---------------------------------------------------------------------

#         # Initial Input State t0
#         current_u = get_state_at_step(batch.x, 0)
        
#         loss = 0.0
        
#         # Metric accumulators
#         sum_d, sum_e, sum_p, sum_m = 0.0, 0.0, 0.0, 0.0
        
#         # Variable to store first step predictions for logging
#         preds_1 = None 

#         for k in range(1, n_unroll_steps + 1):
#             # A. Prepare Target for Step k
#             if k < n_unroll_steps:
#                 # Intermediate targets are available in batch.x
#                 u_target = get_state_at_step(batch.x, k)
#             else:
#                 # Final target (tN) is in batch.y
#                 y_vec = torch.cat([
#                     getattr(batch, "y_density", None).view(-1, 1),
#                     getattr(batch, "y_energy", None).view(-1, 1),
#                     getattr(batch, "y_pressure", None).view(-1, 1),
#                     getattr(batch, "y_momentum", None).view(-1, 2)
#                 ], dim=1)

#                 if target_type == "delta":
#                     # dataset gives y = (tN - tN-1). So tN = u_(tN-1) + y
#                     last_input_idx = int(base_ds.time_window) - 2
#                     u_target = get_state_at_step(batch.x, last_input_idx) + y_vec
#                 else:
#                     u_target = y_vec

#             # B. Prepare Input Batch for Model
#             batch_k = batch.clone()
#             batch_k.x = current_u # Input is prediction from previous step (or t0)

#             # Noise Injection (Only on first step input)
#             if k == 1 and noise_std > 0.0:
#                  noise = torch.randn_like(current_u) * noise_std
#                  batch_k.x = batch_k.x + noise
            
#             # C. Forward Step
#             preds_k = model(batch_k, stats)
            
#             # Save step 1 for logging later
#             if k == 1:
#                 preds_1 = preds_k

#             # D. Reconstruct Prediction
#             u_pred = torch.cat([
#                 preds_k["density"].unsqueeze(-1),
#                 preds_k["energy"].unsqueeze(-1),
#                 preds_k["pressure"].unsqueeze(-1),
#                 preds_k["momentum"]
#             ], dim=1)

#             # E. Compute Loss for Step k
#             l_rho = F.mse_loss(u_pred[:, 0], u_target[:, 0])
#             l_e   = F.mse_loss(u_pred[:, 1], u_target[:, 1])
#             l_p   = F.mse_loss(u_pred[:, 2], u_target[:, 2])
#             l_mom = F.mse_loss(u_pred[:, 3:5], u_target[:, 3:5])

#             step_loss = (loss_weights.get("density") * l_rho +
#                          loss_weights.get("energy") * l_e +
#                          loss_weights.get("pressure") * l_p +
#                          loss_weights.get("momentum") * l_mom)
            
#             # Weighting: Heavily penalize the final step (similar to your 10.0 * loss_2)
#             step_weight = 10.0 if k == n_unroll_steps else 1.0
#             loss += step_weight * step_loss

#             # Update metrics
#             sum_d += l_rho
#             sum_e += l_e
#             sum_p += l_p
#             sum_m += l_mom
            
#             # F. Update current_u for next iteration (Autoregressive)
#             current_u = u_pred

#         # Average metrics across steps for logging
#         mse_density  = sum_d / n_unroll_steps
#         mse_energy   = sum_e / n_unroll_steps
#         mse_pressure = sum_p / n_unroll_steps
#         mse_momentum = sum_m / n_unroll_steps

#         # ---------------------------------------------------------------------
#         # 2. Optimization & Checks
#         # ---------------------------------------------------------------------

#         if not torch.isfinite(loss):
#             print(f"\n⚠️ Non-finite loss (Inf/NaN) at step {step+1}: {loss.item()}. Skipping batch.")
#             optimizer.zero_grad() 
#             continue 

#         optimizer.zero_grad()
#         loss.backward()

#         has_nan_grad = False
#         for param in model.parameters():
#             if param.grad is not None and not torch.isfinite(param.grad).all():
#                 has_nan_grad = True
#                 break
        
#         if has_nan_grad:
#             print(f"\n⚠️ NaN gradients detected at step {step+1}. Skipping optimizer step.")
#             optimizer.zero_grad()
#             continue 

#         if clip_grad is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
#         optimizer.step()

#         # ---------------------------------------------------------------------
#         # 6. Logging
#         # ---------------------------------------------------------------------
        
#         # Check CFL on Step 1 (Diagnostic)
#         dt_used = sum(preds_1["dt_layers"])
#         dt_target = 0.015
#         if dt_used < (dt_target * 0.99):
#             print(f"⚠️ CLIPPING DETECTED: Model evolved {dt_used:.5f}s, target {dt_target:.5f}s")

#         # Monitor Alpha (from Step 1)
#         current_alpha = preds_1.get("mean_alpha", torch.tensor(0.0)).item()

#         print(f"\nCurrent step: {step+1}/{len(dataloader)}")
#         print(f"Batch Loss: {loss.item():.6f} "
#               f"(Density MSE: {mse_density.item():.6f}, "
#               f"Energy MSE: {mse_energy.item():.6f}, "
#               f"Pressure MSE: {mse_pressure.item():.6f}, "
#               f"Momentum MSE: {mse_momentum.item():.6f})")
#         print(f"Current Alpha: {current_alpha:.6f}")
        
#         batch_losses.append(loss.item())

#         n_nodes = current_u.shape[0]
#         total_nodes += n_nodes
#         total_loss += float(loss.detach()) * n_nodes
#         total_mse_density += float(mse_density.detach()) * n_nodes
#         total_mse_energy  += float(mse_energy.detach()) * n_nodes
#         total_mse_pressure+= float(mse_pressure.detach()) * n_nodes
#         total_mse_momentum+= float(mse_momentum.detach()) * n_nodes

#     return {
#         "loss": total_loss / total_nodes,
#         "mse_density": total_mse_density / total_nodes,
#         "mse_energy": total_mse_energy / total_nodes,
#         "mse_pressure": total_mse_pressure / total_nodes,
#         "mse_momentum": total_mse_momentum / total_nodes,
#         "num_nodes": total_nodes,
#         "batch_losses": batch_losses
#     }


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

    # Determine Model's expected history length from node_in_dim
    # C_in = steps * 5. So steps = node_in_dim // 5
    model_in_steps = model.node_in_dim // 5

    # Validation: Dataset must provide enough history for the model + future targets
    # We need 'model_in_steps' to start, plus 'n_unroll_steps' targets.
    # The dataset provides 't_w' total steps (t_w-1 in X, 1 in Y).
    if t_w < model_in_steps + n_unroll_steps:
        raise ValueError(f"Dataset time_window ({t_w}) too small. "
                         f"Model needs {model_in_steps} inputs + {n_unroll_steps} unroll targets. "
                         f"Increase dataset time_window to >= {model_in_steps + n_unroll_steps}.")

    # [Diagram: Sliding Window Update]
    # Old Window (Grouped): [Rho_0..Rho_T | E_0..E_T | P_0..P_T | Mom_0..Mom_T]
    # New Prediction:       [Rho_new,      E_new,     P_new,     Mom_new]
    # New Window:           Drop index 0 of each group, Append new.
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

        # ---------------------------------------------------------------------
        # 1. Unrolled Prediction Loop
        # ---------------------------------------------------------------------

        # Initialize Sliding Window with the first M steps from Batch X
        current_window = get_initial_window(batch.x)
        
        loss = 0.0
        
        # Metric accumulators
        sum_d, sum_e, sum_p, sum_m = 0.0, 0.0, 0.0, 0.0
        
        # Variable to store first step predictions for logging
        preds_1 = None 

        for k in range(1, n_unroll_steps + 1):
            
            # A. Prepare Target for Step k
            # The model has seen steps [0 ... model_in_steps-1].
            # It predicts step [model_in_steps].
            # In general at iter k (1-based), we predict step index: (model_in_steps + k - 1)
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

            # B. Prepare Input Batch for Model
            batch_k = batch.clone()
            batch_k.x = current_window # Sliding window input

            # Noise Injection (Only on first step input)
            if k == 1 and noise_std > 0.0:
                 noise = torch.randn_like(current_window) * noise_std
                 batch_k.x = batch_k.x + noise
            
            # C. Forward Step
            preds_k = model(batch_k, stats)
            
            # Save step 1 for logging later
            if k == 1:
                preds_1 = preds_k

            # D. Reconstruct Prediction
            u_pred = torch.cat([
                preds_k["density"].unsqueeze(-1),
                preds_k["energy"].unsqueeze(-1),
                preds_k["pressure"].unsqueeze(-1),
                preds_k["momentum"]
            ], dim=1)

            # E. Compute Loss for Step k
            l_rho = F.mse_loss(u_pred[:, 0], u_target[:, 0])
            l_e   = F.mse_loss(u_pred[:, 1], u_target[:, 1])
            l_p   = F.mse_loss(u_pred[:, 2], u_target[:, 2])
            l_mom = F.mse_loss(u_pred[:, 3:5], u_target[:, 3:5])

            step_loss = (loss_weights.get("density") * l_rho +
                         loss_weights.get("energy") * l_e +
                         loss_weights.get("pressure") * l_p +
                         loss_weights.get("momentum") * l_mom)
            
            # Weighting: Heavily penalize the final step
            step_weight = 10.0 if k == n_unroll_steps else 1.0
            loss += step_weight * step_loss

            # Update metrics
            sum_d += l_rho
            sum_e += l_e
            sum_p += l_p
            sum_m += l_mom
            
            # F. Update Window for next iteration (Autoregressive Slide)
            # Drop oldest history, append u_pred
            current_window = slide_window(current_window, u_pred, model_in_steps)

        # Average metrics across steps for logging
        mse_density  = sum_d / n_unroll_steps
        mse_energy   = sum_e / n_unroll_steps
        mse_pressure = sum_p / n_unroll_steps
        mse_momentum = sum_m / n_unroll_steps

        # ---------------------------------------------------------------------
        # 2. Optimization & Checks
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # 6. Logging
        # ---------------------------------------------------------------------
        
        # Check CFL on Step 1 (Diagnostic)
        dt_used = sum(preds_1["dt_layers"])
        dt_target = 0.015
        if dt_used < (dt_target * 0.99):
            print(f"⚠️ CLIPPING DETECTED: Model evolved {dt_used:.5f}s, target {dt_target:.5f}s")

        # Monitor Alpha (from Step 1)
        current_alpha = preds_1.get("mean_alpha", torch.tensor(0.0)).item()

        print(f"\nCurrent step: {step+1}/{len(dataloader)}")
        print(f"Batch Loss: {loss.item():.6f} "
              f"(Density MSE: {mse_density.item():.6f}, "
              f"Energy MSE: {mse_energy.item():.6f}, "
              f"Pressure MSE: {mse_pressure.item():.6f}, "
              f"Momentum MSE: {mse_momentum.item():.6f})")
        print(f"Current Alpha: {current_alpha:.6f}")
        
        batch_losses.append(loss.item())

        n_nodes = current_window.shape[0] # roughly
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