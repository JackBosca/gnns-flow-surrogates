import numpy as np
import torch


def teacher_forcing_schedule(epoch, epochs, start=1.0, end=0.0):
    """
    Compute the scheduled sampling probability for the given epoch.
    Linear decay from start to end over the epochs.
    Args:
        epoch (int): Current epoch (1-based).
        epochs (int): Total number of epochs.
        start (float): Starting scheduled sampling probability.
        end (float): Ending scheduled sampling probability.
    Returns:
        float: Scheduled sampling probability for the current epoch.
    """
    denom = max(1, epochs - 1)
    
    # linear fraction from 0 (start) to 1 (end)
    frac = (epoch - 1) / denom
    frac = min(1.0, max(0.0, frac))  # clamp to [0, 1]
    
    # linear interpolation between start and end
    tf_prob = max(end, start + (end - start) * frac)
    return tf_prob

# def autoregressive_pred(model, stats, batch, Hc, Wc, C_in, target_type, start, end):
#     """Perform one autoregressive prediction step by mixing predicted last-step
#     into the input features, dropping the oldest timestep."""
#     # unpack start and end
#     d_start, e_start, p_start, m_start = start[0], start[1], start[2], start[3]
#     d_end, e_end, p_end, m_end = end[0], end[1], end[2], end[3]

#     # run one no-grad forward on the original batch to get predicted last-step
#     with torch.no_grad():
#         preds_model = model(batch, stats)

#     # reshape predictions to grid shapes (Hc, Wc) or (Hc,Wc,2)
#     N = batch.x.shape[0]
#     # sanity check for grid size
#     assert N == Hc * Wc, f"Node count mismatch: N={N}, Hc*Wc={Hc*Wc}"

#     p_d = preds_model["density"].detach().cpu().numpy().reshape(Hc, Wc)
#     p_e = preds_model["energy"].detach().cpu().numpy().reshape(Hc, Wc)
#     p_p = preds_model["pressure"].detach().cpu().numpy().reshape(Hc, Wc)
#     p_m = preds_model["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2)

#     # build input-channel array from batch.x so then can modify last input channels
#     x_nodes = batch.x.detach().cpu().numpy()  # (N, C_in)
#     x_np = x_nodes.T.reshape(C_in, Hc, Wc)    # (C_in, Hc, Wc)

#     # reconstruct predicted "last" in the same (normalized) space the model uses
#     if target_type == "delta":
#         # need the "first" timestep in the input to reconstruct delta -> last when target=="delta"
#         # the first density channel corresponds to time t0
#         first_density = x_np[d_start]  # (Hc,Wc)
#         first_energy = x_np[e_start]
#         first_pressure = x_np[p_start]
#         # momentum channels layout: t0_x, t0_y, t1_x, t1_y, ...
#         # to get momentum[0] (first timestep) need to reconstruct from m_start indices 0 and 1
#         first_mom_x = x_np[m_start]
#         first_mom_y = x_np[m_start + 1]
#         first_momentum = np.stack([first_mom_x, first_mom_y], axis=-1)  # (Hc,Wc,2)

#         pred_last_density  = first_density + p_d
#         pred_last_energy   = first_energy + p_e
#         pred_last_pressure = first_pressure + p_p
#         pred_last_momentum = first_momentum + p_m
#     else:
#         pred_last_density  = p_d
#         pred_last_energy   = p_e
#         pred_last_pressure = p_p
#         pred_last_momentum = p_m

#     # shift-and-append logic: drop the oldest timestep, shift the window forward by one,
#     # and append the predicted last timestep as the new most-recent input
#     # density: shape (d_count, Hc, Wc)
#     x_np[d_start:d_end] = np.concatenate(
#         [x_np[d_start + 1:d_end], np.expand_dims(pred_last_density, axis=0)],
#         axis=0)
#     # energy
#     x_np[e_start:e_end] = np.concatenate(
#         [x_np[e_start + 1:e_end], np.expand_dims(pred_last_energy, axis=0)],
#         axis=0)
#     # pressure
#     x_np[p_start:p_end] = np.concatenate(
#         [x_np[p_start + 1:p_end], np.expand_dims(pred_last_pressure, axis=0)],
#         axis=0)
#     # momentum: channels are [t0_x,t0_y, t1_x,t1_y, ...], so drop first 2 channels and append pred x,y
#     x_np[m_start:m_end] = np.concatenate(
#         [x_np[m_start + 2:m_end],                                 # shifted previous momentum channels
#         np.expand_dims(pred_last_momentum[..., 0], axis=0),       # append predicted momentum x
#         np.expand_dims(pred_last_momentum[..., 1], axis=0)],      # append predicted momentum y
#         axis=0)

#     # convert back to node-feature layout and create a mixed batch
#     x_mixed_nodes = torch.tensor(x_np.reshape(C_in, Hc * Wc).T, dtype=batch.x.dtype, device=batch.x.device)

#     # clone batch and replace x
#     batch_mixed = batch.clone()
#     batch_mixed.x = x_mixed_nodes
#     return batch_mixed

def autoregressive_pred(model, stats, batch, Hc, Wc, C_in, target_type, start, end):
    """
    Perform one autoregressive prediction step.
    
    NOTE: FluxGNN returns absolute values in 'density', 'energy', etc., 
    even if trained on 'delta' targets. Therefore, we simply append 
    the model predictions directly.
    """
    # unpack start and end indices for the input channels
    d_start, e_start, p_start, m_start = start[0], start[1], start[2], start[3]
    d_end, e_end, p_end, m_end = end[0], end[1], end[2], end[3]

    # run one no-grad forward
    with torch.no_grad():
        preds_model = model(batch, stats)

    # 1. Get Predicted Absolute Values (Normalized)
    # FluxGNN returns these as flat tensors, reshape to grid
    p_d = preds_model["density"].detach().cpu().numpy().reshape(Hc, Wc)
    p_e = preds_model["energy"].detach().cpu().numpy().reshape(Hc, Wc)
    p_p = preds_model["pressure"].detach().cpu().numpy().reshape(Hc, Wc)
    p_m = preds_model["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2)

    # 2. Prepare Input Array
    x_nodes = batch.x.detach().cpu().numpy()  # (N, C_in)
    x_np = x_nodes.T.reshape(C_in, Hc, Wc)    # (C_in, Hc, Wc)

    # 3. Shift-and-Append Logic
    # We drop the oldest timestep (index start) and append the new prediction (index end)
    
    # Density
    x_np[d_start:d_end] = np.concatenate(
        [x_np[d_start + 1:d_end], np.expand_dims(p_d, axis=0)], axis=0)
    
    # Energy
    x_np[e_start:e_end] = np.concatenate(
        [x_np[e_start + 1:e_end], np.expand_dims(p_e, axis=0)], axis=0)
    
    # Pressure
    x_np[p_start:p_end] = np.concatenate(
        [x_np[p_start + 1:p_end], np.expand_dims(p_p, axis=0)], axis=0)
    
    # Momentum (channels: t0_x, t0_y, t1_x, t1_y...)
    x_np[m_start:m_end] = np.concatenate(
        [x_np[m_start + 2:m_end], 
         np.expand_dims(p_m[..., 0], axis=0), 
         np.expand_dims(p_m[..., 1], axis=0)], axis=0)

    # 4. Reconstruct Batch
    x_mixed_nodes = torch.tensor(x_np.reshape(C_in, Hc * Wc).T, 
                               dtype=batch.x.dtype, device=batch.x.device)

    batch_mixed = batch.clone()
    batch_mixed.x = x_mixed_nodes
    return batch_mixed


def denorm(rho, e, rhou, stats):
    """De-normalize the model outputs using provided dataset statistics."""
    means = stats["mean"]
    stds = stats["std"]

    # density
    rho_denorm = rho * stds["density"] + means["density"]
    # energy
    e_denorm = e * stds["energy"] + means["energy"]
    
    # momentum
    m_mean = means["momentum"]
    m_std = stds["momentum"]

    # FIX: Convert list to tensor if necessary
    if isinstance(m_mean, list):
        m_mean = torch.tensor(m_mean, device=rhou.device, dtype=rhou.dtype)
    if isinstance(m_std, list):
        m_std = torch.tensor(m_std, device=rhou.device, dtype=rhou.dtype)

    # Now we have (N, 2) * (2,) -> Broadcasts correctly
    rhou_denorm = rhou * m_std + m_mean

    return rho_denorm, e_denorm, rhou_denorm

def norm(rho, e, p, rhou, stats):
    """Normalize the model inputs using provided dataset statistics."""
    means = stats["mean"]
    stds = stats["std"]

    # density
    if rho is not None:
        rho_norm = (rho - means["density"]) / stds["density"]
    else:
        rho_norm = None
        
    # energy
    if e is not None:
        e_norm = (e - means["energy"]) / stds["energy"]
    else:
        e_norm = None
        
    # pressure
    if p is not None:
        p_norm = (p - means["pressure"]) / stds["pressure"]
    else:
        p_norm = None
        
    # momentum
    if rhou is not None:
        m_mean = means["momentum"]
        m_std = stds["momentum"]

        # FIX: Convert list to tensor if necessary
        if isinstance(m_mean, list):
            m_mean = torch.tensor(m_mean, device=rhou.device, dtype=rhou.dtype)
        if isinstance(m_std, list):
            m_std = torch.tensor(m_std, device=rhou.device, dtype=rhou.dtype)

        rhou_norm = (rhou - m_mean) / m_std
    else:
        rhou_norm = None

    return rho_norm, e_norm, p_norm, rhou_norm

def eos(density, energy, momentum, gamma):
    """
    Equation of state: compute pressure from density, energy, momentum.
    All quantities must be in physical (denormalized) units.
    """
    # kinetic energy = 0.5 * (rho*u)^2 / rho
    kinetic_energy = 0.5 * torch.sum(momentum**2, dim=-1) / (density + 1e-8)
    
    internal_energy = energy - kinetic_energy
    
    # internal energy cannot be negative physically
    internal_energy = torch.clamp(internal_energy, min=1e-6)
    
    pressure = (gamma - 1.0) * internal_energy
    return pressure
