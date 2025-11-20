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

def autoregressive_pred(model, batch, Hc, Wc, C_in, target_type, start, end):
    """Perform one autoregressive prediction step by mixing predicted last-step
    into the input features, dropping the oldest timestep."""
    # unpack start and end
    d_start, e_start, p_start, m_start = start[0], start[1], start[2], start[3]
    d_end, e_end, p_end, m_end = end[0], end[1], end[2], end[3]

    # run one no-grad forward on the original batch to get predicted last-step
    with torch.no_grad():
        preds_model = model(batch)

    # reshape predictions to grid shapes (Hc, Wc) or (Hc,Wc,2)
    N = batch.x.shape[0]
    # sanity check for grid size
    assert N == Hc * Wc, f"Node count mismatch: N={N}, Hc*Wc={Hc*Wc}"

    p_d = preds_model["density"].detach().cpu().numpy().reshape(Hc, Wc)
    p_e = preds_model["energy"].detach().cpu().numpy().reshape(Hc, Wc)
    p_p = preds_model["pressure"].detach().cpu().numpy().reshape(Hc, Wc)
    p_m = preds_model["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2)

    # build input-channel array from batch.x so then can modify last input channels
    x_nodes = batch.x.detach().cpu().numpy()  # (N, C_in)
    x_np = x_nodes.T.reshape(C_in, Hc, Wc)    # (C_in, Hc, Wc)

    # reconstruct predicted "last" in the same (normalized) space the model uses
    if target_type == "delta":
        # need the "first" timestep in the input to reconstruct delta -> last when target=="delta"
        # the first density channel corresponds to time t0
        first_density = x_np[d_start]  # (Hc,Wc)
        first_energy = x_np[e_start]
        first_pressure = x_np[p_start]
        # momentum channels layout: t0_x, t0_y, t1_x, t1_y, ...
        # to get momentum[0] (first timestep) need to reconstruct from m_start indices 0 and 1
        first_mom_x = x_np[m_start]
        first_mom_y = x_np[m_start + 1]
        first_momentum = np.stack([first_mom_x, first_mom_y], axis=-1)  # (Hc,Wc,2)

        pred_last_density  = first_density + p_d
        pred_last_energy   = first_energy + p_e
        pred_last_pressure = first_pressure + p_p
        pred_last_momentum = first_momentum + p_m
    else:
        pred_last_density  = p_d
        pred_last_energy   = p_e
        pred_last_pressure = p_p
        pred_last_momentum = p_m

    # shift-and-append logic: drop the oldest timestep, shift the window forward by one,
    # and append the predicted last timestep as the new most-recent input
    # density: shape (d_count, Hc, Wc)
    x_np[d_start:d_end] = np.concatenate(
        [x_np[d_start + 1:d_end], np.expand_dims(pred_last_density, axis=0)],
        axis=0)
    # energy
    x_np[e_start:e_end] = np.concatenate(
        [x_np[e_start + 1:e_end], np.expand_dims(pred_last_energy, axis=0)],
        axis=0)
    # pressure
    x_np[p_start:p_end] = np.concatenate(
        [x_np[p_start + 1:p_end], np.expand_dims(pred_last_pressure, axis=0)],
        axis=0)
    # momentum: channels are [t0_x,t0_y, t1_x,t1_y, ...], so drop first 2 channels and append pred x,y
    x_np[m_start:m_end] = np.concatenate(
        [x_np[m_start + 2:m_end],                                 # shifted previous momentum channels
        np.expand_dims(pred_last_momentum[..., 0], axis=0),       # append predicted momentum x
        np.expand_dims(pred_last_momentum[..., 1], axis=0)],      # append predicted momentum y
        axis=0)

    # convert back to node-feature layout and create a mixed batch
    x_mixed_nodes = torch.tensor(x_np.reshape(C_in, Hc * Wc).T, dtype=batch.x.dtype, device=batch.x.device)

    # clone batch and replace x
    batch_mixed = batch.clone()
    batch_mixed.x = x_mixed_nodes
    return batch_mixed
