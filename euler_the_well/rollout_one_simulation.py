import os
from typing import Optional, Dict, Any, List
import numpy as np
import torch

def _denormalize(arr: np.ndarray, mean, std):
    """
    Denormalize with broadcasting. mean/std can be scalars or arrays.
    """
    if mean is None or std is None:
        return arr
    return arr * np.asarray(std) + np.asarray(mean)


def _read_timestep_from_h5(dataset, sim_idx: int, t_idx: int):
    """
    Read single timestep fields (coarsened) directly from dataset's HDF5.
    Returns:
        dict with keys: density, energy, pressure, momentum (Hc,Wc,2)
    """
    dataset._ensure_h5()
    f = dataset._h5

    Hc, Wc = dataset.Hc, dataset.Wc
    # preallocate
    density = np.empty((Hc, Wc), dtype=np.float32)
    energy  = np.empty((Hc, Wc), dtype=np.float32)
    pressure= np.empty((Hc, Wc), dtype=np.float32)
    momentum= np.empty((Hc, Wc, 2), dtype=np.float32)

    d_density = f["t0_fields"]["density"]
    d_energy  = f["t0_fields"]["energy"]
    d_pressure= f["t0_fields"]["pressure"]
    d_mom     = f["t1_fields"]["momentum"]

    sel_scalar = np.s_[sim_idx, t_idx, ::dataset.sh, ::dataset.sw]
    d_density.read_direct(density, source_sel=sel_scalar)
    d_energy.read_direct(energy, source_sel=sel_scalar)
    d_pressure.read_direct(pressure, source_sel=sel_scalar)

    sel_vector = np.s_[sim_idx, t_idx, ::dataset.sh, ::dataset.sw, :]
    d_mom.read_direct(momentum, source_sel=sel_vector)

    return {"density": density, "energy": energy, "pressure": pressure, "momentum": momentum}


def _compute_rmse(pred: np.ndarray, truth: np.ndarray):
    """
    Compute RMSE over all nodes. Inputs are same-shape ndarrays.
    """
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def rollout_one_simulation(
    model: torch.nn.Module,
    dataset,
    sim_idx: int,
    start_t: int = 0,
    rollout_steps: Optional[int] = None,
    device: str = "cuda",
    return_denormalized: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    - At iteration i, the model is given the current window excluding the last
        (same x as __getitem__). The model predicts the "last" timestep for the
        current window. Then append the predicted last, drop the first, and
        continue -> this advances the window by 1 timestep.
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.eval()
    model.to(device)

    t_w = dataset.time_window
    Hc, Wc = dataset.Hc, dataset.Wc

    # number of rollout steps: if None, roll until end of sim
    if rollout_steps is None:
        rollout_steps = dataset.n_t - (start_t + t_w - 1)

    # load initial window
    window = dataset._load_time_window(sim_idx, start_t)
    # window keys: density (t_w,Hc,Wc), energy, pressure, momentum (t_w,Hc,Wc,2)
    # these are NOT yet normalized

    # normalize initial window if dataset.normalize and stats available (same as __getitem__)
    stats = getattr(dataset, "stats", None)
    if dataset.normalize and stats is not None:
        density = (window["density"]  - stats["mean"]["density"])  / stats["std"]["density"]
        energy  = (window["energy"]   - stats["mean"]["energy"])   / stats["std"]["energy"]
        pressure= (window["pressure"] - stats["mean"]["pressure"]) / stats["std"]["pressure"]
        momentum= (window["momentum"] - stats["mean"]["momentum"]) / stats["std"]["momentum"]
    else:
        density = window["density"].copy()
        energy  = window["energy"].copy()
        pressure= window["pressure"].copy()
        momentum= window["momentum"].copy()

    # storage
    preds_density: List[np.ndarray] = []
    preds_energy: List[np.ndarray] = []
    preds_pressure: List[np.ndarray] = []
    preds_momentum: List[np.ndarray] = []

    gt_density: List[np.ndarray] = []
    gt_energy: List[np.ndarray] = []
    gt_pressure: List[np.ndarray] = []
    gt_momentum: List[np.ndarray] = []

    rmse_density: List[float] = []
    rmse_energy: List[float] = []
    rmse_pressure: List[float] = []
    rmse_momentum_x: List[float] = []
    rmse_momentum_y: List[float] = []

    timesteps: List[int] = []

    # i-th prediction corresponds to absolute timestep:
    # predicted_t = start_t + (t_w - 1) + i      for i = 0..rollout_steps-1
    for i in range(rollout_steps):
        # build x exactly like Dataset.__getitem__
        mom_ch = momentum.transpose(0, 3, 1, 2).reshape(t_w * 2, Hc, Wc)  # shape (t_w*2, Hc, Wc)
        x_np = np.concatenate([density[:-1], energy[:-1], pressure[:-1], mom_ch[:-2]], axis=0)
        # predicted timestep scalar (normalized)
        predicted_t = start_t + (t_w - 1) + i
        time_scalar = float(predicted_t) / float(dataset.n_t - 1)

        # true target arrays for the "last" in the current window
        # later iterations may predict beyond available ground truth -> still predict but do not compute RMSE
        y_density_true  = density[-1].copy()   # normalized if dataset.normalize
        y_energy_true   = energy[-1].copy()
        y_pressure_true = pressure[-1].copy()
        y_momentum_true = momentum[-1].copy()  # (Hc,Wc,2)

        # build Data using dataset helper
        data = dataset._arrays_to_graph(x_np, y_density_true, y_energy_true, y_pressure_true, y_momentum_true, time_step=time_scalar)
        data = data.to(device)

        # forward
        with torch.no_grad():
            out = model(data)

        # outputs on cpu and reshape
        p_density  = out["density"].detach().cpu().numpy().reshape(Hc, Wc) # (N,) -> (Hc,Wc)
        p_energy   = out["energy"].detach().cpu().numpy().reshape(Hc, Wc) # (N,) -> (Hc,Wc)
        p_pressure = out["pressure"].detach().cpu().numpy().reshape(Hc, Wc) # (N,) -> (Hc,Wc)
        p_momentum = out["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2) # (N,2) -> (Hc,Wc,2)

        # - if dataset.target == "delta" then model predicts normalized (last - first)
        # - if dataset.target == "absolute" then model predicts normalized (last)
        if dataset.target == "delta":
            # reconstruct last from first + predicted_delta (all in normalized space if dataset.normalize)
            pred_last_density  = density[0]  + p_density # (Hc,Wc)
            pred_last_energy   = energy[0]   + p_energy # (Hc,Wc)
            pred_last_pressure = pressure[0] + p_pressure # (Hc,Wc)
            pred_last_momentum = momentum[0] + p_momentum # (Hc,Wc,2)
        else:
            pred_last_density  = p_density # (Hc,Wc)
            pred_last_energy   = p_energy # (Hc,Wc)
            pred_last_pressure = p_pressure # (Hc,Wc)
            pred_last_momentum = p_momentum # (Hc,Wc,2)

        # append absolute timestep index of this prediction
        timesteps.append(predicted_t)

        # if we have ground truth in the file for predicted_t, read it and compute RMSE (denormalize first if requested)
        if predicted_t < dataset.n_t:
            gt = _read_timestep_from_h5(dataset, sim_idx, predicted_t)
            # get ground truth (physical, NOT normalized)
            gt_d = gt["density"] # already physical
            gt_e = gt["energy"] # already physical
            gt_p = gt["pressure"] # already physical
            gt_m = gt["momentum"] # already physical

            # append to gt lists
            gt_density.append(gt_d.astype(np.float32))
            gt_energy.append(gt_e.astype(np.float32))
            gt_pressure.append(gt_p.astype(np.float32))
            gt_momentum.append(gt_m.astype(np.float32))

            if dataset.normalize and stats is not None:
                # current predictions are in normalized space if dataset.normalize was True,
                # so denormalize before computing RMSE
                pred_d_den = _denormalize(pred_last_density, stats["mean"]["density"], stats["std"]["density"])
                rmse_d = _compute_rmse(pred_d_den, gt_d)

                pred_e_den = _denormalize(pred_last_energy, stats["mean"]["energy"], stats["std"]["energy"])
                rmse_e = _compute_rmse(pred_e_den, gt_e)

                pred_p_den = _denormalize(pred_last_pressure, stats["mean"]["pressure"], stats["std"]["pressure"])
                rmse_p = _compute_rmse(pred_p_den, gt_p)

                pred_m_den = _denormalize(pred_last_momentum, stats["mean"]["momentum"], stats["std"]["momentum"])
                rmse_mx = _compute_rmse(pred_m_den[..., 0], gt_m[..., 0])
                rmse_my = _compute_rmse(pred_m_den[..., 1], gt_m[..., 1])
            else:
                # no normalization -> predictions are already in physical units (as gt)
                rmse_d = _compute_rmse(pred_last_density, gt_d)
                rmse_e = _compute_rmse(pred_last_energy, gt_e)
                rmse_p = _compute_rmse(pred_last_pressure, gt_p)
                rmse_mx = _compute_rmse(pred_last_momentum[..., 0], gt_m[..., 0])
                rmse_my = _compute_rmse(pred_last_momentum[..., 1], gt_m[..., 1])
        else:
            # no ground truth available
            rmse_d = np.nan
            rmse_e = np.nan
            rmse_p = np.nan
            rmse_mx = np.nan
            rmse_my = np.nan

        # print RMSEs for this step
        if verbose:
            print(f"Rollout sim {sim_idx} t {predicted_t}: RMSE density={rmse_d:.6e}, energy={rmse_e:.6e}",
                f"pressure={rmse_p:.6e}, momentum_x={rmse_mx:.6e}, momentum_y={rmse_my:.6e}")

        rmse_density.append(rmse_d)
        rmse_energy.append(rmse_e)
        rmse_pressure.append(rmse_p)
        rmse_momentum_x.append(rmse_mx)
        rmse_momentum_y.append(rmse_my)

        # store predictions (denormalize if requested, else store normalized predictions)
        if return_denormalized and dataset.normalize and stats is not None:
            store_density = _denormalize(pred_last_density, stats["mean"]["density"], stats["std"]["density"])
            store_energy  = _denormalize(pred_last_energy,  stats["mean"]["energy"],  stats["std"]["energy"])
            store_pressure= _denormalize(pred_last_pressure,stats["mean"]["pressure"],stats["std"]["pressure"])
            store_momentum= _denormalize(pred_last_momentum,stats["mean"]["momentum"],stats["std"]["momentum"])
        else:
            # these are normalized predictions if dataset.normalize and stats is not None
            store_density = pred_last_density.copy()
            store_energy  = pred_last_energy.copy()
            store_pressure= pred_last_pressure.copy()
            store_momentum= pred_last_momentum.copy()

        preds_density.append(store_density.astype(np.float32))
        preds_energy.append(store_energy.astype(np.float32))
        preds_pressure.append(store_pressure.astype(np.float32))
        preds_momentum.append(store_momentum.astype(np.float32))

        # slide the window forward by 1: drop index 0, append the predicted last
        density = np.concatenate([density[1:], np.expand_dims(pred_last_density, 0)], axis=0) # (t_w,Hc,Wc)
        energy  = np.concatenate([energy[1:],  np.expand_dims(pred_last_energy, 0)], axis=0) # (t_w,Hc,Wc)
        pressure= np.concatenate([pressure[1:],np.expand_dims(pred_last_pressure, 0)], axis=0) # (t_w,Hc,Wc)
        momentum= np.concatenate([momentum[1:],np.expand_dims(pred_last_momentum, 0)], axis=0) # (t_w,Hc,Wc,2)

    # collect outputs
    metrics = {
        "rmse_density": np.array(rmse_density, dtype=np.float32), # list to array
        "rmse_energy": np.array(rmse_energy, dtype=np.float32),
        "rmse_pressure": np.array(rmse_pressure, dtype=np.float32),
        "rmse_momentum_x": np.array(rmse_momentum_x, dtype=np.float32),
        "rmse_momentum_y": np.array(rmse_momentum_y, dtype=np.float32),
    }

    results = {
        "predictions": {
            "density": preds_density, # list of arrays (Hc,Wc)
            "energy": preds_energy, 
            "pressure": preds_pressure,
            "momentum": preds_momentum # list of arrays (Hc,Wc,2)
        },
        "ground_truth": {
            "density": gt_density, # list of arrays (Hc,Wc)
            "energy": gt_energy,
            "pressure": gt_pressure,
            "momentum": gt_momentum # list of arrays (Hc,Wc,2)
        },
        "metrics": metrics,
        "timesteps": np.array(timesteps, dtype=int)
    }

    # optional saving
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # prepare arrays for saving: stack along time axis where possible
        try:
            pred_d_stack = np.stack(preds_density, axis=0) if len(preds_density) > 0 else np.empty((0, Hc, Wc), dtype=np.float32)
            pred_e_stack = np.stack(preds_energy, axis=0) if len(preds_energy) > 0 else np.empty((0, Hc, Wc), dtype=np.float32)
            pred_p_stack = np.stack(preds_pressure, axis=0) if len(preds_pressure) > 0 else np.empty((0, Hc, Wc), dtype=np.float32)
            pred_m_stack = np.stack(preds_momentum, axis=0) if len(preds_momentum) > 0 else np.empty((0, Hc, Wc, 2), dtype=np.float32)

            gt_d_stack = np.stack(gt_density, axis=0) if len(gt_density) > 0 else np.empty((0, Hc, Wc), dtype=np.float32)
            gt_e_stack = np.stack(gt_energy, axis=0) if len(gt_energy) > 0 else np.empty((0, Hc, Wc), dtype=np.float32)
            gt_p_stack = np.stack(gt_pressure, axis=0) if len(gt_pressure) > 0 else np.empty((0, Hc, Wc), dtype=np.float32)
            gt_m_stack = np.stack(gt_momentum, axis=0) if len(gt_momentum) > 0 else np.empty((0, Hc, Wc, 2), dtype=np.float32)
            np.savez_compressed(save_path,
                                preds_d=pred_d_stack,
                                preds_e=pred_e_stack,
                                preds_p=pred_p_stack,
                                preds_m=pred_m_stack,
                                gts_d=gt_d_stack,
                                gts_e=gt_e_stack,
                                gts_p=gt_p_stack,
                                gts_m=gt_m_stack,
                                timesteps=results["timesteps"],
                                **metrics)
            print(f"Saved rollout to {save_path}")
        except Exception as e:
            print(f"Warning: failed to save rollout to {save_path}: {e}")

    return results
