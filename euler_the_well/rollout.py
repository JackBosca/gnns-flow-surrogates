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
    - dataset.time_window = number of timesteps loaded per call
    - _load_time_window(sim_idx, t_idx) returns arrays for timesteps
    [t_idx, ..., t_idx + time_window - 1]
    - The Dataset builds model input x from density[:-1], energy[:-1],
    pressure[:-1], and a reshaped momentum stack (same as __getitem__).
    - Training target convention:
        if dataset.target == "delta":
            target = (last - first) in normalized space
        else:
            target = last (absolute, normalized if dataset.normalize)
    - This rollout uses the same convention:
    - At iteration i, the model is given the current window excluding the last
        (same x as __getitem__). The model predicts the "last" timestep for the
        current window. Then append the predicted last, drop the first, and
        continue -> this advances the window by 1 timestep.
    - RMSE metrics are computed only when ground-truth exists (i.e., predicted
    timestep < dataset.n_t). If ground-truth is missing for a step, RMSE is set to np.nan.

    Returns structure:
        {
          "predictions": {
              "density": list of arrays (Hc,Wc) per predicted step (denorm if requested),
              "energy": [...],
              "pressure": [...],
              "momentum": list of arrays (Hc,Wc,2)
          },
          "ground_truth": {
                "density": list of arrays (Hc,Wc) per predicted step (physical units),
                "energy": [...],
                "pressure": [...],
                "momentum": list of arrays (Hc,Wc,2)
            },
          "metrics": {
              "rmse_density": np.array([...]),
              "rmse_energy": ...,
              "rmse_pressure": ...,
              "rmse_momentum_x": ...,
              "rmse_momentum_y": ...
          },
          "timesteps": np.array([...])  # absolute timestep index predicted for each step
        }
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.eval()
    model.to(device)

    t_w = dataset.time_window
    Hc, Wc = dataset.Hc, dataset.Wc

    # number of rollout steps: if None, roll until end of sim
    if rollout_steps is None:
        rollout_steps = dataset.n_t - (start_t + t_w - 1)

    # load initial window (this includes the "last" which is the first predicted step)
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


def evaluate_one_step(
    model: torch.nn.Module,
    dataset,
    sim_idx: int,
    start_t: int = 0,
    steps: Optional[int] = None,
    device: str = "cuda",
    return_denormalized: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate 1-step predictions: for each step i, load ground-truth window starting at
    (start_t + i), feed the model the input window (density[:-1], energy[:-1], pressure[:-1], mom_ch[:-2])
    and predict the last timestep. Compute RMSE against the true last timestep.

    Returns same structure as rollout_one_simulation:
      { "predictions": { "density": [...], ... },
        "ground_truth": { "density": [...], ... },
        "metrics": { "rmse_density": np.array([...]), ... },
        "timesteps": np.array([...]) }
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.eval()
    model.to(device)

    t_w = dataset.time_window
    Hc, Wc = dataset.Hc, dataset.Wc

    # default number of steps until end of simulation
    if steps is None:
        steps = dataset.n_t - (start_t + t_w - 1)

    stats = getattr(dataset, "stats", None)

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

    for i in range(steps):
        current_start = start_t + i
        # load ground-truth window (physical)
        window = dataset._load_time_window(sim_idx, current_start)
        # normalize if needed
        if dataset.normalize and stats is not None:
            density  = (window["density"]  - stats["mean"]["density"])  / stats["std"]["density"]
            energy   = (window["energy"]   - stats["mean"]["energy"])   / stats["std"]["energy"]
            pressure = (window["pressure"] - stats["mean"]["pressure"]) / stats["std"]["pressure"]
            momentum = (window["momentum"] - stats["mean"]["momentum"]) / stats["std"]["momentum"]
        else:
            density  = window["density"].copy()
            energy   = window["energy"].copy()
            pressure = window["pressure"].copy()
            momentum = window["momentum"].copy()

        # build input exactly like Dataset.__getitem__
        mom_ch = momentum.transpose(0, 3, 1, 2).reshape(t_w * 2, Hc, Wc)  # (t_w*2, Hc, Wc)
        x_np = np.concatenate([density[:-1], energy[:-1], pressure[:-1], mom_ch[:-2]], axis=0)

        # last input timestep scalar (normalized to [0,1])
        predicted_t = current_start + (t_w - 1)
        time_scalar = float(predicted_t) / float(dataset.n_t - 1)

        # true last (normalized if dataset.normalize)
        y_density_true  = density[-1].copy()
        y_energy_true   = energy[-1].copy()
        y_pressure_true = pressure[-1].copy()
        y_momentum_true = momentum[-1].copy()

        # build Data and run model
        data = dataset._arrays_to_graph(x_np, y_density_true, y_energy_true, y_pressure_true, y_momentum_true, time_step=time_scalar)
        data = data.to(device)
        with torch.no_grad():
            out = model(data)

        p_density = out["density"].detach().cpu().numpy().reshape(Hc, Wc)
        p_energy  = out["energy"].detach().cpu().numpy().reshape(Hc, Wc)
        p_pressure= out["pressure"].detach().cpu().numpy().reshape(Hc, Wc)
        p_momentum= out["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2)

        # convert model outputs to predicted last (normalized space if dataset.normalize)
        if dataset.target == "delta":
            pred_last_density  = density[0]  + p_density
            pred_last_energy   = energy[0]   + p_energy
            pred_last_pressure = pressure[0] + p_pressure
            pred_last_momentum = momentum[0] + p_momentum
        else:
            pred_last_density  = p_density
            pred_last_energy   = p_energy
            pred_last_pressure = p_pressure
            pred_last_momentum = p_momentum

        timesteps.append(predicted_t)

        # ground truth (physical) for this predicted timestep -> from window (physical)
        gt_d = window["density"][-1]
        gt_e = window["energy"][-1]
        gt_p = window["pressure"][-1]
        gt_m = window["momentum"][-1]

        # compute RMSE (denormalize predictions first if normalized)
        if dataset.normalize and stats is not None:
            pred_d_den = _denormalize(pred_last_density, stats["mean"]["density"], stats["std"]["density"])
            rmse_d = _compute_rmse(pred_d_den, gt_d)

            pred_e_den = _denormalize(pred_last_energy, stats["mean"]["energy"], stats["std"]["energy"])
            rmse_e = _compute_rmse(pred_e_den, gt_e)

            pred_p_den = _denormalize(pred_last_pressure, stats["mean"]["pressure"], stats["std"]["pressure"])
            rmse_p = _compute_rmse(pred_p_den, gt_p)

            pred_m_den = _denormalize(pred_last_momentum, stats["mean"]["momentum"], stats["std"]["momentum"])
            rmse_mx = _compute_rmse(pred_m_den[..., 0], gt_m[..., 0])
            rmse_my = _compute_rmse(pred_m_den[..., 1], gt_m[..., 1])

            # storage values
            store_d = pred_d_den if return_denormalized else pred_last_density.copy()
            store_e = pred_e_den if return_denormalized else pred_last_energy.copy()
            store_p = pred_p_den if return_denormalized else pred_last_pressure.copy()
            store_m = pred_m_den if return_denormalized else pred_last_momentum.copy()
        else:
            rmse_d = _compute_rmse(pred_last_density, gt_d)
            rmse_e = _compute_rmse(pred_last_energy, gt_e)
            rmse_p = _compute_rmse(pred_last_pressure, gt_p)
            rmse_mx = _compute_rmse(pred_last_momentum[..., 0], gt_m[..., 0])
            rmse_my = _compute_rmse(pred_last_momentum[..., 1], gt_m[..., 1])

            store_d = pred_last_density.copy()
            store_e = pred_last_energy.copy()
            store_p = pred_last_pressure.copy()
            store_m = pred_last_momentum.copy()

        # append metrics and storage
        print(f"Eval 1-step sim {sim_idx} t {predicted_t}: RMSE density={rmse_d:.6e}, energy={rmse_e:.6e}, pressure={rmse_p:.6e}, mx={rmse_mx:.6e}, my={rmse_my:.6e}")

        preds_density.append(store_d.astype(np.float32))
        preds_energy.append(store_e.astype(np.float32))
        preds_pressure.append(store_p.astype(np.float32))
        preds_momentum.append(store_m.astype(np.float32))

        gt_density.append(gt_d.astype(np.float32))
        gt_energy.append(gt_e.astype(np.float32))
        gt_pressure.append(gt_p.astype(np.float32))
        gt_momentum.append(gt_m.astype(np.float32))

        rmse_density.append(rmse_d)
        rmse_energy.append(rmse_e)
        rmse_pressure.append(rmse_p)
        rmse_momentum_x.append(rmse_mx)
        rmse_momentum_y.append(rmse_my)

    # prepare outputs
    metrics = {
        "rmse_density": np.array(rmse_density, dtype=np.float32),
        "rmse_energy": np.array(rmse_energy, dtype=np.float32),
        "rmse_pressure": np.array(rmse_pressure, dtype=np.float32),
        "rmse_momentum_x": np.array(rmse_momentum_x, dtype=np.float32),
        "rmse_momentum_y": np.array(rmse_momentum_y, dtype=np.float32),
    }

    results = {
        "predictions": {
            "density": preds_density,
            "energy": preds_energy,
            "pressure": preds_pressure,
            "momentum": preds_momentum
        },
        "ground_truth": {
            "density": gt_density,
            "energy": gt_energy,
            "pressure": gt_pressure,
            "momentum": gt_momentum
        },
        "metrics": metrics,
        "timesteps": np.array(timesteps, dtype=int)
    }

    # optional saving
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
            print(f"Saved 1-step evaluation to {save_path}")
        except Exception as e:
            print(f"Warning: failed to save eval results to {save_path}: {e}")

    return results


if __name__ == "__main__":
    from dataset.euler_coarse import EulerPeriodicDataset
    from model.egnn_state import EGNNStateModel

    # h5_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    h5_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/valid/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"

    ds = EulerPeriodicDataset(h5_path, stats_path=stats_path, time_window=2, target="delta", normalize=True, coarsen=(2,2))

    # build sample to retrieve dimensions
    sample = ds[0]  # torch_geometric.data.Data
    input_node_feats = sample.x.shape[1]        # (time_window-1) * 5
    global_feat_dim = sample.u.shape[1] if getattr(sample, "u", None) is not None else 0

    # instantiate model
    model = EGNNStateModel(
        input_feat_dim = input_node_feats,
        global_feat_dim = global_feat_dim,
        pos_dim = 2,
        edge_attr_dim = 4,
        egnn_hidden_feats = 64,
        n_layers = 6,
        fourier_features = 0,
        soft_edge = 0,
        update_feats = True,
        update_coors = False,
        dropout = 0.0,
        aggr = "add",
        readout_hidden = 128,
        use_separate_heads = True
    )
 
    # --- robust model loader ---
    checkpoint_path = "./checkpoints/model_coarsen_2-2_epoch_2.pt"

    # 1) load the file (map to cpu first)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # 2) extract the state_dict if the file was saved with a wrapper dict
    if isinstance(ckpt, dict):
        # common keys used by training scripts
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # your training saved model.state_dict() directly, so ckpt is the state_dict
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 3) load into model, show missing/unexpected keys if any (use strict=False for diagnostics)
    res = model.load_state_dict(state_dict, strict=False)
    print("load_state_dict result:", res)   # shows missing_keys / unexpected_keys if any

    # 4) move model to device and set eval mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # -------------- QUICK DEBUG --------------
    # sample already created earlier in your rollout script: sample = ds[0]
    # ensure sample is on cpu for this check
    sample_cpu = sample.to("cpu")

    # move model temporarily to cpu for this forward-check (safe)
    model_cpu = model.to("cpu")
    model_cpu.eval()

    with torch.no_grad():
        out = model_cpu(sample_cpu)

    # print stats for each predicted field
    for k, v in out.items():
        a = v.detach().cpu().numpy()
        print(f"{k:12s} shape={a.shape} min={a.min():.3e} max={a.max():.3e} mean={a.mean():.3e} std={a.std():.3e}")
    # -------------- END QUICK DEBUG --------------

    sim_idx = 0
    t_idx = 0
    # save_path = f"./rollouts/rollout_sim{sim_idx}_t{t_idx}.npz"
    save_path = f"./rollouts_1-step/rollout_1-step_sim{sim_idx}_t{t_idx}_valid.npz"

    # perform rollout of first sim starting from t=0
    # rollout_one_simulation(model, ds, sim_idx=sim_idx, start_t=t_idx, save_path=save_path)
    evaluate_one_step(model, ds, sim_idx=sim_idx, start_t=t_idx, save_path=save_path)

    print("Rollout done.")