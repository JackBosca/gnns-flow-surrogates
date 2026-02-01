import os
from typing import Optional, Dict, Any, List, Tuple
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
    if dataset.to_centroids:
        Hc += 1 
        Wc += 1

    density  = np.empty((Hc, Wc), dtype=np.float32)
    energy   = np.empty((Hc, Wc), dtype=np.float32)
    pressure = np.empty((Hc, Wc), dtype=np.float32)
    momentum = np.empty((Hc, Wc, 2), dtype=np.float32)

    d_density  = f["t0_fields"]["density"]
    d_energy   = f["t0_fields"]["energy"]
    d_pressure = f["t0_fields"]["pressure"]
    d_mom      = f["t1_fields"]["momentum"]

    sel_scalar = np.s_[sim_idx, t_idx, ::dataset.sh, ::dataset.sw]
    d_density.read_direct(density, source_sel=sel_scalar)
    d_energy.read_direct(energy, source_sel=sel_scalar)
    d_pressure.read_direct(pressure, source_sel=sel_scalar)

    sel_vector = np.s_[sim_idx, t_idx, ::dataset.sh, ::dataset.sw, :]
    d_mom.read_direct(momentum, source_sel=sel_vector)

    momentum = momentum[np.newaxis, ...]

    if dataset.to_centroids:
        density  = dataset._convert_to_centroids(density)
        energy   = dataset._convert_to_centroids(energy)
        pressure = dataset._convert_to_centroids(pressure)
        momentum = dataset._convert_to_centroids(momentum)

    momentum = momentum[0]

    return {"density": density, "energy": energy, "pressure": pressure, "momentum": momentum}


def _compute_vrmse_and_stats(pred: np.ndarray, truth: np.ndarray, epsilon: float = 1e-8):
    """
    Compute Variance Scaled RMSE (VRMSE) and return stats for debugging.
    """
    mse = np.mean((pred - truth) ** 2)
    rmse = np.sqrt(mse)
    # Spatial variance of the truth at this timestep
    var = np.var(truth)
    vrmse = float(np.sqrt(mse / (var + epsilon)))
    return vrmse, rmse, var


def evaluate_one_step_vrmse(
    model: torch.nn.Module,
    dataset,
    sim_idx: int,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    """
    Evaluate 1-step predictions.
    Input: Normalized Window.
    Output: Normalized Prediction -> Denormalize -> Compare with Physical GT.
    """
    model.eval()
    
    t_w = dataset.time_window
    Hc, Wc = dataset.Hc, dataset.Wc
    stats = getattr(dataset, "stats", None)

    steps = dataset.n_t - t_w 

    vrmse_avg: List[float] = [] 

    for i in range(steps):
        current_start = 0 + i
        
        # 1. Load GT window (physical)
        window = dataset._load_time_window(sim_idx, current_start)
        
        # 2. normalize input
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

        mom_ch = momentum.transpose(0, 3, 1, 2).reshape(t_w * 2, Hc, Wc)
        x_np = np.concatenate([density[:-1], energy[:-1], pressure[:-1], mom_ch[:-2]], axis=0)

        predicted_t = current_start + (t_w - 1)
        time_scalar = float(predicted_t) / float(dataset.n_t - 1)

        # y_true dummies for graph construction
        y_density_true  = density[-1].copy()
        y_energy_true   = energy[-1].copy()
        y_pressure_true = pressure[-1].copy()
        y_momentum_true = momentum[-1].copy()

        data = dataset._arrays_to_graph(x_np, y_density_true, y_energy_true, y_pressure_true, y_momentum_true, time_step=time_scalar)
        data = data.to(device)
        
        with torch.no_grad():
            out = model(data, stats)

        # 3. model output (normalized absoluten
        pred_norm_density  = out["density"].detach().cpu().numpy().reshape(Hc, Wc)
        pred_norm_energy   = out["energy"].detach().cpu().numpy().reshape(Hc, Wc)
        pred_norm_pressure = out["pressure"].detach().cpu().numpy().reshape(Hc, Wc)
        pred_norm_momentum = out["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2)

        # 4. get GT for metrics (physical)
        gt_d = window["density"][-1]
        gt_e = window["energy"][-1]
        gt_p = window["pressure"][-1]
        gt_m = window["momentum"][-1]

        # 5. denormalize prediction for comparison
        if dataset.normalize and stats is not None:
            pred_phys_d = _denormalize(pred_norm_density, stats["mean"]["density"], stats["std"]["density"])
            pred_phys_e = _denormalize(pred_norm_energy, stats["mean"]["energy"], stats["std"]["energy"])
            pred_phys_p = _denormalize(pred_norm_pressure, stats["mean"]["pressure"], stats["std"]["pressure"])
            pred_phys_m = _denormalize(pred_norm_momentum, stats["mean"]["momentum"], stats["std"]["momentum"])
        else:
            pred_phys_d = pred_norm_density
            pred_phys_e = pred_norm_energy
            pred_phys_p = pred_norm_pressure
            pred_phys_m = pred_norm_momentum

        # 6. Compute VRMSE
        v_d, _, _ = _compute_vrmse_and_stats(pred_phys_d, gt_d)
        v_e, _, _ = _compute_vrmse_and_stats(pred_phys_e, gt_e)
        v_p, _, _ = _compute_vrmse_and_stats(pred_phys_p, gt_p)
        v_mx, _, _ = _compute_vrmse_and_stats(pred_phys_m[..., 0], gt_m[..., 0])
        v_my, _, _ = _compute_vrmse_and_stats(pred_phys_m[..., 1], gt_m[..., 1])

        v_step_avg = np.mean([v_d, v_e, v_p, v_mx, v_my])
        vrmse_avg.append(v_step_avg)

    return {"average": vrmse_avg}


def evaluate_autoregressive_vrmse_one_simulation(
    model: torch.nn.Module,
    dataset,
    sim_idx: int,
    eval_windows: List[Tuple[int, int]],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Perform a FULL autoregressive rollout.
    Output is appended to buffer AS IS (Normalized).
    Output is DENORMALIZED for metric calculation.
    """
    model.eval()
    
    t_w = dataset.time_window
    Hc, Wc = dataset.Hc, dataset.Wc
    stats = getattr(dataset, "stats", None)

    max_eval_step = max([w[1] for w in eval_windows])
    if max_eval_step >= dataset.n_t:
        max_eval_step = dataset.n_t - 1

    # 1. Load Initial Window (Physical)
    window = dataset._load_time_window(sim_idx, 0)
    density  = window["density"].copy()
    energy   = window["energy"].copy()
    pressure = window["pressure"].copy()
    momentum = window["momentum"].copy()

    # 2. Normalize Initial Window (This buffer stays Normalized)
    if dataset.normalize and stats is not None:
        density  = (density  - stats["mean"]["density"])  / stats["std"]["density"]
        energy   = (energy   - stats["mean"]["energy"])   / stats["std"]["energy"]
        pressure = (pressure - stats["mean"]["pressure"]) / stats["std"]["pressure"]
        momentum = (momentum - stats["mean"]["momentum"]) / stats["std"]["momentum"]

    window_scores = {f"{w[0]}-{w[1]}": [] for w in eval_windows}

    # Start rollout from t = t_w
    current_t = t_w 
    
    while current_t <= max_eval_step:
        # 3. Build Input from Normalized Buffer
        mom_ch = momentum.transpose(0, 3, 1, 2).reshape(t_w * 2, Hc, Wc)
        x_np = np.concatenate([density[:-1], energy[:-1], pressure[:-1], mom_ch[:-2]], axis=0)
        
        time_scalar = float(current_t) / float(dataset.n_t - 1)

        y_dummy = np.zeros_like(density[-1])
        y_mom_dummy = np.zeros_like(momentum[-1])
        
        data = dataset._arrays_to_graph(x_np, y_dummy, y_dummy, y_dummy, y_mom_dummy, time_step=time_scalar)
        data = data.to(device)

        with torch.no_grad():
            out = model(data, stats)

        # 4. Model Output (Normalized Absolute)
        pred_norm_density  = out["density"].detach().cpu().numpy().reshape(Hc, Wc)
        pred_norm_energy   = out["energy"].detach().cpu().numpy().reshape(Hc, Wc)
        pred_norm_pressure = out["pressure"].detach().cpu().numpy().reshape(Hc, Wc)
        pred_norm_momentum = out["momentum"].detach().cpu().numpy().reshape(Hc, Wc, 2)

        # 5. Check Windows & Compute VRMSE
        for w_start, w_end in eval_windows:
            if w_start <= current_t <= w_end:
                # Load GT (Physical)
                gt_step = _read_timestep_from_h5(dataset, sim_idx, current_t)
                
                # Denormalize Prediction for Metric
                if dataset.normalize and stats is not None:
                    p_d = _denormalize(pred_norm_density, stats["mean"]["density"], stats["std"]["density"])
                    p_e = _denormalize(pred_norm_energy, stats["mean"]["energy"], stats["std"]["energy"])
                    p_p = _denormalize(pred_norm_pressure, stats["mean"]["pressure"], stats["std"]["pressure"])
                    p_m = _denormalize(pred_norm_momentum, stats["mean"]["momentum"], stats["std"]["momentum"])
                else:
                    p_d, p_e, p_p, p_m = pred_norm_density, pred_norm_energy, pred_norm_pressure, pred_norm_momentum

                # Compute VRMSE
                v_d, _, _ = _compute_vrmse_and_stats(p_d, gt_step["density"])
                v_e, _, _ = _compute_vrmse_and_stats(p_e, gt_step["energy"])
                v_p, _, _ = _compute_vrmse_and_stats(p_p, gt_step["pressure"])
                v_mx, _, _ = _compute_vrmse_and_stats(p_m[..., 0], gt_step["momentum"][..., 0])
                v_my, _, _ = _compute_vrmse_and_stats(p_m[..., 1], gt_step["momentum"][..., 1])
                
                avg_vrmse = np.mean([v_d, v_e, v_p, v_mx, v_my])
                window_scores[f"{w_start}-{w_end}"].append(avg_vrmse)

        # 6. Append Normalized Output to Buffer (No Re-normalization!)
        density  = np.concatenate([density[1:], np.expand_dims(pred_norm_density, 0)], axis=0)
        energy   = np.concatenate([energy[1:],  np.expand_dims(pred_norm_energy, 0)], axis=0)
        pressure = np.concatenate([pressure[1:],np.expand_dims(pred_norm_pressure, 0)], axis=0)
        momentum = np.concatenate([momentum[1:],np.expand_dims(pred_norm_momentum, 0)], axis=0)

        current_t += 1

    results = {}
    for key, vals in window_scores.items():
        if len(vals) > 0:
            results[key] = float(np.mean(vals))
        else:
            results[key] = np.nan
            
    return results


if __name__ == "__main__":
    from dataset.euler_coarse import EulerPeriodicDataset
    from model.invariant_gnn_flux import FluxGNN
    
    # 1. evaluation mode: "one_step" or "autoregressive"
    EVALUATION_MODE = "autoregressive" 
    # EVALUATION_MODE = "one_step"

    # 2. windows for autoregressive evaluation (start_t, end_t) inclusive
    AUTO_REG_WINDOWS = [(6, 12), (13, 30)]

    # 3. Data & Model Paths
    test_h5_paths = [
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.13_C3H8_16.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.33_H2O_20.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.365_Dry_air_1000.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.3_CO2_20.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.404_H2_100_Dry_air_-15.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.453_H2_-76.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.4_Dry_air_20.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.597_H2_-181.hdf5",
        "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/euler_multi_quadrants_periodicBC_gamma_1.76_Ar_-180.hdf5",
    ]
    
    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"
    checkpoint_path = "./jobs/Dry_air_20_5UNROLLED_model_flux_n-datasets_1_forcing_1.0_time-window_7_coarsen_4-4_target_delta_centroids_True_layers_12_epoch_7.pt"

    time_window = 2
    coarsen = (4,4)
    target = "delta"
    
    print(f"Initializing Model... Mode: {EVALUATION_MODE}")
    temp_ds = EulerPeriodicDataset(test_h5_paths[0], stats_path=stats_path, time_window=time_window, 
                                   target=target, normalize=True, coarsen=coarsen, to_centroids=True)

    model = FluxGNN(node_in_dim=(time_window-1)*5, 
                    node_embed_dim=64,
                    n_layers=12,
                    dataset_dt=0.015,
                    )

    # Load Weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt: state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt: state_dict = ckpt["state_dict"]
        else: state_dict = ckpt
    else:
        state_dict = ckpt

    res = model.load_state_dict(state_dict, strict=False)
    print("load_state_dict result:", res)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"\nStarting {EVALUATION_MODE} Benchmark on {len(test_h5_paths)} files...")
    print("-" * 60)

    if EVALUATION_MODE == "one_step":
        global_vrmse_agg = []
    else:
        global_vrmse_agg = {f"{w[0]}-{w[1]}": [] for w in AUTO_REG_WINDOWS}

    for path_idx, h5_path in enumerate(test_h5_paths):
        filename = os.path.basename(h5_path)
        print(f"Processing File {path_idx+1}/{len(test_h5_paths)}: {filename}")

        ds = EulerPeriodicDataset(h5_path,
                                stats_path=stats_path,
                                time_window=time_window,
                                target=target,
                                normalize=True,
                                coarsen=coarsen,
                                to_centroids=True)
        
        file_stats = []
        file_window_agg = {}
        for sim_idx in range(ds.n_sims):
            
            if EVALUATION_MODE == "one_step":
                sim_metrics = evaluate_one_step_vrmse(model, ds, sim_idx, device=device)
                global_vrmse_agg.extend(sim_metrics["average"])
                file_stats.extend(sim_metrics["average"])
                sim_mean = np.mean(sim_metrics["average"])
            else:
                sim_results = evaluate_autoregressive_vrmse_one_simulation(
                    model, ds, sim_idx, AUTO_REG_WINDOWS, device=device
                )
                for key, val in sim_results.items():
                    if not np.isnan(val):
                        global_vrmse_agg[key].append(val)
                        file_window_agg.setdefault(key, []).append(val)
                
                valid_vals = [v for v in sim_results.values() if not np.isnan(v)]
                sim_mean = np.mean(valid_vals) if valid_vals else np.nan
                file_stats.append(sim_mean)

            if (sim_idx + 1) % 1 == 0:
                print(f"  > Sim {sim_idx}: Mean VRMSE = {sim_mean:.4f}")

        print(f"  [File {filename} Done] Mean VRMSE: {np.nanmean(file_stats):.4f}")
        
        if EVALUATION_MODE == "autoregressive":
            breakdown = ", ".join([f"Win {k}: {np.mean(v):.4f}" for k, v in file_window_agg.items() if v])
            print(f"    > Breakdown: {breakdown}")

    print("=" * 60)
    print(f"FINAL RESULTS ({EVALUATION_MODE})")
    print("=" * 60)

    if EVALUATION_MODE == "one_step":
        print(f"AVERAGE VRMSE (One-Step): {np.mean(global_vrmse_agg):.6f}")
    else:
        for key in global_vrmse_agg:
            vals = global_vrmse_agg[key]
            mean_val = np.mean(vals) if len(vals) > 0 else np.nan
            print(f"Window [{key}]: VRMSE = {mean_val:.6f}")
    print("=" * 60)