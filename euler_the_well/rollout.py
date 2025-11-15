import os
import numpy as np
import torch

def normalize_fields(do_normalize, stats, d, e, p, m):
    if not do_normalize:
        return d, e, p, m
    mean, std = stats["mean"], stats["std"]
    return (
        (d - mean["density"]) / std["density"],
        (e - mean["energy"]) / std["energy"],
        (p - mean["pressure"]) / std["pressure"],
        (m - mean["momentum"]) / std["momentum"],
    )

def denormalize_fields(do_normalize, stats, d, e, p, m):
    if not do_normalize:
        return d, e, p, m
    mean, std = stats["mean"], stats["std"]
    return (
        d * std["density"] + mean["density"],
        e * std["energy"] + mean["energy"],
        p * std["pressure"] + mean["pressure"],
        m * std["momentum"] + mean["momentum"],
    )

def build_data(dataset, dn, en, pn, mn, t_idx):
    Hc, Wc = dataset.Hc, dataset.Wc
    mom_ch = mn.transpose(2, 0, 1).reshape(2, Hc, Wc)
    x = np.concatenate(
        [dn[None, ...], en[None, ...], pn[None, ...], mom_ch], axis=0
    )
    zeros_scalar = np.zeros((Hc, Wc), np.float32)
    zeros_mom = np.zeros((Hc, Wc, 2), np.float32)
    t_norm = float(t_idx) / float(dataset.n_t - 1) if dataset.n_t > 1 else 0.0
    return dataset._arrays_to_graph(x, zeros_scalar, zeros_scalar, zeros_scalar, zeros_mom, time_step=t_norm)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def rollout_one_simulation(model, dataset, sim_idx, start_t=0, rollout_steps=None,
                           device="cuda", return_denormalized=True, save_path=None):
    """
    Rollout one simulation trajectory using a trained model, comparing to ground truth.
    Prints and returns RMSE per timestep for each quantity.
    """
    model = model.to(device)
    model.eval()

    if dataset.time_window != 1:
        raise NotImplementedError("rollout_one_simulation currently supports time_window == 1 only.")

    max_steps_possible = dataset.n_t - 1 - start_t
    if max_steps_possible < 0:
        raise ValueError(f"start_t={start_t} exceeds available timesteps.")
    rollout_steps = min(rollout_steps or max_steps_possible, max_steps_possible)

    Hc, Wc = dataset.Hc, dataset.Wc
    stats = getattr(dataset, "stats", None)
    do_normalize = bool(dataset.normalize and (stats is not None))

    # load initial state
    initial = dataset._load_time_window(sim_idx, start_t)
    cur_density = initial["density"][0].astype(np.float32)      # shape (Hc,Wc)
    cur_energy = initial["energy"][0].astype(np.float32)        # shape (Hc,Wc)
    cur_pressure = initial["pressure"][0].astype(np.float32)    # shape (Hc,Wc)
    cur_momentum = initial["momentum"][0].astype(np.float32)    # shape (Hc,Wc,2)
 
    cur_density_n, cur_energy_n, cur_pressure_n, cur_momentum_n = normalize_fields(
        do_normalize=do_normalize, stats=stats, d=cur_density, e=cur_energy, p=cur_pressure, m=cur_momentum
    )

    # storage
    dens_traj_n, energy_traj_n, pressure_traj_n, momentum_traj_n = (
        [cur_density_n.copy()],
        [cur_energy_n.copy()],
        [cur_pressure_n.copy()],
        [cur_momentum_n.copy()],
    )
    timesteps = [start_t]

    rmse_density, rmse_energy, rmse_pressure, rmse_momentum_x, rmse_momentum_y = [], [], [], [], []
    gt_density_list, gt_energy_list, gt_pressure_list, gt_momentum_list = [], [], [], []

    print(f"\nRolling out simulation {sim_idx} from t={start_t} for {rollout_steps} steps...\n")

    cur_t = start_t
    for _ in range(rollout_steps):
        data = build_data(dataset, cur_density_n, cur_energy_n, cur_pressure_n, cur_momentum_n, cur_t)
        data = data.to(device)

        with torch.no_grad():
            preds = model(data)

        p_d = preds["density"].cpu().numpy().reshape(Hc, Wc)
        p_e = preds["energy"].cpu().numpy().reshape(Hc, Wc)
        p_p = preds["pressure"].cpu().numpy().reshape(Hc, Wc)
        p_m = preds["momentum"].cpu().numpy().reshape(Hc, Wc, 2)

        if dataset.target == "delta":
            next_d = cur_density_n + p_d
            next_e = cur_energy_n + p_e
            next_p = cur_pressure_n + p_p
            next_m = cur_momentum_n + p_m
        else:
            next_d, next_e, next_p, next_m = p_d, p_e, p_p, p_m

        # get ground truth next state (always denormalized in file)
        gt = dataset._load_time_window(sim_idx, cur_t + 1)
        gt_d, gt_e, gt_p, gt_m = (
            gt["density"][0].astype(np.float32),
            gt["energy"][0].astype(np.float32),
            gt["pressure"][0].astype(np.float32),
            gt["momentum"][0].astype(np.float32),
        )

        # append ground truth
        gt_density_list.append(gt_d.copy())
        gt_energy_list.append(gt_e.copy())
        gt_pressure_list.append(gt_p.copy())
        gt_momentum_list.append(gt_m.copy())

        # denormalize prediction for RMSE comparison in physical units
        pred_d, pred_e, pred_p, pred_m = denormalize_fields(do_normalize=do_normalize, stats=stats, 
                                                            d=next_d, e=next_e, p=next_p, m=next_m)

        r_d = rmse(pred_d, gt_d)
        r_e = rmse(pred_e, gt_e)
        r_p = rmse(pred_p, gt_p)
        r_mx = rmse(pred_m[..., 0], gt_m[..., 0])
        r_my = rmse(pred_m[..., 1], gt_m[..., 1])

        rmse_density.append(r_d)
        rmse_energy.append(r_e)
        rmse_pressure.append(r_p)
        rmse_momentum_x.append(r_mx)
        rmse_momentum_y.append(r_my)

        print(
            f"Step {cur_t+1:4d} | RMSE: density={r_d:.4e}, energy={r_e:.4e}, "
            f"pressure={r_p:.4e}, momentum_x={r_mx:.4e}, momentum_y={r_my:.4e}"
        )

        dens_traj_n.append(next_d.copy())
        energy_traj_n.append(next_e.copy())
        pressure_traj_n.append(next_p.copy())
        momentum_traj_n.append(next_m.copy())
        timesteps.append(cur_t + 1)

        cur_density_n, cur_energy_n, cur_pressure_n, cur_momentum_n = next_d, next_e, next_p, next_m
        cur_t += 1

    # Stack results
    out = dict(
        density_norm=np.stack(dens_traj_n),
        energy_norm=np.stack(energy_traj_n),
        pressure_norm=np.stack(pressure_traj_n),
        momentum_norm=np.stack(momentum_traj_n),
        timesteps=np.array(timesteps),
        metrics=dict(
            rmse_density=np.array(rmse_density),
            rmse_energy=np.array(rmse_energy),
            rmse_pressure=np.array(rmse_pressure),
            rmse_momentum_x=np.array(rmse_momentum_x),
            rmse_momentum_y=np.array(rmse_momentum_y),
        ),
    )

    if return_denormalized and do_normalize:
        den_d, den_e, den_p, den_m = denormalize_fields(do_normalize=do_normalize, stats=stats,
                                                        d=out["density_norm"], e=out["energy_norm"],
                                                        p=out["pressure_norm"], m=out["momentum_norm"])
        out.update(density=den_d, energy=den_e, pressure=den_p, momentum=den_m)

    out.update(
        gt_density=np.stack(gt_density_list),
        gt_energy=np.stack(gt_energy_list),
        gt_pressure=np.stack(gt_pressure_list),
        gt_momentum=np.stack(gt_momentum_list),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez_compressed(save_path, **out)
        print(f"\nSaved rollout to {save_path}\n")

    return out


if __name__ == "__main__":
    from dataset.euler_coarse import EulerPeriodicDataset
    from model.egnn_state import EGNNStateModel

    h5_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"

    ds = EulerPeriodicDataset(h5_path, stats_path=stats_path, time_window=1, normalize=True, coarsen=(1,1))

    # build sample to retrieve dimensions
    sample = ds[0]  # torch_geometric.data.Data
    input_node_feats = sample.x.shape[1]        # time_window * 5
    global_feat_dim = sample.u.shape[1] if getattr(sample, "u", None) is not None else 0

    model = EGNNStateModel(
        input_feat_dim=input_node_feats,
        global_feat_dim=global_feat_dim,
        use_separate_heads=True
    )

    # load saved model
    checkpoint_path = "./checkpoints/model_full-grid_epoch_1.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    sim_idx = 0
    t_idx = 0
    save_path = f"./rollouts/rollout_sim{sim_idx}_t{t_idx}.npz"

    # perform rollout of first sim starting from t=0
    rollout_one_simulation(model, ds, sim_idx=sim_idx, start_t=t_idx, save_path=save_path)

    print("Rollout done.")
