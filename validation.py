import os
import pickle
import numpy as np
import torch
from dataset.cylinder import IndexedTrajectoryDataset  
from rollout import rollout_one_trajectory


def validate_rollouts(
    model,
    dataset_dir,
    split='valid',
    device=torch.device('cpu'),
    n_trajectories=2,
    max_steps=100,
    preserve_one_hot=False,
    cache_static=True,
    transform=None,
    enforce_mask=True,
    save_dir='checkpoint'
):
    """
    Run closed-loop short rollouts on `n_trajectories` from the given split and return aggregated metrics.
    Args:
      model: the Simulator instance to validate
      dataset_dir: root directory of the dataset
      split: which split to use ('valid' or 'test')
      device: torch device to use
      n_trajectories: how many trajectories to rollout (from the start of the split)
      max_steps: maximum number of steps to rollout per trajectory
      preserve_one_hot: whether to keep one-hot encoding of boundary conditions in input features
      cache_static: whether to cache static graph attributes in the dataset
      transform: optional transform to apply to graphs
      enforce_mask: if True, enforce boundary conditions by copying ground-truth velocity for boundary nodes
      save_dir: directory where to save individual rollout results (pickles)
    Returns:
      metrics dict with keys: mean_rmse_per_step (np.array), std_rmse_per_step (np.array), horizon_rmse (dict)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # build a validation dataset instance
    val_ds = IndexedTrajectoryDataset(
        dataset_dir=dataset_dir,
        split=split,
        time_interval=0.01,
        cache_static=cache_static,
        transform=transform,
        preserve_one_hot=preserve_one_hot
    )

    n_trajs = min(n_trajectories, len(val_ds.traj_keys))
    per_traj_rmse = []
    saved_results = []

    for ti in range(n_trajs):
        print(f'[VAL] Rolling out traj {ti}/{n_trajs-1}')
        result, coords = rollout_one_trajectory(
            model=model,
            dataset=val_ds,
            traj_index=ti,
            device=device,
            max_steps=max_steps,
            enforce_mask=enforce_mask
        )
        predicteds, targets = result

        # compute per-step RMSE for this trajectory
        squared_diff = np.square(predicteds - targets).reshape(predicteds.shape[0], -1)
        mse_per_step = np.mean(squared_diff, axis=1)
        rmse_per_step = np.sqrt(mse_per_step)  # (T,)
        per_traj_rmse.append(rmse_per_step)

        # optionally save the raw rollout for inspection/rendering
        out_path = os.path.join(save_dir, f'val_traj_{ti}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump([result, coords], f)
        saved_results.append(out_path)

    # aggregate: pad to longest and compute mean/std
    max_len = max(r.shape[0] for r in per_traj_rmse)
    arr = np.full((len(per_traj_rmse), max_len), np.nan, dtype=float)
    for i, r in enumerate(per_traj_rmse):
        arr[i, : r.shape[0]] = r
    mean_rmse = np.nanmean(arr, axis=0)
    std_rmse = np.nanstd(arr, axis=0)

    # horizon points to report
    horizon_steps = [0, 9, 49, 99]  # 1,10,50,100 (0-based)
    horizon_rmse = {}
    for s in horizon_steps:
        if s < mean_rmse.shape[0]:
            horizon_rmse[s + 1] = float(mean_rmse[s])
        else:
            horizon_rmse[s + 1] = float('nan')

    metrics = {
        'mean_rmse_per_step': mean_rmse,
        'std_rmse_per_step': std_rmse,
        'horizon_rmse': horizon_rmse,
        'saved_results': saved_results
    }
    return metrics
