"""
Rollout script for MeshGraphNets architecture for the flow past a cylinder problem.

Usage example:
    python rollout.py --dataset_dir /path/to/data --test_split test --rollout_num 3 --model_dir checkpoint/simulator.pth --gpu 0
"""
import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch_geometric.transforms as T

from dataset.cylinder import IndexedTrajectoryDataset
from utils.utils import NodeType
from model.model import Simulator


def parse_args():
    p = argparse.ArgumentParser(description='MeshGraphNets rollout for flow past a cylinder')
    p.add_argument("--gpu", type=int, default=0, help="GPU id (0,1,...) or -1 for CPU")
    p.add_argument("--model_dir", type=str, default='checkpoint/simulator.pth', help="path to checkpoint file of trained model")
    p.add_argument("--dataset_dir", type=str, required=True, help="path to dataset directory containing train/val/test .h5 files")
    p.add_argument("--test_split", type=str, default='test', help="which split file to open (train/val/test)")
    p.add_argument("--rollout_num", type=int, default=1, help="how many trajectories to rollout (first N trajectories)")
    p.add_argument("--transform", default=True, action='store_true', help="apply FaceToEdge + Cartesian + Distance transforms to each graph")
    p.add_argument("--preserve_one_hot", action='store_true', help="pass preserve_one_hot to dataset if desired")
    p.add_argument("--cache_static", action='store_true', help="cache static arrays from HDF5 (pos/cells/node_type)")
    return p.parse_args()


def rollout_error(predicteds, targets):
    """
    Compute and print RMSE at intervals, return array of RMSE at each step.
    Args:
      predicteds: (T, N, vel_dim) array of predicted velocities
      targets: (T, N, vel_dim) array of ground-truth velocities
    Returns:
      loss: (T,) array of RMSE at each step
    """
    # retrieve number of steps
    number_len = targets.shape[0]
    # compute squared differences (T, N*vel_dim)
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    # cumulative mean squared error up to each step (T,) -> RMSE
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0) / np.arange(1, number_len + 1))

    # print at intervals
    for show_step in range(0, number_len, 50):
        print(f'testing rmse @ step {show_step} loss: {loss[show_step]:.2e}')
    return loss


@torch.no_grad()
def rollout_one_trajectory(model, dataset, traj_index, transformer=None, device='cpu'):
    """
    Iterate sequentially through frames of trajectory traj_index from the IndexedTrajectoryDataset.
    Args:
      model: the Simulator model (in eval mode)
      dataset: an IndexedTrajectoryDataset instance
      traj_index: which trajectory to rollout (0..len(dataset.traj_keys)-1)
      transformer: optional transform to apply to each graph (e.g. FaceToEdge + Cartesian + Distance)
      device: torch device to use
    Returns:
      result = [predicteds_array, targets_array], crds
      predicteds_array.shape = (T, N, vel_dim), targets_array same
      crds = node positions (from last iter) shape (N, dim)
    """
    # find number of timesteps available for this trajectory
    if traj_index < 0 or traj_index >= len(dataset.traj_keys):
        raise IndexError(f"traj_index {traj_index} out of range (0..{len(dataset.traj_keys)-1})")

    start_idx = int(dataset.traj_cumsum[traj_index])
    count = int(dataset.traj_counts[traj_index])  # number of usable samples (t in 0..n_frames-2)
    if count <= 0:
        raise RuntimeError(f"Trajectory {traj_index} contains no usable samples")

    predicteds = []
    targets = []
    predicted_velocity = None
    mask = None     # boolean mask for boundary nodes (computed once)

    # iterate frames sequentially
    for local_t in tqdm(range(count), total=count, desc=f"traj {traj_index}"):
        global_idx = start_idx + local_t

        # get graph at this timestep
        graph = dataset[global_idx] 
        if transformer is not None:
            graph = transformer(graph)
        # move tensors to device
        graph = graph.to(device)

        # compute mask once (based on node_type feature at column 0)
        if mask is None:
            # node_type might be (N, k)
            node_type_feat = graph.x[:, 0]
            # create boolean mask: True where node_type is NOT (NORMAL or OUTFLOW)
            mask_normal = (node_type_feat == float(NodeType.NORMAL))
            mask_outflow = (node_type_feat == float(NodeType.OUTFLOW))
            mask = torch.logical_not(torch.logical_or(mask_normal, mask_outflow))

        # if a previous prediction exists, replace velocity columns in node features with predicted
        # determine vel_dim from graph.y
        vel_dim = graph.y.shape[1] if graph.y.dim() > 1 else 1
        vel_slice = slice(1, 1 + vel_dim)

        if predicted_velocity is not None:
            graph.x[:, vel_slice] = predicted_velocity.detach()

        # target is next-step velocity stored in graph.y
        next_v = graph.y

        # model forward, do not perform noise injection in rollout
        predicted_velocity = model(graph, velocity_sequence_noise=None)

        # enforce boundary (mask): copy ground-truth for masked nodes
        # mask True = nodes to force -> predicted_velocity[mask] = next_v[mask]
        if mask.any():
            predicted_velocity[mask] = next_v[mask]

        predicteds.append(predicted_velocity.detach().cpu().numpy())
        targets.append(next_v.detach().cpu().numpy())

        crds = graph.pos.detach().cpu().numpy()  # update coords, final value used for saving

    predicteds = np.stack(predicteds)  # shape (T, N, vel_dim)
    targets = np.stack(targets)
    return [predicteds, targets], crds


def main():
    args = parse_args()

    # device selection
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # instantiate model
    simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
    # load checkpoint
    simulator.load_checkpoint(ckp_dir=args.model_dir)
    simulator.eval()
    simulator.to(device)

    # instantiate dataset
    dataset = IndexedTrajectoryDataset(
        dataset_dir=args.dataset_dir,
        split=args.test_split,
        time_interval=0.01,
        cache_static=args.cache_static,
        preserve_one_hot=args.preserve_one_hot
    )

    # optional transforms
    transformer = None
    if args.transform:
        transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    os.makedirs('results', exist_ok=True)

    # rollout over the first args.rollout_num trajectories
    n_trajs = min(args.rollout_num, len(dataset.traj_keys))
    for i in range(n_trajs):
        print(f"Starting rollout on trajectory {i} (key={dataset.traj_keys[i]})")
        result, coords = rollout_one_trajectory(simulator, dataset, traj_index=i, transformer=transformer, device=device)
        predicteds, targets = result

        # save results
        out_path = os.path.join('results', f'results_{i}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump([result, coords], f)
        print(f"Saved rollout results of trajectory {i} to {out_path}")

        # compute & print error summary
        rollout_error(predicteds, targets)
        print('------------------------------------------------------------------')


if __name__ == '__main__':
    main()
