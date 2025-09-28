import os
import argparse
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from dataset.cylinder import IndexedTrajectoryDataset  
from model.model import Simulator
from utils.utils import NodeType, get_velocity_noise


def train(
    model: "Simulator",
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    transformer: T.Compose,
    noise_std: float = 2e-2,
    print_batch: int = 20,
    save_batch: int = 200,
    loss_save_path: str = "losses.pt",
    max_epochs: int = 1,
):
    """
    Train the Simulator model using data from the dataloader.
    Args:
      model: the Simulator model to train
      dataloader: DataLoader providing batches of graphs from IndexedTrajectoryDataset
      optimizer: optimizer for training (e.g. Adam)
      device: torch device to use (e.g. 'cpu' or 'cuda')
      transformer: transform to apply to each graph (e.g. FaceToEdge + Cartesian + Distance)
      noise_std: standard deviation of Gaussian noise to add to input velocities during training
      print_batch: how often (in batches) to print training loss
      save_batch: how often (in batches) to save model checkpoint and loss history
      loss_save_path: filename to save loss history to (in 'checkpoint' directory)
      max_epochs: number of epochs to train for
    Returns:
      None
    """
    model.to(device)

    # global training state
    global_step = 0
    losses = []  # keeps accumulating loss values across all epochs

    # try to compute steps-per-epoch for nicer reporting (not strictly required)
    steps_per_epoch = len(dataloader) if hasattr(dataloader, '__len__') else None

    for epoch in range(max_epochs):
        model.train()

        for batch_index, graph in enumerate(dataloader):
            # apply transforms
            graph = transformer(graph)

            # move graph to device
            graph = graph.to(device)

            node_type = graph.x[:, 0]  # node_type, cur_v, pressure, time

            # noise injection
            velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)

            predicted_delta, target_delta = model(graph, velocity_sequence_noise)

            # mask out nodes at boundaries (NORMAL or OUTFLOW)
            mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)

            # if there are no nodes to compute loss for in this batch, skip
            if not mask.any():
                continue

            errors = ((predicted_delta - target_delta) ** 2)[mask]
            loss = torch.mean(errors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store loss
            losses.append(loss.item())

            # print using global step so prints are spaced across epochs
            if global_step % print_batch == 0:
                print(f"Epoch {epoch+1}/{max_epochs} — batch {batch_index} (global {global_step}) — loss: {loss.item():.2e}")

            # save checkpoints periodically
            if global_step % save_batch == 0:
                try:
                    model.save_checkpoint()
                except Exception:
                    print('Warning: save_checkpoint() failed at global step', global_step)

                # also save intermediate loss history
                try:
                    os.makedirs("checkpoint", exist_ok=True)
                    torch.save(losses, os.path.join("checkpoint", loss_save_path))
                except Exception:
                    print('Warning: could not save loss list at global step', global_step)

            global_step += 1

        # end of epoch reporting
        if steps_per_epoch is not None:
            # take the last `steps_per_epoch` losses to compute epoch mean
            epoch_slice = losses[-steps_per_epoch:]
            if epoch_slice:
                print(f"Finished epoch {epoch+1}/{max_epochs} — mean loss: {sum(epoch_slice)/len(epoch_slice):.2e}")
        else:
            print(f"Finished epoch {epoch+1}/{max_epochs} — total steps so far: {global_step}")

    # final save of losses
    try:
        os.makedirs("checkpoint", exist_ok=True)
        torch.save(losses, os.path.join("checkpoint", loss_save_path))
        print(f"Saved loss history to checkpoint/{loss_save_path}")
    except Exception:
        print('Warning: could not save final loss list')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default="/work/scitas-share/boscario/cylinder_flow_h5")
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--print-batch', type=int, default=20)
    parser.add_argument('--save-batch', type=int, default=200)
    parser.add_argument('--noise-std', type=float, default=2e-2)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max-epochs', type=int, default=5, help='Number of epochs to train for')
    args = parser.parse_args()

    # ----- device -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- model & optimizer -----
    simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
    simulator.to(device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
    print('Optimizer initialized')

    # ----- dataset & loader -----
    dataset_cylinder = IndexedTrajectoryDataset(dataset_dir=args.dataset_dir, split='train',
                                               time_interval=0.01, cache_static=True)

    train_loader = DataLoader(dataset=dataset_cylinder,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              persistent_workers=True)

    # ----- transforms -----
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    # ----- start training -----
    train(
        model=simulator,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        transformer=transformer,
        noise_std=args.noise_std,
        print_batch=args.print_batch,
        save_batch=args.save_batch,
        loss_save_path="losses.pt",
        max_epochs=args.max_epochs,
    )
