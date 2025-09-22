import os
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from dataset.cylinder import TrajectoryIterableDataset
from model.model import Simulator
from utils.utils import NodeType, get_velocity_noise

def train(
    model: Simulator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    transformer: T.Compose,
    noise_std: float = 2e-2,
    print_batch: int = 20,
    save_batch: int = 200,
    loss_save_path: str = "losses.pt",
):
    model.to(device)
    model.train()

    # collect per-batch loss values
    losses = []

    for batch_index, graph in enumerate(dataloader):
        # apply transforms (usually operates on CPU tensors)
        graph = transformer(graph)

        # move graph to device
        graph = graph.to(device)

        node_type = graph.x[:, 0]  # "node_type, cur_v, pressure, time"

        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)

        predicted_delta, target_delta = model(graph, velocity_sequence_noise)

        # mask out nodes we don't want to compute loss for
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

        if batch_index % print_batch == 0:
            print('batch: %d [loss: %.2e]' % (batch_index, loss.item()))

        if batch_index % save_batch == 0:
            try:
                model.save_checkpoint()
            except Exception:
                # warn on save failure
                print('Warning: save_checkpoint() failed at batch', batch_index)

            # also save intermediate loss history
            try:
                os.makedirs("checkpoint", exist_ok=True)
                torch.save(losses, os.path.join("checkpoint", "losses.pt"))
            except Exception:
                print('Warning: could not save loss list at batch', batch_index)
        
    # final save of losses
    try:
        os.makedirs("checkpoint", exist_ok=True)
        torch.save(losses, os.path.join("checkpoint", "losses.pt"))
        print('Saved loss history to checkpoint/losses.pt')
    except Exception:
        print('Warning: could not save final loss list')

if __name__ == '__main__':
    # ----- device -----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----- model & optimizer -----
    simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
    simulator.to(device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
    print('Optimizer initialized')

    # ----- dataset directory -----
    dataset_dir = "/work/scitas-share/boscario/cylinder_flow_h5"

    # ----- batch options -----
    batch_size = 64
    print_batch = 20
    save_batch = 200

    # ----- noise injection std -----
    noise_std = 2e-2

    # ----- dataset & loader -----
    dataset_cylinder = TrajectoryIterableDataset(dataset_dir=dataset_dir, split='train', max_epochs=10)
    train_loader = DataLoader(dataset=dataset_cylinder, batch_size=batch_size, num_workers=10)

    # ----- transforms -----
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])

    # ----- start training -----
    train(
        model=simulator,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        transformer=transformer,
        noise_std=noise_std,
        print_batch=print_batch,
        save_batch=save_batch,
        loss_save_path="losses.pt",
    )
