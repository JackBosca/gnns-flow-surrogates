import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, Optional
from dataset.euler_coarse import EulerPeriodicDataset
from model.egnn_state import EGNNStateModel

def train_one_epoch(model, dataloader, optimizer, device,
                    loss_weights: Optional[Dict[str, float]] = None, clip_grad: Optional[float] = None):
    """
    Args:
        model: model returning dict with keys "density","energy","pressure","momentum".
        dataloader: PyG DataLoader yielding Batch objects with fields:
                    .x, .pos, .edge_index, .edge_attr (opt), .batch (opt),
                    and ground-truth: .y_density, .y_energy, .y_pressure, .y_momentum
        optimizer: optimizer to step.
        device: torch device.
        loss_weights: dict with keys "density","energy","pressure","momentum" -> float.
                      If None, equal weighting is used.
        clip_grad: optional max-norm for gradients.
    Returns:
        dict with averaged metrics: loss, mse_density, mse_energy, mse_pressure, mse_momentum, num_nodes
    """
    model.train()
    model.to(device)

    if loss_weights is None:
        loss_weights = {"density": 1.0, "energy": 1.0, "pressure": 1.0, "momentum": 1.0}

    total_loss = 0.0
    total_mse_density = 0.0
    total_mse_energy = 0.0
    total_mse_pressure = 0.0
    total_mse_momentum = 0.0
    total_nodes = 0

    for step, batch in enumerate(dataloader):
        batch = batch.to(device)

        # ground truth
        y_density = getattr(batch, "y_density", None).view(-1)
        y_energy  = getattr(batch, "y_energy", None).view(-1)
        y_pressure= getattr(batch, "y_pressure", None).view(-1)
        y_momentum= getattr(batch, "y_momentum", None).view(-1, 2)

        # forward
        preds = model(batch)

        p_density = preds["density"].view(-1)
        p_energy  = preds["energy"].view(-1)
        p_pressure= preds["pressure"].view(-1)
        p_momentum= preds["momentum"].view(-1, 2)

        mse_density = F.mse_loss(p_density, y_density, reduction="mean")
        mse_energy  = F.mse_loss(p_energy, y_energy, reduction="mean")
        mse_pressure= F.mse_loss(p_pressure, y_pressure, reduction="mean")
        mse_momentum= F.mse_loss(p_momentum, y_momentum, reduction="mean")

        loss = (
            loss_weights.get("density") * mse_density
            + loss_weights.get("energy") * mse_energy
            + loss_weights.get("pressure") * mse_pressure
            + loss_weights.get("momentum") * mse_momentum
        )

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        print(f"\nCurrent step: {step+1}/{len(dataloader)}")
        print(f"Batch Loss: {loss.item():.6f} "
              f"(Density MSE: {mse_density.item():.6f}, "
              f"Energy MSE: {mse_energy.item():.6f}, "
              f"Pressure MSE: {mse_pressure.item():.6f}, "
              f"Momentum MSE: {mse_momentum.item():.6f})")

        # accumulate (weight by number of nodes so averaging is correct)
        n_nodes = p_density.numel()
        total_nodes += n_nodes
        total_loss += float(loss.detach()) * n_nodes
        total_mse_density += float(mse_density.detach()) * n_nodes
        total_mse_energy  += float(mse_energy.detach()) * n_nodes
        total_mse_pressure+= float(mse_pressure.detach()) * n_nodes
        total_mse_momentum+= float(mse_momentum.detach()) * n_nodes

    if total_nodes == 0:
        return {
            "loss": 0.0,
            "mse_density": 0.0,
            "mse_energy": 0.0,
            "mse_pressure": 0.0,
            "mse_momentum": 0.0,
            "num_nodes": 0
        }

    return {
        "loss": total_loss / total_nodes,
        "mse_density": total_mse_density / total_nodes,
        "mse_energy": total_mse_energy / total_nodes,
        "mse_pressure": total_mse_pressure / total_nodes,
        "mse_momentum": total_mse_momentum / total_nodes,
        "num_nodes": total_nodes
    }


def train(model, train_loader, val_loader=None, optimizer=None, criterion=None, 
          device="cuda", epochs=10, save_dir="./checkpoints", save_every=1, fname="model"):
    """
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: optional DataLoader for validation
        optimizer: PyTorch optimizer
        criterion: loss function
        device: "cuda" or "cpu"
        epochs: number of epochs to train
        save_dir: directory to save checkpoints
        save_every: save model every N epochs
    """
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # train one epoch
        results = train_one_epoch(model, train_loader, optimizer, device)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {results['loss']:.6f}")
        
        # optional validation
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    loss = criterion(output, batch['y'])
                    val_loss_total += loss.item()
            val_loss = val_loss_total / len(val_loader)
            print(f"Epoch {epoch}/{epochs} - Val Loss: {val_loss:.6f}")
            model.train()
        
        # Save checkpoint
        if epoch % save_every == 0:
            # complete fname with epoch
            fname_epoch = f"{fname}_epoch_{epoch}.pt"
            checkpoint_path = os.path.join(save_dir, fname_epoch)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    # h5 files path
    h5_path_train = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    h5_path_valid = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/valid/euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5"
    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"

    # create dataset and dataloader
    coarsen = (2,2)
    train_dataset = EulerPeriodicDataset(h5_path=h5_path_train, stats_path=stats_path, coarsen=coarsen)
    valid_dataset = EulerPeriodicDataset(h5_path=h5_path_valid, stats_path=stats_path)

    # print grid dimensions train
    print(f"Current grid dimension: {len(train_dataset._static_cache['x_coords_coarse'])}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # get a sample for input/global dims
    sample = train_dataset[0]
    input_node_feats = sample.x.shape[1]        # time_window * 5
    global_feat_dim = sample.u.shape[1] if getattr(sample, "u", None) is not None else 0

    # create model
    model = EGNNStateModel(input_feat_dim=input_node_feats, global_feat_dim=global_feat_dim, use_separate_heads=True)

    # use AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    fname = f"model_coarsen_{coarsen[0]}-{coarsen[1]}"

    # train the model
    train(model, train_loader, optimizer=optimizer, fname=fname)

    print("Training complete.")
