import torch
from model.gnn_flux import FluxGNN
from dataset.euler_flux import EulerPeriodicDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

h5_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.76_Ar_-180.hdf5"
stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"
dataset = EulerPeriodicDataset(h5_path, stats_path=stats_path, time_window=2, target='delta', normalize=True, coarsen=(1,1), to_centroids=True)

sample = dataset[0]  # torch_geometric.data.Data
input_node_feats = sample.x.shape[1]        # (time_window-1) * 5
global_feat_dim = sample.u.shape[1] if getattr(sample, "u", None) is not None else 0

model = FluxGNN(
    in_node_dim=input_node_feats
).to(device)

# example forward on a batch
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

batch = next(iter(loader)).to(device)

# FORWARD PASS
out = model(batch, dt_cfl=0.015)  # dict with keys: 'U0', 'U_final', 'delta_U', 'density', 'momentum', 'energy', 'pressure', 'dt_layers'

print("Output keys:", out.keys())
print("U0 shape:", out['U0'].shape)
print("U_final shape:", out['U_final'].shape)
print("delta_U shape:", out['delta_U'].shape)
print("density shape:", out['density'].shape)
print("momentum shape:", out['momentum'].shape)
print("energy shape:", out['energy'].shape)
print("pressure shape:", out['pressure'].shape)

# print some mean values
print("Mean density:", out['density'].mean().item())
print("Mean pressure:", out['pressure'].mean().item())
print("Mean energy:", out['energy'].mean().item())

# print dt_layers
print("dt_layers:", out['dt_layers'])