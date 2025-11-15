import torch
from model.egnn_state import EGNNStateModel
from dataset.euler_coarse import EulerPeriodicDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

h5_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/euler_multi_quadrants_periodicBC_gamma_1.76_Ar_-180.hdf5"
stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"
dataset = EulerPeriodicDataset(h5_path, stats_path=stats_path, time_window=2, target='delta', normalize=True, coarsen=(1,1))

sample = dataset[0]  # torch_geometric.data.Data
input_node_feats = sample.x.shape[1]        # time_window * 5
global_feat_dim = sample.u.shape[1] if getattr(sample, "u", None) is not None else 0

model = EGNNStateModel(
    input_feat_dim=input_node_feats,
    global_feat_dim=global_feat_dim
).to(device)

# example forward on a batch
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

batch = next(iter(loader)).to(device)
preds = model(batch)
# preds is dict with tensors shaped (N,)

print("Output keys:", preds.keys())
# print some values of the outputs
for key, value in preds.items():
    print(f"{key}: mean={value.mean().item():.4f}, std={value.std().item():.4f}")