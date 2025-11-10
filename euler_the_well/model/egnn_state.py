import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Sparse_Network

class EGNNStateModel(nn.Module):
    """
    Wrapper around EGNN_Sparse_Network to predict (density, energy, pressure, momentum_x, momentum_y)
    (or delta values) per node.
    Notes:
    - global feats are concatenated to node features if provided, resulting in (C_in + G) per node.
    - EGNN_Sparse_Network expects input `x` with shape (N, pos_dim + feats_dim).
    - The final readout head maps per-node features (excluding coordinates) to 5 outputs:
      density, energy, pressure, momentum_x, momentum_y.
    """
    def __init__(
        self,
        input_feat_dim: int,                # number of node input channels (C_in)
        global_feat_dim: int = 0,           # number of scalar global features to concatenate (G)
        pos_dim: int = 2,                   # pos dimension
        edge_attr_dim: int = 4,             # edge attribute dimension (currently fixed to 4: [dx, dy, r, wrap_flag])
        egnn_hidden_feats: int = 64,        # internal feature dimensionality used by EGNN layers
        n_layers: int = 6,                  # number of EGNN layers
        fourier_features: int = 0,          # fourier encodings for distance (if desired)
        soft_edge: int = 0,
        update_feats: bool = True,
        update_coors: bool = False,         # positions are fixed (grid)
        dropout: float = 0.0,
        aggr: str = "add",
        readout_hidden: int = 128,          # hidden dim of readout MLP
        use_separate_heads: bool = False    # if True build separate MLPs per target; else one MLP outputs 5 channels
    ):
        super().__init__()

        # store dims
        self.input_feat_dim = int(input_feat_dim)
        self.global_feat_dim = int(global_feat_dim)
        self.pos_dim = int(pos_dim)

        # combined node feature dimension that EGNN will see (excluding pos_dim)
        self.node_feat_dim = self.input_feat_dim + self.global_feat_dim

        # build EGNN sparse network
        self.egnn = EGNN_Sparse_Network(
            n_layers = n_layers,
            feats_dim = self.node_feat_dim, # input node feature dimension
            pos_dim = self.pos_dim,
            edge_attr_dim = edge_attr_dim,            
            m_dim = egnn_hidden_feats,    # message dimension
            fourier_features = fourier_features,
            soft_edge = soft_edge,
            update_coors = update_coors,
            update_feats = update_feats,
            norm_feats = True,
            norm_coors = False,
            dropout = dropout,
            aggr = aggr
        )

        # order: [density, energy, pressure, momentum_x, momentum_y]
        out_channels = 5

        # EGNN returns a tensor of shape (N, pos_dim + feats_dim), so take feature part
        # (x[:, pos_dim:]) as the input to the readout MLPs
        if use_separate_heads:
            self.head_density = nn.Sequential(
                nn.Linear(self.node_feat_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, 1)
            )
            self.head_energy = nn.Sequential(
                nn.Linear(self.node_feat_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, 1)
            )
            self.head_pressure = nn.Sequential(
                nn.Linear(self.node_feat_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, 1)
            )
            self.head_momentum = nn.Sequential(
                nn.Linear(self.node_feat_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, 2)
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(self.node_feat_dim, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, out_channels)
            )

        # apply init
        if use_separate_heads:
            self.head_density.apply(self._init_linear)
            self.head_energy.apply(self._init_linear)
            self.head_pressure.apply(self._init_linear)
            self.head_momentum.apply(self._init_linear)
        else:
            self.readout.apply(self._init_linear)
        
        self.use_separate_heads = bool(use_separate_heads)

    def _init_linear(self, m):
        """Initialize Linear layers with Xavier uniform and zero biases."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _concat_globals_to_nodes(self, x_nodes, u, batch):
        """Broadcast global features u to nodes and concatenate."""
        if u is None:
            return x_nodes

        device = x_nodes.device
        u = u.to(device)
        N = x_nodes.shape[0]

        if u.shape[0] == batch.max().item() + 1:
            u_nodes = u[batch]
        elif u.shape[0] == 1:
            u_nodes = u.repeat(N, 1)
        else:
            raise ValueError(f"Unexpected u shape {u.shape}; expected (num_graphs, G) or (1, G).")

        return torch.cat([x_nodes, u_nodes], dim=-1)

    def forward(self, data):
        """
        Args:
            data: PyG Data or Batch with fields:
                - x: (N, C_in) node features (from dataset)
                - pos: (N, pos_dim) node positions
                - edge_index: (2, E) long tensor
                - edge_attr: (E, edge_attr_dim) float tensor (optional)
                - batch: (N,) long tensor mapping nodes -> graph (optional)
                - u: (num_graphs, G) or (1, G) global features (optional)
        Returns:
            dict with keys "density", "energy", "pressure", "momentum", "feats"
            where density/energy/pressure are (N,), momentum is (N,2), feats is (N, node_feat_dim).
        """
        x_nodes = data.x            # (N, C_in)
        pos = data.pos              # (N, pos_dim)
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        u = getattr(data, "u", None)  # global feats, (num_graphs, G) or (1, G)

        batch = getattr(data, "batch", None)
        if batch is None:
            # make a dummy batch for single graphs with all zeros so forward works
            # also for single graph inputs (not coming from a dataloader batch)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        device = x_nodes.device

        # concatenate global features to node features if present
        node_feats = self._concat_globals_to_nodes(x_nodes, u, batch)  # (N, C_in + G)

        # build EGNN input, concat pos and node_feats
        pos = pos.to(device)
        batch = batch.to(device)
        node_feats = node_feats.to(device)
        x_input = torch.cat([pos, node_feats], dim=-1)  # shape (N, pos_dim + node_feat_dim)

        # EGNN returns (N, pos_dim + feats_dim_out)
        egnn_out = self.egnn(x_input, edge_index.to(device), batch=batch, edge_attr=(edge_attr.to(device) if edge_attr is not None else None))

        # features are the part after positional dims
        feats_out = egnn_out[:, self.pos_dim:]  # (N, feats_dim_out)

        # readout
        if self.use_separate_heads:
            density = self.head_density(feats_out).squeeze(-1)
            energy  = self.head_energy(feats_out).squeeze(-1)
            pressure= self.head_pressure(feats_out).squeeze(-1)
            momentum= self.head_momentum(feats_out)  # (N,2)
        else:
            out = self.readout(feats_out)  # (N, 5)
            density = out[:, 0]
            energy  = out[:, 1]
            pressure= out[:, 2]
            momentum= out[:, 3:5]

        return {
            "density": density,
            "energy": energy,
            "pressure": pressure,
            "momentum": momentum,
            "feats": feats_out
        }
