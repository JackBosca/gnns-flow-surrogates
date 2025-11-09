import torch.nn as nn
from egnn_pytorch import EGNN

class EulerEGNN(nn.Module):
    """
    EGNN for predicting state variables (density, energy, pressure, momentum)
    on 2D vertex grids from EulerPeriodicDataset.
    """
    def __init__(self, 
                 in_node_features,   # C_in from dataset (time_window * 5)
                 hidden_dim=64,
                 n_layers=4,
                 n_edge_features=4,  # dx, dy, r, wrap_flag
                 global_features=2,  # gamma + normalized time
                 out_node_features=5 # density, energy, pressure, momentum_x, momentum_y
                 ):
        super().__init__()

        # EGNN backbone
        self.egnn = EGNN(
            dim=in_node_features,
            edge_dim=n_edge_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            use_edge_features=True,
            dropout=0.0,
            residual=True,
            global_features_dim=global_features
        )

        # final node-wise MLP head to map hidden features -> state variables
        self.node_out = nn.Linear(hidden_dim, out_node_features)

    def forward(self, data):
        """
        Forward pass using a PyG Data object from EulerPeriodicDataset.
        Args:
            data: PyG Data object with:
                  - x: node features (N, C_in)
                  - pos: node positions (N, 2)
                  - edge_index: edges (2, E)
                  - edge_attr: edge features (E, 4)
                  - u: global features (1, global_features)
        Returns:
            y_pred: predicted node states (N, 5)
        """
        # EGNN returns updated node features and optionally positions
        h, _ = self.egnn(
            x=data.x, 
            edge_index=data.edge_index, 
            edge_attr=data.edge_attr, 
            pos=data.pos, 
            u=data.u
        )

        # map to output state variables
        y_pred = self.node_out(h)  # shape: (N, 5)
        return y_pred
