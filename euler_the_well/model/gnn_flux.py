import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Fully-connected MLP.
    - in_dim: input dimension
    - hidden_dims: list/tuple of hidden layer sizes (can be empty => single linear)
    - out_dim: output dimension
    - use_layernorm: if True, apply LayerNorm after each hidden layer
    """
    def __init__(self, in_dim, hidden_dims, out_dim, use_layernorm=False):
        super().__init__()

        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

        # weight init for stability
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (..., in_dim) torch tensor
        returns: (..., out_dim)
        """
        return self.net(x)
    

class EdgeFluxModule(nn.Module):
    """
    Build symmetric pairwise messages and produce per-edge shared fluxes:
    density/energy amplitudes (scalars) and vector momentum flux (2D).
    Inputs:
        - node_u: (N, node_in_dim) conserved variables per node (e.g. [rho, e, p, rho_u_x, rho_u_y])
        - edge_index: (2, E) long tensor
        - edge_attr: (E, 3) float tensor with [dx, dy, r]
    Outputs:
        dict with keys:
            "a_rho": (E,) mass flux scalar amplitude (normal component)
            "a_e": (E,) energy flux scalar amplitude (normal component)
            "a_p": (E,) pressure flux scalar amplitude (normal component)
            "F_rho": (E,2) mass flux vector = a_rho * n
            "F_e": (E,2) energy flux vector = a_e * n
            "F_p": (E,2) pressure flux vector = a_p * n
            "F_rhou": (E,2) momentum flux vector = a_n * n + a_t * t
            "n": (E,2) unit normals (from source->dest)
            "r": (E,) distances (face lengths)
    """
    def __init__(self,
                 node_in_dim=5,
                 node_embed_dim=64,
                 edge_attr_dim=1,   # by default only use r for invariance
                 edge_embed_dim=32,
                 phi_hidden=(64,),
                 msg_hidden=(128,)):
        super().__init__()

        # node encoder
        self.phi_node = MLP(node_in_dim, hidden_dims=phi_hidden, out_dim=node_embed_dim)

        # phi1 and phi2 (deep-set) -> must have same in/out dims
        self.phi1 = MLP(node_embed_dim, hidden_dims=[node_embed_dim//2], out_dim=node_embed_dim)
        self.phi2 = MLP(node_embed_dim, hidden_dims=[node_embed_dim//2], out_dim=node_embed_dim)

        # edge encoder (takes r as attribute for invariance)
        self.phi_edge = MLP(edge_attr_dim, hidden_dims=[edge_embed_dim], out_dim=edge_embed_dim)

        # message network: input = phi1(h_i)+phi1(h_j)+phi2(h_i)+phi2(h_j) + eps_ij
        self.phi_msg = MLP(in_dim=node_embed_dim + edge_embed_dim,
                           hidden_dims=msg_hidden,
                           out_dim=node_embed_dim)

        # psi networks to produce flux amplitudes
        self.psi_rho  = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=1)
        self.psi_e    = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=1)
        self.psi_p    = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=1)
        self.psi_rhou = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=2)  # [a_n, a_t]

    def forward(self, node_u, edge_index, edge_attr):
        """
        node_u: (N, node_in_dim)
        edge_index: (2, E)
        edge_attr: (E, 3) [dx, dy, r]
        """
        src, dst = edge_index[0], edge_index[1]  # (E,)

        # node embeddings
        h = self.phi_node(node_u)                # (N, node_embed_dim)

        # gather per-edge node embeddings
        h_i = h[src]                             # (E, node_embed_dim)
        h_j = h[dst]                             # (E, node_embed_dim)

        # deep-set symmetric combination
        v = self.phi1(h_i) + self.phi1(h_j) + self.phi2(h_i) + self.phi2(h_j)  # (E, node_embed_dim)

        # edge embedding using ONLY r (edge_attr[:, 2:3]) for invariance, so m_ij = m_ji
        eps = self.phi_edge(edge_attr[:, 2:3])           # (E, edge_embed_dim)

        # message latent
        msg_in = torch.cat([v, eps], dim=-1)     # (E, node_embed_dim + edge_embed_dim)
        m_ij = self.phi_msg(msg_in)              # (E, node_embed_dim)

        # flux amplitudes
        a_rho = self.psi_rho(m_ij).squeeze(-1)   # (E,)
        a_e = self.psi_e(m_ij).squeeze(-1)     # (E,)
        a_p = self.psi_p(m_ij).squeeze(-1)     # (E,)
        a_rhou = self.psi_rhou(m_ij)             # (E,2) -> [a_n, a_t] (n, t local components of momentum flux)

        # compute unit normal n and tangential t from dx, dy
        dx = edge_attr[:, 0].unsqueeze(-1)       # (E,1)
        dy = edge_attr[:, 1].unsqueeze(-1)       # (E,1)
        r  = edge_attr[:, 2].unsqueeze(-1)       # (E,1) (distance)
        eps_r = 1e-12
        n = torch.cat([dx, dy], dim=-1) / (r + eps_r)  # (E,2) unit normal (source->dest)
        # tangential vector (90 deg rotation): t = (-n_y, n_x)
        t = torch.stack([-n[:, 1], n[:, 0]], dim=1)    # (E,2)

        # build vector fluxes
        F_rho = a_rho.unsqueeze(-1) * n           # (E,2)
        F_e = a_e.unsqueeze(-1) * n           # (E,2)
        F_p = a_p.unsqueeze(-1) * n           # (E,2)
        F_rhou = a_rhou[:, 0:1] * n + a_rhou[:, 1:2] * t  # (E,2), momentum flux vector not necessarily aligned with n

        return {
            "a_rho": a_rho,       # (E,)
            "a_e": a_e,           # (E,)
            "a_p": a_p,           # (E,)
            "F_rho": F_rho,       # (E,2)
            "F_e": F_e,           # (E,2)
            "F_p": F_p,           # (E,2)
            "F_rhou": F_rhou,     # (E,2)
            "n": n,               # (E,2)
            "r": r.squeeze(-1)    # (E,)
        }
    

class ConservativeMPLayer(nn.Module):
    """
    One conservative message-passing layer.
    Args:
        edge_module: instance of EdgeFluxModule (computes per-edge fluxes)
        dt_max: maximum allowed learned dt (float)
        init_s: initial value for s so sigmoid(s) small (default -5) and below CFL condition
    """ 
    def __init__(self, edge_module: EdgeFluxModule, dt_max: float = 0.015, init_s: float = -5.0):
        super().__init__()
        self.edge_module = edge_module
        self.dt_max = float(dt_max)
        # learnable scalar controlling dt per layer (will be sigmoided)
        self.s = nn.Parameter(torch.tensor(float(init_s), dtype=torch.float32))

    def forward(self, node_u, edge_index, edge_attr, cell_area=None, dt_cfl=None):
        """
        Args:
            node_u: (N, 5) tensor [rho, e, p, rho_u_x, rho_u_y]
            edge_index: (2, E) long tensor (directed edges, both directions present)
            edge_attr: (E, 3) float tensor [dx, dy, r] computed for the directed edge (src->dst)
            cell_area: None or tensor (N,) or float. If None, assumed uniform and area = mean(r)^2 approximated.
            dt_cfl: optional CFL condition scalar upper bound on dt (float). If provided, dt = min(dt, dt_cfl).
        Returns:
            node_u_new: (N,5) updated conserved variables
            diagnostics: dict with dt, optional flux norms
        """
        device = node_u.device
        N = node_u.shape[0]

        # take only edges with src < dst (one representative per undirected edge)
        src = edge_index[0]
        dst = edge_index[1]
        mask = src < dst
        src_u = src[mask]
        dst_u = dst[mask]
        edge_attr_u = edge_attr[mask]   # (E_u, 3)

        # call edge module to compute shared per-edge flux vectors
        out = self.edge_module(node_u, torch.stack([src_u, dst_u], dim=0), edge_attr_u)

        F_rho  = out["F_rho"]    # (E_u,2)
        F_e    = out["F_e"]      # (E_u,2)
        F_p    = out["F_p"]      # (E_u,2)
        F_rhou = out["F_rhou"]   # (E_u,2)
        n      = out["n"]        # (E_u,2) unit normal (from src_u -> dst_u)
        r_edge = out["r"]        # (E_u,)

        # face_length for 4-neighbour uniform grid equals r_edge
        face_length = r_edge     # (E_u,)

        #NOTE: here it might be easier to directly read the a_* values from the out dict
        # compute scalar normal fluxes for scalars: a = <F, n>
        a_rho = (F_rho * n).sum(dim=-1)    # (E_u,)
        a_e = (F_e * n).sum(dim=-1)    # (E_u,)
        a_p = (F_p * n).sum(dim=-1)    # (E_u,)

        # scale by face length to get integrated flux per edge
        scalar_flux_rho = a_rho * face_length    # (E_u,)
        scalar_flux_e = a_e * face_length    # (E_u,)
        scalar_flux_p = a_p * face_length    # (E_u,)
        vector_flux_rhou = F_rhou * face_length.unsqueeze(-1)  # (E_u, 2)

        # build per-edge raw contributions
        # for src node: contribution = - (dt / area_src) * (flux)
        # for dst node: contribution = - (dt / area_dst) * (-flux) = + (dt / area_dst) * flux
        r_raw = scalar_flux_rho        # (E_u,)
        e_raw = scalar_flux_e          # (E_u,)
        p_raw = scalar_flux_p          # (E_u,)
        m_raw = vector_flux_rhou       # (E_u,2)

        # cell_area handling: accept scalar or per-node tensor
        if cell_area is None:
            # approximate uniform cell area -> cell_area = dx^2 with dx = mean(r_edge)
            dx_est = torch.mean(r_edge)
            area = float(dx_est.item() ** 2)
            area_tensor = torch.full((N,), area, device=device, dtype=node_u.dtype)
        else:
            if isinstance(cell_area, (float, int)):
                # specified uniform area
                area_tensor = torch.full((N,), float(cell_area), device=device, dtype=node_u.dtype)
            else:
                area_tensor = cell_area.to(device=device, dtype=node_u.dtype)
                if area_tensor.dim() == 0:
                    area_tensor = area_tensor.repeat(N)

        # compute learned dt and optionally clip to CFL
        dt = self.dt_max * torch.sigmoid(self.s)
        if dt_cfl is not None:
            dt_val = float(dt_cfl)
            if dt.item() > dt_val:
                # clip to CFL value
                dt = torch.tensor(dt_val, device=device, dtype=dt.dtype)

        # compute per-edge contributions (scaled by dt / area of endpoint)
        # src contributions
        contrib_rho_src = - (dt * r_raw) / area_tensor[src_u]         # (E_u,)
        contrib_e_src   = - (dt * e_raw) / area_tensor[src_u]         # (E_u,)
        contrib_p_src   = - (dt * p_raw) / area_tensor[src_u]         # (E_u,)
        contrib_m_src   = - (dt * m_raw) / area_tensor[src_u].unsqueeze(-1)  # (E_u,2)

        # dst contributions (NOTE: sign flip inside formula for conservation)
        contrib_rho_dst = - (dt * (-r_raw)) / area_tensor[dst_u] 
        contrib_e_dst   = - (dt * (-e_raw)) / area_tensor[dst_u]
        contrib_p_dst   = - (dt * (-p_raw)) / area_tensor[dst_u]
        contrib_m_dst   = - (dt * (-m_raw)) / area_tensor[dst_u].unsqueeze(-1)

        # aggregate into node deltas using index_add
        delta_rho = torch.zeros((N,), device=device, dtype=node_u.dtype)
        delta_e = torch.zeros((N,), device=device, dtype=node_u.dtype)
        delta_p = torch.zeros((N,), device=device, dtype=node_u.dtype)
        delta_rhou = torch.zeros((N, 2), device=device, dtype=node_u.dtype)

        # accumulate src contributions
        delta_rho.index_add_(0, src_u, contrib_rho_src)
        delta_e.index_add_(0,   src_u, contrib_e_src)
        delta_p.index_add_(0,   src_u, contrib_p_src)
        delta_rhou.index_add_(0, src_u, contrib_m_src)

        # accumulate dst contributions
        delta_rho.index_add_(0, dst_u, contrib_rho_dst)
        delta_e.index_add_(0, dst_u, contrib_e_dst)
        delta_p.index_add_(0, dst_u, contrib_p_dst)
        delta_rhou.index_add_(0, dst_u, contrib_m_dst)

        # construct updated conserved U: [rho, rhou_x, rhou_y, e]
        rho = node_u[:, 0]
        e = node_u[:, 1]
        p = node_u[:, 2]
        rhou = node_u[:, 3:5]
        

        rho_new = rho + delta_rho
        e_new = e + delta_e
        p_new = p + delta_p
        rhou_new = rhou + delta_rhou

        node_u_new = torch.cat([rho_new.unsqueeze(-1),
                                e_new.unsqueeze(-1),
                                p_new.unsqueeze(-1),
                                rhou_new], dim=-1)

        return node_u_new, float(dt.item())


class FluxGNN(nn.Module):
    """
    - Encodes node input features (Data.x) into conserved U = [rho, e, p, rho_u_x, rho_u_y]
    - Applies `n_layers` conservative message-passing layers (each with its own EdgeFluxModule)
    - Returns predicted change (delta) in conserved variables by default (target='delta').
    - Also returns reconstructed primitives (rho, momentum, energy, pressure).
    Args:
        in_node_dim: input node feature dimension (Data.x.shape[1])
        node_hidden: hidden dims for node encoder MLP (list/tuple)
        node_embed_dim: internal node embedding size used by EdgeFluxModule (defaults 64)
        n_layers: number of conservative message-passing layers
        dt_max: maximum learned dt per layer (passed to ConservativeMPLayer)
        use_residual: whether to add skip connection from input U to final output (recommended True)
    """
    def __init__(self,
                 in_node_dim,
                 node_hidden=(128,),
                 node_embed_dim=64,
                 n_layers=4,
                 dt_max=0.015,
                 gamma=1.4):
        super().__init__()

        self.in_node_dim = in_node_dim
        self.node_embed_dim = node_embed_dim
        self.n_layers = int(n_layers)
        self.dt_max = float(dt_max)
        
        #TODO: remove gamma, not used
        self.gamma = float(gamma) if gamma is not None else None

        # encoder: map arbitrary input node features -> conserved U = [rho, e, p, rho_u_x, rho_u_y]
        # needed for ex for time_window>2
        self.input_encoder = MLP(in_dim=in_node_dim, hidden_dims=list(node_hidden), out_dim=5)

        # build layers (each layer has its own EdgeFluxModule + ConservativeMPLayer)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            edge_mod = EdgeFluxModule(node_in_dim=5,
                                      node_embed_dim=node_embed_dim,
                                      edge_attr_dim=1,
                                      edge_embed_dim=32,
                                      phi_hidden=(node_embed_dim//2,),
                                      msg_hidden=(128,))
            layer = ConservativeMPLayer(edge_module=edge_mod, dt_max=self.dt_max, init_s=-5.0)
            self.layers.append(layer)

    def forward(self, data, dt_cfl=None):
        """
        Args:
            data: PyG Data object with fields:
                  - x: (N, C_in) node features
                  - edge_index: (2, E) long
                  - edge_attr: (E, 3) float [dx, dy, r]
                  - u (optional): global features tensor (1, k)
                  - pos (optional): (N,2)
            dt_cfl: optional float scalar upper bound on dt per layer (CFL condition)
        Returns:
            dict with keys:
              'U0': initial conserved U (N,5)
              'U_final': final conserved U after layers (N,5)
              'delta_U': (N,5) = U_final - U0
              'rho': (N,) reconstructed density from U_final
              'energy': (N,) reconstructed total energy from U_final
              'pressure': (N,) reconstructed pressure from U_final
              'momentum': (N,2) reconstructed momentum from U_final
              'dt_layers': list of dt per layer (floats)
        """
        device = data.x.device if isinstance(data.x, torch.Tensor) else torch.device("cpu")
        x_nodes = data.x.to(device)

        edge_index = data.edge_index.to(device)
        edge_attr  = data.edge_attr.to(device)

        # initial conserved U from encoder
        U0 = self.input_encoder(x_nodes)    # (N,5)

        # optional clamp or positivity enforcement could be added here
        U = U0

        dt_layers = []
        # iterate conservative layers
        for layer in self.layers:
            U, dt_val = layer(U, edge_index, edge_attr, cell_area=None, dt_cfl=dt_cfl)
            dt_layers.append(float(dt_val))

        # split conserved vars
        rho = U[:, 0]                    # (N,)
        e = U[:, 1]                      # (N,)
        p = U[:, 2]                      # (N,)
        rhou = U[:, 3:5]                 # (N,2)
  
        output = {
            "U0": U0,
            "U_final": U,
            "delta_U": U - U0,
            "density": rho,
            "energy": e,
            "pressure": p,
            "momentum": rhou,
            "dt_layers": dt_layers
        }
        return output
