import torch
import torch.nn as nn
# from model_utils import denorm, norm, eos
from utils import denorm, norm, eos

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
            "F_rho": (E,2) mass flux vector = a_rho * n
            "F_e": (E,2) energy flux vector = a_e * n
            "F_rhou": (E,2) momentum flux vector = a_n * n + a_t * t
            "n": (E,2) unit normals (from source->dest)
            "r": (E,) distances (face lengths)
    """
    def __init__(self,
                 node_in_dim=5,
                 node_embed_dim=64,
                 edge_attr_dim=1,
                 edge_embed_dim=32,
                 phi_hidden=(64,),
                 msg_hidden=(128,)):
        super().__init__()

        # node encoder
        self.phi_node = MLP(node_in_dim, hidden_dims=phi_hidden, out_dim=node_embed_dim)

        # embedding for residual connection to match node_embed_dim
        self.input_proj = nn.Linear(node_in_dim, node_embed_dim)

        # --- MODIFICATION START ---
        # edge encoder: NOW TAKES ONLY 1 DIMENSION (r) instead of 3 (dx, dy, r)
        self.phi_edge = MLP(edge_attr_dim, hidden_dims=[edge_embed_dim], out_dim=edge_embed_dim)

        # message network inputs:
        # h_src (D) + h_dst (D) + v_avg (D) + v_diff (D) + eps (E_dim) 
        # + 4 NEW SCALARS: [u_n_src, u_t_src, u_n_dst, u_t_dst]
        self.phi_msg = MLP(in_dim=4*node_embed_dim + edge_embed_dim + 4,
                           hidden_dims=msg_hidden,
                           out_dim=node_embed_dim)
        # --- MODIFICATION END ---

        # # symmetric phi -> must have same in/out dims
        # # self.phi_symm = MLP(node_embed_dim, hidden_dims=(96,), out_dim=node_embed_dim)

        # # edge encoder
        # self.phi_edge = MLP(edge_attr_dim, hidden_dims=[edge_embed_dim], out_dim=edge_embed_dim)

        # # message network: input = h_i | h_j | h_i+h_j | (h_i-h_j)^2 | eps_ij
        # self.phi_msg = MLP(in_dim=4*node_embed_dim + edge_embed_dim,
        #                    hidden_dims=msg_hidden,
        #                    out_dim=node_embed_dim)

        # psi networks to produce flux amplitudes
        self.psi_rho  = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=1)
        self.psi_e    = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=1)
        self.psi_rhou = MLP(in_dim=node_embed_dim, hidden_dims=[64], out_dim=2)  # [a_n, a_t]

        # predicts a scalar coefficient for diffusion (artificial viscosity)
        self.psi_visc = MLP(in_dim=node_embed_dim, hidden_dims=[32], out_dim=1)

        # LayerNorm for the edge memory
        self.memory_norm = nn.LayerNorm(node_embed_dim)

    def forward(self, node_u, edge_index, edge_attr, stats, edge_memory=None):
        """
        node_u: (N, node_in_dim)
        edge_index: (2, E)
        edge_attr: (E, 3) [dx, dy, r]
        """
        src, dst = edge_index[0], edge_index[1]  # (E,)
        dev = node_u.device

        # --- MODIFICATION START: Geometry & Physical Projection ---
        # 1. Geometry
        dx = edge_attr[:, 0].unsqueeze(-1)       # (E,1)
        dy = edge_attr[:, 1].unsqueeze(-1)       # (E,1)
        r  = edge_attr[:, 2].unsqueeze(-1)       # (E,1)

        # 2. Local Basis
        eps_r = 1e-12
        n = torch.cat([dx, dy], dim=-1) / (r + eps_r)  # (E,2) Normal
        t = torch.stack([-n[:, 1], n[:, 0]], dim=1)    # (E,2) Tangent

        # 3. Extract Normalized Momentum
        if node_u.shape[1] > 5:
            curr_state = node_u[:, -5:] 
        else:
            curr_state = node_u
            
        rhou_vec = curr_state[:, 3:5] # (N, 2) Normalized

        # 4. De-normalize to Physical Momentum to get correct Direction
        # We need the physical sign to know if flow is entering/leaving
        # (Handle stats which might be lists/floats)
        if not hasattr(self, '_stats_tensors'):
             self._stats_mean_mom = torch.tensor(stats["mean"]["momentum"], device=dev)
             self._stats_std_mom  = torch.tensor(stats["std"]["momentum"], device=dev)
             # Cache to avoid recreating tensors every forward (optional but good for speed)
             # Or just create them on the fly if stats change per batch (unlikely)
        
        # Robust on-the-fly creation (safer if you switch devices)
        mean_mom = torch.as_tensor(stats["mean"]["momentum"], device=dev)
        std_mom  = torch.as_tensor(stats["std"]["momentum"], device=dev)

        rhou_phys = rhou_vec * std_mom + mean_mom  # (N, 2) Physical units

        # 5. Project PHYSICAL Momentum onto Edge Basis
        # This preserves the correct sign (direction) relative to the wall
        rhou_src = rhou_phys[src] # (E, 2)
        rhou_dst = rhou_phys[dst] # (E, 2)

        mom_n_src = (rhou_src * n).sum(dim=-1, keepdim=True) # (E, 1)
        mom_t_src = (rhou_src * t).sum(dim=-1, keepdim=True) # (E, 1)
        mom_n_dst = (rhou_dst * n).sum(dim=-1, keepdim=True) # (E, 1)
        mom_t_dst = (rhou_dst * t).sum(dim=-1, keepdim=True) # (E, 1)
        
        # 6. Re-scale for the Network
        # rhou_phys is large. We want input ~ O(1).
        # We divide by the magnitude of the momentum std.
        # This is like Z-score normalizing, but WITHOUT shifting the mean.
        # We KEEP the mean shift so that 0.0 really means "no flow".
        scale_factor = torch.norm(std_mom) + 1e-8
        
        vel_proj = torch.cat([mom_n_src, mom_t_src, mom_n_dst, mom_t_dst], dim=-1) / scale_factor
        # --- MODIFICATION END ---

        # node embeddings
        h = self.phi_node(node_u)                # (N, node_embed_dim)

        # residual connection embedding
        res = self.input_proj(node_u)            # (N, node_embed_dim)
        h = h + res                             # (N, node_embed_dim)

        # gather per-edge node embeddings
        #NOTE: pytorch effectively "expands" node features h onto the edges. 
        # Because there are more edges than nodes, this duplicates the node features 
        # for every edge that node participates in
        h_src = h[src]                             # (E, node_embed_dim)
        h_dst = h[dst]                             # (E, node_embed_dim)

        # # compute transformed node features once
        # h_transformed = self.phi_symm(h)  # (N, node_embed_dim)

        # # gather for edges
        # h_src_trans = h_transformed[src]
        # h_dst_trans = h_transformed[dst]

        # 1) deep-set symmetric combination -> tells the net: "we have high pressure here"
        # v_avg = h_src_trans + h_dst_trans # (E, node_embed_dim)
        v_avg = h_src + h_dst # (E, node_embed_dim)

        # 2) symmetric difference ("gradient" magnitude) -> tells the net: "there is a shockwave here"
        # (h_src - h_dst)^2 is strictly symmetric
        v_diff = (h_src - h_dst).pow(2) # (E, node_embed_dim)

        # 2.1) symmetric difference ("gradient" magnitude) -> tells the net: "there is a shockwave here"
        # |h_src - h_dst| is strictly symmetric (shock sensor)
        # v_diff = torch.abs(h_src - h_dst) # (E, node_embed_dim)

        # --- MODIFICATION START: Symmetric Edge Encoding ---
        # Only use 'r' for the geometry embedding. 
        # The network now gets orientation info from 'vel_proj', not 'dx/dy'.
        eps = self.phi_edge(r)                   # (E, edge_embed_dim)
        
        # Add vel_proj to the message input
        msg_in = torch.cat([h_src, h_dst, v_avg, v_diff, eps, vel_proj], dim=-1) 
        # --- MODIFICATION END ---

        m_ij = self.phi_msg(msg_in)              # (E, node_embed_dim)

        # Edge Memory Logic
        if edge_memory is not None:
            # Apply Residual + LayerNorm: LayerNorm(New + Old)
            m_ij = self.memory_norm(m_ij + edge_memory)
            # pass

        # flux amplitudes
        a_rho = self.psi_rho(m_ij).squeeze(-1)   # (E,)
        a_e = self.psi_e(m_ij).squeeze(-1)     # (E,)
        a_rhou = self.psi_rhou(m_ij)             # (E,2) -> [a_n, a_t] (n, t local components of momentum flux)

        # # compute unit normal n and tangential t from dx, dy
        # dx = edge_attr[:, 0].unsqueeze(-1)       # (E,1)
        # dy = edge_attr[:, 1].unsqueeze(-1)       # (E,1)
        # r  = edge_attr[:, 2].unsqueeze(-1)       # (E,1) (distance)
        # eps_r = 1e-12
        # n = torch.cat([dx, dy], dim=-1) / (r + eps_r)  # (E,2) unit normal (source->dest)
        # # tangential vector (90 deg rotation): t = (-n_y, n_x)
        # t = torch.stack([-n[:, 1], n[:, 0]], dim=1)    # (E,2)

        # build vector fluxes
        F_rho = a_rho.unsqueeze(-1) * n           # (E,2)
        F_e = a_e.unsqueeze(-1) * n           # (E,2)
        F_rhou = a_rhou[:, 0:1] * n + a_rhou[:, 1:2] * t  # (E,2), momentum flux vector not necessarily aligned with n

        # diffusion coefficient (artificial viscosity) shared among quantities
        alpha = torch.sigmoid(self.psi_visc(m_ij).squeeze(-1)).unsqueeze(-1)  # (E,1), in [0,1]

        # MAX_ALPHA = 0.15
        # alpha = alpha * MAX_ALPHA

        # # density scale
        # scale_rho = 1.0 + torch.abs(a_rho).unsqueeze(-1).detach()
        
        # # energy scale
        # scale_e = 1.0 + torch.abs(a_e).unsqueeze(-1).detach()
        
        # # momentum scaler
        # scale_rhou = 1.0 + torch.norm(F_rhou, dim=-1, keepdim=True).detach()

        # # if node_u is 5 dim, use it all, if >5, take last 5
        # if node_u.shape[1] > 5:
        #      current_u = node_u[:, -5:] 
        # else:
        #      current_u = node_u

        # # gather state
        # u_src = current_u[src]
        # u_dst = current_u[dst]
        
        # # calculate jump (vector of size 5)
        # diff_u = u_dst - u_src # [rho, e, p, rhou_x, rhou_y]

        u_src_full = curr_state[src]
        u_dst_full = curr_state[dst]
        diff_u = u_dst_full - u_src_full

        # F_rho corresponds to diff_u[:, 0] (density)
        # F_e corresponds to diff_u[:, 1] (energy)
        # F_rhou corresponds to diff_u[:, 3:5] (momentum)
        #NOTE: skip pressure (index 2) because flux of pressure isn't a conserved variable update
        diff_rho  = diff_u[:, 0].unsqueeze(-1)  # (E, 1)
        diff_e    = diff_u[:, 1].unsqueeze(-1)  # (E, 1)
        diff_rhou = diff_u[:, 3:5]              # (E, 2)
        
        # apply adaptive diffusion (per-channel)
        # density: scale by density flux
        # F_rho_final  = F_rho  - (alpha * scale_rho) * diff_rho * n
        F_rho_final  = F_rho  - alpha * diff_rho * n
        
        # energy: scale by energy flux
        # F_e_final    = F_e    - (alpha * scale_e) * diff_e * n
        F_e_final    = F_e    - alpha * diff_e * n
        
        # momentum: scale by momentum flux
        # F_rhou_final = F_rhou - (alpha * scale_rhou) * diff_rhou # momentum is already a vector, no need to multiply by n
        F_rhou_final = F_rhou - alpha * diff_rhou # momentum is already a vector, no need to multiply by n

        return {
            "F_rho": F_rho_final,
            "F_e": F_e_final,
            "F_rhou": F_rhou_final,
            "n": n,
            "r": r.squeeze(-1),
            "alpha": alpha.squeeze(-1),
            "edge_memory": m_ij  # return updated edge memory
        }
    

class ConservativeMPLayer(nn.Module):
    """
    One conservative message-passing layer.
    Args:
        edge_module: instance of EdgeFluxModule (computes per-edge fluxes)
    """ 
    def __init__(self, edge_module: EdgeFluxModule):
        super().__init__()
        self.edge_module = edge_module

    def forward(self, node_state, node_feat, edge_index, edge_attr, gamma, stats, dt, cell_area=None, dt_cfl=None, edge_memory=None):
        """
        Args:
            node_state: (N, 5) tensor [rho, e, p, rho_u_x, rho_u_y]
            node_feat: (N, F) tensor of additional node features (time_window > 2)
            edge_index: (2, E) long tensor (directed edges, both directions present)
            edge_attr: (E, 3) float tensor [dx, dy, r] computed for the directed edge (src->dst)
            gamma: specific heat ratio (float)
            stats: dict with dataset statistics for (de)normalization
            dt: time step (float)
            cell_area: None or tensor (N,) or float. If None, assumed uniform and area = mean(r)^2 approximated.
            dt_cfl: optional CFL condition scalar upper bound on dt (float). If provided, dt = min(dt, dt_cfl).
        Returns:
            node_u_new: (N,5) updated conserved variables
            diagnostics: dict with dt, optional flux norms
        """
        device = node_state.device
        N = node_state.shape[0]
 
        # take only edges with src < dst (one representative per undirected edge)
        src = edge_index[0]
        dst = edge_index[1]
        mask = src < dst
        src_u = src[mask] # (E_u,) = (E/2,)
        dst_u = dst[mask] # (E_u,) = (E/2,)
        edge_attr_u = edge_attr[mask]   # (E_u, 3)

        # call edge module to compute shared per-edge flux vectors (pass full history node_feat)
        out = self.edge_module(node_feat, torch.stack([src_u, dst_u], dim=0), edge_attr_u, stats, edge_memory=edge_memory)

        F_rho  = out["F_rho"]    # (E_u,2)
        F_e    = out["F_e"]      # (E_u,2)
        F_rhou = out["F_rhou"]   # (E_u,2)
        n      = out["n"]        # (E_u,2) unit normal (from src_u -> dst_u)
        r_edge = out["r"]        # (E_u,)
        alpha = out["alpha"]  # (E_u,)
        new_edge_memory = out["edge_memory"]

        mean_alpha = alpha.mean()

        # face_length for 4-neighbour uniform grid equals r_edge
        face_length = r_edge     # (E_u,)

        #NOTE: here it might be easier to directly read the a_* values from the out dict
        # compute scalar normal fluxes for scalars: a = <F, n>
        a_rho = (F_rho * n).sum(dim=-1)    # (E_u,)
        a_e = (F_e * n).sum(dim=-1)    # (E_u,)

        # scale by face length to get integrated flux per edge
        scalar_flux_rho = a_rho * face_length    # (E_u,)
        scalar_flux_e = a_e * face_length    # (E_u,)
        vector_flux_rhou = F_rhou * face_length.unsqueeze(-1)  # (E_u, 2)

        # build per-edge raw contributions
        # for src node: contribution = - (dt / area_src) * (flux)
        # for dst node: contribution = - (dt / area_dst) * (-flux) = + (dt / area_dst) * flux
        r_raw = scalar_flux_rho        # (E_u,)
        e_raw = scalar_flux_e          # (E_u,)
        m_raw = vector_flux_rhou       # (E_u,2)

        # cell_area handling: accept scalar or per-node tensor
        if cell_area is None:
            # approximate uniform cell area -> cell_area = dx^2 with dx = mean(r_edge)
            dx_est = torch.mean(r_edge)
            area = float(dx_est.item() ** 2)
            area_tensor = torch.full((N,), area, device=device, dtype=node_state.dtype)
        else:
            if isinstance(cell_area, (float, int)):
                # specified uniform area
                area_tensor = torch.full((N,), float(cell_area), device=device, dtype=node_state.dtype)
            else:
                area_tensor = cell_area.to(device=device, dtype=node_state.dtype)
                if area_tensor.dim() == 0:
                    area_tensor = area_tensor.repeat(N)

        # differentiable clipping
        if dt_cfl is not None:
            # ensure dt_cfl is a tensor on the correct device
            if not torch.is_tensor(dt_cfl):
                dt_cfl_tensor = torch.tensor(dt_cfl, device=device, dtype=dt.dtype)
            else:
                dt_cfl_tensor = dt_cfl.to(device)
                
            # if dt > dt_cfl, dt becomes dt_cfl
            dt = torch.clamp(dt, max=dt_cfl_tensor) # torch.clamp preserves the graph structure

        # compute per-edge contributions (scaled by dt / area of endpoint)
        # src contributions
        contrib_rho_src = - (dt * r_raw) / area_tensor[src_u]         # (E_u,)
        contrib_e_src   = - (dt * e_raw) / area_tensor[src_u]         # (E_u,)
        contrib_m_src   = - (dt * m_raw) / area_tensor[src_u].unsqueeze(-1)  # (E_u,2)

        # dst contributions (NOTE: sign flip inside formula for conservation)
        contrib_rho_dst = - (dt * (-r_raw)) / area_tensor[dst_u] 
        contrib_e_dst   = - (dt * (-e_raw)) / area_tensor[dst_u]
        contrib_m_dst   = - (dt * (-m_raw)) / area_tensor[dst_u].unsqueeze(-1)

        # aggregate into node deltas using index_add
        delta_rho = torch.zeros((N,), device=device, dtype=node_state.dtype)
        delta_e = torch.zeros((N,), device=device, dtype=node_state.dtype)
        delta_rhou = torch.zeros((N, 2), device=device, dtype=node_state.dtype)

        # accumulate src contributions
        delta_rho.index_add_(0, src_u, contrib_rho_src)
        delta_e.index_add_(0, src_u, contrib_e_src)
        delta_rhou.index_add_(0, src_u, contrib_m_src)

        # accumulate dst contributions
        delta_rho.index_add_(0, dst_u, contrib_rho_dst)
        delta_e.index_add_(0, dst_u, contrib_e_dst)
        delta_rhou.index_add_(0, dst_u, contrib_m_dst)

        # construct updated conserved U: [rho, e, rhou_x, rhou_y]
        rho = node_state[:, 0]
        e = node_state[:, 1]
        rhou = node_state[:, 3:5] # indexes shifted because p is at 2

        rho_new = rho + delta_rho
        e_new = e + delta_e
        rhou_new = rhou + delta_rhou

        # PHYSICAL CLAMPING
        # 1) denormalize to physical space
        rho_phys, e_phys, rhou_phys = denorm(rho_new, e_new, rhou_new, stats)

        # 2) apply constraints on physical values
        eps = 1e-6
        # POSITIVITY
        rho_phys = torch.clamp(rho_phys, min=eps)
        e_phys   = torch.clamp(e_phys, min=eps)

        # # VELOCITY CAP (prevent infinite speed explosion)
        # u_vec = rhou_phys / rho_phys.unsqueeze(-1)
        # u_mag = torch.norm(u_vec, dim=-1)
        
        # MAX_VEL = 100.0  # safe cap
        # scale_factor = torch.clamp(MAX_VEL / (u_mag + 1e-8), max=1.0)
        # rhou_phys = rhou_phys * scale_factor.unsqueeze(-1)

        # 3) re-normalize back
        rho_new, e_new, _, rhou_new = norm(rho_phys, e_phys, None, rhou_phys, stats)

        # compute pressure using physically valid (clamped) variables
        p_phys = eos(rho_phys, e_phys, rhou_phys, gamma)

        # normalize pressure
        _, _, p_norm, _ = norm(None, None, p_phys, None, stats)

        node_u_new = torch.cat([
                    rho_new.unsqueeze(-1),
                    e_new.unsqueeze(-1),
                    p_norm.unsqueeze(-1),
                    rhou_new
                ], dim=-1) # (N,5)

        return node_u_new, dt, mean_alpha, new_edge_memory


class FluxGNN(nn.Module):
    """
    - Applies `n_layers` conservative message-passing layers (each with its own EdgeFluxModule)
    - Returns predicted change (delta) in conserved variables by default (target='delta').
    - Also returns reconstructed primitives (rho, energy, momentum).
    Args:
        node_embed_dim: internal node embedding size used by EdgeFluxModule (defaults 64)
        n_layers: number of conservative message-passing layers
        dt_max: maximum learned dt per layer (passed to ConservativeMPLayer)
    """
    def __init__(self,
                 node_in_dim=5,
                 node_embed_dim=64,
                 n_layers=4,
                 dataset_dt=0.015,
                 gamma=None):
        super().__init__()

        self.node_in_dim = node_in_dim # > 5 if time_window>2
        self.node_embed_dim = node_embed_dim
        self.n_layers = int(n_layers)
        self.dataset_dt = float(dataset_dt)
        # self.gamma = float(gamma) if gamma is not None else None
        
        # calculate the required step per layer
        self.fixed_dt_per_layer = self.dataset_dt / self.n_layers

        # build layers (each layer has its own EdgeFluxModule + ConservativeMPLayer)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            edge_mod = EdgeFluxModule(node_in_dim=node_in_dim,
                                      node_embed_dim=node_embed_dim,
                                      edge_attr_dim=1,
                                      edge_embed_dim=32,
                                      phi_hidden=(node_embed_dim//2,),
                                      msg_hidden=(128,))
            layer = ConservativeMPLayer(edge_module=edge_mod)
            self.layers.append(layer)

    def _extract_initial_state(self, x):
        """
        Deterministically extract the LAST timestep from the input window `x`.
        
        Dataset Layout:
        x = np.concatenate([
            density[:-1],   # shape (T_in, H, W)
            energy[:-1],    # shape (T_in, H, W)
            pressure[:-1],  # shape (T_in, H, W)
            mom_ch[:-2]     # shape (2 * T_in, H, W) -> [x0, y0, x1, y1...]
        ], axis=0)
        
        Where T_in = time_window - 1.
        """
        C_total = x.shape[1]
        
        # calculate T_in (number of history steps in input)
        T_in = C_total // 5 # 3 scalars + 1 vector (2 components) = 5
        
        # 1. density is x[:, 0 : T_in] -> last density is at index (T_in - 1)
        rho_idx = T_in - 1
        
        # 2. energy is x[:, T_in : 2*T_in] -> last energy is at index (2*T_in - 1)
        e_idx = (2 * T_in) - 1
        
        # 3. pressure is x[:, 2*T_in : 3*T_in] -> last pressure is at index (3*T_in - 1)
        p_idx = (3 * T_in) - 1
        
        # 4. momentum is x[:, 3*T_in : 5*T_in] (length 2 * T_in), layout is [x0, y0, x1, y1, ...]
        mom_start = 3 * T_in
        mx_idx = mom_start + 2 * (T_in - 1) # last x component is at mom_start + 2*(T_in-1)
        my_idx = mx_idx + 1 # last y component is at mom_start + 2*(T_in-1) + 1
        
        # extract tensors
        rho0 = x[:, rho_idx]
        e0   = x[:, e_idx]
        p0   = x[:, p_idx]
        rhou0 = torch.stack([x[:, mx_idx], x[:, my_idx]], dim=-1)
        
        return rho0, e0, p0, rhou0

    def compute_cfl_limit(self, U, edge_attr, stats, gamma, cfl_factor=0.6):
        """
        Computes the maximum allowable dt based on the CFL condition.
        dt <= CFL * min( dx / (|u| + c) )
        """
        with torch.no_grad(): # detach from graph -> don't backprop through the limit constraint
            # recover physical primitives
            rho_norm = U[:, 0]
            p_norm = U[:, 2]
            rhou_norm = U[:, 3:5]

            # un-normalize using the global stats
            rho = rho_norm * stats["std"]["density"] + stats["mean"]["density"]
            p = p_norm   * stats["std"]["pressure"]   + stats["mean"]["pressure"]
            
            # compute velocity and sound speed
            rhou_x = rhou_norm[:, 0] * stats["std"]["momentum"][0] + stats["mean"]["momentum"][0]
            rhou_y = rhou_norm[:, 1] * stats["std"]["momentum"][1] + stats["mean"]["momentum"][1]
            
            u_sq = (rhou_x**2 + rhou_y**2) / (rho**2)
            u_mag = torch.sqrt(u_sq)

            # c = sqrt(gamma * p / rho)
            c_sq = (gamma * p) / rho
            c = torch.sqrt(torch.clamp(c_sq, min=1e-8))

            # max wave speed per node
            max_wave_speed = u_mag + c

            # get grid spacing dx
            # use min(r) of the whole mesh (conservative)
            min_dx = edge_attr[:, 2].min()

            # compute limit
            max_s = max_wave_speed.max()
            limit = cfl_factor * min_dx / (max_s + 1e-8)
            
            return limit

    def forward(self, data, stats):
        """
        Args:
            data: PyG Data object with fields:
                  - x: (N, C_in) node features
                  - edge_index: (2, E) long
                  - edge_attr: (E, 3) float [dx, dy, r]
                  - u (optional): global features tensor (1, k)
                  - pos (optional): (N,2)
            stats: dict with dataset statistics for (de)normalization
        Returns:
            dict with keys:
              'U0': initial conserved U (N,4)
              'U_final': final conserved U after layers (N,4)
              'delta_U': (N,4) = U_final - U0
              'rho': (N,) reconstructed density from U_final
              'energy': (N,) reconstructed total energy from U_final
              'momentum': (N,2) reconstructed momentum from U_final
              'dt_layers': list of dt per layer (floats)
        """
        device = data.x.device if isinstance(data.x, torch.Tensor) else torch.device("cpu")
        x_nodes = data.x.to(device)

        # data.u is (Batch_Size, n_globals), where index 0 is gamma
        # data.batch is (Total_Nodes,) mapping each node to its graph index
        batch_idx = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        gamma = data.u[batch_idx, 0] # Shape: (Total_Nodes,)

        edge_index = data.edge_index.to(device)
        edge_attr  = data.edge_attr.to(device)

        # extract U0 directly from the input features
        rho0, e0, p0, rhou0 = self._extract_initial_state(x_nodes) # this fixes U0 to be exactly the input state

        # PHYSICAL CLAMPING BEFORE LAYERS
        # denorm
        rho_phys, e_phys, rhou_phys = denorm(rho0, e0, rhou0, stats)

        # POSITIVITY
        eps = 1e-6
        rho_phys = torch.clamp(rho_phys, min=eps)
        e_phys   = torch.clamp(e_phys, min=eps)
        
        # # VELOCITY CAP
        # u_vec = rhou_phys / rho_phys.unsqueeze(-1)
        # u_mag = torch.norm(u_vec, dim=-1)
        # MAX_VEL = 100.0 
        # scale_factor = torch.clamp(MAX_VEL / (u_mag + 1e-8), max=1.0)
        # rhou_phys = rhou_phys * scale_factor.unsqueeze(-1)

        # renorm
        rho0, e0, _, rhou0 = norm(rho_phys, e_phys, None, rhou_phys, stats)
        
        # recalculate p0 consistent with clamped variables
        p_phys = eos(rho_phys, e_phys, rhou_phys, gamma)
        _, _, p_norm, _ = norm(None, None, p_phys, None, stats)
        
        # this is the safe starting state
        U = torch.cat([
             rho0.unsqueeze(-1), 
             e0.unsqueeze(-1), 
             p_norm.unsqueeze(-1), 
             rhou0
        ], dim=-1) 

        # keep a copy of the starting U to compute delta later
        U_conserved_start = torch.cat([rho0.unsqueeze(-1), e0.unsqueeze(-1), rhou0], dim=-1)

        # calculate dynamic CFL limit based on current state U
        cfl_limit = self.compute_cfl_limit(U, data.edge_attr, stats, gamma=gamma, cfl_factor=0.6)

        # dt_to_use = self.fixed_dt_per_layer
        dt_to_use = min(self.fixed_dt_per_layer, cfl_limit.item())

        current_U = U # state that will be integrated

        dt_layers = []
        alpha_layers = []

        # initialize edge memory container
        current_edge_memory = None

        # iterate conservative layers
        for layer in self.layers:
            if self.node_in_dim == 5:
                node_feat = current_U
            else:
                # need to take the static history from x_nodes
                # and concatenate the UPDATED current state (current_U)

                history_part = x_nodes[:, :-5] # static

                # append the EVOLVING current state (current_U)
                # current_U is (N, 5) and ALREADY NORMALIZED
                node_feat = torch.cat([history_part, current_U], dim=-1)

            current_U, _, mean_alpha, current_edge_memory = layer(
                node_state=current_U, 
                node_feat=node_feat, 
                edge_index=edge_index, 
                edge_attr=edge_attr, 
                gamma=gamma, 
                stats=stats, 
                dt=dt_to_use, 
                cell_area=None,
                edge_memory=current_edge_memory
            )

            dt_layers.append(float(dt_to_use))
            alpha_layers.append(mean_alpha)

        global_mean_alpha = torch.stack(alpha_layers).mean()

        # split conserved vars
        rho = current_U[:, 0]                    # (N,)
        e = current_U[:, 1]                      # (N,)
        p = current_U[:, 2]                      # (N,)
        rhou = current_U[:, 3:5]                 # (N,2)
  
        # Final conserved vars
        U_conserved_final = torch.cat([current_U[:, 0:2], current_U[:, 3:5]], dim=-1)

        output = {
            "U0": U_conserved_start,
            "U_final": current_U,
            "delta_U": U_conserved_final - U_conserved_start,
            "density": rho,
            "energy": e,
            "pressure": p,
            "momentum": rhou,
            "dt_layers": dt_layers,
            "mean_alpha": global_mean_alpha
        }
        return output
