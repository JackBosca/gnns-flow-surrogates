"""
This code implements the Euler dataset class with a patching strategy instead of a grid-coarsening one.
This code was never used for training and evaluating models in the final results.
"""

import atexit
from pathlib import Path
import numpy as np
import h5py
import yaml
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class EulerPeriodicDataset(Dataset):
    """
    Dataset for Euler periodicBC HDF5 files where top-level groups are fields
    and first dimension is simulation index: e.g. t0_fields/density shape = (n_sims, n_t, H, W).
    """
    def __init__(
        self,
        h5_path,
        stats_path=None,
        time_window=2,
        patch_size=None,         # (h, w) or None for full grid
        patch_stride=None,       # stride between patch origins (None -> equals patch_size)
        normalize=True,
        target="delta"           # "delta" or "absolute"
    ):
        # validate paths
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # optional yaml dataset stats for normalization
        self.stats = None
        if stats_path is not None:
            with open(stats_path, "r") as fh:
                self.stats = yaml.safe_load(fh)

        # prediction time window (in timesteps)
        self.time_window = int(time_window)
        if self.time_window < 1:
            raise ValueError("time_window must be >= 1")
        if self.time_window == 1 and target == "delta":
            print("Warning: time_window=1 with target='delta' means target is always zero.")

        # patch configuration (None => full-grid behavior)
        self.patch_size = None if patch_size is None else tuple(patch_size)
        # patch_stride: None means stride == patch_size (non-overlapping)
        self.patch_stride = patch_stride

        # these will be populated only if patch_size is provided
        self.patches_per_row = None
        self.patches_per_col = None
        self.patches_per_timestep = None
        self.patch_stride_h = None
        self.patch_stride_w = None

        self.normalize = bool(normalize)
        self.target = str(target)
        assert self.target in ("delta", "absolute")

        # lazy HDF5 handle
        self._h5 = None

        # dictionary to cache static data
        self._static_cache = {}

        # retrieve number of simulations, timesteps, grid size from file
        with h5py.File(self.h5_path, "r") as f:
            # use density to get shapes
            d_density = f["t0_fields"]["density"]
            dens_shape = tuple(d_density.shape)

            # density shape: (n_sims, n_t, H, W)
            self.n_sims, self.n_t, self.H, self.W = int(dens_shape[0]), int(dens_shape[1]), int(dens_shape[2]), int(dens_shape[3])

            # each simulations has a regular grid with (y, x) coordinates in [0,1]
            # see: https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_periodicBC/
            # read x coordinates (length W)
            d_x = f["dimensions"]["x"]
            x_buf = np.empty(d_x.shape, dtype=np.float32)
            d_x.read_direct(x_buf)
            # read y coordinates (length H)
            d_y = f["dimensions"]["y"]
            y_buf = np.empty(d_y.shape, dtype=np.float32)
            d_y.read_direct(y_buf)

            # read scalar gamma
            d_gamma = f["scalars"]["gamma"]
            gamma_buf = np.empty((), dtype=np.float32)
            d_gamma.read_direct(gamma_buf)
            gamma_val = float(gamma_buf)

            # pos[i,j] = (x[j], y[i])
            X, Y = np.meshgrid(x_buf, y_buf, indexing="xy")   # X.shape = (H, W)
            pos_template = np.stack([X, Y], axis=-1).astype(np.float32).reshape(self.H * self.W, 2) # (H, W , 2) -> (N, 2)

            # read boundary masks
            d_xmask = f["boundary_conditions"]["x_periodic"]["mask"]
            xmask = np.empty(d_xmask.shape, dtype=bool)
            d_xmask.read_direct(xmask)

            d_ymask = f["boundary_conditions"]["y_periodic"]["mask"]
            ymask = np.empty(d_ymask.shape, dtype=bool)
            d_ymask.read_direct(ymask)

            # determine if periodic in each direction
            x_periodic = bool(xmask[0] and xmask[-1])
            y_periodic = bool(ymask[0] and ymask[-1])

            # cache for reuse (static and same accross simulations)
            self._static_cache["pos_template"] = pos_template
            self._static_cache["x_coords"] = x_buf
            self._static_cache["y_coords"] = y_buf
            self._static_cache["gamma"] = gamma_val
            self._static_cache["x_periodic_mask"] = xmask
            self._static_cache["y_periodic_mask"] = ymask
            self._static_cache["x_periodic"] = x_periodic
            self._static_cache["y_periodic"] = y_periodic

        if self.patch_size is None:
            # full-grid training will be heavy, warn user
            print(f"EulerPeriodicDataset: using full-grid samples ({self.H}x{self.W}), this is large ({self.H*self.W} nodes). Consider patching.")
        else:
            ph, pw = self.patch_size
            if ph > self.H or pw > self.W:
                raise ValueError(f"patch_size {self.patch_size} larger than grid {self.H, self.W}")

        # number of usable start times per simulation (need t+time_window for target)
        self.n_per_sim = max(0, self.n_t - self.time_window + 1)
        if self.n_per_sim == 0:
            raise RuntimeError("time_window too large for available timesteps")

        # if patching is requested, compute patch counts and update total_samples
        if self.patch_size is not None:
            # compute patches per sim and stride (populates self.patches_per_*)
            self._compute_patch_grid()
            # total samples across all simulations, timesteps and patch locations
            self.total_samples = int(self.n_sims * self.n_per_sim * self.patches_per_timestep)
        else:
            # default full-grid sample count
            self.total_samples = int(self.n_sims * self.n_per_sim)

        # ensure file closes on process exit
        atexit.register(self.close)

    def _ensure_h5(self):
        """
        Ensure a valid h5py.File is available on `self._h5`.

        - Opens the file lazily (read-only) the first time it is needed in a process.
        - If an existing handle becomes invalid (e.g., after fork), closes it and reopens.
        - Safe to call at the start of __getitem__ or in worker init.
        """
        # open file lazily if not yet opened
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            return
        
    def close(self):
        """
        Close the internal h5py.File handle (if open) and release it.
        Safe to call multiple times.
        """
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
        self._h5 = None

    def _compute_patch_grid(self):
        """
        Compute patch parameters based on self.patch_size and self.patch_stride.
        Populates and returns (patches_per_row, patches_per_col, patches_per_timestep,
                              patch_stride_h, patch_stride_w).
        Requires that self.patch_size is not None.
        """
        if self.patch_size is None:
            raise ValueError("_compute_patch_grid called but self.patch_size is None")

        # unify patch_size
        if isinstance(self.patch_size, int):
            ph = pw = int(self.patch_size)
        else:
            ph, pw = map(int, self.patch_size)

        # stride default -> non-overlapping
        if self.patch_stride is None:
            sh, sw = ph, pw
        elif isinstance(self.patch_stride, int):
            sh = sw = int(self.patch_stride)
        else:
            sh, sw = map(int, self.patch_stride)

        # compute number of patches that fit (sliding-window)
        patches_per_col = ((self.H - ph) // sh) + 1
        patches_per_row = ((self.W - pw) // sw) + 1
        patches_per_timestep = patches_per_row * patches_per_col

        # store into self for reuse
        self.patches_per_row = int(patches_per_row)
        self.patches_per_col = int(patches_per_col)
        self.patches_per_timestep = int(patches_per_timestep)
        self.patch_stride_h = int(sh)
        self.patch_stride_w = int(sw)

        return (self.patches_per_row, self.patches_per_col,
                self.patches_per_timestep, self.patch_stride_h, self.patch_stride_w)

    def _decode_index(self, idx):
        """
        Convert a flat index idx into (sim_idx, t_idx, i0, j0).
        - If patching is not active (self.patch_size is None), returns (sim_idx, t_idx, None, None).
        - If patching is active, returns the patch origin (i0,j0) for the given idx.
        Assumes idx in [0, len(self)-1].

        Example:
        H=W=512, p=64, stride=64, patches_per_row=8, patches_per_col=8, patches_per_timestep=64, n_per_sim=100, n_sims=400:
            per_sim = 100 * 64 = 6400.
            idx = 0 -> sim_idx = 0, rem=0 -> t_idx=0, patch_id=0 -> row=0,col=0 -> i0=0, j0=0
            idx = 1 -> sim 0, t 0, patch_id 1 -> row 0, col 1 -> i0=0, j0=64
            idx = 64 -> sim 0, t 1, patch_id 0 -> sim=0, t=1, i0=0, j0=0
            idx = per_sim -> sim_idx =1, rem=0 -> sim 1, t 0, patch_id 0.
        """
        idx = int(idx)
        if idx < 0:
            raise IndexError("Negative index not supported")

        if self.patch_size is None:
            # idx -> (sim_idx, t_idx) via divmod with n_per_sim
            sim_idx, t_idx = divmod(idx, self.n_per_sim)
            return int(sim_idx), int(t_idx), None, None

        # for sim in sims: for t in 0..n_per_sim-1: for each patch origin...
        per_sim = int(self.n_per_sim * self.patches_per_timestep)
        sim_idx = idx // per_sim
        rem = idx % per_sim
        t_idx = rem // self.patches_per_timestep
        patch_id = rem % self.patches_per_timestep

        # decode patch_id to (row, col)
        row = patch_id // self.patches_per_row
        col = patch_id % self.patches_per_row

        i0 = int(row * self.patch_stride_h)
        j0 = int(col * self.patch_stride_w)

        # sanity checks
        if sim_idx < 0 or sim_idx >= self.n_sims:
            raise IndexError(f"sim_idx out of range: {sim_idx}")
        if t_idx < 0 or t_idx >= self.n_per_sim:
            raise IndexError(f"t_idx out of range: {t_idx}")

        return int(sim_idx), int(t_idx), int(i0), int(j0)

    def _build_grid_edges(self, H, W, pos_grid, x_periodic=False, y_periodic=False, cache_key=None):
        """
        Edge builder for an HxW grid given pos_grid (H,W,2), respecting periodic wrap as indicated by masks.
        If cache_key is provided and present in self._static_cache, return cached tensors.
        Returns:
            edge_index: torch.LongTensor of shape (2, E)
            edge_attr:  torch.FloatTensor of shape (E, 4) containing [dx, dy, r, wrap_flag]
                        where wrap_flag is 1.0 if the edge is a wrap-around edge.
        """
        # check cache: because every simulation and timestep shares the same uniform 512×512 grid layout, 
        # a patch of size pxp always has the same local topology and neighbor connectivity
        # --> can reuse them for all timesteps and all simulations (same for full grid)
        if cache_key is not None and cache_key in self._static_cache:
            return self._static_cache[cache_key]["edge_index"], self._static_cache[cache_key]["edge_attr"]

        # normalize pos_grid to a numpy array shaped (H, W, 2)
        if isinstance(pos_grid, torch.Tensor):
            pos_np = pos_grid.cpu().numpy()
        else:
            pos_np = np.asarray(pos_grid, dtype=np.float32)

        # accept flattened (N,2) or (H,W,2)
        if pos_np.ndim == 2 and pos_np.shape[1] == 2:
            pos_np = pos_np.reshape(H, W, 2)
        elif pos_np.ndim == 3:
            # check shape matches
            if pos_np.shape[0] != H or pos_np.shape[1] != W or pos_np.shape[2] != 2:
                raise ValueError(f"pos_grid shape {pos_np.shape} incompatible with H={H}, W={W}")
        else:
            raise ValueError(f"pos_grid must be (H,W,2) or (N,2); got shape {pos_np.shape}")

        # domain extents for modular arithmetic (for handling periodic)
        x_flat = pos_np[..., 0].ravel() # selects the first element of the last dimension for every other dimension
        y_flat = pos_np[..., 1].ravel()

        # --------------------------------- fixing some problems with my numpy package
        try:
            # numpy road
            Lx = float(x_flat.max() - x_flat.min()) if x_flat.size > 0 else 1.0
            Ly = float(y_flat.max() - y_flat.min()) if y_flat.size > 0 else 1.0
        except Exception:
            # fallback with lists
            x_list = list(x_flat.tolist())
            y_list = list(y_flat.tolist())
            Lx = float(max(x_list) - min(x_list)) if len(x_list) > 0 else 1.0
            Ly = float(max(y_list) - min(y_list)) if len(y_list) > 0 else 1.0
        # ---------------------------------

        src_list = []
        dst_list = []
        attr_list = []  # [dx, dy, r, wrap_flag]

        # neighbor offsets: north, south, west, east
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # the double loop automatically yields bi-directional edges
        for i in range(H):
            for j in range(W):
                a_id = i * W + j    # flattened index of node a (source)
                pa = pos_np[i, j]   # [x_a, y_a]

                for di, dj in nbrs:
                    # compute candidate neighbor indices
                    ni = i + di
                    nj = j + dj
                    wrapped = False

                    # handle y-axis (vertical) overflow
                    if ni < 0 or ni >= H:
                        if y_periodic:
                            # if periodic, wrap around: neighbor of node at top goes to bottom and vice versa
                            ni = ni % H
                            wrapped = True
                        else:
                            continue    # skip neighbor outside domain for non-periodic

                    # handle x-axis (horizontal) overflow
                    if nj < 0 or nj >= W:
                        if x_periodic:
                            # if periodic, wrap around: neighbor of node at left goes to right and vice versa
                            nj = nj % W
                            wrapped = True
                        else:
                            continue

                    # compute flattened index of valid neighbor node b (destination)
                    b_id = ni * W + nj
                    pb = pos_np[ni, nj] # [x_b, y_b]

                    # compute distance components
                    dx_raw = float(pb[0] - pa[0])
                    dy_raw = float(pb[1] - pa[1])

                    # modular reduction to [-L/2, L/2] if periodic
                    if x_periodic and Lx > 0:
                        dx = ((dx_raw + 0.5 * Lx) % Lx) - 0.5 * Lx
                    else:
                        dx = dx_raw

                    if y_periodic and Ly > 0:
                        dy = ((dy_raw + 0.5 * Ly) % Ly) - 0.5 * Ly
                    else:
                        dy = dy_raw

                    # Euclidean distance
                    r = (dx * dx + dy * dy) ** 0.5
                    # wrap flag: 1.0 if wrapped on either axis else 0.0
                    wrap_flag = 1.0 if wrapped else 0.0

                    src_list.append(a_id)
                    dst_list.append(b_id)
                    attr_list.append([dx, dy, r, wrap_flag])

        # convert to tensors and cache
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) # shape (2, E)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32) # shape (E, 4)

        if cache_key is not None:
            # store as a dict
            self._static_cache[cache_key] = {"edge_index": edge_index, "edge_attr": edge_attr}

        return edge_index, edge_attr

    def _build_full_grid_edges(self):
        """
        Returns cached full-grid edges or builds them by calling _build_grid_edges() on full grid.
        Returns:
            edge_index: torch.LongTensor of shape (2, E)
            edge_attr:  torch.FloatTensor of shape (E, 4) containing [dx, dy, r, wrap_flag]
                        where wrap_flag is 1.0 if the edge is a wrap-around edge.
        """
        x_periodic = self._static_cache.get("x_periodic", False)
        y_periodic = self._static_cache.get("y_periodic", False)

        # create cache key for full grid edges
        cache_key = f"edge_full_{self.H}_{self.W}_{int(x_periodic)}_{int(y_periodic)}"
        # retrieve global pos template as (H, W, 2)
        pos_template = self._static_cache.get("pos_template", None)
        pos_grid = pos_template.reshape(self.H, self.W, 2)  # pos_grid[i,j] = (x_j, y_i)

        return self._build_grid_edges(self.H, self.W, pos_grid,
                                      x_periodic=x_periodic,
                                      y_periodic=y_periodic,
                                      cache_key=cache_key)
    
    def _load_time_window(self, sim_idx, t_idx, i0=None, j0=None, p_h=None, p_w=None):
        """
        Load scalar and vector fields for a given simulation and starting timestep.
        If p_h/p_w are provided, loads a node patch starting at (i0, j0) of shape (p_h, p_w).
        Args:
            sim_idx: int, simulation index (0 .. n_sims-1)
            t_idx: int, starting timestep index (0 .. n_t - time_window - 1)
            i0, j0: int or None, patch origin (row, col) if patching is used
            p_h, p_w: int or None, patch height and width if patching is used
        Returns:
            dict with keys "density", "energy", "pressure", "momentum"
            Each value is a numpy array with shape:
                - scalar fields: (time_window, H, W)
                - momentum: (time_window, H, W, 2)
        """
        # open the h5 file
        self._ensure_h5()
        f = self._h5

        t_w = self.time_window

        # decide whether to read full grid or patch
        if p_h is None or p_w is None:
            H, W = self.H, self.W

            # preallocate arrays
            density = np.empty((t_w, H, W), dtype=np.float32)
            energy  = np.empty((t_w, H, W), dtype=np.float32)
            pressure= np.empty((t_w, H, W), dtype=np.float32)
            momentum= np.empty((t_w, H, W, 2), dtype=np.float32)

            d_density = f["t0_fields"]["density"]
            d_energy  = f["t0_fields"]["energy"]
            d_pressure= f["t0_fields"]["pressure"]
            d_mom     = f["t1_fields"]["momentum"]

            # read each timestep with read_direct
            for i in range(t_w):
                # slice for scalar fields: (sim_idx, t_idx+i, :, :)
                src_sel_scalar = np.s_[sim_idx, t_idx+i, :, :]
                d_density.read_direct(density[i], source_sel=src_sel_scalar)
                d_energy.read_direct(energy[i], source_sel=src_sel_scalar)
                d_pressure.read_direct(pressure[i], source_sel=src_sel_scalar)

                # slice for vector field: (sim_idx, t_idx+i, :, :, :)
                src_sel_vector = np.s_[sim_idx, t_idx+i, :, :, :]
                d_mom.read_direct(momentum[i], source_sel=src_sel_vector)

        else:
            # patch read
            if i0 is None or j0 is None:
                raise ValueError("i0,j0 must be provided when p_h,p_w are specified")

            H_patch = int(p_h)
            W_patch = int(p_w)

            density = np.empty((t_w, H_patch, W_patch), dtype=np.float32)
            energy  = np.empty((t_w, H_patch, W_patch), dtype=np.float32)
            pressure= np.empty((t_w, H_patch, W_patch), dtype=np.float32)
            momentum= np.empty((t_w, H_patch, W_patch, 2), dtype=np.float32)

            d_density = f["t0_fields"]["density"]
            d_energy  = f["t0_fields"]["energy"]
            d_pressure= f["t0_fields"]["pressure"]
            d_mom     = f["t1_fields"]["momentum"]

            # slices: sim_idx, t_idx+i, i0:i0+H_patch, j0:j0+W_patch
            for i in range(t_w):
                src_sel_scalar = np.s_[sim_idx, t_idx+i, i0:i0+H_patch, j0:j0+W_patch]
                d_density.read_direct(density[i], source_sel=src_sel_scalar)
                d_energy.read_direct(energy[i], source_sel=src_sel_scalar)
                d_pressure.read_direct(pressure[i], source_sel=src_sel_scalar)

                src_sel_vector = np.s_[sim_idx, t_idx+i, i0:i0+H_patch, j0:j0+W_patch, :]
                d_mom.read_direct(momentum[i], source_sel=src_sel_vector)

        return {
            "density": density,
            "energy": energy,
            "pressure": pressure,
            "momentum": momentum
        }

    def _arrays_to_graph(self, x, y_density, y_energy, y_pressure, y_momentum,
                        time_step=None,
                        pos_template_override=None,
                        edge_index_override=None,
                        edge_attr_override=None):
        """
        Convert arrays to a PyG Data object.
        Args:
            x: np.ndarray of shape (C_in, H, W) - stacked input channels (time_window*5)
            y_density, y_energy, y_pressure: np.ndarray of shape (H, W) - scalar targets
            y_momentum: np.ndarray of shape (H, W, 2) - vector target
            time_step: int or None - current timestep (optional global feature)
            pos_template_override: np.ndarray of shape (H, W, 2) or None - override for node positions
            edge_index_override: torch.LongTensor of shape (2, E) or None - override for edge_index
            edge_attr_override: torch.FloatTensor of shape (E, 4) or None - override for edge_attr
        Returns:
            Data: PyG Data object with x, pos, edge_index (if desired), y, and optional globals.
        """
        C_in, H, W = x.shape
        N = H * W

        # flatten node features
        x_nodes = torch.tensor(x.reshape(C_in, N).T, dtype=torch.float)  # shape (N, C_in)

        if pos_template_override is not None:
            # accepts numpy array (H, W, 2)
            pos_arr = pos_template_override.reshape(N, 2)
            pos = torch.as_tensor(pos_arr, dtype=torch.float)
        else:
            # use cached pos
            pos_template = self._static_cache.get("pos_template", None)
            # pos_template is numpy array (N,2) -> convert to torch
            pos = torch.as_tensor(pos_template, dtype=torch.float)

        if edge_index_override is not None and edge_attr_override is not None:
            edge_index = edge_index_override
            edge_attr = edge_attr_override
        else:
            # attach precomputed full-grid 4-neighbour edges and attributes
            edge_index, edge_attr = self._build_full_grid_edges()

        # flatten targets and turn into tensors
        y_density = torch.tensor(y_density.reshape(N), dtype=torch.float)
        y_energy  = torch.tensor(y_energy.reshape(N), dtype=torch.float)
        y_pressure= torch.tensor(y_pressure.reshape(N), dtype=torch.float)
        y_momentum= torch.tensor(y_momentum.reshape(N, 2), dtype=torch.float)

        # construct global features
        global_feat = []
        
        # retrieve gamma from cache
        gamma_val = self._static_cache.get("gamma", None)
        if gamma_val is not None:
            global_feat.append(torch.tensor([float(gamma_val)], dtype=torch.float))
        if time_step is not None:
            global_feat.append(torch.tensor([float(time_step)], dtype=torch.float))

        if global_feat:
            global_feat = torch.cat(global_feat).unsqueeze(0)  # shape (1, # global features)
        else:
            global_feat = None

        data = Data(x=x_nodes,
                    pos=pos,
                    y_density=y_density,
                    y_energy=y_energy,
                    y_pressure=y_pressure,
                    y_momentum=y_momentum,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    u=global_feat)  # global features in PyG convention

        return data
    
    def __getitem__(self, idx):
        """
        Return a sample from the Euler periodic dataset.
        If self.patch_size is set, returns a patch-graph, otherwise returns the full-grid graph.
        Args:
            idx: int, flat index in range [0, len(self)-1]
        Returns:
            dict with keys:
                "x": input array of shape (C_in, H, W) where C_in = time_window * 3 + time_window * 2
                "y_density": target density array of shape (H, W)
                "y_energy": target energy array of shape (H, W)
                "y_pressure": target pressure array of shape (H, W)
                "y_momentum": target momentum array of shape (H, W, 2)
        """
        # decode index into sim, t and optional patch origin
        sim_idx, t_idx, i0, j0 = self._decode_index(idx)

        # load time window of data (full grid or patch depending on patch_size)
        if self.patch_size is None:
            # full-grid read
            data = self._load_time_window(sim_idx, t_idx)
            density  = data["density"]   # shape: (time_window, H, W)
            energy   = data["energy"]    # shape: (time_window, H, W)
            pressure = data["pressure"]  # shape: (time_window, H, W)
            momentum = data["momentum"]  # shape: (time_window, H, W, 2)
            H_use, W_use = self.H, self.W
        else:
            # patch read: read p_h x p_w node patch starting at (i0, j0)
            p_h, p_w = self.patch_size
            data = self._load_time_window(sim_idx, t_idx, i0=i0, j0=j0, p_h=p_h, p_w=p_w)
            density  = data["density"]   # (time_window, p_h, p_w)
            energy   = data["energy"]    # (time_window, p_h, p_w)
            pressure = data["pressure"]  # (time_window, p_h, p_w)
            momentum = data["momentum"]  # (time_window, p_h, p_w, 2)
            H_use, W_use = int(p_h), int(p_w)
        
        # optionally normalize
        if self.normalize and self.stats is not None:
            density  = (density  - self.stats["mean"]["density"])  / self.stats["std"]["density"]
            energy   = (energy   - self.stats["mean"]["energy"])   / self.stats["std"]["energy"]
            pressure = (pressure - self.stats["mean"]["pressure"]) / self.stats["std"]["pressure"]
            momentum = (momentum - self.stats["mean"]["momentum"]) / self.stats["std"]["momentum"]

        # compute target depending on "delta" or "absolute"
        if self.target == "delta":
            # target = difference between last step and first step
            y_density  = density[-1]  - density[0]
            y_energy   = energy[-1]   - energy[0]
            y_pressure = pressure[-1] - pressure[0]
            y_momentum = momentum[-1] - momentum[0]
        else:  # "absolute"
            y_density  = density[-1]
            y_energy   = energy[-1]
            y_pressure = pressure[-1]
            y_momentum = momentum[-1]

        # reshape momentum to (time_window, 2, H, W) and flatten into (time_window*2, H_use, W_use) for concatenation
        mom_ch = momentum.transpose(0, 3, 1, 2).reshape(self.time_window * 2, H_use, W_use)
        x = np.concatenate([density, energy, pressure, mom_ch], axis=0)  # shape: (C_in=time_window*5, H_use, W_use)

        # last input timestep to be attached as a global feature
        last_input_t = t_idx + self.time_window - 1 # if time_window=1, last_input_t = t_idx
        time_scalar = float(last_input_t) / float(self.n_t - 1)   # normalization to [0,1]

        # build graph Data: if patching, build per-patch pos_template and edges 
        # and pass them to _arrays_to_graph
        if self.patch_size is not None:
            # construct pos_template for the patch using global coord arrays
            x_coords = self._static_cache.get("x_coords", None)
            y_coords = self._static_cache.get("y_coords", None)

            # extract coordinate slices for the patch (no wrapping needed due to tiling logic:
            # j0+W_use <= W and i0+H_use <= H)
            j0 = int(j0)
            i0 = int(i0)
            x_patch = x_coords[j0:j0+W_use]
            y_patch = y_coords[i0:i0+H_use]
            Xp, Yp = np.meshgrid(x_patch, y_patch, indexing="xy")   # (H_use, W_use)
            pos_template_patch = np.stack([Xp, Yp], axis=-1).astype(np.float32)  # (H_use, W_use, 2)

            # build per-patch edges and attributes
            x_periodic = bool(self._static_cache.get("x_periodic", False))
            y_periodic = bool(self._static_cache.get("y_periodic", False))
            edge_cache_key = f"edge_patch_{H_use}_{W_use}_{int(x_periodic)}_{int(y_periodic)}"
            edge_index_patch, edge_attr_patch = self._build_grid_edges(H_use, W_use, pos_template_patch,
                                                                    x_periodic=x_periodic, y_periodic=y_periodic,
                                                                    cache_key=edge_cache_key)

            # call arrays_to_graph with overrides
            data = self._arrays_to_graph(x, y_density, y_energy, y_pressure, y_momentum,
                                         time_step=time_scalar,
                                         pos_template_override=pos_template_patch,
                                         edge_index_override=edge_index_patch,
                                         edge_attr_override=edge_attr_patch)
        else:
            # full-grid case (_build_full_grid_edges used internally)
            data = self._arrays_to_graph(x, y_density, y_energy, y_pressure, y_momentum,
                                    time_step=time_scalar)

        return data
    
    def __len__(self):
        """
        Return the number of samples (flat index space).
        If patch_size was provided, this is n_sims * n_per_sim * patches_per_timestep.
        Otherwise the original full-grid total_samples (n_sims * n_per_sim).
        """
        return int(self.total_samples)
