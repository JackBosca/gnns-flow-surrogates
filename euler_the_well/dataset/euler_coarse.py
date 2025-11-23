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
        normalize=True,
        target="delta",          # "delta" or "absolute"
        coarsen=(1, 1),          # (sh, sw) stride: (rows, cols)
        to_centroids=False       # convert vertex to cell-centered values
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
        if self.time_window < 2:
            raise ValueError("time_window must be >= 2 to have input and target timesteps.")

        self.normalize = bool(normalize)
        self.target = str(target)
        assert self.target in ("delta", "absolute")

        # coarsening stride (rows, cols)
        if not (isinstance(coarsen, (tuple, list)) and len(coarsen) == 2):
            raise ValueError("coarsen must be a tuple (sh, sw)")
        self.sh, self.sw = int(coarsen[0]), int(coarsen[1])
        if self.sh < 1 or self.sw < 1:
            raise ValueError("coarsen strides must be >= 1")
        
        # centroids conversion flag
        self.to_centroids = bool(to_centroids)

        # lazy HDF5 handle
        self._h5 = None

        # dictionary to cache static data
        self._static_cache = {}

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

            # compute coarsened grid sizes, must be exact divisibile
            if (self.H % self.sh != 0) or (self.W % self.sw != 0):
                raise ValueError(f"coarsen {self.sh, self.sw} must divide grid size {self.H, self.W}")

            self.Hc = self.H // self.sh
            self.Wc = self.W // self.sw

            # coarsened coordinates (strided) at vertex locations
            x_coords_coarse = x_buf[::self.sw].astype(np.float32)   # length Wc_vertices
            y_coords_coarse = y_buf[::self.sh].astype(np.float32)   # length Hc_vertices

            if self.to_centroids:
                # convert to cell-centered coordinates and adjust Hc, Wc accordingly
                # original vertex grid sizes (coarsened)
                Hc_verts = y_coords_coarse.size
                Wc_verts = x_coords_coarse.size

                # compute cell-centered coordinates (centroids) by averaging adjacent vertex coords
                x_coords_cent = 0.5 * (x_coords_coarse[:-1] + x_coords_coarse[1:])   # length Wc_cells = Wc_verts - 1
                y_coords_cent = 0.5 * (y_coords_coarse[:-1] + y_coords_coarse[1:])   # length Hc_cells = Hc_verts - 1

                # grid of centroid positions
                Xc_cent, Yc_cent = np.meshgrid(x_coords_cent, y_coords_cent, indexing="xy")
                pos_template_coarse_cells = np.stack([Xc_cent, Yc_cent], axis=-1).astype(np.float32).reshape((Hc_verts-1) * (Wc_verts-1), 2) # (Hc_cells, Wc_cells, 2) -> (Nc_cells, 2)

                # update self.Hc and self.Wc to be cell counts (not vertex counts)
                self.Hc = Hc_verts - 1
                self.Wc = Wc_verts - 1

                # cache centroid templates and centroid coordinate arrays
                self._static_cache["pos_template_coarse"] = pos_template_coarse_cells
                self._static_cache["x_coords_coarse"] = x_coords_cent
                self._static_cache["y_coords_coarse"] = y_coords_cent
            else:
                # coarsened pos template: pos_coarse[i,j] = (x_coords_coarse[j], y_coords_coarse[i])
                Xc, Yc = np.meshgrid(x_coords_coarse, y_coords_coarse, indexing="xy")
                pos_template_coarse = np.stack([Xc, Yc], axis=-1).astype(np.float32).reshape(self.Hc * self.Wc, 2) # (Hc, Wc, 2) -> (Nc, 2)

                # cache vertex templates and vertex coordinate arrays
                self._static_cache["pos_template_coarse"] = pos_template_coarse
                self._static_cache["x_coords_coarse"] = x_coords_coarse
                self._static_cache["y_coords_coarse"] = y_coords_coarse

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

        # number of usable start times per simulation (need t+time_window for target)
        self.n_per_sim = max(0, self.n_t - self.time_window + 1)
        if self.n_per_sim == 0:
            raise RuntimeError("time_window too large for available timesteps")
 
        # total samples across all simulations (flat index space)
        self.total_samples = int(self.n_sims * self.n_per_sim)

        if self.sh == 1 and self.sw == 1:
            # full-grid training will be heavy, warn user
            print(f"using full-grid samples ({self.H}x{self.W}), this is large ({self.H*self.W} nodes). Consider coarsening.")

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

    def _convert_to_centroids(self, arr):
        """
        Convert vertex-centered array to cell-centered by averaging 2x2 vertex blocks.
        Args:
            arr: numpy array with last two dims equal to (Hv, Wv) vertex grid.
                Leading dims (e.g. time) are preserved. Works for scalar fields
                shape (..., Hv, Wv) and vector fields shape (..., Hv, Wv, C).
        Returns:
            numpy array with last two dims (Hv-1, Wv-1) containing cell-centered values.
        """
        # handle vector field case (e.g. momentum (..., Hv, Wv, 2))
        if arr.ndim > 3:
            # average over 2x2 blocks on Hv, Wv
            top_left     = arr[..., :-1, :-1, :]
            top_right    = arr[..., :-1, 1:, :]
            bottom_left  = arr[..., 1:, :-1, :]
            bottom_right = arr[..., 1:, 1:, :]
            return 0.25 * (top_left + top_right + bottom_left + bottom_right)
        # scalar case
        top_left     = arr[..., :-1, :-1]
        top_right    = arr[..., :-1, 1:]
        bottom_left  = arr[..., 1:,  :-1]
        bottom_right = arr[..., 1:,  1:]
        return 0.25 * (top_left + top_right + bottom_left + bottom_right)

    def _build_full_grid_edges(self):
        """
        Build and cache full-grid 4-neighbor directed edges and edge attributes
        for the entire (H x W) grid, respecting periodic wrap as indicated by masks.
        Caches outputs so it's built only once.
        Returns:
            edge_index: torch.LongTensor of shape (2, E)
            edge_attr:  torch.FloatTensor of shape (E, 4) containing [dx, dy, r, wrap_flag]
                        where wrap_flag is 1.0 if the edge is a wrap-around edge (crosses seam).
        """
        # cache keys that include the stride
        cache_key_idx = f"edge_index_full_s{self.sh}x{self.sw}"
        cache_key_attr = f"edge_attr_full_s{self.sh}x{self.sw}"
        if cache_key_idx in self._static_cache and cache_key_attr in self._static_cache:
            return self._static_cache[cache_key_idx], self._static_cache[cache_key_attr]

        # use coarsened sizes and positions when available, else fallback to full resolution
        H = int(getattr(self, "Hc", self.H))
        W = int(getattr(self, "Wc", self.W))

        # retrieve pos template as (Hc, Wc, 2)
        pos_template = self._static_cache.get("pos_template_coarse",
                                          self._static_cache.get("pos_template", None))
        if pos_template is None:
            raise RuntimeError("pos_template not found in cache (coarse or full).")
    
        pos_grid = pos_template.reshape(H, W, 2)  # pos_grid[i,j] = (x_j, y_i)

        # periodic flags
        x_periodic = bool(self._static_cache.get("x_periodic", False))
        y_periodic = bool(self._static_cache.get("y_periodic", False))

        # get x and y coordinate arrays and then actual domain extensions
        x_coords = self._static_cache.get("x_coords_coarse",
                                      self._static_cache.get("x_coords", None))
        y_coords = self._static_cache.get("y_coords_coarse",
                                      self._static_cache.get("y_coords", None))
        if x_coords is not None:
            Lx = float(x_coords.max() - x_coords.min())
        else:
            Lx = 1.0    # see: https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_periodicBC/
        if y_coords is not None:
            Ly = float(y_coords.max() - y_coords.min())
        else:
            Ly = 1.0    # see: https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_periodicBC/

        src_list = []
        dst_list = []
        attr_list = []  # [dx, dy, r, wrap_flag]

        # neighbor offsets: north, south, west, east
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # the double loop automatically yields bi-directional edges
        for i in range(H):
            for j in range(W):
                a_id = i * W + j    # flattened index of node a (source)
                pa = pos_grid[i, j]  # [x_a, y_a]

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
                            continue  # skip neighbor outside domain for non-periodic

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
                    pb = pos_grid[ni, nj]  # [x_b, y_b]

                    # compute distance components
                    dx_raw = float(pb[0] - pa[0])
                    dy_raw = float(pb[1] - pa[1])

                    # modular reduction to [-L/2, L/2] if periodic
                    if x_periodic:
                        dx = ((dx_raw + 0.5 * Lx) % Lx) - 0.5 * Lx
                    else:
                        dx = dx_raw

                    if y_periodic:
                        dy = ((dy_raw + 0.5 * Ly) % Ly) - 0.5 * Ly
                    else:
                        dy = dy_raw

                    # Euclidean distance
                    r = (dx * dx + dy * dy) ** 0.5

                    src_list.append(a_id)
                    dst_list.append(b_id)
                    attr_list.append([dx, dy, r])

        # convert to tensors and cache
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) # shape (2, E)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32) # shape (E, 3)

        self._static_cache[cache_key_idx] = edge_index
        self._static_cache[cache_key_attr] = edge_attr

        return edge_index, edge_attr
    
    def _load_time_window(self, sim_idx, t_idx):
        """
        Load scalar and vector fields for a given simulation and starting timestep.
        Args:
            sim_idx: int, simulation index (0 .. n_sims-1)
            t_idx: int, starting timestep index (0 .. n_t - time_window - 1)
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
        Hc, Wc = self.Hc, self.Wc   # if to_centroids is True, these are cell counts

        if self.to_centroids:
            # convert from cell-centered to vertex-centered sizes for reading
            Hc += 1 
            Wc += 1

        # preallocate arrays
        density  = np.empty((t_w, Hc, Wc), dtype=np.float32)
        energy   = np.empty((t_w, Hc, Wc), dtype=np.float32)
        pressure = np.empty((t_w, Hc, Wc), dtype=np.float32)
        momentum = np.empty((t_w, Hc, Wc, 2), dtype=np.float32)

        d_density  = f["t0_fields"]["density"]
        d_energy   = f["t0_fields"]["energy"]
        d_pressure = f["t0_fields"]["pressure"]
        d_mom      = f["t1_fields"]["momentum"]

        # read each timestep with read_direct
        for i in range(t_w):
            # slice for scalar fields with stride: (sim_idx, t_idx+i, ::sh, ::sw)
            src_sel_scalar = np.s_[sim_idx, t_idx+i, ::self.sh, ::self.sw]
            d_density.read_direct(density[i], source_sel=src_sel_scalar)
            d_energy.read_direct(energy[i], source_sel=src_sel_scalar)
            d_pressure.read_direct(pressure[i], source_sel=src_sel_scalar)

            # slice for vector field with stride: (sim_idx, t_idx+i, ::sh, ::sw, :)
            src_sel_vector = np.s_[sim_idx, t_idx+i, ::self.sh, ::self.sw, :]
            d_mom.read_direct(momentum[i], source_sel=src_sel_vector)

        if self.to_centroids:
            # convert vertex-centered arrays (shape: (t_w, Hcv, Wcv)) -> cell-centered (t_w, Hc, Wc)
            density  = self._convert_to_centroids(density)
            energy   = self._convert_to_centroids(energy)
            pressure = self._convert_to_centroids(pressure)
            momentum = self._convert_to_centroids(momentum)

        return {
            "density": density,
            "energy": energy,
            "pressure": pressure,
            "momentum": momentum
        }

    def _arrays_to_graph(self, x, y_density, y_energy, y_pressure, y_momentum,
                        time_step=None):
        """
        Convert arrays to a PyG Data object.
        Args:
            x: np.ndarray of shape (C_in, H, W) - stacked input channels ((time_window-1)*5)
            y_density, y_energy, y_pressure: np.ndarray of shape (H, W) - scalar targets
            y_momentum: np.ndarray of shape (H, W, 2) - vector target
            time_step: int or None - current timestep (optional global feature)
        Returns:
            Data: PyG Data object with x, pos, edge_index (if desired), y, and optional globals.
        """
        C_in, H, W = x.shape
        N = H * W

        # flatten node features
        x_nodes = torch.tensor(x.reshape(C_in, N).T, dtype=torch.float)  # shape (N, C_in)

        # use cached coarse pos if available, else fall back to full-resolution pos
        pos_template = self._static_cache.get("pos_template_coarse",
                                              self._static_cache.get("pos_template", None))
        if pos_template is not None:
            # pos is numpy array (N,2) -> convert to torch
            pos = torch.as_tensor(pos_template, dtype=torch.float)
        else:
            pos = None

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
        
        # check if empty
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
        self._ensure_h5()

        # map global flat index to simulation and starting timestep
        sim_idx, t_idx = divmod(idx, self.n_per_sim)

        # load time window of data
        data = self._load_time_window(sim_idx, t_idx)
        density  = data["density"]   # shape: (time_window, H, W)
        energy   = data["energy"]    # shape: (time_window, H, W)
        pressure = data["pressure"]  # shape: (time_window, H, W)
        momentum = data["momentum"]  # shape: (time_window, H, W, 2)

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

        # reshape momentum into (time_window*2, Hc, Wc)
        mom_ch = momentum.transpose(0, 3, 1, 2).reshape(self.time_window * 2, self.Hc, self.Wc)

        # build x by considering all time window steps EXCEPT the last one (which is target)
        x = np.concatenate([density[:-1], energy[:-1], pressure[:-1], mom_ch[:-2]], axis=0)  # shape: (C_in=(time_window-1)*5, Hc, Wc)

        # last input timestep to be attached as a global feature
        last_input_t = t_idx + self.time_window - 1 # if time_window=1, last_input_t = t_idx
        time_scalar = float(last_input_t) / float(self.n_t - 1)   # normalization to [0,1]

        # build graph Data
        data = self._arrays_to_graph(x, y_density, y_energy, y_pressure, y_momentum,
                                    time_step=time_scalar)

        return data
    
    def __len__(self):
        """
        Return the number of samples (flat index space).
        """
        return int(self.total_samples)
    