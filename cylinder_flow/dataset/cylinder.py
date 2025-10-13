import os
import atexit
import numpy as np
import h5py
import torch 
from torch.utils.data import Dataset
from torch_geometric.data import Data

class IndexedTrajectoryDataset(Dataset):
    """
    Dataset of trajectories stored in a single HDF5 file. Each trajectory is a group
    containing datasets for node positions, velocities, pressures, cell connectivity,
    and node types. The dataset supports indexing into individual time steps across
    all trajectories.

    Each trajectory group should contain the following datasets:
    - 'pos': Node positions, shape (n_frames, n_nodes, dim) or (n_nodes, dim) if static.
    - 'velocity': Node velocities, shape (n_frames, n_nodes, vel_dim).
    - 'pressure': Node pressures, shape (n_frames, n_nodes) or (n_frames,).
    - 'cells': Mesh connectivity, shape (n_frames, n_faces, 3) or (n_faces, 3) if static.
    - 'node_type': Node types, shape (n_frames, n_nodes), (n_nodes,), or (n_nodes, k).

    The dataset returns PyG Data objects with:
    - x: Node features tensor of shape (N, F) with columns [node_type, cur_v, pressure, time].
    - pos: Node positions tensor of shape (N, dim).
    - face: Face connectivity tensor of shape (3, F).
    - y: Target velocities tensor of shape (N, vel_dim) for the next time step.

    Args:
        dataset_dir (str): Directory containing the HDF5 dataset file.
        split (str): Dataset split to use ('train', 'val', 'test').
        time_interval (float): Time interval between frames in seconds.
        cache_static (bool): Whether to cache static data (pos, cells, node_type).
        preserve_one_hot (bool): Whether to preserve one-hot encoding in node_type.
    Returns:
        Data: A PyG Data object for the specified time step.
    """
    def __init__(self, dataset_dir, split='train', time_interval=0.01, cache_static=False,
                 preserve_one_hot=False, transform=None):
        super().__init__()
        self.dataset_path = os.path.join(dataset_dir, f'{split}.h5')
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(self.dataset_path)

        self.time_interval = float(time_interval)
        self.cache_static = bool(cache_static)
        self.preserve_one_hot = bool(preserve_one_hot)

        # allow passing a PyG transform (e.g. T.Compose([...]))
        self.transform = transform

        # lazy HDF5 file handle per-process
        self._h5 = None

        # static cache per trajectory key
        # traj_key -> dict with 'pos','cells','node_type', and (cached transform outputs)
        self._static_cache = {}  # traj_key -> dict

        # Build metadata (open file only briefly to inspect shapes)
        traj_keys = []
        traj_counts = []
        traj_meta = []
        with h5py.File(self.dataset_path, 'r') as f:
            for key in f.keys():
                grp = f[key]
                n_frames = int(grp['velocity'].shape[0])

                # number of usable samples from this trajectory (t in [0..n_frames-1])
                usable = max(0, n_frames - 1)
                if usable == 0:
                    continue

                # inspect shapes to determine time-varying vs static datasets
                def _ds_info(name):
                    ds = grp[name]
                    return tuple(ds.shape), ds.ndim

                pos_shape, pos_ndim = _ds_info('pos')
                cells_shape, cells_ndim = _ds_info('cells')
                node_type_shape, node_type_ndim = _ds_info('node_type')

                meta = {
                    'n_frames': n_frames,
                    'pos_shape': pos_shape, 'pos_ndim': pos_ndim,
                    'cells_shape': cells_shape, 'cells_ndim': cells_ndim,
                    'node_type_shape': node_type_shape, 'node_type_ndim': node_type_ndim,
                }

                traj_keys.append(key)
                traj_counts.append(usable)
                traj_meta.append(meta)

        if len(traj_keys) == 0:
            raise RuntimeError(f"No usable trajectories in {self.dataset_path}")

        self.traj_keys = traj_keys
        self.traj_counts = np.asarray(traj_counts, dtype=np.int64)
        self.traj_meta = traj_meta
        # cumulative sum of trajectory time steps to map global idx -> (traj, t)
        self.traj_cumsum = np.concatenate(([0], np.cumsum(self.traj_counts)))
        self.total_samples = int(self.traj_cumsum[-1])

        # Ensure file closed on process exit
        atexit.register(self.close)

    def __len__(self):
        return int(self.total_samples)

    # HDF5 helpers
    def _ensure_h5(self):
        if self._h5 is None:
            # open read-only
            self._h5 = h5py.File(self.dataset_path, 'r')

    def close(self):
        if getattr(self, '_h5', None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None

    def __getstate__(self):
        # don't pickle the h5py file handle (unsafe). keep the path & metadata.
        state = self.__dict__.copy()
        state['_h5'] = None
        # don't pickle static cache unless you want that duplicated across workers
        # (we can choose to keep it, but safer to rebuild per-process)
        state['_static_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None

    # index mapping
    def _global_to_traj_and_t(self, idx):
        """Map global index to (trajectory index, local time index)."""
        # support negative indexing (ex. -1 is last sample)
        if idx < 0:
            idx += self.total_samples

        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f'Index {idx} out of bounds for dataset of size {self.total_samples}')
        
        # find trajectory: self.traj_cumsum has offsets: [0, n1, n1+n2, ...]
        # traj_pos = last index where cumsum <= idx
        traj_pos = int(np.searchsorted(self.traj_cumsum, idx, side='right') - 1)

        # compute local t in that trajectory
        # local_offset is in [0, traj_counts[traj_pos]-1] and corresponds to frame t
        local_t = int(idx - self.traj_cumsum[traj_pos])
        return traj_pos, local_t

    # normalization helpers
    @staticmethod
    def _ensure_2d_firstdim(arr, N, name):
        """Ensure arr is numpy array with shape (N, ...) or raise.
        This is needed for correctly stacking node features later."""
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return np.full((N, 1), float(arr), dtype=arr.dtype)
        if arr.ndim == 1:
            if arr.shape[0] == N:
                return arr.reshape(N, 1)
            else:
                raise ValueError(f"{name} length {arr.shape[0]} != N={N}")
        if arr.shape[0] != N:
            # try squeeze and re-check
            a = np.squeeze(arr)
            if a.ndim >= 1 and a.shape[0] == N:
                if a.ndim == 1:
                    return a.reshape(N, 1)
                return a
            raise ValueError(f"{name} first dimension {arr.shape[0]} != N={N}")
        return arr

    def _normalize_node_type(self, raw, N):
        """Return either (N,1) scalar label or (N,K) one-hot depending on preserve_one_hot.
        This is needed for correctly stacking node features later."""
        a = np.asarray(raw)
        a = np.squeeze(a)  # drop unit axes
        # if node_type is per-frame, the caller sliced already,
        # here just align dims
        if a.ndim == 0:
            out = np.full((N, 1), float(a), dtype=np.float32)
        elif a.ndim == 1:
            if a.shape[0] == N:
                out = a.reshape(N, 1)
            else:
                raise ValueError(f"node_type length {a.shape[0]} != N={N}")
        elif a.ndim == 2:
            # either (N,K) or (K,N) (transposed). prefer (N,K)
            if a.shape[0] == N:
                out = a
            elif a.shape[1] == N:
                out = a.T
            else:
                raise ValueError(f"Unhandled node_type shape {a.shape} for N={N}")
        else:
            # higher dims: try squeeze again
            s = np.squeeze(a)
            if s.ndim == 2 and s.shape[0] == N:
                out = s
            else:
                raise ValueError(f"Unhandled node_type shape {a.shape} for N={N}")

        # if user wants scalar label and input is one-hot/multi-channel -> argmax
        if not self.preserve_one_hot and out.ndim == 2 and out.shape[1] > 1:
            # argmax returns shape (N,), convert to (N,1)
            labels = np.argmax(out, axis=1).reshape(N, 1)
            return labels.astype(np.int32)
        
        return out.astype(np.float32)
    
    def _load_and_maybe_cache(self, traj_key, ds, t, n_nodes, meta, name):
        """
        Generic loader that:
        - Returns the array for the current time/frame if needed, or the static array if it's static.
        - If dataset appears static (ndim==2 OR ndim==3 with first==last) and cache_static=True,
            it stores the static array in self._static_cache[traj_key][name].
        Args:
            traj_key: str, trajectory group name (cache key)
            ds: h5py.Dataset
            t: int, current time index
            n_nodes: int, number of nodes inferred from velocity (for disambiguation)
            meta: dict, per-trajectory metadata (contains 'n_frames')
            name: str, logical feature name to use for caching ('pos','cells','node_type', ...)
        Returns:
            numpy array corresponding to this feature at the requested frame (or static).
        """
        # 1) return cached value if already present for this trajectory
        if self.cache_static and traj_key in self._static_cache:
            cached = self._static_cache[traj_key]
            if name in cached:
                return cached[name]

        # 2) shape-based handling
        try:
            ndim = ds.ndim
            shape = tuple(ds.shape)
        except Exception:
            # in the case ds isn't a proper h5py.Dataset, fall back to reading
            arr = ds[()] if hasattr(ds, '__call__') or hasattr(ds, '__array__') else ds
            if self.cache_static:
                self._static_cache.setdefault(traj_key, {})[name] = arr
            return arr

        # number of frames in this trajectory
        n_frames = int(meta.get('n_frames', -1))

        # -- 0-d or scalar datasets --
        if ndim == 0:
            arr = ds[()]  # scalar -> broadcast downstream if needed
            if self.cache_static:
                self._static_cache.setdefault(traj_key, {})[name] = arr
            return arr

        # -- 1-d datasets: either per-node or per-frame --
        if ndim == 1:
            if shape[0] == n_nodes:
                # per-node static vector
                arr = ds[()]
                if self.cache_static:
                    self._static_cache.setdefault(traj_key, {})[name] = arr
                return arr
            elif shape[0] == n_frames:
                # per-frame 1D -> pick frame t
                return ds[t]
            else:
                # unknown: read everything (fallback)
                arr = ds[()]
                if self.cache_static:
                    self._static_cache.setdefault(traj_key, {})[name] = arr
                return arr

        # -- 2-d datasets: usually static per-node (n_nodes, k) or (n_faces,3) --
        if ndim == 2:
            # many cases: (n_nodes, dim) static pos, (n_faces,3) static cells, (n_nodes, k) node_type
            # treat 2D as static and cache it.
            arr = ds[()]  # read static
            if self.cache_static:
                self._static_cache.setdefault(traj_key, {})[name] = arr
            return arr

        # -- 3-d datasets: (n_frames, n_nodes, dim) or (n_frames, n_faces, 3) or (n_frames, n_nodes, k) --
        if ndim == 3:
            # compare first and last frames to decide if static (cheap check)
            first = ds[0]
            last = ds[-1]

            # choose comparison method by dtype
            if np.issubdtype(first.dtype, np.floating):
                equal = np.allclose(first, last, rtol=1e-6, atol=1e-8)
            else:
                equal = np.array_equal(first, last)

            if equal:
                # cache & return 'first' (representative static mesh)
                if self.cache_static:
                    self._static_cache.setdefault(traj_key, {})[name] = first
                return first
            else:
                # truly time-varying: return requested frame
                return ds[t]

        # -- other dims (unexpected) --
        # fallback: read everything (rare)
        arr = ds[()]
        if self.cache_static:
            self._static_cache.setdefault(traj_key, {})[name] = arr
        return arr

    # graph construction
    @staticmethod
    def _datas_to_graph(pos, node_type, vel_t, vel_tp1, cells, pressure_t, time_scalar):
        """Convert raw arrays to PyG Data object. All arrays are numpy arrays.
        Args:
            pos: (N, dim) node positions
            node_type: (N, 1) or (N, K) node type features
            vel_t: (N, vel_dim) current velocities
            vel_tp1: (N, vel_dim) next velocities (target)
            cells: (F, 3) or (3, F) face connectivity
            pressure_t: (N,) or (N, 1) or (,) pressures at current time
            time_scalar: float scalar time value to add as feature
        Returns:
            Data: PyG Data object with x, pos, face, y
        """
        # retrieve number of nodes
        N = int(vel_t.shape[0])

        # node_type already normalized by caller
        node_type = np.asarray(node_type).astype(np.float32)
        pos = np.asarray(pos).astype(np.float32)
        v_t = np.asarray(vel_t).squeeze().astype(np.float32)
        v_tp1 = np.asarray(vel_tp1).squeeze().astype(np.float32)

        # sanity checks and reshapes on velocities
        if v_t.ndim == 1:
            v_t = v_t.reshape(N, 1)
        if v_tp1.ndim == 1:
            v_tp1 = v_tp1.reshape(N, 1)

        # treat pressure: (N,) or (N,1) or (,) -> (N,1)
        p = np.asarray(pressure_t).squeeze().astype(np.float32)
        if p.ndim == 0:
            p = np.full((N, 1), float(p), dtype=np.float32)
        elif p.ndim == 1:
            p = p.reshape(N, 1)

        # construct time feature vector (N, 1)
        time_vec = np.full((N, 1), float(time_scalar), dtype=np.float32)

        # cells -> (3, F) tensor
        cells = np.asarray(cells)
        if cells.ndim != 2:
            raise ValueError(f"cells must be 2D (faces), got shape {cells.shape}")
        if cells.shape[1] == 3 and cells.shape[0] != 3:
            face = torch.as_tensor(cells.T.astype(np.int64), dtype=torch.long)
        elif cells.shape[0] == 3:
            face = torch.as_tensor(cells.astype(np.int64), dtype=torch.long)
        else:
            raise ValueError(f"Unrecognized cells shape {cells.shape}; expecting (F,3) or (3,F)")

        # stack node features: [node_type, v_t, pressure, time]
        # each should be (N, k) where k may vary
        node_attr = np.hstack((node_type.astype(np.float32), v_t.astype(np.float32), p.astype(np.float32), time_vec.astype(np.float32)))

        # convert to tensors
        x = torch.as_tensor(node_attr, dtype=torch.float)
        pos_t = torch.as_tensor(pos.astype(np.float32), dtype=torch.float)
        y = torch.as_tensor(v_tp1.astype(np.float32), dtype=torch.float)

        # sanity checks
        if x.shape[0] != pos_t.shape[0] or x.shape[0] != y.shape[0]:
            raise ValueError(f"Inconsistent node counts after build: x {x.shape}, pos {pos_t.shape}, y {y.shape}")

        # return graph Data object
        return Data(x=x, pos=pos_t, face=face, y=y)

    def __getitem__(self, idx):
        """Get graph data for global index idx."""
        # retrieve trajectory, local time index and metadata
        # idx in [0..total_samples-1] maps to (traj_pos, t)
        # where t in [0..n_frames-2] (model predicts t+1) and
        # traj_pos in [0..n_trajectories-1]
        traj_pos, local_t = self._global_to_traj_and_t(idx)
        traj_key = self.traj_keys[traj_pos]
        t = int(local_t)
        meta = self.traj_meta[traj_pos]

        # open HDF5 file if needed and get current group
        self._ensure_h5()
        grp = self._h5[traj_key]

        # velocities & pressure (read per-frame slices)
        vel_t = grp['velocity'][t]
        vel_tp1 = grp['velocity'][t + 1]
        press_t = grp['pressure'][t]

        n_nodes = int(vel_t.shape[0])

        # pos
        pos_ds = grp['pos']
        pos = self._load_and_maybe_cache(traj_key, pos_ds, t, n_nodes, meta, 'pos')

        # cells
        cells_ds = grp['cells']
        cells = self._load_and_maybe_cache(traj_key, cells_ds, t, n_nodes, meta, 'cells')

        # node_type (unlike pos and cells needs normalization)
        node_type = None
        if self.cache_static and traj_key in self._static_cache:
            node_type = self._static_cache[traj_key].get('node_type', None)
        if node_type is None:
            nt_ds = grp['node_type']
            raw_nt = self._load_and_maybe_cache(traj_key, nt_ds, t, n_nodes, meta, 'node_type')

            # normalize shape / one-hot / scalar label
            node_type = self._normalize_node_type(raw_nt, n_nodes)
            # note: _load_and_maybe_cache has already cached raw_nt if it decided it's static
            # and cache_static=True -> cache the normalized node_type
            if self.cache_static and traj_key in self._static_cache and 'node_type' not in self._static_cache[traj_key]:
                # store normalized version
                self._static_cache.setdefault(traj_key, {})['node_type'] = node_type

        # compute time scalar feature
        time_scalar = float(self.time_interval * t)
        
        data = self._datas_to_graph(pos, node_type, vel_t, vel_tp1, cells, press_t, time_scalar)

        # ---- apply transform (if any) and optionally cache edge_index/edge_attr ----
        # only cache edge_index/edge_attr (transform outputs) per trajectory when cache_static=True
        if self.transform is not None:
            traj_key = self.traj_keys[traj_pos]
            # ensure there is a dict for this trajectory
            traj_cache = self._static_cache.setdefault(traj_key, {})

            # if already cached transform outputs for this trajectory, reuse them
            if self.cache_static and ('edge_index' in traj_cache):
                # attach cached fields to data (avoid re-running the transform)
                data.edge_index = traj_cache['edge_index']
                # edge_attr may be None if transform didn't produce it
                if 'edge_attr' in traj_cache and traj_cache['edge_attr'] is not None:
                    data.edge_attr = traj_cache['edge_attr']
            else:
                # run the transform (this will create edge_index/edge_attr, etc.)
                data = self.transform(data)

                # if cache_static -> store the transform outputs for this trajectory
                if self.cache_static:
                    # save edge_index and edge_attr (if present) for reuse
                    traj_cache['edge_index'] = data.edge_index
                    traj_cache['edge_attr'] = getattr(data, 'edge_attr', None)
                    # also keep pos/cells/node_type if not already cached (your _load_and_maybe_cache may have done this)
                    traj_cache.setdefault('pos', pos)
                    traj_cache.setdefault('cells', cells)
                    traj_cache.setdefault('node_type', node_type)
        
        # ensure 'face' attribute is always present so DataLoader batching is consistent
        # face may be None if transform removed it
        # `cells` is the array loaded earlier in this __getitem__ (n_faces x 3 or 3 x n_faces)
        if not hasattr(data, 'face') or data.face is None:
            try:
                # make sure cells is numpy array with shape (F,3) or (3,F)
                cells_arr = np.asarray(cells)
                if cells_arr.ndim == 2:
                    # if cells shape is (F,3) convert to (3,F) for PyG face convention, then to tensor
                    if cells_arr.shape[1] == 3 and cells_arr.shape[0] != 3:
                        face_t = torch.as_tensor(cells_arr.T, dtype=torch.long)
                    elif cells_arr.shape[0] == 3:
                        face_t = torch.as_tensor(cells_arr, dtype=torch.long)
                    else:
                        # fallback: try to transpose
                        face_t = torch.as_tensor(cells_arr.T, dtype=torch.long)
                    data.face = face_t
            except Exception as e:
                # don't crash dataset on weird cells; just warn and continue
                import warnings
                warnings.warn(f"Could not attach face attribute in dataset __getitem__: {e}")

        return data
