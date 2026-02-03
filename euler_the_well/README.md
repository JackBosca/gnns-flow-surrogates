# Flux Conservative GNN: Compressible Euler Gases

## Overview

This folder contains a PyTorch implementation of the **Flux Conservative Graph Neural Network** model designed for 2D compressible Euler gases using the *The Well* periodic multi-quadrant HDF5 dataset.
The model enforces flux-conservative updates on a coarsened grid represented as a graph (4-neighbour connectivity), and is trained with N-step unrolled teacher-forcing to improve autoregressive rollout stability.

---

## Repository layout

```
.
├── train_flux.py                 # training script
├── rollout_flux.py               # rollout / evaluation script
├── rollout_one_simulation.py     # rollout helper for optional validation
├── compute_vrmse.py              # VRMSE / RMSE utilities (evaluate rollouts)
├── render.py                     # render high-quality videos (8-panel style)
├── render_report.py              # report-quality rendering script (frames & video)
├── plots.py                      # loss plotting utilities
├── model/
│   ├── invariant_gnn_flux.py     # model (final model used)
│   ├── egnn_state.py             # discarded EGNN wrapper (for reference)
│   └── __init__.py
├── dataset/
│   ├── euler_coarse.py           # main Dataset class used for experiments
│   ├── euler_patched.py          # alternate patching dataset (not used in final runs)
│   └── compute_means.py          # helper to compute dataset statistics (stats.yaml)
├── data/
│   └── hdf5_inspect.ipynb        # inspection notebooks
├── utils.py                      # normalization, EOS, teacher-forcing schedule, helpers
├── utils_flux.py                 # unrolled training loop and helper utilities
├── jobs/                         # contains model weights, rollout files and VRMSE jobs logs
└── loss_plot.png                 # example loss plot
```

---

## Dataset

The dataset can be downloaded from <https://polymathic-ai.org/the_well/tutorials/dataset/>. `dataset.euler_coarse.EulerPeriodicDataset` expects **periodic, multi-simulation HDF5 files** with the following structure (the code reads these keys directly):

Top-level groups used:

* `t0_fields/`: scalar fields used as inputs:

  * `t0_fields/density`: shape `(n_sims, n_t, H, W)` (float32)
  * `t0_fields/energy`: shape `(n_sims, n_t, H, W)`
  * `t0_fields/pressure`: shape `(n_sims, n_t, H, W)`
* `t1_fields/`: vector fields (momentum):

  * `t1_fields/momentum`: shape `(n_sims, n_t, H, W, 2)` (float32)
* `geometry/`: `x`, `y` coordinate arrays used to build pos templates (the dataset class caches `pos_template`).
* `boundary_conditions/`: periodic masks (used to handle boundaries when coarsening).
* `scalars/gamma`: scalar (e.g. specific gas gamma) used as a global feature.
* `scalars` and `boundary_conditions` keys are used by `EulerPeriodicDataset` when present.

Notes about `EulerPeriodicDataset`:

* It supports **coarsening** via `coarsen=(sh, sw)` (stride downsampling on the H×W grid).
* `time_window` indicates how many timesteps are returned per sample (used for history & unrolled training).
* `target` can be `"delta"` (predict increments) or `"absolute"`.
* `to_centroids=True` converts vertex-centered fields to cell-centered values when needed.
* The dataset returns PyG `Data` objects with:

  * `x`: per-node stacked input channels (history window flattened)
  * `edge_index`, `edge_attr`: 4-neighbour grid connectivity and attributes (dx, dy, r)
  * optional global features (e.g. gamma/time)
  * `y_*`: targets for density, energy, pressure, momentum

---

## Training

The repository contains the training script `train_flux.py`. Edit the HDF5 paths and the hyperparameters at the bottom of `train_flux.py` (the script contains a `__main__` section with example settings), then run:

```bash
# run training (edit paths & parameters inside train_flux.py before running)
python train_flux.py
```

Example configuration used in the experiments (these are the defaults in the script):

* `time_window = 7` (history + prediction window used to build nodes)
* `coarsen = (4, 4)` (coarsening stride on the H×W grid)
* `target = "delta"` (predict delta updates)
* `to_centroids = True` (use cell-centered values)
* `batch_size = 1` (DataLoader batch size)
* `n_layers = 12` (model depth)
* optimizer: `AdamW(lr=1e-4, weight_decay=1e-5)`
* `epochs = 30` (script default)
* `teacher_forcing_start = 1.0` (full teacher forcing at epoch 0)
* `noise_std` injected into node states during training (controlled in the unrolled training helper; see `utils_flux.train_one_epoch_unrolled`)

Training details:

* The training loop supports **N-step unrolled training** implemented in `utils_flux.train_one_epoch_unrolled`. This runs the model rollout for `n_unroll_steps` inside the training loop and accumulates losses for stable long-horizon behavior.
* A scheduled sampling / teacher-forcing schedule is available via `utils.teacher_forcing_schedule`.
* Gaussian state noise (`noise_std`) is injected to improve autoregressive robustness (noise is applied only to node states, not global or edge features).

**Outputs**

* Model objects / losses / checkpoints are written by the script into the working directory (names are composed in the script; check the `fname` and `floss` variables near the bottom of `train_flux.py` to see the exact filenames used for your run).

---

## Rollout / evaluation

`rollout_flux.py` and `rollout_one_simulation.py` provide evaluation utilities and example usage. The file `rollout_flux.py` includes an example `__main__` that demonstrates:

* how to instantiate `EulerPeriodicDataset` for validation
* how to instantiate `FluxGNN` and load your trained weights (edit model load path in the script)
* how to call `rollout_one_simulation(model, ds, sim_idx, start_t, save_path=...)`

Example (edit the paths inside the script and run):

```bash
python rollout_flux.py
```

Behavior / outputs:

* `rollout_one_simulation` will perform autoregressive prediction for a single simulation index and save a `.npz` file containing:

  * predicted arrays for density/energy/pressure/momentum across rollout timesteps
  * ground-truth arrays when available
  * per-step timesteps and RMSE metrics (if ground truth present)
* `compute_vrmse.py` contains helpers to compute VRMSE/stepwise RMSE from `.npz` rollouts.

---

## Rendering and plots

* `render.py`: renders an 8-panel video for a rollout `.npz`.
* `render_report.py`: convenience script to produce frames and a video from rollouts for the final report.
* `plots.py`: plots training losses found in the checkpoint/loss files.

Typical rendering usage (edit paths inside `render_report.py` to point to your `.npz` rollouts):

```bash
python render_report.py \
  --results-dir ./rollouts/ \
  --out-dir ./videos/ \
  --size 1920 1080 \
  --skip 1
```

Render scripts automatically compute robust color limits (2%-98% GT percentiles) and produce consistent colormaps for each physical variable.

---

## Model architecture

`model/invariant_gnn_flux.py` (final model):

* Input: per-node state vector `u = [rho, energy, pressure, momentum_x, momentum_y]` (plus optional history/time/global features).
* Graph: structured 4-neighbour grid edges (edge_attr = `[dx, dy, r]`).
* Core idea: compute **edge fluxes** from node states (MLP readouts that are invariant to rotation), assemble a conservative flux at nodes, and predict node updates accordingly. Edge memory and node embedding updates with residual connections and normalization are implemented.
* Normalization and EOS:

  * The model uses normalization helpers in `utils.py` (`norm`, `denorm`) and a simple ideal-gas EOS helper `eos`.

There is a experimental EGNN wrapper (`model/egnn_state.py`) included for reference only: final experiments used the Flux Conservative GNN.

---

## Normalization & statistics

* `dataset/compute_means.py` helps compute per-dataset statistics (mean/std) used to normalize inputs and de-normalize outputs for metrics and plotting.
* `utils.norm` / `utils.denorm` are used throughout to ensure the model operates on normalized inputs and evaluations report physical quantities.
* `stats.yaml` (path set in `train_flux.py`) must include `mean` and `std` dictionaries for:

  * `density`, `energy`, `pressure`, `momentum`

---

## Checkpoints & reproducibility

The training script composes checkpoint/loss filenames from the `fname` and `floss` variables. Inspect the bottom of `train_flux.py` to see exact naming. Save locations can be changed inside that script.

---

## References

For additional details, read the report in this repository root.
