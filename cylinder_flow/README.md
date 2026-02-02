# MeshGraphNets: Incompressible Flow Past a Cylinder

## Overview

This folder contains a PyTorch implementation of a MeshGraphNets model for the 2D incompressible flow-past-a-cylinder problem.

## Repository layout

```
.
├── train.py                # training script
├── rollout.py              # inference / rollout
├── validation.py           # validation utilities
├── render.py               # low-level rendering helpers
├── render_report.py        # render frames/videos for final report
├── plots.py                # generic plotting utilities
├── model/
│   ├── blocks.py           # EdgeBlock, NodeBlock, MLP builder utilities
│   └── model.py            # Simulator wrapper + Encoder/Processor/Decoder
├── dataset/
│   └── cylinder.py         # IndexedTrajectoryDataset
├── data/
│   ├── download_dataset.sh # helper: downloads DeepMind tfrecord dataset
│   └── parse_tfrecord.py   # converts tfrecord -> HDF5 (one group per trajectory)
├── utils/
│   └── utils.py            # NodeType enum, Normalizer(s), graph utils, noise helper
├── jobs/                   # Contains model weights, rollouts and videos
└── results_*/              # result plots 50-100% training data
```

---

## Dataset: preparation

The code expects trajectories stored in HDF5 files. Each trajectory group must include datasets:

* `pos`         : node positions per time step (T × N × 2)
* `velocity`    : velocities per time step (T × N × 2)
* `pressure`    : pressure per time step (T × N)
* `cells`       : cell connectivity indices (mesh faces/cells)
* `node_type`   : per-node integer or one-hot node types (N or N×K)

**Get dataset (recommended workflow):**

1. Download DeepMind MeshGraphNets `cylinder_flow` tfrecord files:

```bash
# usage: ./data/download_dataset.sh <dataset_name> <output_dir>
# example:
./data/download_dataset.sh cylinder_flow /path/to/data
```

2. Convert TFRecords to HDF5:

```bash
python data/parse_tfrecord.py --input /path/to/data/cylinder_flow --output /path/to/h5_output
```

(`parse_tfrecord.py` in this repo contains the conversion script; it uses TensorFlow to read examples and writes one HDF5 file.)

3. Your final `dataset_dir` should contain HDF5 files expected by `IndexedTrajectoryDataset` (see `dataset/cylinder.py`).

**Notes on the HDF5 layout:**
`IndexedTrajectoryDataset` expects trajectories as groups under the HDF5 root. Each group should contain the datasets listed above. The dataset class supports caching of static data (positions, cells, node types) and returns PyG `Data` objects suitable for the model.

---

## Training

Basic training command (examples):

```bash
# minimal example (adjust dataset path, GPU environment, workers)
python train.py \
  --dataset-dir /path/to/h5_output \
  --batch-size 1 \
  --max-epochs 10 \
  --save-dir checkpoint \
  --noise-std 0.02 \
  --workers 8
```

Important `train.py` flags (non-exhaustive):

* `--dataset-dir` : path to HDF5 dataset directory
* `--batch-size`  : training batch size
* `--max-epochs`  : number of epochs
* `--save-dir`    : directory where checkpoints and losses are saved
* `--noise-std`   : std of velocity noise injected during training (see `utils.get_velocity_noise`)
* `--workers`     : DataLoader num workers
* `--fraction`    : fraction of dataset to use (for quick debugging)
* `--print-batch`, `--save-batch` : logging & checkpoint cadence

Output:

* model checkpoints (saved by `Simulator.save` / `torch.save` into `--save-dir`)
* `losses.pt` (per-batch loss history)

---

## Rollout

Use `rollout.py` to run trajectories and save predictions for later analysis/rendering.

Example:

```bash
python rollout.py \
  --model_dir checkpoint/simulator.pth \
  --dataset_dir /path/to/h5_output \
  --test_split test \
  --rollout_num 5 \
  --gpu 0 \
  --cache_static
```

Important `rollout.py` flags:

* `--model_dir`    : path to saved model checkpoint (simulator file)
* `--dataset_dir`  : path to dataset HDF5
* `--test_split`   : `train|valid|test`
* `--rollout_num`  : how many trajectories to rollout (first N)
* `--preserve_one_hot` : pass `preserve_one_hot` to dataset (if used)
* `--cache_static` : cache static arrays (positions/cells/node_type) in memory

Outputs:

* Pickled result files (per rollout) that include predicted velocities per step, reference targets, masks, and optionally connectivity for rendering. These result `.pkl` files are suitable inputs to the render scripts.

Validation helper:

* `validation.validate_rollouts(...)` is provided for aggregating RMSE metrics across rollouts and extracting horizon RMSE values.

---

## Rendering and plots

Generate frames/videos from rollout `.pkl` results with `render_report.py` (suitable for report-quality images) or `render.py` for lower-level utilities.

Example (render directory of pickles -> video frames/video):

```bash
python render_report.py \
  --results-dir rollout/ \
  --out-dir videos/ \
  --fps 25 \
  --size 1920 1080 \
  --skip 1
```

`plots.py` contains utilities to plot loss curves (used to produce the example `results_*/loss_plot_*.png`).

---

## Model architecture

Implementation split across `model/blocks.py` and `model/model.py`:

* **Encoder**: MLPs map raw node features and raw edge features into a fixed latent dimension `H`.
* **Processor**: Repeated message-passing blocks (`K` times). Each block consists of:

  * `EdgeBlock`: updates edge attributes using sender, receiver node latents and edge attributes.
  * `NodeBlock`: aggregates incoming edge messages and updates node latents.
  * Residual connections across each processor layer are used for stability.
* **Decoder**: MLP that maps node latent to output (2D velocity delta).
* **Simulator wrapper** (`Simulator` in `model/model.py`):

  * Manages normalization.
  * Injects training noise when requested (`get_velocity_noise` in utils).
  * Differentiates behavior between training (single-step, batched) and inference (rollouts).
  * Save/load helpers to persist both model weights and normalizer state.

---

## Normalization & noise

* `utils/utils.py` implements `Normalizer` objects used by `Simulator` to normalize node/edge features and targets.
* Training can inject gaussian velocity noise (controlled by `--noise-std`) to improve robustness in autoregressive rollouts.

---

## Checkpoints & reproducibility

* Checkpoints saved under `--save-dir` contain:

  * model weights (`model` or `simulator` object)
  * normalizer parameters (mean, std/scale) so evaluation uses identical normalization
* To resume training, load the checkpoint path in `Simulator.load` (implemented in `model/model.py`).

* The `jobs/` directory contains final model weights, rollouts and videos for the 50% and 100% training data experiments.

---

## References

This folder contains research code; check the top-level repository LICENSE. The `data/download_dataset.sh` script and dataset originate from DeepMind’s MeshGraphNets dataset: <https://github.com/echowve/meshGraphNets_pytorch> (Apache License 2.0).
