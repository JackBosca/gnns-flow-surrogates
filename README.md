# GNNs Flow Surrogates

## Overview

This is the repository for the EPFL semester project on **Compressible and Inviscid Fluid Flow Modeling Using Graph Neural Networks** carried out at the Intelligent Maintenance and Operations Systems Laboratory (IMOS) lab. The repository is divided as follows:

* `cylinder_flow/`: **MeshGraphNets** implementation and experiments for the 2D incompressible flow past a cylinder.
* `euler_the_well/`: **Flux Conservative GNN** implementation for 2D compressible (inviscid) Euler gases using *The Well* dataset.

This README is intentionally concise, as each subfolder contains a detailed `README.md` with detailed implementation notes.

---

## Repository layout

```
.
├── cylinder_flow/           # MeshGraphNets (incompressible cylinder flow)
├── euler_the_well/          # Flux Conservative GNN (compressible Euler)
├── .gitignore
├── LICENSE
├── env.yaml                 # Conda environment file for reproducibility
├── report.pdf               # Project report
└── README.md                # (this file)
```

---

## Quick start

1. Create environment (recommended)

```bash
conda env create -f env.yaml
conda activate gnn_flows    
```

2. Read the detailed instructions for the part you want to run:

* `cylinder_flow/README.md`: training, rollout, rendering for MeshGraphNets.
* `euler_the_well/README.md`: training, rollout, rendering for Flux Conservative GNN.

3. Example commands (run from repo root):

Train MeshGraphNets (cylinder flow):

```bash
cd cylinder_flow
python train.py --dataset-dir /path/to/h5_file --batch-size 1 --max-epochs 10 --save-dir checkpoint
```

Train Flux Conservative GNN (Euler / The Well):

```bash
cd euler_the_well
python train_flux.py   # edit paths & params at bottom of script
```

Run a rollout:

```bash
# MeshGraphNets rollout
cd cylinder_flow
python rollout.py --model_dir checkpoint/simulator.pth --dataset_dir /path/to/h5_output --test_split test --rollout_num 5

# Flux Conservative GNN rollout
cd ../euler_the_well
python rollout_flux.py
```

Render outputs:

```bash
# Example (adapt paths)
cd cylinder_flow
python render.py --results-dir rollout/ --out-dir videos/ --size 1920 1080

cd ../euler_the_well
python render.py --results-dir ./rollouts/ --out-dir ./videos/ --size 1920 1080
```

---

## Report

`report.pdf` contains the full project report. Key takeaways:

* Implemented and benchmarked MeshGraphNets on an incompressible cylinder-flow baseline: shows good one-step accuracy but long-horizon drift.
* Designed **Flux Conservative GNN** that predicts interface fluxes within an FVM backbone, enforcing conservation by construction.
* The new model incorporates SE(2)-equivariant edge projections, spatio-temporal edge memory, learned artificial viscosity, and a dynamic CFL condition.
* The new model yields substantially better autoregressive stability and strong zero-shot generalization across gases in *The Well* dataset (while being trained just on a single gas).

Read the full `report.pdf` for precise details.

---

## Citation & license

See `LICENSE` at repo root for the license terms.

---

## Contact

Author: Boscariol Jacopo, MSc Mechanical Engineering, EPFL.
(See `report.pdf` for supervisors details.)

---
