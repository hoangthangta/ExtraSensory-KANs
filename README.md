# ExtraSensory_KANs

Official implementation for the paper:

> **Kolmogorov-Arnold Networks for Personal Context Recognition on ExtraSensory**

This repository provides the code and experimental configurations for evaluating KAN variants on the ExtraSensory personal context recognition dataset. The experiments compare KAN models with conventional **MLP** and **TabM**.

## Overview

The repository evaluates the following models:

* **MLP** — Multilayer Perceptron
* **TabM** — Parameter-efficient tabular deep-learning model
* **EfficientKAN** — Efficient Kolmogorov-Arnold Network
* **FastKAN** — Fast Kolmogorov-Arnold Network
* **BSRBF-KAN** — B-spline Radial Basis Function KAN

One classification task is considered:
* **Multilabel classification** — 51 context labels
---

## Repository Structure

```text
ExtraSensory_KANs/
│
├── data/
│   └── ...                         # ExtraSensory dataset
│
├── mlp.py                          # MLP
├── tabm.py                         # TabM
├── efficient_kan.py                # EfficientKAN
├── fast_kan.py                     # FastKAN
├── bsrbf_kan.py                    # BSRBF-KAN
│
├── schedulers.py                   # Learning-rate schedulers
├── utils.py                        # Utility functions
│
├── run_extra_sen.py                # ExtraSensory experiments
├── run_extra_sen_ex.sh             # Experiment commands
│
├── README.md
└── ...
```

---

# Dataset

The experiments use the **ExtraSensory** dataset, containing:

* **225 input features**
* **51 context labels**
* **60 users**

One task are evaluated:

* **Multilabel classification**

The experiments follow the official five-fold user-level partition. A validation set is further selected from the training users.

Place the dataset under the `data/` directory according to the structure expected by `run_extra_sen.py`.

---

# Running the Experiments

The main experiment script is:

```text
run_extra_sen.py
```

The corresponding shell script contains the experimental configurations:

```text
run_extra_sen_ex.sh
```

For the complete experimental settings, please refer to the shell script.

---

# Common Parameters

The main experiment script supports:

* `--task`: Task to train: `multilabel`.
* `--model`: Model to train: `mlp`, `tab_m`, `efficient_kan`, `fast_kan`, or `bsrbf_kan`.
* `--fold`: Five-fold user-level cross-validation fold.
* `--hidden_layers`: Hidden-layer configuration. Default: `256`.
* `--batch_size`: Training batch size. Default: `64`.
* `--epochs`: Number of training epochs. Default: `20`.
* `--lr`: Learning rate. Default: `1e-3`.
* `--weight_decay`: Weight decay. Default: `1e-4`.
* `--scheduler`: Learning-rate scheduler. Default: `OneCycleLR`.
* `--norm1_type`: First normalization layer.
* `--norm2_type`: Second normalization layer.
* `--norm_mode`: Normalization mode. Default: `all`.
* `--norm_type`: General normalization type. Default: `layer`.
* `--activation`: Activation function. Default: `silu`.
* `--num_grids`: Number of grid points used by KAN models.
* `--balance`: Enable class balancing.
* `--select_metric`: Validation metric used for model selection.
* `--seed`: Random seed.
* `--device`: Computing device. Default: `cuda`.
* `--note`: Optional experiment identifier.

---

# Example Commands

```
python run_extra_sen.py --task "multilabel" --model "fast_kan" --fold 0 --hidden_layers "29" --num_grids 8 --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "run_0"
```
```
python run_extra_sen.py --task "multilabel" --model "bsrbf_kan" --fold 0 --hidden_layers "29" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --norm_type "layer" --seed 0 --note "run_0"
```
```
python run_extra_sen.py --task "multilabel" --model "efficient_kan" --fold 0 --hidden_layers "26" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "run_0"
```
```
python run_extra_sen.py --task "multilabel" --model "mlp" --fold 0 --hidden_layers "256" --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "run_0"
```
```
python run_extra_sen.py --task "multilabel" --model "tab_m" --fold 0 --batch_size 64 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --scheduler "OneCycleLR" --balance --select_metric "balanced_accuracy" --seed 0 --note "run_0"
```
---

# Reproducibility

To reproduce the experiments reported in the paper, use:

```text
run_extra_sen_ex.sh
```

The script contains the model configurations, hyperparameters, folds, random seeds, and other experimental settings used in the study.

The complete benchmark consists of:

```text
6 models × 2 tasks × 5 folds × 3 seeds = 180 runs
```

---
