# Compute-Aware Bidding

Stage 1 is a minimal PyTorch research baseline for CIFAR-10. It includes a dense small ResNet-style CNN, a YAML config, a plain training loop, evaluation, checkpoint saving, and metrics logging.

## What Stage 1 Implements

- CIFAR-10 data loading with standard training augmentation
- A custom dense residual CNN baseline
- Training and evaluation with accuracy and loss tracking
- Reproducibility utilities for seeds and device selection
- Checkpoint and metrics saving

This stage does not include experts, routing, bidding, compute-aware logic, FLOPs estimation, or multiple model variants.

## Setup

```bash
cd compute_aware_bidding
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config configs/dense.yaml
```

Outputs are written to the directory specified by `output_dir` in the config. By default this includes:

- `model.pt`
- `metrics.json`
