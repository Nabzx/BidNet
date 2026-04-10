from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.cifar10 import build_dataloaders
from models.dense_model import SmallResNet
from training.trainer import Trainer
from utils.config import load_config
from utils.device import get_device
from utils.logging import SimpleLogger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dense CIFAR-10 baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])
    device = get_device(config["device"])

    output_dir = str((PROJECT_ROOT / config["output_dir"]).resolve())
    data_dir = str((PROJECT_ROOT / config["data_dir"]).resolve())
    logger = SimpleLogger(output_dir)
    logger.log(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        debug_subset_train=config.get("debug_subset_train"),
        debug_subset_val=config.get("debug_subset_val"),
    )

    model = SmallResNet(
        num_classes=config["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        logger=logger,
        config=config,
        print_every=config.get("print_every", 100),
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config["epochs"],
    )


if __name__ == "__main__":
    main()
