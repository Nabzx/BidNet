from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from training.metrics import AverageMeter, compute_accuracy
from utils.logging import SimpleLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        output_dir: str,
        logger: SimpleLogger,
        print_every: int = 100,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()
        self.history: List[Dict[str, Any]] = []
        self.print_every = print_every

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> List[Dict[str, Any]]:
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.logger.log(f"Starting epoch {epoch}/{epochs}")
            train_loss, train_accuracy, train_compute = self.train_one_epoch(train_loader, epoch=epoch)
            val_loss, val_accuracy, val_compute = self.evaluate(val_loader)
            epoch_time = time.time() - start_time

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_compute": train_compute,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_compute": val_compute,
                "epoch_time": epoch_time,
            }
            self.history.append(metrics)
            self.logger.log_epoch(metrics)
            self.logger.save_metrics(self.history)

        self.save_checkpoint()
        return self.history

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        compute_meter = AverageMeter()

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits, compute = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(compute_accuracy(logits, targets), batch_size)
            compute_meter.update(compute.item(), batch_size)

            if self.print_every > 0 and (batch_idx % self.print_every == 0 or batch_idx == len(train_loader)):
                self.logger.log(
                    f"Epoch {epoch:>3} | "
                    f"batch {batch_idx:>4}/{len(train_loader):<4} | "
                    f"loss={loss_meter.average:.4f} | "
                    f"acc={accuracy_meter.average:.4f} | "
                    f"compute={compute_meter.average:.4f}"
                )

        return loss_meter.average, accuracy_meter.average, compute_meter.average

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        compute_meter = AverageMeter()

        for images, targets in data_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits, compute = self.model(images)
            loss = self.criterion(logits, targets)

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(compute_accuracy(logits, targets), batch_size)
            compute_meter.update(compute.item(), batch_size)

        return loss_meter.average, accuracy_meter.average, compute_meter.average

    def save_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / "model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
