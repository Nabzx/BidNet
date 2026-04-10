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
        config: Dict[str, Any],
        print_every: int = 100,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.history: List[Dict[str, Any]] = []
        self.print_every = print_every
        self.lambda_compute = self.config["training"]["lambda_compute"]
        self.target_compute = self.config["training"]["target_compute"]
        self.lambda_lr = self.config["training"]["lambda_lr"]

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> List[Dict[str, Any]]:
        if self.lambda_compute > 0:
            self.logger.log(
                "Using budget controller: "
                f"lambda_compute={self.lambda_compute}, "
                f"target_compute={self.target_compute}, "
                f"lambda_lr={self.lambda_lr}"
            )
        else:
            self.logger.log("Using lambda_compute = 0.0 (no compute penalty, controller disabled)")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.logger.log(f"Starting epoch {epoch}/{epochs}")
            train_metrics = self.train_one_epoch(train_loader, epoch=epoch)
            val_metrics = self.evaluate(val_loader)
            epoch_time = time.time() - start_time

            metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_compute": train_metrics["compute"],
                "train_flops": train_metrics["flops"],
                "train_sparsity": train_metrics["sparsity"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_compute": val_metrics["compute"],
                "val_flops": val_metrics["flops"],
                "val_sparsity": val_metrics["sparsity"],
                "lambda_compute": self.lambda_compute,
                "epoch_time": epoch_time,
            }
            self.history.append(metrics)
            self.logger.log_epoch(metrics)
            self.logger.save_metrics(self.history)
            self.logger.log(f"Average expert usage (train): {train_metrics['avg_expert_usage'].cpu().numpy()}")
            self.logger.log(f"Average expert usage (val): {val_metrics['avg_expert_usage'].cpu().numpy()}")
            for class_index, avg_usage in val_metrics["class_expert_usage"].items():
                self.logger.log(f"Class {class_index} expert usage: {avg_usage.cpu().numpy()}")

        self.save_checkpoint()
        return self.history

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, Any]:
        self.model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        compute_meter = AverageMeter()
        flops_meter = AverageMeter()
        sparsity_meter = AverageMeter()
        gate_sum = None
        total_samples = 0

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits, compute, flops, gates = self.model(images)
            loss = self.criterion(logits, targets) + self.lambda_compute * compute
            loss.backward()
            self.optimizer.step()

            if self.lambda_compute > 0:
                self.lambda_compute += self.lambda_lr * (compute.item() - self.target_compute)
                self.lambda_compute = max(self.lambda_compute, 0.0)

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(compute_accuracy(logits, targets), batch_size)
            compute_meter.update(compute.item(), batch_size)
            flops_meter.update(flops.item(), batch_size)
            sparsity = (gates < 0.1).float().mean()
            sparsity_meter.update(sparsity.item(), batch_size)
            batch_gate_sum = gates.detach().sum(dim=0)
            gate_sum = batch_gate_sum if gate_sum is None else gate_sum + batch_gate_sum
            total_samples += batch_size

            if self.print_every > 0 and (batch_idx % self.print_every == 0 or batch_idx == len(train_loader)):
                self.logger.log(
                    f"Epoch {epoch:>3} | "
                    f"batch {batch_idx:>4}/{len(train_loader):<4} | "
                    f"loss={loss_meter.average:.4f} | "
                    f"acc={accuracy_meter.average:.4f} | "
                    f"compute={compute_meter.average:.4f} | "
                    f"flops={flops_meter.average:.4f} | "
                    f"sparsity={sparsity_meter.average:.4f} | "
                    f"lambda_compute={self.lambda_compute:.4f}"
                )

        avg_expert_usage = gate_sum / total_samples
        return {
            "loss": loss_meter.average,
            "accuracy": accuracy_meter.average,
            "compute": compute_meter.average,
            "flops": flops_meter.average,
            "sparsity": sparsity_meter.average,
            "avg_expert_usage": avg_expert_usage,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        compute_meter = AverageMeter()
        flops_meter = AverageMeter()
        sparsity_meter = AverageMeter()
        gate_sum = None
        total_samples = 0
        all_gates = []
        all_labels = []

        for images, targets in data_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits, compute, flops, gates = self.model(images)
            loss = self.criterion(logits, targets)

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(compute_accuracy(logits, targets), batch_size)
            compute_meter.update(compute.item(), batch_size)
            flops_meter.update(flops.item(), batch_size)
            sparsity = (gates < 0.1).float().mean()
            sparsity_meter.update(sparsity.item(), batch_size)
            batch_gate_sum = gates.detach().sum(dim=0)
            gate_sum = batch_gate_sum if gate_sum is None else gate_sum + batch_gate_sum
            total_samples += batch_size
            all_gates.append(gates.detach().cpu())
            all_labels.append(targets.detach().cpu())

        avg_expert_usage = gate_sum / total_samples
        class_expert_usage: Dict[int, torch.Tensor] = {}
        all_gates_tensor = torch.cat(all_gates, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        num_classes = int(all_labels_tensor.max().item()) + 1
        for class_index in range(num_classes):
            class_mask = all_labels_tensor == class_index
            if class_mask.sum().item() > 0:
                class_gates = all_gates_tensor[class_mask]
                class_expert_usage[class_index] = class_gates.mean(dim=0)

        return {
            "loss": loss_meter.average,
            "accuracy": accuracy_meter.average,
            "compute": compute_meter.average,
            "flops": flops_meter.average,
            "sparsity": sparsity_meter.average,
            "avg_expert_usage": avg_expert_usage,
            "class_expert_usage": class_expert_usage,
        }

    def save_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / "model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
