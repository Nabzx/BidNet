from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class SimpleLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.json"

    def log(self, message: str) -> None:
        print(message)

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        message = (
            f"Epoch {metrics['epoch']:>3} | "
            f"train_loss={metrics['train_loss']:.4f} | "
            f"train_acc={metrics['train_accuracy']:.4f} | "
            f"train_compute={metrics['train_compute']:.4f} | "
            f"train_flops={metrics['train_flops']:.4f} | "
            f"train_sparsity={metrics['train_sparsity']:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_acc={metrics['val_accuracy']:.4f} | "
            f"val_compute={metrics['val_compute']:.4f} | "
            f"val_flops={metrics['val_flops']:.4f} | "
            f"val_sparsity={metrics['val_sparsity']:.4f} | "
            f"time={metrics['epoch_time']:.2f}s"
        )
        self.log(message)

    def save_metrics(self, history: List[Dict[str, Any]]) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
