from __future__ import annotations

import torch
from torch import nn


class Router(nn.Module):
    def __init__(self, in_channels: int, num_experts: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channels, num_experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(z)
        pooled = torch.flatten(pooled, start_dim=1)
        scores = self.linear(pooled)
        return torch.softmax(scores, dim=-1)
