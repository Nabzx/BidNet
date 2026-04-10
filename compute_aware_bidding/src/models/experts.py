from __future__ import annotations

import torch
from torch import nn

from models.router import Router


class ResidualExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class ExpertBlock(nn.Module):
    def __init__(self, channels: int, num_experts: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = Router(in_channels=channels, num_experts=num_experts)
        self.experts = nn.ModuleList(
            [ResidualExpert(channels) for _ in range(num_experts)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        weights = self.router(z)
        batch_size = z.size(0)
        alpha = 0.1

        outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(z)
            weight = weights[:, i].view(batch_size, 1, 1, 1)
            outputs.append(weight * out)

        final = sum(outputs)
        return z + alpha * final
