from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .gates import build_gate


def _odd_kernel(k: int) -> int:
    return k if k % 2 == 1 else k + 1


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 32,
        branch_channels: int = 32,
        kernel_sizes: Iterable[int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        kernels = [_odd_kernel(k) for k in kernel_sizes]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels),
            nn.GELU(),
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        out_channels = branch_channels * (len(kernels) + 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        pieces = [branch(z) for branch in self.branches]
        pieces.append(self.pool_branch(x))
        return self.act(self.norm(torch.cat(pieces, dim=1)))


class ResidualInceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        branch_channels: int,
        kernel_sizes: Iterable[int],
    ) -> None:
        super().__init__()
        self.inception = InceptionModule(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            branch_channels=branch_channels,
            kernel_sizes=kernel_sizes,
        )
        out_channels = self.inception.out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.act = nn.GELU()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.inception(x) + self.shortcut(x))


class InceptionBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int = 4,
        bottleneck_channels: int = 32,
        branch_channels: int = 24,
        kernel_sizes: Iterable[int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        blocks = []
        channels = in_channels
        for _ in range(depth):
            block = ResidualInceptionBlock(
                in_channels=channels,
                bottleneck_channels=bottleneck_channels,
                branch_channels=branch_channels,
                kernel_sizes=kernel_sizes,
            )
            channels = block.out_channels
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@dataclass
class ModelConfig:
    num_channels: int
    num_classes: int
    gate: str = "pdsi"
    backbone: str = "inception"
    depth: int = 4
    bottleneck_channels: int = 32
    branch_channels: int = 24
    hidden_channels: int = 96
    num_bands: int = 16
    max_delta: float = 0.5
    dropout: float = 0.1
    multilabel: bool = False


class FCNBackbone(nn.Module):
    """Compact fully convolutional TSC baseline."""

    def __init__(self, in_channels: int, hidden_channels: int = 96) -> None:
        super().__init__()
        channels = [hidden_channels, hidden_channels * 2, hidden_channels]
        kernels = [8, 5, 3]
        layers = []
        current = in_channels
        for out_channels, kernel in zip(channels, kernels):
            layers.extend(
                [
                    nn.Conv1d(current, out_channels, kernel_size=kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ]
            )
            current = out_channels
        self.net = nn.Sequential(*layers)
        self.out_channels = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels))
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.shortcut(x))


class ResNetBackbone1D(nn.Module):
    """Small residual CNN baseline for time-series classification."""

    def __init__(self, in_channels: int, hidden_channels: int = 96, depth: int = 4) -> None:
        super().__init__()
        blocks = []
        current = in_channels
        for idx in range(depth):
            out_channels = hidden_channels * (2 if idx >= max(depth - 1, 1) else 1)
            blocks.append(ResNetBlock1D(current, out_channels))
            current = out_channels
        self.net = nn.Sequential(*blocks)
        self.out_channels = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(cfg: ModelConfig) -> nn.Module:
    name = cfg.backbone.lower()
    if name == "inception":
        return InceptionBackbone(
            in_channels=cfg.num_channels,
            depth=cfg.depth,
            bottleneck_channels=cfg.bottleneck_channels,
            branch_channels=cfg.branch_channels,
        )
    if name == "fcn":
        return FCNBackbone(in_channels=cfg.num_channels, hidden_channels=cfg.hidden_channels)
    if name in {"resnet", "resnet1d"}:
        return ResNetBackbone1D(in_channels=cfg.num_channels, hidden_channels=cfg.hidden_channels, depth=cfg.depth)
    raise ValueError(f"Unknown backbone: {cfg.backbone}")


class PDSIClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = build_gate(
            gate=cfg.gate,
            channels=cfg.num_channels,
            num_bands=cfg.num_bands,
            max_delta=cfg.max_delta,
        )
        self.backbone = build_backbone(cfg)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.backbone.out_channels, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(x)
        z = self.backbone(x)
        return self.head(z)

    def regularization_loss(self, lambda_identity: float = 0.0, lambda_tv: float = 0.0) -> torch.Tensor:
        if hasattr(self.gate, "regularization_loss"):
            return self.gate.regularization_loss(lambda_identity=lambda_identity, lambda_tv=lambda_tv)
        return torch.zeros((), device=next(self.parameters()).device)
