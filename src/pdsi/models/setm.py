from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _rfft_fp32(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft(x.float(), dim=-1, norm="ortho")


def _rbf_basis(num_bins: int, num_bands: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num_bands < 2:
        raise ValueError("num_bands must be >= 2")
    freqs = torch.linspace(0.0, 1.0, num_bins, device=device, dtype=dtype)
    centers = torch.linspace(0.0, 1.0, num_bands, device=device, dtype=dtype)
    width = 1.5 / max(num_bands - 1, 1)
    basis = torch.exp(-0.5 * ((freqs[None, :] - centers[:, None]) / width) ** 2)
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class SpectralSummary(nn.Module):
    """Compresses a multichannel waveform into a small frequency-band descriptor."""

    def __init__(self, num_bands: int) -> None:
        super().__init__()
        self.num_bands = num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = _rfft_fp32(x)
        magnitude = spectrum.abs().mean(dim=1)
        basis = _rbf_basis(spectrum.shape[-1], self.num_bands, spectrum.device, magnitude.dtype)
        return torch.log1p(magnitude) @ basis.T


class PrototypeRouter(nn.Module):
    """Maps spectral summaries to block-wise expert weights via learned prototypes."""

    def __init__(
        self,
        num_bands: int,
        num_prototypes: int,
        num_blocks: int,
        num_experts: int,
        hidden_channels: int,
        gain_limit: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        self.num_experts = num_experts
        self.hidden_channels = hidden_channels
        self.gain_limit = gain_limit

        self.summary_norm = nn.LayerNorm(num_bands)
        self.prototype_proj = nn.Linear(num_bands, num_bands, bias=False)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, num_bands) * 0.02)
        self.route_library = nn.Parameter(torch.zeros(num_prototypes, num_blocks, num_experts))
        self.gain_library = nn.Parameter(torch.zeros(num_prototypes, num_blocks, hidden_channels))
        self._last_assignments: Optional[torch.Tensor] = None
        self._last_routes: Optional[torch.Tensor] = None

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self.summary_norm(summary)
        routed_summary = self.prototype_proj(summary)
        logits = -((routed_summary[:, None, :] - self.prototypes[None, :, :]) ** 2).mean(dim=-1)
        assignments = torch.softmax(logits * math.sqrt(summary.shape[-1]), dim=-1)
        route_logits = torch.einsum("bp,pde->bde", assignments, self.route_library)
        routes = torch.softmax(route_logits, dim=-1)
        gains = 1.0 + self.gain_limit * torch.tanh(torch.einsum("bp,pdc->bdc", assignments, self.gain_library))
        self._last_assignments = assignments
        self._last_routes = routes
        return routes, gains

    def regularization_loss(self, lambda_balance: float = 0.0, lambda_temporal_smooth: float = 0.0) -> torch.Tensor:
        if self._last_assignments is None or self._last_routes is None:
            return self.prototypes.new_zeros(())
        loss = self.prototypes.new_zeros(())
        if lambda_balance:
            usage = self._last_assignments.mean(dim=0)
            target = torch.full_like(usage, 1.0 / usage.numel())
            loss = loss + lambda_balance * ((usage - target) ** 2).mean()
        if lambda_temporal_smooth and self._last_routes.shape[1] > 1:
            loss = loss + lambda_temporal_smooth * (
                self._last_routes[:, 1:, :] - self._last_routes[:, :-1, :]
            ).abs().mean()
        return loss

    def detached_assignments(self) -> Optional[torch.Tensor]:
        return None if self._last_assignments is None else self._last_assignments.detach()


class RoutedTemporalBlock(nn.Module):
    """Sequential mixer block with depthwise temporal experts and channel mixing."""

    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        expert_kernel_size: int = 7,
        expert_dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.expert_dilations = expert_dilations
        self.norm1 = nn.GroupNorm(1, channels)
        self.experts = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=expert_kernel_size,
                    dilation=dilation,
                    padding=(expert_kernel_size // 2) * dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in expert_dilations
            ]
        )
        self.expert_mix = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        hidden = int(channels * mlp_ratio)
        self.norm2 = nn.GroupNorm(1, channels)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, route_weights: torch.Tensor, channel_gain: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        pieces = torch.stack([expert(z) for expert in self.experts], dim=1)
        mixed = torch.einsum("be,bect->bct", route_weights, pieces)
        mixed = self.expert_mix(mixed * channel_gain.unsqueeze(-1))
        x = x + mixed
        x = x + self.channel_mlp(self.norm2(x))
        return x


class SETMBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 96,
        depth: int = 4,
        dropout: float = 0.1,
        expert_kernel_size: int = 7,
        expert_dilations: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                RoutedTemporalBlock(
                    channels=hidden_channels,
                    dropout=dropout,
                    expert_kernel_size=expert_kernel_size,
                    expert_dilations=expert_dilations,
                )
                for _ in range(depth)
            ]
        )
        self.out_channels = hidden_channels

    def forward(self, x: torch.Tensor, routes: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        for idx, block in enumerate(self.blocks):
            z = block(z, routes[:, idx, :], gains[:, idx, :])
        return z


@dataclass
class ModelConfig:
    num_channels: int
    num_classes: int
    hidden_channels: int = 96
    depth: int = 4
    dropout: float = 0.1
    num_bands: int = 16
    num_prototypes: int = 8
    router_mode: str = "prototype"
    expert_kernel_size: int = 7
    expert_dilations: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8))
    gain_limit: float = 0.5
    multilabel: bool = False

    def __post_init__(self) -> None:
        self.router_mode = self.router_mode.lower()
        self.expert_dilations = tuple(self.expert_dilations)


class DirectRouter(nn.Module):
    """Ablation that predicts routes directly from the spectral summary."""

    def __init__(
        self,
        num_bands: int,
        num_blocks: int,
        num_experts: int,
        hidden_channels: int,
        gain_limit: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = max(num_bands * 2, 16)
        self.summary_norm = nn.LayerNorm(num_bands)
        self.route_head = nn.Sequential(
            nn.Linear(num_bands, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_blocks * num_experts),
        )
        self.gain_head = nn.Sequential(
            nn.Linear(num_bands, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_blocks * hidden_channels),
        )
        self.num_blocks = num_blocks
        self.num_experts = num_experts
        self.hidden_channels = hidden_channels
        self.gain_limit = gain_limit
        self._last_routes: Optional[torch.Tensor] = None

        nn.init.zeros_(self.route_head[-1].weight)
        nn.init.zeros_(self.route_head[-1].bias)
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.zeros_(self.gain_head[-1].bias)

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self.summary_norm(summary)
        routes = self.route_head(summary).view(-1, self.num_blocks, self.num_experts)
        routes = torch.softmax(routes, dim=-1)
        gains = self.gain_head(summary).view(-1, self.num_blocks, self.hidden_channels)
        gains = 1.0 + self.gain_limit * torch.tanh(gains)
        self._last_routes = routes
        return routes, gains

    def regularization_loss(self, lambda_balance: float = 0.0, lambda_temporal_smooth: float = 0.0) -> torch.Tensor:
        if self._last_routes is None:
            return self.route_head[-1].weight.new_zeros(())
        if not lambda_temporal_smooth or self._last_routes.shape[1] <= 1:
            return self.route_head[-1].weight.new_zeros(())
        return lambda_temporal_smooth * (self._last_routes[:, 1:, :] - self._last_routes[:, :-1, :]).abs().mean()


class StaticRouter(nn.Module):
    """Ablation with one dataset-level route schedule shared by all samples."""

    def __init__(self, num_blocks: int, num_experts: int, hidden_channels: int, gain_limit: float = 0.5) -> None:
        super().__init__()
        self.route_logits = nn.Parameter(torch.zeros(num_blocks, num_experts))
        self.gain_logits = nn.Parameter(torch.zeros(num_blocks, hidden_channels))
        self.gain_limit = gain_limit
        self._last_routes: Optional[torch.Tensor] = None

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = summary.shape[0]
        routes = torch.softmax(self.route_logits, dim=-1).unsqueeze(0).expand(batch, -1, -1)
        gains = 1.0 + self.gain_limit * torch.tanh(self.gain_logits).unsqueeze(0).expand(batch, -1, -1)
        self._last_routes = routes
        return routes, gains

    def regularization_loss(self, lambda_balance: float = 0.0, lambda_temporal_smooth: float = 0.0) -> torch.Tensor:
        if self._last_routes is None or not lambda_temporal_smooth or self._last_routes.shape[1] <= 1:
            return self.route_logits.new_zeros(())
        return lambda_temporal_smooth * (self._last_routes[:, 1:, :] - self._last_routes[:, :-1, :]).abs().mean()


class UniformRouter(nn.Module):
    """Ablation that averages experts uniformly for every sample and block."""

    def __init__(self, num_blocks: int, num_experts: int, hidden_channels: int) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_experts = num_experts
        self.hidden_channels = hidden_channels

    def forward(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = summary.shape[0]
        routes = summary.new_full((batch, self.num_blocks, self.num_experts), 1.0 / self.num_experts)
        gains = summary.new_ones((batch, self.num_blocks, self.hidden_channels))
        return routes, gains

    def regularization_loss(self, lambda_balance: float = 0.0, lambda_temporal_smooth: float = 0.0) -> torch.Tensor:
        return torch.zeros(())


def build_router(cfg: ModelConfig) -> nn.Module:
    num_blocks = cfg.depth
    num_experts = len(cfg.expert_dilations)
    if cfg.router_mode == "prototype":
        return PrototypeRouter(
            num_bands=cfg.num_bands,
            num_prototypes=cfg.num_prototypes,
            num_blocks=num_blocks,
            num_experts=num_experts,
            hidden_channels=cfg.hidden_channels,
            gain_limit=cfg.gain_limit,
        )
    if cfg.router_mode == "direct":
        return DirectRouter(
            num_bands=cfg.num_bands,
            num_blocks=num_blocks,
            num_experts=num_experts,
            hidden_channels=cfg.hidden_channels,
            gain_limit=cfg.gain_limit,
        )
    if cfg.router_mode == "static":
        return StaticRouter(
            num_blocks=num_blocks,
            num_experts=num_experts,
            hidden_channels=cfg.hidden_channels,
            gain_limit=cfg.gain_limit,
        )
    if cfg.router_mode == "uniform":
        return UniformRouter(
            num_blocks=num_blocks,
            num_experts=num_experts,
            hidden_channels=cfg.hidden_channels,
        )
    raise ValueError(f"Unknown router_mode: {cfg.router_mode}")


class SETMClassifier(nn.Module):
    """Spectral Episode Temporal Mixer.

    The model extracts a small FFT-based summary, softly assigns the sample to
    learned spectral prototypes, and uses the resulting episode mixture to route
    a stack of temporal expert blocks.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.uses_fft = True
        self.fft_passes = 1
        self.summary = SpectralSummary(cfg.num_bands)
        self.router = build_router(cfg)
        self.backbone = SETMBackbone(
            in_channels=cfg.num_channels,
            hidden_channels=cfg.hidden_channels,
            depth=cfg.depth,
            dropout=cfg.dropout,
            expert_kernel_size=cfg.expert_kernel_size,
            expert_dilations=cfg.expert_dilations,
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(self.backbone.out_channels, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = self.summary(x)
        routes, gains = self.router(summary)
        z = self.backbone(x, routes, gains)
        return self.head(z)

    def regularization_loss(self, lambda_balance: float = 0.0, lambda_temporal_smooth: float = 0.0) -> torch.Tensor:
        if hasattr(self.router, "regularization_loss"):
            return self.router.regularization_loss(
                lambda_balance=lambda_balance,
                lambda_temporal_smooth=lambda_temporal_smooth,
            )
        return torch.zeros((), device=next(self.parameters()).device)
