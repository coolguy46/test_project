from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as Fnn


class IdentityGate(nn.Module):
    """No-op gate used for matched InceptionTime baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def regularization_loss(self, lambda_identity: float = 0.0, lambda_tv: float = 0.0) -> torch.Tensor:
        return self._zero


def _rbf_basis(num_bins: int, num_bands: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num_bands < 2:
        raise ValueError("num_bands must be >= 2")
    freqs = torch.linspace(0.0, 1.0, num_bins, device=device, dtype=dtype)
    centers = torch.linspace(0.0, 1.0, num_bands, device=device, dtype=dtype)
    width = 1.0 / max(num_bands - 1, 1)
    basis = torch.exp(-0.5 * ((freqs[None, :] - centers[:, None]) / width) ** 2)
    basis = basis / basis.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return basis


class AdaptiveSpectralGate(nn.Module):
    """Bounded sample-adaptive magnitude gate that preserves Fourier phase.

    For input x in R^(B,C,T), the gate computes X = rFFT(x), predicts a smooth
    low-rank real mask M in [1-max_delta, 1+max_delta], and returns
    iRFFT(M * X). Multiplication by a nonnegative real mask changes amplitude
    but leaves the phase of every retained Fourier coefficient unchanged.
    """

    def __init__(
        self,
        channels: int,
        num_bands: int = 16,
        hidden_multiplier: int = 2,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < max_delta < 1.0:
            raise ValueError("max_delta must be in (0, 1) to keep the mask positive")
        hidden = max(num_bands * hidden_multiplier, 8)
        self.channels = channels
        self.num_bands = num_bands
        self.max_delta = max_delta
        self.controller = nn.Sequential(
            nn.LayerNorm(num_bands),
            nn.Linear(num_bands, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_bands),
        )
        self.channel_offsets = nn.Parameter(torch.zeros(channels, num_bands))
        self._last_mask: Optional[torch.Tensor] = None

        last = self.controller[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _band_energy(self, magnitude: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        band_basis = basis / basis.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_power = torch.log1p(magnitude.mean(dim=1))
        return log_power @ band_basis.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
        magnitude = spectrum.abs()
        bins = spectrum.shape[-1]
        basis = _rbf_basis(bins, self.num_bands, spectrum.device, magnitude.dtype)
        band_energy = self._band_energy(magnitude, basis)
        sample_weights = self.controller(band_energy)
        raw = sample_weights[:, None, :] + self.channel_offsets[None, :, :]
        delta = torch.einsum("bck,kf->bcf", raw, basis)
        mask = 1.0 + self.max_delta * torch.tanh(delta)
        self._last_mask = mask
        gated = spectrum * mask.to(dtype=spectrum.dtype)
        return torch.fft.irfft(gated, n=x.shape[-1], dim=-1, norm="ortho")

    def regularization_loss(self, lambda_identity: float = 0.0, lambda_tv: float = 0.0) -> torch.Tensor:
        if self._last_mask is None:
            return torch.zeros((), device=self.channel_offsets.device)
        loss = torch.zeros((), device=self._last_mask.device)
        if lambda_identity:
            loss = loss + lambda_identity * ((self._last_mask - 1.0) ** 2).mean()
        if lambda_tv and self._last_mask.shape[-1] > 1:
            loss = loss + lambda_tv * (self._last_mask[..., 1:] - self._last_mask[..., :-1]).abs().mean()
        return loss

    def detached_mask(self) -> Optional[torch.Tensor]:
        return None if self._last_mask is None else self._last_mask.detach()


class ComplexSpectralGate(nn.Module):
    """Ablation gate that learns a complex multiplier and can perturb phase."""

    def __init__(self, channels: int, max_bins: int = 4097, max_delta: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < max_delta < 1.0:
            raise ValueError("max_delta must be in (0, 1)")
        self.channels = channels
        self.max_bins = max_bins
        self.max_delta = max_delta
        self.real_delta = nn.Parameter(torch.zeros(1, channels, max_bins))
        self.imag_delta = nn.Parameter(torch.zeros(1, channels, max_bins))
        self._last_multiplier: Optional[torch.Tensor] = None

    def _resize(self, param: torch.Tensor, bins: int) -> torch.Tensor:
        if param.shape[-1] == bins:
            return param
        return Fnn.interpolate(param, size=bins, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
        bins = spectrum.shape[-1]
        real = 1.0 + self.max_delta * torch.tanh(self._resize(self.real_delta, bins))
        imag = self.max_delta * torch.tanh(self._resize(self.imag_delta, bins))
        multiplier = torch.complex(real, imag)
        self._last_multiplier = multiplier
        gated = spectrum * multiplier
        return torch.fft.irfft(gated, n=x.shape[-1], dim=-1, norm="ortho")

    def regularization_loss(self, lambda_identity: float = 0.0, lambda_tv: float = 0.0) -> torch.Tensor:
        if self._last_multiplier is None:
            return torch.zeros((), device=self.real_delta.device)
        real = self._last_multiplier.real
        imag = self._last_multiplier.imag
        loss = torch.zeros((), device=real.device)
        if lambda_identity:
            loss = loss + lambda_identity * (((real - 1.0) ** 2) + (imag**2)).mean()
        if lambda_tv and real.shape[-1] > 1:
            loss = loss + lambda_tv * (
                (real[..., 1:] - real[..., :-1]).abs().mean()
                + (imag[..., 1:] - imag[..., :-1]).abs().mean()
            )
        return loss


class FixedButterworthGate(nn.Module):
    """Fixed frequency-domain Butterworth-style bandpass control."""

    def __init__(self, low_cut: float = 0.01, high_cut: float = 0.45, order: int = 4) -> None:
        super().__init__()
        if not 0.0 <= low_cut < high_cut <= 1.0:
            raise ValueError("Require 0 <= low_cut < high_cut <= 1 in Nyquist-normalized units")
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
        bins = spectrum.shape[-1]
        freqs = torch.linspace(0.0, 1.0, bins, device=x.device, dtype=x.dtype)
        low_pass = 1.0 / torch.sqrt(1.0 + (freqs / self.high_cut).clamp_min(0.0) ** (2 * self.order))
        if self.low_cut > 0:
            high_pass = 1.0 / torch.sqrt(1.0 + (self.low_cut / freqs.clamp_min(1e-6)) ** (2 * self.order))
        else:
            high_pass = torch.ones_like(freqs)
        mask = (low_pass * high_pass)[None, None, :]
        gated = spectrum * mask.to(dtype=spectrum.dtype)
        return torch.fft.irfft(gated, n=x.shape[-1], dim=-1, norm="ortho")

    def regularization_loss(self, lambda_identity: float = 0.0, lambda_tv: float = 0.0) -> torch.Tensor:
        return self._zero


def build_gate(
    gate: str,
    channels: int,
    num_bands: int = 16,
    max_delta: float = 0.5,
    butter_low: float = 0.01,
    butter_high: float = 0.45,
) -> nn.Module:
    gate = gate.lower()
    if gate in {"none", "identity", "baseline"}:
        return IdentityGate()
    if gate == "pdsi":
        return AdaptiveSpectralGate(channels=channels, num_bands=num_bands, max_delta=max_delta)
    if gate == "complex":
        return ComplexSpectralGate(channels=channels, max_delta=max_delta)
    if gate == "butterworth":
        return FixedButterworthGate(low_cut=butter_low, high_cut=butter_high)
    raise ValueError(f"Unknown gate mode: {gate}")
