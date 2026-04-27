from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_forward_flops(model: nn.Module, seq_len: int, num_channels: int, batch_size: int = 1) -> dict[str, Any]:
    """Rough forward-pass FLOP estimate using Conv1d/Linear hooks plus FFT cost."""

    conv_flops = 0
    linear_flops = 0
    hooks = []

    def conv_hook(module: nn.Conv1d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal conv_flops
        batch = output.shape[0]
        out_channels = output.shape[1]
        out_len = output.shape[2]
        kernel = module.kernel_size[0]
        groups = module.groups
        in_per_group = module.in_channels // groups
        conv_flops += 2 * batch * out_channels * out_len * in_per_group * kernel

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal linear_flops
        batch = output.shape[0] if output.ndim > 1 else 1
        linear_flops += 2 * batch * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(batch_size, num_channels, seq_len, device=device)
        _ = model(dummy)
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()

    gate_name = model.gate.__class__.__name__ if hasattr(model, "gate") else ""
    fft_flops = 0
    if gate_name not in {"IdentityGate", ""}:
        fft_flops = int(10 * batch_size * num_channels * seq_len * math.log2(max(seq_len, 2)))
    return {
        "params": count_parameters(model),
        "conv_flops": int(conv_flops),
        "linear_flops": int(linear_flops),
        "fft_flops": int(fft_flops),
        "total_forward_flops": int(conv_flops + linear_flops + fft_flops),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_channels": num_channels,
    }
