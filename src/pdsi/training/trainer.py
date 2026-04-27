from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 0
    amp: bool = True
    amp_dtype: str = "bfloat16"
    lambda_identity: float = 1e-4
    lambda_tv: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 0
    multilabel: bool = False
    progress: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _loss_fn(multilabel: bool) -> nn.Module:
    return nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()


def _loader(dataset: TensorDataset, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_all = []
    target_all = []
    total_loss = 0.0
    criterion = _loss_fn(cfg.multilabel)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y.float() if cfg.multilabel else y.long())
            total_loss += float(loss.item()) * x.shape[0]
            logits_all.append(logits.detach().cpu())
            target_all.append(y.detach().cpu())
    logits_cat = torch.cat(logits_all, dim=0)
    target_cat = torch.cat(target_all, dim=0)
    metrics = classification_metrics(logits_cat, target_cat, multilabel=cfg.multilabel)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_model(
    model: nn.Module,
    datasets: dict[str, TensorDataset],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    model.to(device)
    train_loader = _loader(datasets["train"], cfg, shuffle=True)
    val_loader = _loader(datasets["val"], cfg, shuffle=False)
    test_loader = _loader(datasets["test"], cfg, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))
    criterion = _loss_fn(cfg.multilabel)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and cfg.amp_dtype == "float16")

    best_state = copy.deepcopy(model.state_dict())
    best_val = -float("inf")
    best_epoch = 0
    history = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        iterator = train_loader
        progress_bar = None
        if cfg.progress:
            try:
                from tqdm.auto import tqdm

                progress_bar = tqdm(
                    train_loader,
                    desc=f"epoch {epoch}/{cfg.epochs}",
                    leave=False,
                    dynamic_ncols=True,
                )
                iterator = progress_bar
            except ModuleNotFoundError:
                progress_bar = None
        for x, y in iterator:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(cfg.amp_dtype), enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y.float() if cfg.multilabel else y.long())
                if hasattr(model, "regularization_loss"):
                    loss = loss + model.regularization_loss(cfg.lambda_identity, cfg.lambda_tv)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().item()) * x.shape[0]
            if progress_bar is not None:
                progress_bar.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

        scheduler.step()
        val_metrics = evaluate(model, val_loader, cfg, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)
        score = val_metrics["macro_f1"]
        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, cfg, device)
    elapsed = time.perf_counter() - start
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "test": test_metrics,
        "history": history,
        "wall_clock_sec": elapsed,
        "peak_memory_mb": peak_memory_mb,
    }
