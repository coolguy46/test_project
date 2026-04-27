from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required for profiling.") from exc

from pdsi.models.complexity import estimate_forward_flops
from pdsi.models.inception import ModelConfig, PDSIClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-channels", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--gate", default="pdsi", choices=["none", "pdsi", "complex", "butterworth"])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(num_channels=args.num_channels, num_classes=args.num_classes, gate=args.gate)
    model = PDSIClassifier(cfg).to(device)
    print(json.dumps(estimate_forward_flops(model, args.seq_len, args.num_channels, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
