from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


DEFAULT_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]


def _import_mne():
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise SystemExit("Install optional dependency first: pip install mne") from exc
    return mne


def _parse_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    current: str | None = None
    pending_start: float | None = None
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("File Name:"):
            current = line.split(":", 1)[1].strip()
            intervals.setdefault(current, [])
            pending_start = None
        elif "Seizure" in line and "Start Time" in line and current:
            match = re.search(r"(\d+(?:\.\d+)?)\s+seconds", line)
            if match:
                pending_start = float(match.group(1))
        elif "Seizure" in line and "End Time" in line and current and pending_start is not None:
            match = re.search(r"(\d+(?:\.\d+)?)\s+seconds", line)
            if match:
                intervals[current].append((pending_start, float(match.group(1))))
                pending_start = None
    return intervals


def _subject_id(path: Path) -> int:
    match = re.search(r"chb(\d+)", path.as_posix())
    if not match:
        raise ValueError(f"Cannot infer CHB subject id from {path}")
    return int(match.group(1))


def _split_name(subject: int) -> str:
    if subject in {20, 21, 22, 23, 24}:
        return "test"
    if subject in {17, 18, 19}:
        return "val"
    return "train"


def _canonical_name(name: str) -> str:
    """Undo MNE duplicate suffixes such as T8-P8-0 for channel matching."""

    match = re.match(r"^(.*)-\d+$", name)
    return match.group(1) if match else name


def _pick_canonical_channels(raw, desired: list[str]) -> bool:
    selected: list[str] = []
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for wanted in desired:
        match = None
        for channel in raw.ch_names:
            if channel in used:
                continue
            if _canonical_name(channel) == wanted:
                match = channel
                break
        if match is None:
            return False
        selected.append(match)
        used.add(match)
        rename_map[match] = wanted
    raw.pick(selected)
    raw.rename_channels(rename_map)
    return True


def _window_label(start_sec: float, end_sec: float, seizures: list[tuple[float, float]], min_overlap: float) -> int:
    for seizure_start, seizure_end in seizures:
        overlap = max(0.0, min(end_sec, seizure_end) - max(start_sec, seizure_start))
        if overlap >= min_overlap:
            return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CHB-MIT EDF files into the PDSI NPZ format.")
    parser.add_argument("--root", type=Path, required=True, help="CHB-MIT root containing chb?? folders.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=256.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--min-seizure-overlap-sec", type=float, default=1.0)
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    args = parser.parse_args()
    mne = _import_mne()

    all_intervals: dict[Path, list[tuple[float, float]]] = {}
    for summary in args.root.glob("chb??/chb??-summary.txt"):
        parsed = _parse_summary(summary)
        for filename, intervals in parsed.items():
            all_intervals[summary.parent / filename] = intervals

    split_x: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_y: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for edf in sorted(args.root.glob("chb??/*.edf")):
        raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
        if not _pick_canonical_channels(raw, args.channels):
            continue
        raw.resample(args.target_hz, verbose=False)
        data = raw.get_data().astype(np.float32, copy=False)
        sfreq = float(raw.info["sfreq"])
        win = int(round(args.window_sec * sfreq))
        stride = int(round(args.stride_sec * sfreq))
        seizures = all_intervals.get(edf, [])
        split = _split_name(_subject_id(edf))
        for start in range(0, max(data.shape[1] - win + 1, 0), stride):
            end = start + win
            start_sec = start / sfreq
            end_sec = end / sfreq
            split_x[split].append(data[:, start:end])
            split_y[split].append(_window_label(start_sec, end_sec, seizures, args.min_seizure_overlap_sec))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split in ["train", "val", "test"]:
        payload[f"x_{split}"] = np.stack(split_x[split], axis=0)
        payload[f"y_{split}"] = np.array(split_y[split], dtype=np.int64)
    payload["channels"] = np.array(args.channels)
    payload["target_hz"] = args.target_hz
    np.savez_compressed(args.out, **payload)
    print(
        f"saved {args.out}: "
        + ", ".join(f"{split}={payload[f'x_{split}'].shape}" for split in ["train", "val", "test"])
    )


if __name__ == "__main__":
    main()
