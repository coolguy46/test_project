from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def _import_mne():
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise SystemExit("Install optional dependency first: pip install mne") from exc
    return mne


def _subject_id(path: Path) -> int:
    # Sleep-cassette files look like SC4xxyy..., where `xx` is the subject id
    # and `yy` is the recording/night identifier.
    match = re.search(r"SC4(\d{2})\d", path.name)
    if not match:
        return abs(hash(path.name)) % 1000
    return int(match.group(1))


def _split_name(subject: int) -> str:
    if subject >= 17:
        return "test"
    if subject >= 14:
        return "val"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Sleep-EDF Expanded into PDSI NPZ format.")
    parser.add_argument("--root", type=Path, required=True, help="Sleep-EDF root containing PSG and Hypnogram EDF files.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--channel", default="EEG Fpz-Cz")
    parser.add_argument("--target-hz", type=float, default=100.0)
    parser.add_argument("--epoch-sec", type=float, default=30.0)
    args = parser.parse_args()
    mne = _import_mne()

    psg_files = sorted(args.root.rglob("*PSG.edf"))
    split_x: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_y: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for psg in psg_files:
        # Hypnogram suffixes differ from PSG suffixes (for example E0 vs EC),
        # so pair on the shared record prefix instead of an exact replacement.
        hyp_matches = sorted(psg.parent.glob(f"{psg.stem[:6]}*-Hypnogram.edf"))
        if not hyp_matches:
            continue
        hyp = hyp_matches[0]
        raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
        if args.channel not in raw.ch_names:
            continue
        raw.pick([args.channel])
        raw.resample(args.target_hz, verbose=False)
        annotations = mne.read_annotations(hyp)
        raw.set_annotations(annotations, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=STAGE_MAP, chunk_duration=args.epoch_sec, verbose=False)
        if len(events) == 0:
            continue
        epochs = mne.Epochs(
            raw,
            events,
            event_id=None,
            tmin=0.0,
            tmax=args.epoch_sec - 1.0 / args.target_hz,
            baseline=None,
            preload=True,
            verbose=False,
        )
        x = epochs.get_data().astype(np.float32, copy=False)
        y = events[:, 2].astype(np.int64, copy=False)
        split = _split_name(_subject_id(psg))
        split_x[split].extend(list(x))
        split_y[split].extend(list(y))

    payload = {}
    for split in ["train", "val", "test"]:
        if not split_x[split]:
            raise SystemExit(f"No Sleep-EDF epochs found for split {split}; check root path and file layout.")
        payload[f"x_{split}"] = np.stack(split_x[split], axis=0)
        payload[f"y_{split}"] = np.array(split_y[split], dtype=np.int64)
    payload["classes"] = np.array(["W", "N1", "N2", "N3", "R"])
    payload["channel"] = args.channel
    payload["target_hz"] = args.target_hz
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(
        f"saved {args.out}: "
        + ", ".join(f"{split}={payload[f'x_{split}'].shape}" for split in ["train", "val", "test"])
    )


if __name__ == "__main__":
    main()
