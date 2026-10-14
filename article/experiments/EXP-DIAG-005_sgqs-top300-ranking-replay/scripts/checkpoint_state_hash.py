"""Hash checkpoint containers and model states without saving model weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import struct
import sys
from pathlib import Path

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if not torch.is_tensor(value):
            digest.update(repr(value).encode("utf-8"))
            digest.update(b"\0")
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<q", dimension))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "file",
            "bytes",
            "last_epoch",
            "file_sha256",
            "raw_state_sha256",
            "ema_state_sha256",
        ]
    )
    for path in args.checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ema = payload.get("ema")
        ema_state = ema.get("module") if isinstance(ema, dict) else None
        writer.writerow(
            [
                path.name,
                path.stat().st_size,
                payload.get("last_epoch", ""),
                file_sha256(path),
                state_sha256(payload["model"]),
                state_sha256(ema_state) if ema_state is not None else "",
            ]
        )


if __name__ == "__main__":
    main()
