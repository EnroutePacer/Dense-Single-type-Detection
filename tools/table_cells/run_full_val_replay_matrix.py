"""Run the local SGQS e28--e34 full-validation replay matrix safely.

The matrix is intentionally serial: each checkpoint gets a fresh child
process and that process evaluates ``raw`` followed by ``ema``.  A failed
checkpoint stops the matrix.  Re-running the command resumes from the durable
memmap boundary or skips a checkpoint whose two state markers and artifacts
have already been verified.

The defaults name only ``checkpoint0028.pth`` through
``checkpoint0034.pth``.  ``best_stg1.pth`` and ``last.pth`` are duplicate
containers and are therefore outside the plan by construction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Iterator, Mapping, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = tuple(range(28, 35))
STATES = ("raw", "ema")
LOCAL_ANNOTATION_SHA256 = "f855dd4b9dded934479a54aad70f01b3c678d0ae1fd91e3f8ca1cd6890bd693e"
REPLAY_SCRIPT_SHA256 = "ae2ef375387800fa1b38a995d1923c63fc2793b93a9fef872beaf8b29d8e9416"
SUPERVISION_PROBE_SHA256 = "853f93dba19b2be2a56d0932d496fdf2e69efef764c3f360ad014da103801429"
CHECKPOINT_IDENTITY_SHA256 = "0d80f2670fa87edf6cbf8aff427ca9f02336633948b91861bd11fbfff71aa8cf"
CHECKPOINT_HASH_SCRIPT_SHA256 = "a4012237aa5df1fee9a045fe1d63dd85ac225a8c3ec793c3f0158d424240dae0"
ENGINE_PYTHON_TREE_SHA256 = "f5f83671fb95c8f79fd10ff488102a5fcc37d7128a70959c7ae91a04fa0b2a51"
EXPECTED_IMAGES = 12_841
EXPECTED_ANNOTATIONS = 1_458_060


class MatrixError(RuntimeError):
    """A precondition, child process, or output postcondition failed."""


@dataclass(frozen=True)
class RunSpec:
    epoch: int
    checkpoint: Path
    output: Path
    log: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def build_specs(weights_root: Path, output_root: Path) -> list[RunSpec]:
    logs = output_root / "logs"
    return [
        RunSpec(
            epoch=epoch,
            checkpoint=weights_root / f"checkpoint{epoch:04d}.pth",
            output=output_root / f"checkpoint{epoch:04d}",
            log=logs / f"checkpoint{epoch:04d}.log",
        )
        for epoch in EPOCHS
    ]


def replay_command(args: argparse.Namespace, spec: RunSpec) -> list[str]:
    command = [
        os.fspath(args.python),
        "-u",
        os.fspath(args.replay_script),
        "--config",
        os.fspath(args.config),
        "--checkpoint",
        os.fspath(spec.checkpoint),
        "--annotation",
        os.fspath(args.annotation),
        "--images-root",
        os.fspath(args.images_root),
        "--expected-annotation-sha256",
        args.expected_annotation_sha256,
        "--split-name",
        args.split_name,
        "--output",
        os.fspath(spec.output),
        "--device",
        args.device,
        "--states",
        *STATES,
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--durable-images",
        str(args.durable_images),
        "--progress-every",
        str(args.progress_every),
        "--resume",
    ]
    return command


def command_text(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise MatrixError(f"missing {label}: {path}")


def preflight(args: argparse.Namespace, specs: Sequence[RunSpec]) -> None:
    require_file(args.python, "Python interpreter")
    require_file(args.replay_script, "replay script")
    require_file(args.supervision_probe, "supervision probe module")
    require_file(args.config, "resolved SGQS config")
    require_file(args.annotation, "merged validation annotation")
    require_file(args.checkpoint_identity, "frozen checkpoint identity table")
    require_file(args.checkpoint_hash_script, "checkpoint state hash script")
    if not args.images_root.is_dir():
        raise MatrixError(f"missing merged image view: {args.images_root}")
    for spec in specs:
        require_file(spec.checkpoint, f"epoch-{spec.epoch} checkpoint")
    observed = sha256(args.annotation)
    if observed.lower() != args.expected_annotation_sha256.lower():
        raise MatrixError(
            "merged validation annotation SHA-256 mismatch: "
            f"expected {args.expected_annotation_sha256}, got {observed}"
        )
    for label, path, expected in (
        ("replay script", args.replay_script, args.expected_replay_sha256),
        ("supervision probe", args.supervision_probe, args.expected_supervision_probe_sha256),
        ("checkpoint identity", args.checkpoint_identity, args.expected_checkpoint_identity_sha256),
        ("checkpoint hash script", args.checkpoint_hash_script, args.expected_checkpoint_hash_script_sha256),
    ):
        observed = sha256(path)
        if observed.lower() != expected.lower():
            raise MatrixError(
                f"{label} SHA-256 mismatch: expected {expected}, got {observed}; "
                "do not mix code revisions inside a resumed memmap"
            )
    engine_identity = engine_tree_identity()
    if engine_identity["sha256"] != args.expected_engine_tree_sha256:
        raise MatrixError(
            "engine Python source tree changed: expected "
            f"{args.expected_engine_tree_sha256}, got {engine_identity['sha256']}"
        )


def engine_tree_identity() -> dict:
    files = sorted((ROOT / "engine").rglob("*.py"))
    digest = hashlib.sha256()
    entries = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        file_hash = sha256(path)
        entries[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "files": len(files), "entries": entries}


def expected_checkpoint_contract(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row.get("file", ""): row for row in rows}
    expected_names = {f"checkpoint{epoch:04d}.pth" for epoch in EPOCHS}
    missing = expected_names - set(by_name)
    if missing:
        raise MatrixError(f"checkpoint identity table is missing {sorted(missing)}")
    result = {}
    for epoch in EPOCHS:
        name = f"checkpoint{epoch:04d}.pth"
        row = by_name[name]
        normalized = {
            "file": name,
            "bytes": int(row["bytes"]),
            "last_epoch": int(row["last_epoch"]),
            "file_sha256": row["file_sha256"].lower(),
            "raw_state_sha256": row["raw_state_sha256"].lower(),
            "ema_state_sha256": row["ema_state_sha256"].lower(),
        }
        if normalized["last_epoch"] != epoch:
            raise MatrixError(f"checkpoint identity epoch mismatch for {name}")
        for key in ("file_sha256", "raw_state_sha256", "ema_state_sha256"):
            value = normalized[key]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise MatrixError(f"invalid {key} for {name}")
        result[name] = normalized
    return result


def compare_checkpoint_contracts(
    expected: Mapping[str, dict], observed: Mapping[str, dict]
) -> dict[str, dict]:
    if set(observed) != set(expected):
        raise MatrixError(
            "observed checkpoint set differs from frozen identity table: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    verified = {}
    fields = (
        "bytes",
        "last_epoch",
        "file_sha256",
        "raw_state_sha256",
        "ema_state_sha256",
    )
    for name in sorted(expected):
        expected_row = expected[name]
        observed_row = observed[name]
        for field in fields:
            expected_value = expected_row[field]
            observed_value = observed_row[field]
            if isinstance(expected_value, str):
                observed_value = str(observed_value).lower()
            else:
                observed_value = int(observed_value)
            if observed_value != expected_value:
                raise MatrixError(
                    f"checkpoint identity mismatch for {name}/{field}: "
                    f"expected {expected_value}, got {observed_value}"
                )
        verified[name] = {**expected_row, "verification": "recomputed_equal"}
    return verified


def verify_checkpoint_contract(
    args: argparse.Namespace, specs: Sequence[RunSpec]
) -> dict[str, dict]:
    expected = expected_checkpoint_contract(args.checkpoint_identity)
    command = [
        os.fspath(args.python),
        os.fspath(args.checkpoint_hash_script),
        *(os.fspath(spec.checkpoint) for spec in specs),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise MatrixError(
            "checkpoint state hash subprocess failed with exit code "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )
    try:
        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        observed = {
            row["file"]: {
                "file": row["file"],
                "bytes": int(row["bytes"]),
                "last_epoch": int(row["last_epoch"]),
                "file_sha256": row["file_sha256"].lower(),
                "raw_state_sha256": row["raw_state_sha256"].lower(),
                "ema_state_sha256": row["ema_state_sha256"].lower(),
            }
            for row in rows
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise MatrixError("invalid checkpoint state hash subprocess output") from exc
    return compare_checkpoint_contracts(expected, observed)


def probe_backend(args: argparse.Namespace) -> dict:
    probe = r'''
import json, os, platform, subprocess, sys
import torch
requested = sys.argv[1]
result = {
    "requested_device": requested,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "torch_version": str(torch.__version__),
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": bool(torch.cuda.is_available()),
    "torch_num_threads": int(torch.get_num_threads()),
    "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
}
if requested.startswith("cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    result.update(
        backend="cuda",
        device_index=int(index),
        device_name=str(properties.name),
        compute_capability=list(torch.cuda.get_device_capability(index)),
        total_memory_bytes=int(properties.total_memory),
    )
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[index].strip()
    except Exception:
        driver = None
    result["cuda_driver_version"] = driver
else:
    result.update(
        backend="cpu",
        device_index=None,
        device_name=platform.processor() or platform.machine(),
        compute_capability=None,
        total_memory_bytes=None,
        cuda_driver_version=None,
    )
print(json.dumps(result, sort_keys=True))
'''
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS=str(args.cpu_threads),
        MKL_NUM_THREADS=str(args.cpu_threads),
    )
    completed = subprocess.run(
        [os.fspath(args.python), "-c", probe, args.device],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise MatrixError(
            f"backend probe failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MatrixError("backend probe returned invalid JSON") from exc
    if identity.get("backend") == "cuda" and not identity.get("cuda_driver_version"):
        raise MatrixError("CUDA backend probe could not determine the driver version")
    return identity


def check_run_identity(
    identity: dict, args: argparse.Namespace, spec: RunSpec, *, context: str
) -> None:
    expected_paths = {
        "checkpoint": spec.checkpoint,
        "config": args.config,
    }
    for key, expected in expected_paths.items():
        if normalized_path(identity.get(key, "")) != normalized_path(expected):
            raise MatrixError(f"{context}: {key} identity mismatch")
    source = identity.get("source")
    if not isinstance(source, dict) or source.get("mode") != "direct_coco":
        raise MatrixError(f"{context}: expected direct COCO source identity")
    for key, expected in (("annotation", args.annotation), ("images_root", args.images_root)):
        if normalized_path(source.get(key, "")) != normalized_path(expected):
            raise MatrixError(f"{context}: source {key} identity mismatch")
    if source.get("annotation_sha256", "").lower() != args.expected_annotation_sha256.lower():
        raise MatrixError(f"{context}: annotation SHA-256 identity mismatch")
    if identity.get("states") != list(STATES):
        raise MatrixError(f"{context}: state matrix is not raw+ema")
    if identity.get("max_images") is not None or identity.get("image_ids") is not None:
        raise MatrixError(f"{context}: output is a subset rather than full validation")
    inference = identity.get("inference_identity", {})
    if inference.get("device") != args.device:
        raise MatrixError(f"{context}: device identity mismatch")
    if int(inference.get("batch_size", -1)) != args.batch_size:
        raise MatrixError(f"{context}: inference batch-size identity mismatch")
    runtime = getattr(args, "runtime_backend", None)
    if runtime is not None:
        if inference.get("torch_version") != runtime.get("torch_version"):
            raise MatrixError(f"{context}: torch runtime identity mismatch")
        if inference.get("torch_cuda_version") != runtime.get("torch_cuda_version"):
            raise MatrixError(f"{context}: CUDA runtime identity mismatch")


def verify_artifact(path: Path, descriptor: dict, *, context: str) -> None:
    require_file(path, f"{context} artifact")
    expected_bytes = int(descriptor.get("bytes", -1))
    if path.stat().st_size != expected_bytes:
        raise MatrixError(f"{context}: artifact size mismatch: {path}")
    expected_hash = descriptor.get("sha256", "").lower()
    if len(expected_hash) != 64 or sha256(path) != expected_hash:
        raise MatrixError(f"{context}: artifact SHA-256 mismatch: {path}")


def validate_complete(
    args: argparse.Namespace,
    spec: RunSpec,
    expected_checkpoint_sha256: str | None = None,
) -> dict | None:
    """Return the validated summary, or ``None`` for a resumable output."""
    summary_path = spec.output / "summary.json"
    marker_paths = {
        state: spec.output / f"{args.split_name}.{state}.complete.json" for state in STATES
    }
    if not summary_path.exists() or not all(path.exists() for path in marker_paths.values()):
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"invalid completed summary {summary_path}: {exc}") from exc
    check_run_identity(summary, args, spec, context=str(summary_path))
    checkpoint_hash = sha256(spec.checkpoint)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_hash != expected_checkpoint_sha256.lower()
    ):
        raise MatrixError(
            f"{summary_path}: checkpoint differs from the frozen checkpoint contract"
        )
    if summary.get("checkpoint_sha256", "").lower() != checkpoint_hash:
        raise MatrixError(f"{summary_path}: checkpoint content changed")
    config_hash = sha256(args.config)
    if summary.get("config_sha256", "").lower() != config_hash:
        raise MatrixError(f"{summary_path}: config content changed")
    if int(summary.get("last_epoch", -1)) != spec.epoch:
        raise MatrixError(f"{summary_path}: checkpoint epoch mismatch")
    split = summary.get("splits", {}).get(args.split_name, {})
    if (
        int(split.get("images", -1)) != EXPECTED_IMAGES
        or int(split.get("noncrowd_annotations", -1)) != EXPECTED_ANNOTATIONS
    ):
        raise MatrixError(f"{summary_path}: full-validation population mismatch")
    metadata = spec.output / split.get("metadata", f"{args.split_name}.metadata.npz")
    require_file(metadata, "full-validation metadata")
    if split.get("metadata_sha256", "").lower() != sha256(metadata):
        raise MatrixError(f"{summary_path}: metadata SHA-256 mismatch")

    for state, marker_path in marker_paths.items():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixError(f"invalid completion marker {marker_path}: {exc}") from exc
        identity = marker.get("identity", {})
        marker_run = identity.get("run_identity", {})
        check_run_identity(marker_run, args, spec, context=str(marker_path))
        if (
            marker_run.get("checkpoint_sha256", "").lower() != checkpoint_hash
            or marker_run.get("config_sha256", "").lower() != config_hash
        ):
            raise MatrixError(f"{marker_path}: checkpoint/config SHA-256 identity mismatch")
        if identity.get("state") != state or identity.get("split") != args.split_name:
            raise MatrixError(f"{marker_path}: state/split identity mismatch")
        if (
            int(identity.get("images", -1)) != EXPECTED_IMAGES
            or int(identity.get("annotations", -1)) != EXPECTED_ANNOTATIONS
        ):
            raise MatrixError(f"{marker_path}: full-validation population mismatch")
        result = marker.get("result", {})
        summary_result = summary.get("results", {}).get(f"{args.split_name}/{state}")
        if result != summary_result:
            raise MatrixError(f"{marker_path}: result differs from summary.json")
        if (
            int(result.get("images", -1)) != EXPECTED_IMAGES
            or int(result.get("noncrowd_annotations", -1)) != EXPECTED_ANNOTATIONS
            or int(result.get("predictions", -1)) != EXPECTED_IMAGES * 640
        ):
            raise MatrixError(f"{marker_path}: result population mismatch")
        for budget in (300, 640):
            if len(result.get(f"coco_maxdets{budget}", ())) != 12:
                raise MatrixError(f"{marker_path}: invalid COCO@{budget} metric vector")
            if len(result.get(f"coco_small_per_iou_maxdets{budget}", ())) != 10:
                raise MatrixError(f"{marker_path}: missing ten COCO-small IoU slices")
        artifacts = result.get("artifacts", {})
        for artifact_name in ("predictions", "gt_stages"):
            descriptor = artifacts.get(artifact_name)
            if not isinstance(descriptor, dict) or not descriptor.get("file"):
                raise MatrixError(f"{marker_path}: missing {artifact_name} descriptor")
            verify_artifact(
                spec.output / descriptor["file"],
                descriptor,
                context=f"e{spec.epoch}/{state}/{artifact_name}",
            )
    return summary


def verify_checkpoint_container(
    spec: RunSpec, checkpoint_identity: Mapping[str, object], *, context: str
) -> None:
    expected_bytes = int(checkpoint_identity["bytes"])
    if spec.checkpoint.stat().st_size != expected_bytes:
        raise MatrixError(f"{context}: checkpoint size differs from frozen contract")
    expected_hash = str(checkpoint_identity["file_sha256"]).lower()
    if sha256(spec.checkpoint) != expected_hash:
        raise MatrixError(f"{context}: checkpoint SHA-256 differs from frozen contract")


def newest_output_mtime(directory: Path) -> float | None:
    if not directory.exists():
        return None
    newest = directory.stat().st_mtime
    for path in directory.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except FileNotFoundError:
            # A live writer may atomically replace a progress file while the
            # directory is scanned.  The parent directory mtime still makes
            # this output recent, so retrying the disappearing path is not
            # necessary.
            continue
    return newest


def recent_incomplete(args: argparse.Namespace, spec: RunSpec) -> tuple[bool, float | None]:
    newest = newest_output_mtime(spec.output)
    if newest is None:
        return False, None
    age = max(0.0, time.time() - newest)
    return age < args.active_grace_seconds, age


def validate_partial_progress(args: argparse.Namespace, spec: RunSpec) -> None:
    """Fail closed on impossible or foreign durable progress markers."""
    for state in STATES:
        partial = spec.output / f".{args.split_name}.{state}.partial"
        if not partial.exists():
            continue
        progress_path = partial / "progress.json"
        require_file(progress_path, f"e{spec.epoch}/{state} progress marker")
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixError(f"invalid progress marker {progress_path}: {exc}") from exc
        identity = progress.get("identity", {})
        check_run_identity(
            identity.get("run_identity", {}), args, spec, context=str(progress_path)
        )
        if identity.get("state") != state or identity.get("split") != args.split_name:
            raise MatrixError(f"{progress_path}: state/split identity mismatch")
        if (
            int(identity.get("images", -1)) != EXPECTED_IMAGES
            or int(identity.get("annotations", -1)) != EXPECTED_ANNOTATIONS
        ):
            raise MatrixError(f"{progress_path}: full-validation population mismatch")
        completed = int(progress.get("completed_images", -1))
        if not 0 <= completed <= EXPECTED_IMAGES:
            raise MatrixError(
                f"{progress_path}: completed_images={completed} is outside "
                f"[0,{EXPECTED_IMAGES}]"
            )


def memory_status_bytes() -> dict[str, int | None]:
    """Return physical and commit headroom using only the stdlib.

    On Windows, ``avail_page_file`` from ``GlobalMemoryStatusEx`` is the
    remaining system commit headroom (RAM plus pagefile backing), not merely
    the number of unused bytes inside ``pagefile.sys``.  That is the relevant
    fail-closed signal for errors such as Win32 ``0x800705AF``.
    """
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "available_physical": int(status.avail_phys),
                "available_commit": int(status.avail_page_file),
                "commit_limit": int(status.total_page_file),
            }
        return {
            "available_physical": None,
            "available_commit": None,
            "commit_limit": None,
        }
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        available_physical = None
    else:
        available_physical = int(pages * page_size)
    return {
        "available_physical": available_physical,
        "available_commit": None,
        "commit_limit": None,
    }


def available_memory_bytes() -> int | None:
    """Backward-compatible physical-memory accessor."""
    return memory_status_bytes()["available_physical"]


def resource_gate(args: argparse.Namespace) -> dict[str, float | None]:
    gib = 1024**3
    memory = memory_status_bytes()
    available = memory["available_physical"]
    if available is None:
        raise MatrixError("cannot determine available physical memory")
    available_commit = memory["available_commit"]
    if os.name == "nt" and available_commit is None:
        raise MatrixError("cannot determine Windows commit/pagefile headroom")
    free_disk = shutil.disk_usage(args.output_root).free
    observed = {
        "available_memory_gib": available / gib,
        "available_commit_gib": (
            available_commit / gib if available_commit is not None else None
        ),
        "commit_limit_gib": (
            memory["commit_limit"] / gib if memory["commit_limit"] is not None else None
        ),
        "free_disk_gib": free_disk / gib,
    }
    if observed["available_memory_gib"] < args.minimum_available_gib:
        raise MatrixError(
            f"only {observed['available_memory_gib']:.2f} GiB RAM is available; "
            f"require {args.minimum_available_gib:.2f} GiB before starting a child process"
        )
    if (
        observed["available_commit_gib"] is not None
        and observed["available_commit_gib"] < args.minimum_available_commit_gib
    ):
        raise MatrixError(
            f"only {observed['available_commit_gib']:.2f} GiB Windows commit/pagefile "
            f"headroom is available; require {args.minimum_available_commit_gib:.2f} GiB "
            "before starting a child process"
        )
    if observed["free_disk_gib"] < args.minimum_free_disk_gib:
        raise MatrixError(
            f"only {observed['free_disk_gib']:.2f} GiB disk is free; "
            f"require {args.minimum_free_disk_gib:.2f} GiB"
        )
    return observed


@contextmanager
def exclusive_matrix_lock(path: Path) -> Iterator[TextIO]:
    """Hold a non-blocking process lock; OS release makes crash recovery safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise MatrixError(f"another matrix orchestrator holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()},
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        yield handle
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip("\n") + "\n")


def run_child(command: Sequence[str], log_path: Path, env: dict[str, str]) -> int:
    header = f"\n[{utc_now()}] START {command_text(command)}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(header)
        log.flush()
        print(header.rstrip(), flush=True)
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
        footer = f"[{utc_now()}] EXIT {return_code}\n"
        print(footer.rstrip(), flush=True)
        log.write(footer)
        log.flush()
        return return_code


def base_audit(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    checkpoint_contract: Mapping[str, dict],
    runtime_backend: Mapping[str, object],
    engine_identity: Mapping[str, object],
) -> dict:
    return {
        "schema": "sgqs_full_val_matrix_orchestration_v1",
        "epochs": list(EPOCHS),
        "states": list(STATES),
        "serial_jobs": 1,
        "duplicate_containers_excluded": ["best_stg1.pth", "last.pth"],
        "annotation": os.fspath(args.annotation),
        "annotation_sha256": args.expected_annotation_sha256,
        "replay_script_sha256": args.expected_replay_sha256,
        "supervision_probe_sha256": args.expected_supervision_probe_sha256,
        "images_root": os.fspath(args.images_root),
        "config": os.fspath(args.config),
        "config_sha256": sha256(args.config),
        "checkpoint_identity_file": os.fspath(args.checkpoint_identity),
        "checkpoint_identity_sha256": args.expected_checkpoint_identity_sha256,
        "checkpoint_contract": dict(checkpoint_contract),
        "runtime_backend": dict(runtime_backend),
        "engine_python_tree": dict(engine_identity),
        "device": args.device,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "durable_images": args.durable_images,
        "cpu_threads": args.cpu_threads,
        "minimum_available_gib": args.minimum_available_gib,
        "minimum_available_commit_gib": args.minimum_available_commit_gib,
        "minimum_free_disk_gib": args.minimum_free_disk_gib,
        "runs": [
            {
                **asdict(spec),
                "checkpoint": os.fspath(spec.checkpoint),
                "output": os.fspath(spec.output),
                "log": os.fspath(spec.log),
            }
            for spec in specs
        ],
    }


def dry_run(args: argparse.Namespace, specs: Sequence[RunSpec]) -> None:
    for spec in specs:
        summary = validate_complete(args, spec)
        if summary is not None:
            status = "skip-complete"
        else:
            validate_partial_progress(args, spec)
            recent, age = recent_incomplete(args, spec)
            if recent:
                status = f"refuse-recent-incomplete(age={age:.1f}s)"
            elif spec.output.exists():
                status = "resume"
            else:
                status = "start"
        print(f"e{spec.epoch}: {status}")
        print(f"  {command_text(replay_command(args, spec))}")


def orchestrate(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    checkpoint_contract: Mapping[str, dict],
    runtime_backend: Mapping[str, object],
    engine_identity: Mapping[str, object],
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "matrix_status.json"
    complete_path = args.output_root / "matrix.complete.json"
    audit = base_audit(args, specs, checkpoint_contract, runtime_backend, engine_identity)
    audit.update(status="running", started_at=utc_now(), checkpoints={})
    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS=str(args.cpu_threads),
        MKL_NUM_THREADS=str(args.cpu_threads),
    )

    with exclusive_matrix_lock(args.output_root / ".matrix.lock"):
        if status_path.is_file():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            for key in (
                "config_sha256",
                "checkpoint_identity_sha256",
                "checkpoint_contract",
                "runtime_backend",
                "engine_python_tree",
                "minimum_available_gib",
                "minimum_available_commit_gib",
                "minimum_free_disk_gib",
            ):
                if existing.get(key) != audit.get(key):
                    raise MatrixError(f"existing matrix status has a different {key}")
        atomic_json(status_path, audit)
        for spec in specs:
            state = {"epoch": spec.epoch, "updated_at": utc_now()}
            checkpoint_identity = checkpoint_contract[spec.checkpoint.name]
            state.update(
                checkpoint_sha256=checkpoint_identity["file_sha256"],
                raw_state_sha256=checkpoint_identity["raw_state_sha256"],
                ema_state_sha256=checkpoint_identity["ema_state_sha256"],
            )
            audit["checkpoints"][str(spec.epoch)] = state
            summary = validate_complete(
                args, spec, checkpoint_identity["file_sha256"]
            )
            if summary is not None:
                state.update(status="skipped_verified_complete")
                append_log(spec.log, f"[{utc_now()}] SKIP verified complete e{spec.epoch}")
                atomic_json(status_path, audit)
                continue

            validate_partial_progress(args, spec)
            verify_checkpoint_container(
                spec, checkpoint_identity, context=f"e{spec.epoch} pre-run"
            )

            recent, age = recent_incomplete(args, spec)
            if recent and not args.resume_recent:
                state.update(status="refused_recent_incomplete", newest_age_seconds=age)
                audit.update(status="failed", failed_epoch=spec.epoch, finished_at=utc_now())
                atomic_json(status_path, audit)
                raise MatrixError(
                    f"e{spec.epoch} output changed {age:.1f}s ago and may have a live writer; "
                    "wait for it to finish, or use --resume-recent only after confirming no process "
                    "is writing that directory"
                )

            command = replay_command(args, spec)
            resources = resource_gate(args)
            state.update(
                status="running",
                command=command_text(command),
                started_at=utc_now(),
                resource_gate=resources,
            )
            atomic_json(status_path, audit)
            return_code = run_child(command, spec.log, env)
            if return_code:
                state.update(status="failed", return_code=return_code, finished_at=utc_now())
                audit.update(status="failed", failed_epoch=spec.epoch, finished_at=utc_now())
                atomic_json(status_path, audit)
                raise MatrixError(f"e{spec.epoch} replay failed with exit code {return_code}")
            summary = validate_complete(
                args, spec, checkpoint_identity["file_sha256"]
            )
            if summary is None:
                state.update(status="failed_postcondition", finished_at=utc_now())
                audit.update(status="failed", failed_epoch=spec.epoch, finished_at=utc_now())
                atomic_json(status_path, audit)
                raise MatrixError(f"e{spec.epoch} exited successfully without complete raw/ema output")
            state.update(
                status="verified_complete",
                finished_at=utc_now(),
                checkpoint_sha256=summary["checkpoint_sha256"],
            )
            atomic_json(status_path, audit)

        audit.update(status="complete", finished_at=utc_now())
        atomic_json(status_path, audit)
        atomic_json(complete_path, audit)
        print(complete_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--replay-script", type=Path, default=ROOT / "tools/table_cells/full_val_replay.py"
    )
    parser.add_argument(
        "--supervision-probe",
        type=Path,
        default=ROOT / "tools/table_cells/supervision_probe.py",
    )
    parser.add_argument(
        "--checkpoint-identity",
        type=Path,
        default=(
            ROOT
            / "article/experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/data/checkpoint_identity.csv"
        ),
    )
    parser.add_argument(
        "--checkpoint-hash-script",
        type=Path,
        default=(
            ROOT
            / "article/experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/scripts/checkpoint_state_hash.py"
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "_local/server_sgqs_probe_20261014/resolved.yml"
    )
    parser.add_argument(
        "--weights-root", type=Path, default=ROOT / "_local/server_sgqs_probe_20261014"
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=ROOT / "_local/fullval_dataset_view/annotations/instance_val.json",
    )
    parser.add_argument(
        "--images-root", type=Path, default=ROOT / "_local/fullval_dataset_view/images"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "_local/fullval_cpu")
    parser.add_argument(
        "--expected-annotation-sha256", default=LOCAL_ANNOTATION_SHA256
    )
    parser.add_argument("--expected-replay-sha256", default=REPLAY_SCRIPT_SHA256)
    parser.add_argument(
        "--expected-supervision-probe-sha256", default=SUPERVISION_PROBE_SHA256
    )
    parser.add_argument(
        "--expected-checkpoint-identity-sha256", default=CHECKPOINT_IDENTITY_SHA256
    )
    parser.add_argument(
        "--expected-checkpoint-hash-script-sha256", default=CHECKPOINT_HASH_SCRIPT_SHA256
    )
    parser.add_argument("--expected-engine-tree-sha256", default=ENGINE_PYTHON_TREE_SHA256)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--durable-images", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=20)
    parser.add_argument("--minimum-available-gib", type=float, default=12.0)
    parser.add_argument(
        "--minimum-available-commit-gib",
        type=float,
        default=6.0,
        help=(
            "minimum Windows system commit headroom (RAM + pagefile backing) "
            "required before each child process; 6 GiB covers the observed ~4 GiB "
            "single-replay footprint with additional margin"
        ),
    )
    parser.add_argument("--minimum-free-disk-gib", type=float, default=20.0)
    parser.add_argument("--active-grace-seconds", type=float, default=900.0)
    parser.add_argument(
        "--resume-recent",
        action="store_true",
        help="resume a recently modified incomplete output only after confirming it has no live writer",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("batch_size", "eval_batch_size", "durable_images", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.active_grace_seconds < 0:
        parser.error("--active-grace-seconds cannot be negative")
    if (
        args.minimum_available_gib <= 0
        or args.minimum_available_commit_gib <= 0
        or args.minimum_free_disk_gib <= 0
    ):
        parser.error("resource thresholds must be positive")
    for name in (
        "expected_annotation_sha256",
        "expected_replay_sha256",
        "expected_supervision_probe_sha256",
        "expected_checkpoint_identity_sha256",
        "expected_checkpoint_hash_script_sha256",
        "expected_engine_tree_sha256",
    ):
        value = getattr(args, name)
        if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
            parser.error(f"--{name.replace('_', '-')} must contain 64 hex digits")
    for name in (
        "python",
        "replay_script",
        "supervision_probe",
        "checkpoint_identity",
        "checkpoint_hash_script",
        "config",
        "weights_root",
        "annotation",
        "images_root",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = build_specs(args.weights_root, args.output_root)
    try:
        preflight(args, specs)
        checkpoint_contract = verify_checkpoint_contract(args, specs)
        runtime_backend = probe_backend(args)
        engine_identity = engine_tree_identity()
        args.runtime_backend = runtime_backend
        if args.dry_run:
            dry_run(args, specs)
        else:
            orchestrate(
                args,
                specs,
                checkpoint_contract,
                runtime_backend,
                engine_identity,
            )
    except MatrixError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
