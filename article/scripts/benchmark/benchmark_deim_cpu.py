#!/usr/bin/env python3
"""Fixed CPU visualization and latency benchmark for the registered DEIM model.

The checkpoint path is resolved from models/which_weight_to_use/path.txt, whose
format is documented by the adjacent README.md.  Both wired and wireless inputs
use the same DEIM checkpoint by design.

Primary comparison metric:
    eager PyTorch model + DEIM PostProcessor forward latency.

Disk decode, checkpoint/model loading, visualization, and output writes are
excluded.  Preprocessing and NumPy/result formatting are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
METRIC_FIELDS = (
    "preprocess_ms",
    "inference_ms",
    "output_materialize_postprocess_ms",
    "runtime_with_output_ms",
    "pipeline_no_io_ms",
)
PROTOCOL_VERSION = (
    "deim_pytorch_cpu_b1_t1_aff2_fp32_mkldnn0_eager_640_v1"
)
REGISTRY_LINE_RE = re.compile(r'^\s*(\S+)\s+"([^"]*)"\s*$')


def default_baseline_root() -> Path:
    """Resolve local benchmark assets without embedding a machine path."""
    fallback = Path(__file__).resolve().parents[2] / "_local" / "baseline"
    return Path(os.environ.get("BASELINE_ROOT", str(fallback))).expanduser()


def parse_args() -> argparse.Namespace:
    root = default_baseline_root()
    parser = argparse.ArgumentParser(
        description="Visualize and benchmark the registered DEIM model on CPU."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "DATA")
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=root / "models" / "which_weight_to_use" / "path.txt",
    )
    parser.add_argument("--model-name", default="DEIM")
    parser.add_argument(
        "--config",
        type=Path,
        default=root
        / "models"
        / "DEIM"
        / "configs"
        / "table_cells"
        / "dfine_hgnetv2_m_table_cell.yml",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "DEIM")
    parser.add_argument("--groups", nargs="+", default=["wired", "wireless"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--affinity",
        default="2",
        help="One logical CPU index; fixed to CPU 2 by the comparison protocol.",
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=640)
    return parser.parse_args()


def parse_affinity(value: str) -> list[int]:
    cpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cpus or any(cpu < 0 for cpu in cpus):
        raise ValueError("--affinity must be one non-negative logical CPU index")
    return cpus


def configure_process(args: argparse.Namespace) -> dict[str, Any]:
    """Pin the process and set thread controls before importing PyTorch/NumPy."""
    value = str(args.threads)
    fixed_env = {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    for key, env_value in fixed_env.items():
        os.environ[key] = env_value

    import psutil

    process = psutil.Process()
    requested = parse_affinity(args.affinity)
    available = process.cpu_affinity()
    unavailable = sorted(set(requested) - set(available))
    if unavailable:
        raise RuntimeError(
            f"Requested CPUs are unavailable: {unavailable}; available={available}"
        )
    process.cpu_affinity(requested)

    priority: dict[str, Any] = {"requested": "ABOVE_NORMAL"}
    try:
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        priority["applied"] = str(process.nice())
    except (psutil.AccessDenied, OSError) as exc:
        priority["applied"] = "unchanged"
        priority["warning"] = str(exc)

    return {
        "environment": fixed_env,
        "affinity_requested": requested,
        "affinity_applied": process.cpu_affinity(),
        "priority": priority,
        "psutil_version": psutil.__version__,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def get_cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor or "unknown"


def get_power_plan() -> str:
    if sys.platform != "win32":
        return "not_applicable"
    try:
        result = subprocess.run(
            ["powercfg", "/getactivescheme"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        raw_value = (result.stdout or result.stderr).strip()
        for encoding in ("utf-8", "gbk", "mbcs"):
            try:
                return raw_value.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return raw_value.decode("utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def read_model_registry(registry_path: Path) -> dict[str, str]:
    if not registry_path.is_file():
        raise FileNotFoundError(f"Model registry does not exist: {registry_path}")
    registry: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        registry_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = REGISTRY_LINE_RE.fullmatch(raw_line)
        if not match:
            raise ValueError(
                f"Invalid model registry line {line_number}; expected: MODEL \"PATH\""
            )
        name, path_value = match.groups()
        if name in registry:
            raise ValueError(f"Duplicate model registry entry: {name}")
        registry[name] = path_value
    return registry


def resolve_checkpoint(registry_path: Path, model_name: str) -> Path:
    registry = read_model_registry(registry_path)
    if model_name not in registry:
        raise KeyError(f"Model {model_name!r} is absent from {registry_path}")
    value = registry[model_name].strip()
    if not value:
        raise ValueError(f"Model {model_name!r} has an empty path in {registry_path}")
    checkpoint = Path(value).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Registered checkpoint does not exist: {checkpoint}")
    return checkpoint


def validate_torch_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Fail early on zero-byte, changing, or truncated PyTorch ZIP checkpoints."""
    first = checkpoint.stat()
    if first.st_size <= 0:
        raise RuntimeError(f"Registered checkpoint is empty: {checkpoint}")
    if not zipfile.is_zipfile(checkpoint):
        raise RuntimeError(
            "Registered DEIM checkpoint is not a complete PyTorch ZIP archive. "
            "The file is likely truncated or still being copied: "
            f"{checkpoint} ({first.st_size} bytes)"
        )
    digest = sha256_file(checkpoint)
    second = checkpoint.stat()
    if first.st_size != second.st_size or first.st_mtime_ns != second.st_mtime_ns:
        raise RuntimeError(f"Checkpoint changed while it was being verified: {checkpoint}")
    return {
        "path": str(checkpoint),
        "size_bytes": second.st_size,
        "last_write_time_ns": second.st_mtime_ns,
        "sha256": digest,
        "zip_central_directory_valid": True,
    }


def find_images(data_dir: Path, groups: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in groups:
        group_dir = data_dir / group
        if not group_dir.is_dir():
            raise FileNotFoundError(f"Image group directory does not exist: {group_dir}")
        paths = sorted(
            (
                path
                for path in group_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda item: str(item.relative_to(group_dir)).casefold(),
        )
        if not paths:
            raise RuntimeError(f"No supported images found in: {group_dir}")
        for path in paths:
            records.append({"group": group, "path": path})
    return records


def load_images(records: list[dict[str, Any]], Image: Any) -> None:
    for record in records:
        start_ns = time.perf_counter_ns()
        with Image.open(record["path"]) as source:
            image = source.convert("RGB").copy()
        end_ns = time.perf_counter_ns()
        width, height = image.size
        record.update(
            {
                "image": image,
                "width": int(width),
                "height": int(height),
                "decode_ms": (end_ns - start_ns) / 1_000_000.0,
                "key": f"{record['group']}/{record['path'].name}",
            }
        )


def load_deim_model(
    repo_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    torch: Any,
) -> tuple[Any, float, dict[str, Any]]:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from engine.core import YAMLConfig

    start_ns = time.perf_counter_ns()
    cfg = YAMLConfig(str(config_path))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint).__name__}")
    if isinstance(checkpoint.get("ema"), dict) and "module" in checkpoint["ema"]:
        state = checkpoint["ema"]["module"]
        state_source = "ema.module"
    elif "model" in checkpoint:
        state = checkpoint["model"]
        state_source = "model"
    else:
        raise KeyError("Checkpoint has neither ema.module nor model state")

    model = cfg.model
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = model.deploy().eval().cpu()
    postprocessor = cfg.postprocessor.deploy().eval().cpu()

    class InferenceWrapper(torch.nn.Module):
        def __init__(self, detector: Any, processor: Any) -> None:
            super().__init__()
            self.detector = detector
            self.processor = processor

        def forward(self, images: Any, original_sizes: Any) -> Any:
            return self.processor(self.detector(images), original_sizes)

    wrapper = InferenceWrapper(model, postprocessor).eval()
    end_ns = time.perf_counter_ns()
    metadata = {
        "checkpoint_keys": sorted(str(key) for key in checkpoint.keys()),
        "checkpoint_last_epoch": checkpoint.get(
            "last_epoch", checkpoint.get("epoch")
        ),
        "state_source": state_source,
        "strict_state_load": True,
        "deploy_conversion": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "postprocessor_num_top_queries": getattr(postprocessor, "num_top_queries", None),
        "postprocessor_num_classes": getattr(postprocessor, "num_classes", None),
    }
    return wrapper, (end_ns - start_ns) / 1_000_000.0, metadata


def materialize_detections(
    output: Any,
    width: int,
    height: int,
    threshold: float,
    np: Any,
) -> tuple[list[dict[str, Any]], tuple[Any, Any, Any]]:
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise RuntimeError("Expected deployed PostProcessor output: labels, boxes, scores")
    labels, boxes, scores = output
    labels_np = labels[0].detach().cpu().numpy()
    boxes_np = boxes[0].detach().cpu().numpy()
    scores_np = scores[0].detach().cpu().numpy()
    keep = scores_np > threshold
    labels_np = labels_np[keep]
    boxes_np = boxes_np[keep]
    scores_np = scores_np[keep]

    detections: list[dict[str, Any]] = []
    for class_value, box, score_value in zip(labels_np, boxes_np, scores_np):
        class_id = int(class_value)
        score = float(score_value)
        x1 = max(0.0, float(box[0]))
        y1 = max(0.0, float(box[1]))
        x2 = min(float(width), float(box[2]))
        y2 = min(float(height), float(box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "class_id": class_id,
                "class_name": "cell" if class_id == 0 else f"class_{class_id}",
                "score": score,
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return detections, (labels_np, boxes_np, scores_np)


def output_signature(arrays: tuple[Any, Any, Any]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def summarize_values(values: list[float], np: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean_ms": float(array.mean()),
        "median_p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "std_ms": float(array.std(ddof=0)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def summarize_rows(rows: list[dict[str, Any]], np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "unique_images": len({row["image_key"] for row in rows}),
        "latency": {},
    }
    for field in METRIC_FIELDS:
        result["latency"][field] = summarize_values(
            [float(row[field]) for row in rows], np
        )
    mean_ms = result["latency"]["inference_ms"]["mean_ms"]
    result["inference_fps_from_mean"] = 1000.0 / mean_ms
    return result


def draw_visualization(
    image: Any,
    detections: list[dict[str, Any]],
    ImageDraw: Any,
    ImageFont: Any,
) -> Any:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    thickness = max(1, int(round(min(width, height) / 420.0)))
    font_size = max(10, min(16, int(round(min(width, height) / 60.0))))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        coordinates = [round(x1), round(y1), round(x2), round(y2)]
        draw.rectangle(coordinates, outline=(0, 220, 0), width=thickness)
        if x2 - x1 < 42 or y2 - y1 < 15:
            continue
        label = f"{detection['score']:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = max(0, min(width - text_width - 4, int(round(x1)) + 1))
        text_y = int(round(y1)) - text_height - 4
        if text_y < 0:
            text_y = min(height - text_height - 2, int(round(y1)) + 2)
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 3, text_y + text_height + 2],
            fill=(0, 90, 0),
        )
        draw.text((text_x + 1, text_y), label, fill=(255, 255, 255), font=font)
    return canvas


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    if args.threads != 1:
        raise ValueError("This comparison protocol is fixed to --threads 1")
    affinity = parse_affinity(args.affinity)
    if affinity != [2]:
        raise ValueError("This comparison protocol is fixed to --affinity 2")
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("Use at least one warm-up and two measured repeats")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    process_config = configure_process(args)

    import numpy as np
    import PIL
    from PIL import Image, ImageDraw, ImageFont
    import psutil
    import torch
    import torchvision
    import torchvision.transforms as transforms

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.mkldnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    data_dir = args.data_dir.resolve()
    registry_path = args.model_registry.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"DEIM config does not exist: {config_path}")
    checkpoint_path = resolve_checkpoint(registry_path, args.model_name)
    checkpoint_metadata = validate_torch_checkpoint(checkpoint_path)
    repo_dir = config_path.parents[2]

    output_dir.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        (output_dir / group).mkdir(parents=True, exist_ok=True)

    records = find_images(data_dir, args.groups)
    load_images(records, Image)
    image_counts = {
        group: sum(record["group"] == group for record in records) for group in args.groups
    }
    print(
        f"Resolved {args.model_name} checkpoint from registry: {checkpoint_path}",
        flush=True,
    )
    print(f"Loaded {len(records)} images: {image_counts}", flush=True)

    wrapper, model_load_ms, model_metadata = load_deim_model(
        repo_dir, config_path, checkpoint_path, torch
    )
    resize_to_tensor = transforms.Compose(
        [
            transforms.Resize(
                (args.input_height, args.input_width),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

    median_record = sorted(
        records, key=lambda item: item["width"] * item["height"]
    )[len(records) // 2]
    with torch.inference_mode():
        for _ in range(args.warmup):
            image_tensor = resize_to_tensor(median_record["image"]).unsqueeze(0)
            original_size = torch.tensor(
                [[median_record["width"], median_record["height"]]],
                dtype=torch.int64,
            )
            warm_output = wrapper(image_tensor, original_size)
            materialize_detections(
                warm_output,
                median_record["width"],
                median_record["height"],
                args.threshold,
                np,
            )
    print(f"Warm-up complete: {args.warmup} runs", flush=True)

    rows: list[dict[str, Any]] = []
    detections_by_key: dict[str, list[dict[str, Any]]] = {}
    signatures_by_key: dict[str, str] = {}
    non_deterministic_keys: set[str] = set()
    sequence_index = 0
    rng = random.Random(args.seed)

    gc.collect()
    gc.disable()
    benchmark_start = datetime.now().astimezone()
    try:
        with torch.inference_mode():
            for repeat_index in range(1, args.repeats + 1):
                order = list(records)
                rng.shuffle(order)
                repeat_inference: list[float] = []

                for record in order:
                    sequence_index += 1
                    t0 = time.perf_counter_ns()
                    image_tensor = resize_to_tensor(record["image"]).unsqueeze(0)
                    original_size = torch.tensor(
                        [[record["width"], record["height"]]], dtype=torch.int64
                    )
                    t1 = time.perf_counter_ns()
                    output = wrapper(image_tensor, original_size)
                    t2 = time.perf_counter_ns()
                    detections, materialized_arrays = materialize_detections(
                        output,
                        record["width"],
                        record["height"],
                        args.threshold,
                        np,
                    )
                    t3 = time.perf_counter_ns()

                    signature = output_signature(materialized_arrays)
                    if record["key"] not in signatures_by_key:
                        signatures_by_key[record["key"]] = signature
                        detections_by_key[record["key"]] = detections
                    elif signatures_by_key[record["key"]] != signature:
                        non_deterministic_keys.add(record["key"])

                    row = {
                        "sequence_index": sequence_index,
                        "repeat": repeat_index,
                        "group": record["group"],
                        "image_key": record["key"],
                        "image_name": record["path"].name,
                        "width": record["width"],
                        "height": record["height"],
                        "raw_query_count": int(output[2].shape[1]),
                        "detection_count_at_threshold": len(detections),
                        "preprocess_ms": (t1 - t0) / 1_000_000.0,
                        "inference_ms": (t2 - t1) / 1_000_000.0,
                        "output_materialize_postprocess_ms": (t3 - t2) / 1_000_000.0,
                        "runtime_with_output_ms": (t3 - t1) / 1_000_000.0,
                        "pipeline_no_io_ms": (t3 - t0) / 1_000_000.0,
                    }
                    rows.append(row)
                    repeat_inference.append(row["inference_ms"])

                print(
                    f"Measured repeat {repeat_index:02d}/{args.repeats}: "
                    f"inference mean={statistics.fmean(repeat_inference):.3f} ms",
                    flush=True,
                )
    finally:
        gc.enable()
    benchmark_end = datetime.now().astimezone()

    if non_deterministic_keys:
        print(
            "WARNING: output bytes changed across repeats for: "
            + ", ".join(sorted(non_deterministic_keys)),
            flush=True,
        )

    with (output_dir / "latency_raw.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped_rows = {
        group: [row for row in rows if row["group"] == group] for group in args.groups
    }
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": "DEIM detector + deployed PostProcessor forward",
        "overall": summarize_rows(rows, np),
        "groups": {
            group: summarize_rows(group_rows, np)
            for group, group_rows in grouped_rows.items()
        },
        "per_image": {
            record["key"]: summarize_rows(
                [row for row in rows if row["image_key"] == record["key"]], np
            )
            for record in records
        },
        "detection_count_at_threshold": {
            record["key"]: len(detections_by_key[record["key"]]) for record in records
        },
        "output_deterministic_across_repeats": not non_deterministic_keys,
        "non_deterministic_images": sorted(non_deterministic_keys),
    }
    write_json(output_dir / "latency_summary.json", summary)

    detection_document = {
        "model": "DEIM / D-FINE-M table-cell detector",
        "checkpoint": str(checkpoint_path),
        "checkpoint_state": model_metadata["state_source"],
        "threshold_rule": f"score > {args.threshold}",
        "images": [],
    }
    for record in records:
        detections = detections_by_key[record["key"]]
        visual = draw_visualization(
            record["image"], detections, ImageDraw, ImageFont
        )
        visual_path = output_dir / record["group"] / f"{record['path'].stem}_vis.png"
        visual.save(visual_path, format="PNG")
        detection_document["images"].append(
            {
                "group": record["group"],
                "source_path": str(record["path"].resolve()),
                "visualization_path": str(visual_path.resolve()),
                "width": record["width"],
                "height": record["height"],
                "decode_ms_not_in_benchmark": record["decode_ms"],
                "detection_count": len(detections),
                "detections": detections,
            }
        )
    write_json(output_dir / "detections.json", detection_document)

    config_document = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "benchmark_started_at": benchmark_start.isoformat(),
        "benchmark_finished_at": benchmark_end.isoformat(),
        "model_name": args.model_name,
        "architecture": "DEIM with HGNetv2-B2 / D-FINE transformer",
        "paths": {
            "data_dir": str(data_dir),
            "model_registry": str(registry_path),
            "checkpoint_from_registry": str(checkpoint_path),
            "config": str(config_path),
            "repo_dir": str(repo_dir),
            "output_dir": str(output_dir),
            "script": str(Path(__file__).resolve()),
        },
        "registry": {
            "format_source": str((registry_path.parent / "README.md").resolve()),
            "model_entry": args.model_name,
            "registry_sha256": sha256_file(registry_path),
        },
        "checkpoint": checkpoint_metadata,
        "config_sha256": sha256_file(config_path),
        "model": model_metadata,
        "data": {
            "groups": list(args.groups),
            "image_counts": image_counts,
            "total_images": len(records),
            "fixed_input_shape_nchw": [1, 3, args.input_height, args.input_width],
            "input_images": [
                {
                    "key": record["key"],
                    "width": record["width"],
                    "height": record["height"],
                    "size_bytes": record["path"].stat().st_size,
                    "sha256": sha256_file(record["path"]),
                }
                for record in records
            ],
        },
        "preprocessing": [
            "Pillow decode and RGB conversion (excluded from benchmark)",
            f"torchvision Resize to {args.input_width}x{args.input_height}; BILINEAR, antialias=true, keep_ratio=false",
            "torchvision ToTensor: CHW float32 in [0,1]",
            "add batch dimension",
        ],
        "postprocessing": {
            "implementation": "repository PostProcessor.deploy()",
            "labels": ["cell"],
            "score_threshold": args.threshold,
            "comparison": "strict greater-than",
            "boxes_clipped_to_original_image_for_serialization": True,
            "extra_nms": False,
        },
        "benchmark": {
            "warmup_runs": args.warmup,
            "measured_repeats_per_image": args.repeats,
            "measured_samples": len(rows),
            "shuffle_seed": args.seed,
            "ordering": "deterministic seeded shuffle of all wired+wireless images per repeat",
            "timer": "time.perf_counter_ns",
            "primary_metric": "model + repository PostProcessor forward",
            "runtime_with_output_metric": "forward + CPU tensor to NumPy + threshold/clip/result formatting",
            "pipeline_metric": "torchvision preprocessing + runtime_with_output",
            "excluded_from_latency": [
                "checkpoint/model/config loading and deploy conversion",
                "image disk read/decode",
                "visualization drawing",
                "output file encoding/writes",
            ],
            "model_load_ms_reported_separately": model_load_ms,
        },
        "runtime": {
            "device": "CPU",
            "batch_size": 1,
            "dtype": "float32",
            "execution": "PyTorch eager inference_mode",
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_available": torch.cuda.is_available(),
        },
        "process": process_config,
        "host": {
            "cpu": get_cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "power_plan_observed": get_power_plan(),
            "operating_system": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "psutil": psutil.__version__,
        },
        "comparability_note": (
            "This run is fixed to the same single logical P-core (CPU 2), one "
            "thread, FP32, batch 1, input 640x640, warm-up count, repeat count, "
            "image order seed, and score threshold as the PaddleX-L benchmark. "
            "PyTorch MKLDNN is explicitly disabled to match the Paddle benchmark's "
            "required no-oneDNN backend."
        ),
    }
    write_json(output_dir / "benchmark_config.json", config_document)

    overall_inference = summary["overall"]["latency"]["inference_ms"]
    overall_pipeline = summary["overall"]["latency"]["pipeline_no_io_ms"]
    log_lines = [
        "DEIM fixed CPU benchmark",
        f"Protocol: {PROTOCOL_VERSION}",
        f"Registered checkpoint: {checkpoint_path}",
        f"Checkpoint state: {model_metadata['state_source']}",
        f"Images: {len(records)} ({image_counts})",
        f"Warm-up: {args.warmup}; repeats/image: {args.repeats}; samples: {len(rows)}",
        f"Threshold: score > {args.threshold}",
        f"CPU affinity: {process_config['affinity_applied']}; threads: {args.threads}",
        "Backend: PyTorch eager FP32; MKLDNN: OFF; batch size: 1",
        (
            "Inference-only overall: "
            f"mean={overall_inference['mean_ms']:.3f} ms, "
            f"P50={overall_inference['median_p50_ms']:.3f} ms, "
            f"P90={overall_inference['p90_ms']:.3f} ms, "
            f"P95={overall_inference['p95_ms']:.3f} ms, "
            f"std={overall_inference['std_ms']:.3f} ms"
        ),
        (
            "Pipeline no-I/O overall: "
            f"mean={overall_pipeline['mean_ms']:.3f} ms, "
            f"P50={overall_pipeline['median_p50_ms']:.3f} ms, "
            f"P90={overall_pipeline['p90_ms']:.3f} ms, "
            f"P95={overall_pipeline['p95_ms']:.3f} ms, "
            f"std={overall_pipeline['std_ms']:.3f} ms"
        ),
    ]
    for group in args.groups:
        group_stats = summary["groups"][group]["latency"]["inference_ms"]
        log_lines.append(
            f"{group} inference-only: mean={group_stats['mean_ms']:.3f} ms, "
            f"P50={group_stats['median_p50_ms']:.3f} ms, "
            f"P95={group_stats['p95_ms']:.3f} ms"
        )
    with (output_dir / "run.log").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(log_lines) + "\n")

    print("\n".join(log_lines), flush=True)
    print(f"Artifacts written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
