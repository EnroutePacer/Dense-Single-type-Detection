#!/usr/bin/env python3
"""Reproducible CPU benchmark and visualization for a PaddleX RT-DETR export.

The benchmark deliberately uses ``paddle.inference`` directly.  This avoids
unrelated PaddleX top-level dependencies while preserving the exact exported
model, preprocessing, inputs, outputs, and score filtering used by PaddleX.

Primary comparison metric:
    predictor.run() latency, after model load and warm-up.

Also reported:
    preprocessing, input copy, output copy + postprocessing, and their
    no-disk-I/O pipeline total.  Image decode, model loading, drawing, and
    writing output files are never included in those latency measurements.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
METRIC_FIELDS = (
    "preprocess_ms",
    "input_copy_ms",
    "inference_ms",
    "output_copy_ms",
    "postprocess_ms",
    "runtime_with_tensor_io_ms",
    "pipeline_no_io_ms",
)
PROTOCOL_VERSION = (
    "paddlex_cpu_b1_t1_aff2_paddle_fp32_onednn0_newir1_opt3_640_v1"
)
REGISTRY_LINE_RE = re.compile(r'^\s*(\S+)\s+"([^"]*)"\s*$')


def default_baseline_root() -> Path:
    fallback = Path(__file__).resolve().parents[3] / "_local" / "baseline"
    return Path(os.environ.get("BASELINE_ROOT", str(fallback))).expanduser()


def parse_args() -> argparse.Namespace:
    root = default_baseline_root()
    parser = argparse.ArgumentParser(
        description="Visualize and benchmark PaddleX-L on wired and wireless images."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "DATA")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Explicit exported-model directory. By default it is resolved from the registry.",
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=root / "models" / "which_weight_to_use" / "path.txt",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Registry key. Defaults to PaddleX-<group>; one group per benchmark run.",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "paddlex_L")
    parser.add_argument("--groups", nargs="+", default=["wired", "wireless"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--affinity",
        default="2",
        help="One logical CPU index. Default: 2 (a fixed P-core logical CPU on this host).",
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=640)
    return parser.parse_args()


def parse_affinity(value: str) -> list[int]:
    cpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cpus:
        raise ValueError("--affinity must contain at least one logical CPU index")
    if any(cpu < 0 for cpu in cpus):
        raise ValueError("--affinity CPU indices must be non-negative")
    return cpus


def configure_process(args: argparse.Namespace) -> dict[str, Any]:
    """Set thread controls before importing NumPy, OpenCV, or Paddle."""
    thread_value = str(args.threads)
    fixed_env = {
        "OMP_NUM_THREADS": thread_value,
        "MKL_NUM_THREADS": thread_value,
        "OPENBLAS_NUM_THREADS": thread_value,
        "NUMEXPR_NUM_THREADS": thread_value,
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "FLAGS_use_mkldnn": "0",
    }
    for key, value in fixed_env.items():
        os.environ[key] = value

    import psutil

    process = psutil.Process()
    requested_affinity = parse_affinity(args.affinity)
    available_affinity = process.cpu_affinity()
    unavailable = sorted(set(requested_affinity) - set(available_affinity))
    if unavailable:
        raise RuntimeError(
            f"Requested CPUs are unavailable: {unavailable}; available={available_affinity}"
        )
    process.cpu_affinity(requested_affinity)

    priority_status: dict[str, Any] = {"requested": "ABOVE_NORMAL"}
    try:
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        priority_status["applied"] = str(process.nice())
    except (psutil.AccessDenied, OSError) as exc:
        priority_status["applied"] = "unchanged"
        priority_status["warning"] = str(exc)

    return {
        "environment": fixed_env,
        "affinity_requested": requested_affinity,
        "affinity_applied": process.cpu_affinity(),
        "priority": priority_status,
        "psutil_version": psutil.__version__,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_model_registry(registry_path: Path) -> dict[str, str]:
    """Parse the registry format documented by which_weight_to_use/README.md."""
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


def resolve_model_dir(args: argparse.Namespace) -> tuple[Path, str, Path]:
    registry_path = args.model_registry.resolve()
    if args.model_dir is not None:
        model_name = args.model_name or "explicit-model-dir"
        return args.model_dir.resolve(), model_name, registry_path
    if len(args.groups) != 1:
        raise ValueError(
            "Registry-driven PaddleX runs require exactly one group because wired and "
            "wireless use different weights. Run each group separately, then assemble them."
        )
    model_name = args.model_name or f"PaddleX-{args.groups[0]}"
    registry = read_model_registry(registry_path)
    if model_name not in registry:
        raise KeyError(f"Model {model_name!r} is absent from {registry_path}")
    value = registry[model_name].strip()
    if not value:
        raise ValueError(f"Model {model_name!r} has an empty path in {registry_path}")
    return Path(value).expanduser().resolve(), model_name, registry_path


def read_export_model_name(inference_yml: Path) -> str:
    """Read Global.model_name without importing a YAML dependency."""
    for raw_line in inference_yml.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*model_name:\s*(.+?)\s*$", raw_line)
        if match:
            return match.group(1).strip(" '\"")
    raise ValueError(f"Global.model_name is missing from {inference_yml}")


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
        value = ""
        for encoding in ("utf-8", "gbk", "mbcs"):
            try:
                value = raw_value.decode(encoding)
                break
            except (LookupError, UnicodeDecodeError):
                continue
        if not value:
            value = raw_value.decode("utf-8", errors="replace")
        return value or f"powercfg_exit_{result.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


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


def load_images(records: list[dict[str, Any]], cv2: Any) -> None:
    for record in records:
        start_ns = time.perf_counter_ns()
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        end_ns = time.perf_counter_ns()
        if image is None:
            raise RuntimeError(f"OpenCV could not decode: {record['path']}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"Expected a 3-channel image: {record['path']}")
        height, width = image.shape[:2]
        record.update(
            {
                "image": image,
                "height": int(height),
                "width": int(width),
                "decode_ms": (end_ns - start_ns) / 1_000_000.0,
                "key": f"{record['group']}/{record['path'].name}",
            }
        )


def preprocess(
    bgr_image: Any,
    cv2: Any,
    np: Any,
    input_width: int,
    input_height: int,
) -> tuple[Any, Any, Any]:
    """Exact PaddleX export preprocessing from inference.yml."""
    original_height, original_width = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb_image,
        (input_width, input_height),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = resized.astype(np.float32)
    normalized *= np.float32(1.0 / 255.0)
    image_tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])
    image_shape = np.asarray([[input_height, input_width]], dtype=np.float32)
    scale_factor = np.asarray(
        [[input_height / original_height, input_width / original_width]],
        dtype=np.float32,
    )
    return image_tensor, image_shape, scale_factor


def create_predictor(
    model_dir: Path,
    threads: int,
    paddle_infer: Any,
) -> tuple[Any, float, dict[str, Any]]:
    model_path = model_dir / "inference.pdmodel"
    params_path = model_dir / "inference.pdiparams"
    if not model_path.is_file() or not params_path.is_file():
        raise FileNotFoundError(
            f"Missing inference.pdmodel or inference.pdiparams under {model_dir}"
        )

    start_ns = time.perf_counter_ns()
    config = paddle_infer.Config(str(model_path), str(params_path))
    config.disable_gpu()
    if hasattr(config, "disable_mkldnn"):
        config.disable_mkldnn()
        onednn_method = "disable_mkldnn"
    elif hasattr(config, "disable_onednn"):
        config.disable_onednn()
        onednn_method = "disable_onednn"
    else:
        raise RuntimeError("This Paddle build cannot explicitly disable oneDNN/MKLDNN")
    config.set_cpu_math_library_num_threads(threads)
    if hasattr(config, "enable_new_ir"):
        config.enable_new_ir(True)
        new_ir_enabled = True
    else:
        config.switch_ir_optim(True)
        new_ir_enabled = False
    if hasattr(config, "enable_new_executor"):
        config.enable_new_executor()
        new_executor_enabled = True
    else:
        new_executor_enabled = False
    if hasattr(config, "set_optimization_level"):
        config.set_optimization_level(3)
        optimization_level = 3
    else:
        optimization_level = None
    config.enable_memory_optim()
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)
    end_ns = time.perf_counter_ns()

    runtime_config = {
        "device": "CPU",
        "batch_size": 1,
        "cpu_math_threads": threads,
        "backend": "paddle_fp32_no_mkldnn",
        "onednn_enabled": False,
        "onednn_disable_method": onednn_method,
        "new_ir_enabled": new_ir_enabled,
        "new_executor_enabled": new_executor_enabled,
        "optimization_level": optimization_level,
        "memory_optimization_enabled": True,
        "glog_enabled": False,
        "input_names": list(predictor.get_input_names()),
        "output_names": list(predictor.get_output_names()),
    }
    return predictor, (end_ns - start_ns) / 1_000_000.0, runtime_config


def copy_inputs(input_handles: dict[str, Any], tensors: tuple[Any, Any, Any]) -> None:
    image_tensor, image_shape, scale_factor = tensors
    values = {
        "image": image_tensor,
        "im_shape": image_shape,
        "scale_factor": scale_factor,
    }
    missing = sorted(set(input_handles) - set(values))
    if missing:
        raise RuntimeError(f"Unexpected model inputs: {missing}")
    for name, handle in input_handles.items():
        # PaddleX's static runner reshapes immediately before each tensor copy.
        handle.reshape(values[name].shape)
        handle.copy_from_cpu(values[name])


def identify_outputs(predictor: Any, np: Any) -> tuple[str, str, dict[str, Any]]:
    arrays: dict[str, Any] = {}
    for name in predictor.get_output_names():
        arrays[name] = predictor.get_output_handle(name).copy_to_cpu()

    box_names = [
        name
        for name, value in arrays.items()
        if value.ndim == 2 and value.shape[1] in (6, 10) and np.issubdtype(value.dtype, np.floating)
    ]
    count_names = [
        name
        for name, value in arrays.items()
        if value.ndim == 1 and np.issubdtype(value.dtype, np.integer)
    ]
    if len(box_names) != 1 or len(count_names) != 1:
        shapes = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        }
        raise RuntimeError(f"Cannot identify boxes/bbox_num outputs: {shapes}")
    metadata = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    return box_names[0], count_names[0], metadata


def postprocess_boxes(
    raw_boxes: Any,
    raw_count: Any,
    width: int,
    height: int,
    threshold: float,
) -> list[dict[str, Any]]:
    count = int(raw_count.reshape(-1)[0])
    boxes = raw_boxes[:count]
    detections: list[dict[str, Any]] = []
    for box in boxes:
        class_id = int(box[0])
        score = float(box[1])
        # PaddleX DetPostProcess uses strict greater-than and class_id > -1.
        if class_id < 0 or not score > threshold:
            continue
        x1 = max(0.0, float(box[2]))
        y1 = max(0.0, float(box[3]))
        x2 = min(float(width), float(box[4]))
        y2 = min(float(height), float(box[5]))
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
    return detections


def output_signature(raw_boxes: Any, raw_count: Any) -> str:
    digest = hashlib.sha256()
    digest.update(raw_boxes.tobytes(order="C"))
    digest.update(raw_count.tobytes(order="C"))
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
    inference_mean = result["latency"]["inference_ms"]["mean_ms"]
    result["inference_fps_from_mean"] = 1000.0 / inference_mean
    return result


def draw_visualization(image: Any, detections: list[dict[str, Any]], cv2: Any) -> Any:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, int(round(min(width, height) / 420.0)))
    font_scale = max(0.30, min(0.55, min(width, height) / 1050.0))
    font_thickness = 1

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, p1, p2, (0, 220, 0), thickness, cv2.LINE_AA)

        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 42 or box_height < 15:
            continue
        label = f"{detection['score']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        text_x = max(0, min(width - text_width - 2, p1[0] + 1))
        text_bottom = p1[1] - 2
        if text_bottom - text_height - baseline < 0:
            text_bottom = min(height - baseline - 1, p1[1] + text_height + baseline + 2)
        top = max(0, text_bottom - text_height - baseline)
        right = min(width - 1, text_x + text_width + 2)
        bottom = min(height - 1, text_bottom + baseline)
        cv2.rectangle(canvas, (text_x, top), (right, bottom), (0, 90, 0), -1)
        cv2.putText(
            canvas,
            label,
            (text_x + 1, text_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
    return canvas


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    if args.threads != 1:
        raise ValueError(
            "This comparison protocol is fixed to one inference thread; use --threads 1."
        )
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("Use at least one warm-up and two measured repeats")
    if len(parse_affinity(args.affinity)) != 1:
        raise ValueError("This protocol requires exactly one fixed logical CPU")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    process_config = configure_process(args)

    # These imports intentionally happen only after all thread controls and CPU
    # affinity are fixed.
    import cv2
    import numpy as np
    import paddle
    import paddle.inference as paddle_infer
    import psutil

    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    data_dir = args.data_dir.resolve()
    model_dir, registry_key, registry_path = resolve_model_dir(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        (output_dir / group).mkdir(parents=True, exist_ok=True)

    records = find_images(data_dir, args.groups)
    load_images(records, cv2)
    image_counts = {
        group: sum(record["group"] == group for record in records) for group in args.groups
    }
    print(f"Loaded {len(records)} images: {image_counts}", flush=True)

    required_model_files = [
        model_dir / "inference.pdmodel",
        model_dir / "inference.pdiparams",
        model_dir / "inference.pdiparams.info",
        model_dir / "inference.yml",
    ]
    missing_files = [str(path) for path in required_model_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    export_model_name = read_export_model_name(model_dir / "inference.yml")
    expected_suffix = f"_{args.groups[0]}_table_cell_det" if len(args.groups) == 1 else None
    if expected_suffix and not export_model_name.endswith(expected_suffix):
        raise RuntimeError(
            f"Registry/model mismatch: group={args.groups[0]!r}, "
            f"registry_key={registry_key!r}, export_model_name={export_model_name!r}"
        )
    model_files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in required_model_files
    }

    predictor, model_load_ms, runtime_config = create_predictor(
        model_dir, args.threads, paddle_infer
    )
    required_inputs = {"image", "im_shape", "scale_factor"}
    actual_inputs = set(predictor.get_input_names())
    if actual_inputs != required_inputs:
        raise RuntimeError(
            f"Expected inputs {sorted(required_inputs)}, got {sorted(actual_inputs)}"
        )
    input_handles = {
        name: predictor.get_input_handle(name) for name in predictor.get_input_names()
    }

    median_record = sorted(
        records, key=lambda item: item["width"] * item["height"]
    )[len(records) // 2]
    box_output_name: str | None = None
    count_output_name: str | None = None
    output_metadata: dict[str, Any] | None = None
    for warmup_index in range(args.warmup):
        tensors = preprocess(
            median_record["image"],
            cv2,
            np,
            args.input_width,
            args.input_height,
        )
        copy_inputs(input_handles, tensors)
        # Paddle Inference's Python API returns None on successful synchronous CPU runs.
        predictor.run()
        if warmup_index == 0:
            box_output_name, count_output_name, output_metadata = identify_outputs(
                predictor, np
            )
        else:
            predictor.get_output_handle(box_output_name).copy_to_cpu()
            predictor.get_output_handle(count_output_name).copy_to_cpu()
    assert box_output_name is not None and count_output_name is not None
    assert output_metadata is not None
    box_handle = predictor.get_output_handle(box_output_name)
    count_handle = predictor.get_output_handle(count_output_name)
    runtime_config["identified_boxes_output"] = box_output_name
    runtime_config["identified_bbox_num_output"] = count_output_name
    runtime_config["warmup_output_metadata"] = output_metadata
    print(
        f"Warm-up complete: {args.warmup} runs; boxes={box_output_name}, "
        f"bbox_num={count_output_name}",
        flush=True,
    )

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
        for repeat_index in range(1, args.repeats + 1):
            order = list(records)
            rng.shuffle(order)
            repeat_inference: list[float] = []

            for record in order:
                sequence_index += 1
                t0 = time.perf_counter_ns()
                tensors = preprocess(
                    record["image"],
                    cv2,
                    np,
                    args.input_width,
                    args.input_height,
                )
                t1 = time.perf_counter_ns()
                copy_inputs(input_handles, tensors)
                t2 = time.perf_counter_ns()
                predictor.run()
                t3 = time.perf_counter_ns()
                raw_boxes = box_handle.copy_to_cpu()
                raw_count = count_handle.copy_to_cpu()
                t4 = time.perf_counter_ns()
                detections = postprocess_boxes(
                    raw_boxes,
                    raw_count,
                    record["width"],
                    record["height"],
                    args.threshold,
                )
                t5 = time.perf_counter_ns()

                signature = output_signature(raw_boxes, raw_count)
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
                    "raw_bbox_count": int(raw_count.reshape(-1)[0]),
                    "detection_count_at_threshold": len(detections),
                    "preprocess_ms": (t1 - t0) / 1_000_000.0,
                    "input_copy_ms": (t2 - t1) / 1_000_000.0,
                    "inference_ms": (t3 - t2) / 1_000_000.0,
                    "output_copy_ms": (t4 - t3) / 1_000_000.0,
                    "postprocess_ms": (t5 - t4) / 1_000_000.0,
                    "runtime_with_tensor_io_ms": (t4 - t1) / 1_000_000.0,
                    "pipeline_no_io_ms": (t5 - t0) / 1_000_000.0,
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

    csv_path = output_dir / "latency_raw.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped_rows = {
        group: [row for row in rows if row["group"] == group] for group in args.groups
    }
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "primary_metric": "inference_ms = predictor.run() only",
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
        "model": f"{export_model_name} (PaddleX export)",
        "registry_key": registry_key,
        "threshold_rule": f"score > {args.threshold}",
        "images": [],
    }
    for record in records:
        detections = detections_by_key[record["key"]]
        visual = draw_visualization(record["image"], detections, cv2)
        visual_path = output_dir / record["group"] / f"{record['path'].stem}_vis.png"
        if not cv2.imwrite(str(visual_path), visual):
            raise RuntimeError(f"Failed to write visualization: {visual_path}")
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
        "model_name": export_model_name,
        "registry_key": registry_key,
        "execution_path": "paddle.inference on the PaddleX-exported model",
        "paths": {
            "data_dir": str(data_dir),
            "model_dir": str(model_dir),
            "model_registry": str(registry_path),
            "registry_format_source": str((registry_path.parent / "README.md").resolve()),
            "output_dir": str(output_dir),
            "script": str(Path(__file__).resolve()),
        },
        "model_files": model_files,
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
            "OpenCV decode to BGR (excluded from benchmark)",
            "BGR to RGB",
            f"stretch resize to {args.input_width}x{args.input_height} with INTER_CUBIC; keep_ratio=false",
            "float32 * (1/255); mean=[0,0,0], std=[1,1,1]",
            "HWC to contiguous NCHW",
        ],
        "postprocessing": {
            "labels": ["cell"],
            "score_threshold": args.threshold,
            "comparison": "strict greater-than, matching PaddleX DetPostProcess",
            "boxes_clipped_to_original_image": True,
        },
        "benchmark": {
            "warmup_runs": args.warmup,
            "measured_repeats_per_image": args.repeats,
            "measured_samples": len(rows),
            "shuffle_seed": args.seed,
            "ordering": "deterministic seeded shuffle of selected-group images per repeat",
            "timer": "time.perf_counter_ns",
            "primary_metric": "predictor.run() only",
            "runtime_with_tensor_io_metric": "input reshape/copy + predictor.run + output copy",
            "pipeline_metric": "preprocess + input reshape/copy + predictor.run + output copy + postprocess",
            "excluded_from_latency": [
                "model/config loading",
                "image disk read/decode",
                "visualization drawing",
                "output file encoding/writes",
            ],
            "model_load_ms_reported_separately": model_load_ms,
        },
        "runtime": runtime_config,
        "process": process_config,
        "host": {
            "cpu": get_cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "power_plan_observed": get_power_plan(),
            "operating_system": platform.platform(),
            "python": sys.version,
            "paddle": paddle.__version__,
            "paddle_commit": getattr(paddle.version, "commit", None),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "psutil": psutil.__version__,
        },
        "compatibility_note": (
            "oneDNN is explicitly disabled because Paddle 3.2.2 on this host fails "
            "this exported graph with a fused_conv2d Filter error when oneDNN and "
            "IR optimization are enabled together. Do not mix results from a different "
            "oneDNN/IR/thread/affinity configuration."
        ),
    }
    write_json(output_dir / "benchmark_config.json", config_document)

    overall_inference = summary["overall"]["latency"]["inference_ms"]
    overall_pipeline = summary["overall"]["latency"]["pipeline_no_io_ms"]
    log_lines = [
        "PaddleX-L fixed CPU benchmark",
        f"Protocol: {PROTOCOL_VERSION}",
        f"Registry model: {registry_key} -> {model_dir}",
        f"Export model: {export_model_name}",
        f"Images: {len(records)} ({image_counts})",
        f"Warm-up: {args.warmup}; repeats/image: {args.repeats}; samples: {len(rows)}",
        f"Threshold: score > {args.threshold}",
        f"CPU affinity: {process_config['affinity_applied']}; threads: {args.threads}",
        "Backend: Paddle FP32; oneDNN: OFF; new IR: ON; optimization level: 3; batch size: 1",
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
