#!/usr/bin/env python3
"""Assemble group-specific PaddleX-L runs into one canonical result directory.

PaddleX uses separate wired and wireless weights.  This script replaces one
group in an existing canonical run without rerunning the already-correct group,
then recomputes every aggregate latency field from the combined raw samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


METRIC_FIELDS = (
    "preprocess_ms",
    "input_copy_ms",
    "inference_ms",
    "output_copy_ms",
    "postprocess_ms",
    "runtime_with_tensor_io_ms",
    "pipeline_no_io_ms",
)


def parse_args() -> argparse.Namespace:
    fallback = Path(__file__).resolve().parents[3] / "_local" / "baseline"
    root = Path(os.environ.get("BASELINE_ROOT", str(fallback))).expanduser()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-dir", type=Path, default=root / "output" / "paddlex_L"
    )
    parser.add_argument(
        "--replacement-dir",
        type=Path,
        default=root / "output" / "paddlex_L_wired_fix",
    )
    parser.add_argument("--replace-group", choices=("wired", "wireless"), default="wired")
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=root / "models" / "which_weight_to_use" / "path.txt",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_registry(path: Path) -> dict[str, Path]:
    registry: dict[str, Path] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ' "' not in line or not line.endswith('"'):
            raise ValueError(f"Invalid registry line {line_number}: {raw}")
        name, quoted = line.split(None, 1)
        value = quoted.strip()
        if len(value) < 2 or value[0] != '"' or value[-1] != '"':
            raise ValueError(f"Invalid registry line {line_number}: {raw}")
        registry[name] = Path(value[1:-1]).expanduser().resolve()
    return registry


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader.fieldnames), rows


def summarize_values(values: list[float]) -> dict[str, float | int]:
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


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty row set")
    result: dict[str, Any] = {
        "samples": len(rows),
        "unique_images": len({row["image_key"] for row in rows}),
        "latency": {},
    }
    for field in METRIC_FIELDS:
        result["latency"][field] = summarize_values([float(row[field]) for row in rows])
    mean_ms = result["latency"]["inference_ms"]["mean_ms"]
    result["inference_fps_from_mean"] = 1000.0 / mean_ms
    return result


def model_files(model_dir: Path) -> dict[str, dict[str, Any]]:
    names = (
        "inference.pdmodel",
        "inference.pdiparams",
        "inference.pdiparams.info",
        "inference.yml",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def config_model_entry(config: dict[str, Any], registry_key: str, model_dir: Path) -> dict[str, Any]:
    return {
        "registry_key": registry_key,
        "export_model_name": config["model_name"],
        "model_dir": str(model_dir),
        "model_files": model_files(model_dir),
        "benchmark_started_at": config["benchmark_started_at"],
        "benchmark_finished_at": config["benchmark_finished_at"],
        "model_load_ms": config["benchmark"]["model_load_ms_reported_separately"],
        "runtime": config["runtime"],
        "process": config["process"],
    }


def main() -> int:
    args = parse_args()
    canonical = args.canonical_dir.resolve()
    replacement = args.replacement_dir.resolve()
    group = args.replace_group
    other_group = "wireless" if group == "wired" else "wired"
    registry_path = args.model_registry.resolve()
    registry = read_registry(registry_path)

    old_config = read_json(canonical / "benchmark_config.json")
    new_config = read_json(replacement / "benchmark_config.json")
    old_summary = read_json(canonical / "latency_summary.json")
    new_summary = read_json(replacement / "latency_summary.json")
    old_detections = read_json(canonical / "detections.json")
    new_detections = read_json(replacement / "detections.json")

    if new_config["data"]["groups"] != [group]:
        raise RuntimeError(
            f"Replacement run must contain only {group!r}; got {new_config['data']['groups']}"
        )
    expected_new_key = f"PaddleX-{group}"
    if new_config.get("registry_key") != expected_new_key:
        raise RuntimeError(
            f"Replacement registry key must be {expected_new_key!r}; "
            f"got {new_config.get('registry_key')!r}"
        )
    if old_config["protocol_version"] != new_config["protocol_version"]:
        raise RuntimeError("Protocol versions differ; results cannot be assembled")
    if old_config["postprocessing"]["score_threshold"] != 0.5:
        raise RuntimeError("Existing canonical threshold is not 0.5")
    if new_config["postprocessing"]["score_threshold"] != 0.5:
        raise RuntimeError("Replacement threshold is not 0.5")

    preserved_registry_key = f"PaddleX-{other_group}"
    if preserved_registry_key not in registry:
        raise KeyError(f"{preserved_registry_key!r} is absent from {registry_path}")
    preserved_files = model_files(registry[preserved_registry_key])
    recorded_files = old_config.get("model_files")
    if not isinstance(recorded_files, dict):
        raise RuntimeError("Existing canonical config has no recorded model-file hashes")
    for file_name, metadata in preserved_files.items():
        recorded = recorded_files.get(file_name, {})
        if recorded.get("sha256") != metadata["sha256"]:
            raise RuntimeError(
                f"Preserved {other_group} model hash differs for {file_name}; rerun is required"
            )

    old_fields, old_rows = read_csv(canonical / "latency_raw.csv")
    new_fields, new_rows = read_csv(replacement / "latency_raw.csv")
    if old_fields != new_fields:
        raise RuntimeError("Latency CSV schemas differ")
    preserved_rows = [row for row in old_rows if row["group"] == other_group]
    replacement_rows = [row for row in new_rows if row["group"] == group]
    if not preserved_rows or not replacement_rows:
        raise RuntimeError("Missing preserved or replacement latency rows")
    combined_rows = replacement_rows + preserved_rows
    combined_rows.sort(
        key=lambda row: (
            int(row["repeat"]),
            0 if row["group"] == "wired" else 1,
            int(row["sequence_index"]),
        )
    )
    for sequence_index, row in enumerate(combined_rows, start=1):
        row["sequence_index"] = str(sequence_index)

    with (canonical / "latency_raw.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=old_fields)
        writer.writeheader()
        writer.writerows(combined_rows)

    old_images = {
        (item["group"], Path(item["source_path"]).name): item
        for item in old_detections["images"]
        if item["group"] == other_group
    }
    new_images = {
        (item["group"], Path(item["source_path"]).name): item
        for item in new_detections["images"]
        if item["group"] == group
    }
    if not old_images or not new_images:
        raise RuntimeError("Missing preserved or replacement detections")

    destination_group_dir = canonical / group
    destination_group_dir.mkdir(parents=True, exist_ok=True)
    for item in new_images.values():
        source_visualization = Path(item["visualization_path"])
        destination_visualization = destination_group_dir / source_visualization.name
        shutil.copy2(source_visualization, destination_visualization)
        item["visualization_path"] = str(destination_visualization.resolve())

    combined_images = list(old_images.values()) + list(new_images.values())
    combined_images.sort(
        key=lambda item: (item["group"], Path(item["source_path"]).name.casefold())
    )
    detections_document = {
        "model": "PaddleX-L group-specific table-cell detectors",
        "models_by_group": {
            group: new_config["model_name"],
            other_group: old_config["model_name"],
        },
        "registry_keys_by_group": {
            group: expected_new_key,
            other_group: f"PaddleX-{other_group}",
        },
        "threshold_rule": "score > 0.5",
        "images": combined_images,
    }
    write_json(canonical / "detections.json", detections_document)

    groups = ("wired", "wireless")
    by_group = {name: [row for row in combined_rows if row["group"] == name] for name in groups}
    image_keys = sorted({row["image_key"] for row in combined_rows})
    combined_summary: dict[str, Any] = {
        "protocol_version": new_config["protocol_version"],
        "primary_metric": "inference_ms = predictor.run() only",
        "overall": summarize_rows(combined_rows),
        "groups": {name: summarize_rows(by_group[name]) for name in groups},
        "per_image": {
            key: summarize_rows([row for row in combined_rows if row["image_key"] == key])
            for key in image_keys
        },
        "detection_count_at_threshold": {
            f"{item['group']}/{Path(item['source_path']).name}": int(item["detection_count"])
            for item in combined_images
        },
        "output_deterministic_across_repeats": bool(
            old_summary["output_deterministic_across_repeats"]
            and new_summary["output_deterministic_across_repeats"]
        ),
        "non_deterministic_images": sorted(
            set(old_summary["non_deterministic_images"])
            | set(new_summary["non_deterministic_images"])
        ),
        "models_by_group": detections_document["models_by_group"],
        "assembly": {
            "preserved_group": other_group,
            "replaced_group": group,
            "replacement_run": str(replacement),
        },
    }
    write_json(canonical / "latency_summary.json", combined_summary)

    group_configs = {group: new_config, other_group: old_config}
    model_entries: dict[str, Any] = {}
    for name in groups:
        key = f"PaddleX-{name}"
        if key not in registry:
            raise KeyError(f"{key!r} is absent from {registry_path}")
        model_entries[name] = config_model_entry(group_configs[name], key, registry[key])

    input_images = []
    for name in groups:
        source_config = group_configs[name]
        input_images.extend(
            item
            for item in source_config["data"]["input_images"]
            if item["key"].startswith(f"{name}/")
        )
    input_images.sort(key=lambda item: item["key"].casefold())
    combined_config = {
        "protocol_version": new_config["protocol_version"],
        "created_at": datetime.now().astimezone().isoformat(),
        "model_name": "PaddleX-L table-cell detectors (group-specific weights)",
        "execution_path": "paddle.inference on PaddleX-exported models",
        "models_by_group": model_entries,
        "paths": {
            "data_dir": new_config["paths"]["data_dir"],
            "model_registry": str(registry_path),
            "registry_format_source": str((registry_path.parent / "README.md").resolve()),
            "model_dirs_by_group": {
                name: str(registry[f"PaddleX-{name}"]) for name in groups
            },
            "output_dir": str(canonical),
            "benchmark_script": new_config["paths"]["script"],
            "assembly_script": str(Path(__file__).resolve()),
        },
        "data": {
            "groups": list(groups),
            "image_counts": {
                name: sum(item["key"].startswith(f"{name}/") for item in input_images)
                for name in groups
            },
            "total_images": len(input_images),
            "fixed_input_shape_nchw": new_config["data"]["fixed_input_shape_nchw"],
            "input_images": input_images,
        },
        "preprocessing": new_config["preprocessing"],
        "postprocessing": new_config["postprocessing"],
        "benchmark": {
            "warmup_runs_per_model": {
                name: group_configs[name]["benchmark"]["warmup_runs"] for name in groups
            },
            "measured_repeats_per_image": new_config["benchmark"][
                "measured_repeats_per_image"
            ],
            "measured_samples": len(combined_rows),
            "shuffle_seed": new_config["benchmark"]["shuffle_seed"],
            "ordering": "each group measured independently with deterministic seeded shuffle",
            "timer": new_config["benchmark"]["timer"],
            "primary_metric": new_config["benchmark"]["primary_metric"],
            "runtime_with_tensor_io_metric": new_config["benchmark"][
                "runtime_with_tensor_io_metric"
            ],
            "pipeline_metric": new_config["benchmark"]["pipeline_metric"],
            "excluded_from_latency": new_config["benchmark"]["excluded_from_latency"],
        },
        "host": new_config["host"],
        "assembly": {
            "preserved_group": other_group,
            "preserved_source_created_at": old_config["created_at"],
            "replaced_group": group,
            "replacement_source_created_at": new_config["created_at"],
            "replacement_run": str(replacement),
            "preserved_model_hashes_verified_against_registry": True,
            "reason": "PaddleX uses separate wired and wireless weights per registry README",
        },
        "compatibility_note": new_config["compatibility_note"],
    }
    write_json(canonical / "benchmark_config.json", combined_config)

    overall = combined_summary["overall"]["latency"]["inference_ms"]
    log_lines = [
        "PaddleX-L fixed CPU benchmark (group-specific weights)",
        f"Protocol: {new_config['protocol_version']}",
        f"Registry: {registry_path}",
        f"Models: wired={model_entries['wired']['export_model_name']}; "
        f"wireless={model_entries['wireless']['export_model_name']}",
        f"Threshold: score > {new_config['postprocessing']['score_threshold']}",
        f"Samples: {len(combined_rows)}; images: {len(input_images)}",
        f"Inference overall mean={overall['mean_ms']:.3f} ms, "
        f"P50={overall['median_p50_ms']:.3f} ms, P95={overall['p95_ms']:.3f} ms",
    ]
    for name in groups:
        values = combined_summary["groups"][name]["latency"]["inference_ms"]
        log_lines.append(
            f"{name} inference mean={values['mean_ms']:.3f} ms, "
            f"P50={values['median_p50_ms']:.3f} ms, P95={values['p95_ms']:.3f} ms"
        )
    (canonical / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"Replaced {group} and preserved {other_group} under {canonical}")
    print(f"Combined samples: {len(combined_rows)}; images: {len(input_images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
