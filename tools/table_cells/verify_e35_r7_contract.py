"""Fail closed when an e35/r7 run drifts from the frozen experiment contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


REFERENCE_COMMIT = "6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a"
RESUME_EXPERIMENTS = {
    "resume_b0_baseline": (False, False),
    "resume_b1_sgqs": (True, False),
    "resume_b2_efgl": (False, True),
    "resume_b3_sgqs_e": (True, True),
}


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label}: expected {expected!r}, got {actual!r}")


def canonical_lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_method_snapshot(project_root: Path, provenance: dict[str, Any]) -> None:
    require_equal(provenance["reference"]["commit"], REFERENCE_COMMIT,
                  "normative SGQS-E commit")
    require_equal(provenance["hash_policy"], "sha256_after_canonical_lf",
                  "method hash policy")

    delta = provenance["allowed_runtime_delta"]
    delta_path = delta["path"]
    reference_files = provenance["reference_files"]
    require(delta_path in reference_files,
            "allowed runtime delta is absent from reference_files")

    for relative, reference_hash in reference_files.items():
        path = project_root / relative
        require(path.is_file(), f"method snapshot file is missing: {relative}")
        actual_hash = canonical_lf_sha256(path)
        expected_hash = (delta["runtime_sha256"]
                         if relative == delta_path else reference_hash)
        require_equal(actual_hash, expected_hash,
                      f"canonical method SHA-256 for {relative}")


def verify_common_budget(config: dict[str, Any]) -> None:
    require_equal(config["epoches"], 35, "absolute target epoches")
    require_equal(config["seed"], 0, "seed")
    require_equal(config["num_classes"], 1, "number of classes")
    require_equal(config["eval_spatial_size"], [640, 640], "evaluation size")
    require_equal(config["DFINETransformer"]["num_queries"], 640,
                  "decoder query budget")
    require_equal(config["PostProcessor"]["num_top_queries"], 640,
                  "postprocessor query budget")

    for split in ("train_dataloader", "val_dataloader"):
        loader = config[split]
        require_equal(loader["total_batch_size"], 64, f"{split} global batch")
        collate = loader["collate_fn"]
        require_equal(collate["base_size"], 640, f"{split} collate base size")
        require_equal(collate.get("scales"), None, f"{split} multiscale setting")
        resize_ops = [op for op in loader["dataset"]["transforms"]["ops"]
                      if op["type"] == "Resize"]
        require_equal(len(resize_ops), 1, f"{split} Resize count")
        require_equal(resize_ops[0]["size"], [640, 640], f"{split} Resize size")

    require_equal(config.get("quick_probe", False), False, "quick-probe flag")
    require(config.get("max_train_steps") in (None, 0),
            "full epoch runs must not set max_train_steps")


def verify_base_vfl(config: dict[str, Any]) -> None:
    criterion = config["DEIMCriterion"]
    require_equal(criterion["losses"], ["vfl", "boxes", "local"],
                  "resume base losses")
    require_equal(criterion["weight_dict"], {
        "loss_vfl": 1,
        "loss_bbox": 5,
        "loss_giou": 2,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }, "resume base loss coefficients")
    require_equal(criterion["alpha"], 0.75, "VFL alpha")
    require_equal(criterion["gamma"], 2.0, "VFL gamma")
    require_equal(criterion["small_object_weight"], 1.0,
                  "classification small-object weight")
    require_equal(criterion["small_object_box_weight"], 1.0,
                  "box small-object weight")
    require_equal(criterion["small_object_local_weight"], 1.0,
                  "local small-object weight")


def verify_sgqs(config: dict[str, Any]) -> None:
    require_equal(config["criterion"], "SGQSCriterion", "SGQS criterion type")
    require_equal(config["DFINETransformer"]["sgqs_supervisor"], {
        "type": "SGQSQuerySupervisor",
        "scale_priors": [32.0, 64.0, 128.0],
        "initial_radius": 1,
        "max_radius": 4,
        "ignore_radius": 1,
    }, "SGQS query supervisor")
    sgqs = config["SGQSCriterion"]
    require_equal(sgqs["base_criterion"], "DEIMCriterion", "SGQS base criterion")
    require_equal(sgqs["weight_dict"], {
        "loss_sgqs_dense": 1.0,
        "loss_sgqs_bbox": 2.5,
        "loss_sgqs_giou": 1.0,
        "loss_sgqs_survival": 0.25,
    }, "SGQS loss coefficients")
    expected = {
        "q_min": 0.25,
        "dense_alpha": 0.75,
        "dense_gamma": 2.0,
        "negative_weight": 1.0,
        "survival_margin": 0.2,
        "survival_temperature": 0.1,
        "survival_quality_power": 1.0,
        "survival_start_epoch": 28.0,
        "survival_ramp_epochs": 1.0,
        "survival_threshold": "non_owner_budget",
        "protect_encoder_supervision": True,
    }
    for key, value in expected.items():
        require_equal(sgqs[key], value, f"SGQS {key}")


def verify_no_sgqs(config: dict[str, Any]) -> None:
    require_equal(config["criterion"], "DEIMCriterion", "base criterion type")
    require(config["DFINETransformer"].get("sgqs_supervisor") is None,
            "non-SGQS arm unexpectedly enables the SGQS supervisor")


def verify_efgl(config: dict[str, Any], enabled: bool) -> None:
    efgl = config.get("efgl", {})
    require_equal(bool(efgl.get("enabled", False)), enabled, "EFGL enabled flag")
    if not enabled:
        return
    expected = {
        "strength": 0.5,
        "neighbor_strength": 0.5,
        "short_side_scale": 16.0,
        "pixel_floor": 2.0,
        "neighbors": 4,
        "neighbor_radius": 4.0,
        "compensation_fraction": 0.3,
        "normal_branch": True,
        "dn_branch": True,
        "compensation_start_epoch": 28.0,
        "compensation_duration_epochs": 2.1,
    }
    for key, value in expected.items():
        require_equal(efgl[key], value, f"EFGL {key}")


def verify_resume(config: dict[str, Any], experiment: str) -> None:
    sgqs_enabled, efgl_enabled = RESUME_EXPERIMENTS[experiment]
    require_equal(Path(config.get("resume", "")).name, "checkpoint0027.pth",
                  "resume checkpoint basename")
    require("tuning" not in config, "resume arm unexpectedly contains tuning")
    require_equal(config["lrsheduler"], "flatcosine", "resume LR scheduler")
    require_equal(config["lr_gamma"], 1, "resume LR gamma")
    require_equal(config["flat_epoch"], 4_000_000, "resume flat LR horizon")
    require_equal(config["no_aug_epoch"], 0, "resume no-augmentation horizon")

    train_ops = [op["type"] for op in
                 config["train_dataloader"]["dataset"]["transforms"]["ops"]]
    require_equal(train_ops, [
        "RandomPhotometricDistort",
        "RandomHorizontalFlip",
        "Resize",
        "SanitizeBoundingBoxes",
        "ConvertPILImage",
        "ConvertBoxes",
    ], "resume training transforms")
    require_equal(config["train_dataloader"]["collate_fn"]["mixup_prob"], 0.0,
                  "resume MixUp probability")

    verify_base_vfl(config)
    verify_sgqs(config) if sgqs_enabled else verify_no_sgqs(config)
    verify_efgl(config, efgl_enabled)


def verify_deim_core(config: dict[str, Any]) -> None:
    require("resume" not in config, "fresh DEIM-core arm unexpectedly contains resume")
    require_equal(Path(config.get("tuning", "")).name,
                  "deim_dfine_hgnetv2_m_coco_90e.pth",
                  "DEIM-core tuning checkpoint basename")
    verify_no_sgqs(config)
    verify_efgl(config, False)

    criterion = config["DEIMCriterion"]
    require_equal(criterion["losses"], ["mal", "boxes", "local"],
                  "DEIM-core losses")
    require_equal(criterion["weight_dict"], {
        # Deep YAML merging retains the inherited key, but `losses` is the
        # authoritative dispatch list, so VFL is not computed in this arm.
        "loss_vfl": 1,
        "loss_mal": 1,
        "loss_bbox": 5,
        "loss_giou": 2,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }, "DEIM-core loss coefficients")
    require_equal(criterion["alpha"], 0.75, "MAL alpha")
    require_equal(criterion["gamma"], 1.5, "MAL gamma")

    train = config["train_dataloader"]
    ops = [op["type"] for op in train["dataset"]["transforms"]["ops"]]
    require_equal(ops, [
        "Mosaic",
        "RandomPhotometricDistort",
        "RandomZoomOut",
        "RandomIoUCrop",
        "SanitizeBoundingBoxes",
        "RandomHorizontalFlip",
        "Resize",
        "SanitizeBoundingBoxes",
        "ConvertPILImage",
        "ConvertBoxes",
    ], "DEIM-core Dense-O2O transforms")
    require_equal(train["dataset"]["transforms"]["policy"]["epoch"], [1, 17, 31],
                  "DEIM-core augmentation stages")
    require_equal(train["collate_fn"]["mixup_epochs"], [1, 17],
                  "DEIM-core MixUp stages")
    require_equal(train["collate_fn"]["stop_epoch"], 35,
                  "DEIM-core stop epoch")


def verify_resolved_config(config: dict[str, Any], experiment: str) -> None:
    verify_common_budget(config)
    if experiment in RESUME_EXPERIMENTS:
        verify_resume(config, experiment)
    elif experiment == "deim_core_e35":
        verify_deim_core(config)
    else:
        raise ContractError(f"unknown experiment: {experiment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    provenance = yaml.safe_load(args.provenance.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))
    verify_method_snapshot(project_root, provenance)
    verify_resolved_config(config, args.experiment)
    print(
        "e35/r7 contract OK: "
        f"experiment={args.experiment}, reference={REFERENCE_COMMIT}, "
        "fixed640/query640/global_batch64/maxDets300"
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ContractError) as exc:
        raise SystemExit(f"e35/r7 contract FAILED: {exc}") from exc
