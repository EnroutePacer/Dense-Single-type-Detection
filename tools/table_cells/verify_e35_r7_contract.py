"""Fail closed when an e35/r7 run drifts from the frozen experiment contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


REFERENCE_COMMIT = "6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a"
RESUME_EXPERIMENTS = {
    "resume_b0_baseline": dict(sgqs=False, efgl=False, loss="vfl", augmentation="clean"),
    "resume_b1_sgqs": dict(sgqs=True, efgl=False, loss="vfl", augmentation="clean"),
    "resume_b2_efgl": dict(sgqs=False, efgl=True, loss="vfl", augmentation="clean"),
    "resume_b3_sgqs_e": dict(sgqs=True, efgl=True, loss="vfl", augmentation="clean"),
    "resume_c1_small_reg": dict(sgqs=False, efgl=False, loss="small_reg", augmentation="clean"),
    "resume_c2_mal_only": dict(sgqs=False, efgl=False, loss="mal", augmentation="clean"),
    "resume_c3_mosaic_only": dict(sgqs=False, efgl=False, loss="vfl", augmentation="mosaic_only"),
    "resume_c4_deim_core": dict(sgqs=False, efgl=False, loss="mal", augmentation="dense_o2o"),
}

CONFIG_BASENAMES = {
    "deim_core_e35": "deim_core_e35.yml",
    **{name: f"{name}.yml" for name in RESUME_EXPERIMENTS},
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
                         if relative == delta_path else provenance.get("compatible_runtime_deltas", {}).get(
                             relative, {}).get("runtime_sha256", reference_hash))
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
        require_equal(loader["num_workers"], 4, f"{split} worker count")
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


def verify_base_vfl(
    config: dict[str, Any],
    *,
    box_weight: float = 1.0,
    local_weight: float = 1.0,
) -> None:
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
    require_equal(criterion["small_object_box_weight"], box_weight,
                  "box small-object weight")
    require_equal(criterion["small_object_local_weight"], local_weight,
                  "local small-object weight")


def verify_base_mal(config: dict[str, Any]) -> None:
    criterion = config["DEIMCriterion"]
    require_equal(criterion["losses"], ["mal", "boxes", "local"],
                  "MAL losses")
    require_equal(criterion["weight_dict"], {
        # The inactive VFL key remains after the repository's deep YAML merge.
        "loss_vfl": 1,
        "loss_mal": 1,
        "loss_bbox": 5,
        "loss_giou": 2,
        "loss_fgl": 0.15,
        "loss_ddf": 1.5,
    }, "MAL loss coefficients")
    require_equal(criterion["alpha"], 0.75, "MAL alpha")
    require_equal(criterion["gamma"], 1.5, "MAL gamma")
    require_equal(criterion["small_object_weight"], 1.0,
                  "MAL classification small-object weight")
    require_equal(criterion["small_object_box_weight"], 1.0,
                  "MAL box small-object weight")
    require_equal(criterion["small_object_local_weight"], 1.0,
                  "MAL local small-object weight")


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


def verify_clean_augmentation(config: dict[str, Any]) -> None:
    train = config["train_dataloader"]
    train_ops = [op["type"] for op in train["dataset"]["transforms"]["ops"]]
    require_equal(train_ops, [
        "RandomPhotometricDistort",
        "RandomHorizontalFlip",
        "Resize",
        "SanitizeBoundingBoxes",
        "ConvertPILImage",
        "ConvertBoxes",
    ], "clean resume training transforms")
    require_equal(train["collate_fn"]["mixup_prob"], 0.0,
                  "clean resume MixUp probability")


def verify_mosaic_only_augmentation(config: dict[str, Any]) -> None:
    train = config["train_dataloader"]
    transforms = train["dataset"]["transforms"]
    require_equal([op["type"] for op in transforms["ops"]], [
        "Mosaic",
        "RandomPhotometricDistort",
        "RandomHorizontalFlip",
        "Resize",
        "SanitizeBoundingBoxes",
        "ConvertPILImage",
        "ConvertBoxes",
    ], "Mosaic-only training transforms")
    require_equal(transforms["policy"], {
        "name": "stop_epoch",
        "epoch": [28, 35, 35],
        "ops": ["Mosaic"],
    }, "Mosaic-only policy")
    require_equal(transforms["mosaic_prob"], 0.5, "Mosaic-only probability")
    require_equal(train["collate_fn"]["mixup_prob"], 0.0,
                  "Mosaic-only MixUp probability")


def verify_dense_o2o_augmentation(config: dict[str, Any], *, fresh: bool) -> None:
    train = config["train_dataloader"]
    transforms = train["dataset"]["transforms"]
    require_equal([op["type"] for op in transforms["ops"]], [
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
    ], "Dense-O2O transforms")
    expected_policy = [1, 17, 31] if fresh else [28, 35, 35]
    expected_mixup = [1, 17] if fresh else [28, 35]
    require_equal(transforms["policy"]["epoch"], expected_policy,
                  "Dense-O2O augmentation stages")
    require_equal(transforms["policy"]["ops"], [
        "Mosaic", "RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop",
    ], "Dense-O2O scheduled transforms")
    require_equal(transforms["mosaic_prob"], 0.5, "Dense-O2O Mosaic probability")
    require_equal(train["collate_fn"]["mixup_prob"], 0.5,
                  "Dense-O2O MixUp probability")
    require_equal(train["collate_fn"]["mixup_epochs"], expected_mixup,
                  "Dense-O2O MixUp stages")
    require_equal(train["collate_fn"]["stop_epoch"], 35,
                  "Dense-O2O stop epoch")


def verify_resume(config: dict[str, Any], experiment: str) -> None:
    profile = RESUME_EXPERIMENTS[experiment]
    require_equal(Path(config.get("resume", "")).name, "checkpoint0027.pth",
                  "resume checkpoint basename")
    require("tuning" not in config, "resume arm unexpectedly contains tuning")
    require_equal(config["lrsheduler"], "flatcosine", "resume LR scheduler")
    require_equal(config["lr_gamma"], 1, "resume LR gamma")
    require_equal(config["flat_epoch"], 4_000_000, "resume flat LR horizon")
    require_equal(config["no_aug_epoch"], 0, "resume no-augmentation horizon")

    if profile["augmentation"] == "clean":
        verify_clean_augmentation(config)
    elif profile["augmentation"] == "mosaic_only":
        verify_mosaic_only_augmentation(config)
    elif profile["augmentation"] == "dense_o2o":
        verify_dense_o2o_augmentation(config, fresh=False)
    else:
        raise ContractError(f"unknown augmentation profile: {profile['augmentation']}")

    if profile["loss"] == "vfl":
        verify_base_vfl(config)
    elif profile["loss"] == "small_reg":
        verify_base_vfl(config, box_weight=1.30, local_weight=1.15)
    elif profile["loss"] == "mal":
        verify_base_mal(config)
    else:
        raise ContractError(f"unknown loss profile: {profile['loss']}")

    verify_sgqs(config) if profile["sgqs"] else verify_no_sgqs(config)
    verify_efgl(config, profile["efgl"])


def verify_deim_core(config: dict[str, Any]) -> None:
    require("resume" not in config, "fresh DEIM-core arm unexpectedly contains resume")
    require_equal(Path(config.get("tuning", "")).name,
                  "deim_dfine_hgnetv2_m_coco_90e.pth",
                  "DEIM-core tuning checkpoint basename")
    verify_no_sgqs(config)
    verify_efgl(config, False)

    verify_base_mal(config)
    verify_dense_o2o_augmentation(config, fresh=True)


def verify_resolved_config(config: dict[str, Any], experiment: str) -> None:
    verify_common_budget(config)
    if experiment in RESUME_EXPERIMENTS:
        verify_resume(config, experiment)
    elif experiment == "deim_core_e35":
        verify_deim_core(config)
    else:
        raise ContractError(f"unknown experiment: {experiment}")


def verify_frozen_inputs(args: argparse.Namespace, frozen: dict[str, Any]) -> None:
    require_equal(args.config_source.name, CONFIG_BASENAMES[args.experiment],
                  "config source basename")

    expected_mode = "tuning" if args.experiment == "deim_core_e35" else "resume"
    require_equal(args.run_mode, expected_mode, "checkpoint mode")
    require_equal(args.world_size, 8, "DDP world size")
    require_equal(args.batch_size_per_gpu, 8, "per-GPU batch size")
    require_equal(args.num_workers, 4, "DataLoader worker count")
    require_equal(args.quick_steps, 0, "quick-step override")

    train = frozen["train_annotation"]
    val = frozen["val_annotation"]
    checkpoint_key = ("resume_checkpoint" if args.run_mode == "resume"
                      else "deim_tuning_checkpoint")
    checkpoint = frozen[checkpoint_key]

    require(args.train_ann.is_file(), f"training annotation missing: {args.train_ann}")
    require(args.val_ann.is_file(), f"validation annotation missing: {args.val_ann}")
    require(args.source_checkpoint.is_file(),
            f"source checkpoint missing: {args.source_checkpoint}")
    require_equal(args.source_checkpoint.name, checkpoint["basename"],
                  "source checkpoint basename")

    if args.run_mode == "resume":
        require_equal(args.source_checkpoint_last_epoch,
                      str(checkpoint["last_epoch"]), "resume last_epoch")
    else:
        require_equal(args.source_checkpoint_last_epoch, "not_applicable",
                      "tuning checkpoint last_epoch marker")

    if not args.verify_frozen_inputs:
        return

    require_equal(args.train_ann.stat().st_size, train["bytes"],
                  "training annotation bytes")
    require_equal(args.train_ann_sha256, train["sha256"],
                  "training annotation SHA-256")
    require_equal(args.val_ann.stat().st_size, val["bytes"],
                  "validation annotation bytes")
    require_equal(args.val_ann_sha256, val["sha256"],
                  "validation annotation SHA-256")
    require_equal(args.source_checkpoint.stat().st_size, checkpoint["bytes"],
                  "source checkpoint bytes")
    require_equal(args.source_checkpoint_sha256, checkpoint["sha256"],
                  "source checkpoint SHA-256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--config-source", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-mode", choices=("resume", "tuning"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--quick-steps", type=int, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--source-checkpoint-last-epoch", required=True)
    parser.add_argument("--train-ann", type=Path, required=True)
    parser.add_argument("--train-ann-sha256", required=True)
    parser.add_argument("--val-ann", type=Path, required=True)
    parser.add_argument("--val-ann-sha256", required=True)
    parser.add_argument("--verify-frozen-inputs", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    provenance = yaml.safe_load(args.provenance.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(args.frozen_inputs.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.resolved_config.read_text(encoding="utf-8"))
    args.verify_frozen_inputs = bool(args.verify_frozen_inputs)
    require(args.experiment in CONFIG_BASENAMES,
            f"unknown experiment: {args.experiment}")
    verify_method_snapshot(project_root, provenance)
    verify_frozen_inputs(args, frozen)
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
