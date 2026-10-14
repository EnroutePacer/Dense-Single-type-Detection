from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RUNS = OrderedDict(
    [
        ("SGQS", ROOT / "resume_b1_sgqs_seed0" / "console.log"),
        ("SGQS-E", ROOT / "resume_b3_sgqs_e_seed0" / "console.log"),
    ]
)
OUTPUT_DIR = Path(__file__).resolve().parent

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EPOCH_PROGRESS_RE = re.compile(r"^Epoch: \[(\d+)\]")
EPOCH_TOTAL_RE = re.compile(r"^Epoch: \[(\d+)\] Total time:")
PAIR_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_]*):\s+({FLOAT})\s+\(({FLOAT})\)")
SIMPLE_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_]*):\s+({FLOAT})(?!\s*\()")
COCO_VALUE_RE = re.compile(rf"=\s+({FLOAT})\s*$")

COCO_METRICS = [
    "mAP@50:95",
    "AP50",
    "AP75",
    "AP-small",
    "AP-medium",
    "AP-large",
    "AR@1",
    "AR@10",
    "AR@300",
    "AR-small",
    "AR-medium",
    "AR-large",
    "AR50",
    "AR75",
]


def parse_log(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    train: dict[int, dict[str, float]] = {}
    eval_data: dict[int | None, dict[str, float]] = {}
    current_train_epoch: int | None = None
    completed_epoch: int | None = None
    pending_coco_epoch: int | None = None
    coco_values: list[float] | None = None

    for line in lines:
        progress = EPOCH_PROGRESS_RE.match(line)
        if progress:
            current_train_epoch = int(progress.group(1))

        total = EPOCH_TOTAL_RE.match(line)
        if total:
            completed_epoch = int(total.group(1))

        if line.startswith("Averaged stats:") and "loss:" in line:
            epoch = completed_epoch if completed_epoch is not None else current_train_epoch
            if epoch is None:
                continue
            metrics = {name: float(avg) for name, _, avg in PAIR_RE.findall(line)}
            for name, value in SIMPLE_RE.findall(line):
                metrics.setdefault(name, float(value))
            train[epoch] = metrics

        if line.strip() == "IoU metric: bbox":
            pending_coco_epoch = completed_epoch
            coco_values = []
            continue

        if coco_values is not None and (
            line.startswith(" Average Precision") or line.startswith(" Average Recall")
        ):
            match = COCO_VALUE_RE.search(line)
            if match:
                coco_values.append(float(match.group(1)))
            if len(coco_values) == len(COCO_METRICS):
                eval_data[pending_coco_epoch] = dict(zip(COCO_METRICS, coco_values))
                coco_values = None

    if not train:
        raise RuntimeError(f"No training epochs parsed from {path}")
    first_epoch = min(train)
    if None in eval_data:
        eval_data[first_epoch - 1] = eval_data.pop(None)

    return {"train": train, "eval": dict(sorted(eval_data.items()))}


def save_long_csv(parsed: dict[str, dict[str, dict[int, dict[str, float]]]]) -> None:
    destination = OUTPUT_DIR / "sgqs_vs_sgqse_metrics_long.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "phase", "epoch", "metric", "value"])
        for run, phases in parsed.items():
            for phase, epochs in phases.items():
                for epoch, metrics in sorted(epochs.items()):
                    for metric, value in sorted(metrics.items()):
                        writer.writerow([run, phase, epoch, metric, f"{value:.12g}"])


def padded_limits(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        pad = max(abs(low) * 0.04, 0.01)
    else:
        pad = span * 0.14
    return low - pad, high + pad


def plot_grid(
    parsed: dict[str, dict[str, dict[int, dict[str, float]]]],
    *,
    phase: str,
    metrics: list[str],
    title: str,
    filename: str,
    columns: int,
    percent_like: set[str] | None = None,
    runs: tuple[str, ...] = ("SGQS", "SGQS-E"),
) -> plt.Figure:
    percent_like = percent_like or set()
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.25 * rows), squeeze=False)
    colors = {"SGQS": "#2A6FBB", "SGQS-E": "#D95F02"}
    markers = {"SGQS": "o", "SGQS-E": "s"}
    all_handles = []
    all_labels = []

    for index, metric in enumerate(metrics):
        ax = axes[index // columns][index % columns]
        panel_values: list[float] = []
        for run in runs:
            epochs = parsed[run][phase]
            points = [(epoch, values[metric]) for epoch, values in sorted(epochs.items()) if metric in values]
            if not points:
                continue
            x = [item[0] for item in points]
            y = [item[1] for item in points]
            panel_values.extend(y)
            line = ax.plot(
                x,
                y,
                label=run,
                color=colors[run],
                marker=markers[run],
                linewidth=2.0,
                markersize=5,
            )[0]
            if run not in all_labels:
                all_handles.append(line)
                all_labels.append(run)

        ax.set_title(metric, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
        if panel_values:
            ax.set_ylim(*padded_limits(panel_values))
        if metric in percent_like:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    for index in range(len(metrics), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="semibold", y=0.995)
    if all_handles:
        fig.legend(all_handles, all_labels, loc="upper center", ncol=len(all_labels), frameon=False, bbox_to_anchor=(0.5, 0.972))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    return fig


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    parsed = {run: parse_log(path) for run, path in RUNS.items()}
    save_long_csv(parsed)

    figures: list[plt.Figure] = []
    figures.append(
        plot_grid(
            parsed,
            phase="eval",
            metrics=COCO_METRICS,
            title="SGQS vs SGQS-E: COCO evaluation trends",
            filename="sgqs_vs_sgqse_coco_metrics.png",
            columns=3,
            percent_like=set(COCO_METRICS),
        )
    )

    figures.append(
        plot_grid(
            parsed,
            phase="train",
            metrics=[
                "loss",
                "loss_vfl",
                "loss_bbox",
                "loss_giou",
                "loss_fgl",
                "loss_vfl_enc_0",
                "loss_bbox_enc_0",
                "loss_giou_enc_0",
                "loss_sgqs_dense",
                "loss_sgqs_bbox",
                "loss_sgqs_giou",
                "loss_sgqs_survival",
            ],
            title="SGQS vs SGQS-E: training-loss trends",
            filename="sgqs_vs_sgqse_training_losses.png",
            columns=3,
        )
    )

    figures.append(
        plot_grid(
            parsed,
            phase="train",
            metrics=[
                "sgqs_owner_coverage",
                "sgqs_owner_iou",
                "sgqs_pre_topk_survival_rate",
                "sgqs_owner_cutoff_gap",
                "sgqs_conflicting_owner_fraction",
                "sgqs_owner_count",
                "sgqs_owner_p3_count",
                "sgqs_owner_p4_count",
                "sgqs_owner_p5_count",
                "sgqs_assignment_miss_rate",
                "sgqs_budget_overflow_rate",
            ],
            title="SGQS vs SGQS-E: SGQS diagnostic trends",
            filename="sgqs_vs_sgqse_sgqs_diagnostics.png",
            columns=3,
            percent_like={
                "sgqs_owner_coverage",
                "sgqs_owner_iou",
                "sgqs_pre_topk_survival_rate",
                "sgqs_conflicting_owner_fraction",
                "sgqs_assignment_miss_rate",
                "sgqs_budget_overflow_rate",
            },
        )
    )

    figures.append(
        plot_grid(
            parsed,
            phase="train",
            metrics=["efgl_weight_bonus_mean", "efgl_compensation_ramp"],
            title="SGQS-E: EFGL diagnostic trends",
            filename="sgqse_efgl_diagnostics.png",
            columns=2,
            runs=("SGQS-E",),
        )
    )

    with PdfPages(OUTPUT_DIR / "sgqs_vs_sgqse_all_trends.pdf") as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")

    for figure in figures:
        plt.close(figure)

    for run, path in RUNS.items():
        eval_epochs = sorted(parsed[run]["eval"])
        train_epochs = sorted(parsed[run]["train"])
        print(f"{run}: eval={eval_epochs}, train={train_epochs}, source={path}")


if __name__ == "__main__":
    main()
