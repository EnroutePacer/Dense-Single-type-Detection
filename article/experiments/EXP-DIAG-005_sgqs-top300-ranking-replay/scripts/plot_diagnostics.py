#!/usr/bin/env python3
"""Render the static figure pack for the SGQS e28--e34 diagnosis.

The quantitative figures deliberately keep these notions separate:

* full validation vs. the fixed val128 diagnostic subset;
* COCO-small vs. the local hybrid-small definition;
* true COCO AP/AR vs. independent per-GT best-IoU coverage.

The two contact sheets are qualitative only.  Their 13-image source set has no
ground-truth annotations and therefore cannot support GT attribution.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
from PIL import Image


EPOCHS = list(range(28, 35))

# Color-blind-friendly palette (Okabe-Ito plus neutral variants).
COLORS = {
    "purple": "#6F57D9",
    "orange": "#D55E00",
    "teal": "#009E73",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "gray": "#6B7280",
    "light_gray": "#B8BEC8",
    "dark": "#263238",
    "green_fill": "#DDF4EA",
    "red_fill": "#FBE4E1",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_experiment = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=default_experiment,
        help="EXP-DIAG-005 directory containing data/ and figures/.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help=(
            "Optional directory containing SGQS_probe_checkpoint00XX_* folders. "
            "When omitted, only the three self-contained quantitative figures are rendered."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=default_experiment / "figures")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#9AA1AA",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#DDE1E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_epochs(mapping: dict[int, float], label: str) -> list[float]:
    missing = [epoch for epoch in EPOCHS if epoch not in mapping]
    if missing:
        raise ValueError(f"{label} is missing epochs: {missing}")
    return [mapping[epoch] for epoch in EPOCHS]


def read_full_val_curves(experiment: Path) -> tuple[list[float], list[float]]:
    sgqs_rows = read_csv(experiment / "data" / "full_val_metrics.csv")
    sgqs = {int(row["epoch"]): float(row["AP_small"]) for row in sgqs_rows}
    b0_rows = read_csv(experiment / "data" / "b0_full_val_metrics.csv")
    b0 = {int(row["epoch"]): float(row["ap_small"]) for row in b0_rows}
    return require_epochs(sgqs, "SGQS full-val"), require_epochs(b0, "B0 full-val")


def read_val128_coco_metrics(experiment: Path) -> dict[str, list[float]]:
    rows = read_csv(experiment / "data" / "val128_raw_ema_metrics.csv")
    output: dict[str, list[float]] = {}
    for state in ("ema", "raw"):
        selected = {int(row["epoch"]): row for row in rows if row["state"] == state}
        for metric, column in (
            ("ap300", "AP_small_at300"),
            ("ap640", "AP_small_at640"),
            ("ar300", "AR_small_at300"),
            ("ar640", "AR_small_at640"),
        ):
            mapping = {epoch: float(row[column]) for epoch, row in selected.items()}
            output[f"{state}_{metric}"] = require_epochs(
                mapping, f"val128/{state}/{metric}"
            )
    return output


def finish_axis(ax: plt.Axes, y_label: str | None = None) -> None:
    ax.set_xticks(EPOCHS)
    ax.set_xlabel("Checkpoint epoch")
    if y_label:
        ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def annotate_peak(
    ax: plt.Axes,
    epoch: int,
    value: float,
    label: str,
    color: str,
    xytext: tuple[int, int],
) -> None:
    ax.annotate(
        label,
        xy=(epoch, value),
        xytext=xytext,
        textcoords="offset points",
        color=color,
        fontsize=9.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
    )


def plot_metric_trajectories(experiment: Path, out_dir: Path) -> Path:
    sgqs_full, b0_full = read_full_val_curves(experiment)
    val128 = read_val128_coco_metrics(experiment)

    fig = plt.figure(figsize=(13.2, 8.6), dpi=200)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.86, 1.14), hspace=0.42, wspace=0.24)
    ax_full = fig.add_subplot(grid[0, :])
    ax_ap = fig.add_subplot(grid[1, 0])
    ax_ar = fig.add_subplot(grid[1, 1])

    sgqs_pct = [100 * value for value in sgqs_full]
    b0_pct = [100 * value for value in b0_full]
    ax_full.plot(
        EPOCHS,
        sgqs_pct,
        color=COLORS["purple"],
        marker="o",
        lw=2.5,
        ms=6,
        label="SGQS (EMA)",
    )
    ax_full.plot(
        EPOCHS,
        b0_pct,
        color=COLORS["gray"],
        marker="s",
        lw=2.0,
        ms=5,
        label="B0 fixed-640 VFL control (EMA)",
    )
    ax_full.axvspan(28.85, 30.15, color=COLORS["purple"], alpha=0.07, lw=0)
    annotate_peak(
        ax_full,
        30,
        sgqs_pct[2],
        f"SGQS peak {sgqs_pct[2]:.2f}%",
        COLORS["purple"],
        (10, 9),
    )
    ax_full.set_ylim(67.1, 71.15)
    ax_full.yaxis.set_major_locator(MultipleLocator(0.5))
    ax_full.set_title(
        "A  Full validation: COCO AP-small (true one-to-one COCO matching)",
        loc="left",
    )
    ax_full.legend(loc="upper right", ncol=2)
    finish_axis(ax_full, "AP-small (%)")

    styles = {
        "ema300": (COLORS["orange"], "o", "-", "EMA, maxDets=300"),
        "ema640": (COLORS["teal"], "s", "-", "EMA, maxDets=640"),
        "raw300": (COLORS["gray"], "^", "--", "Raw, maxDets=300"),
        "raw640": (COLORS["sky"], "D", ":", "Raw, maxDets=640"),
    }
    for ax, metric, panel in ((ax_ap, "ap", "B"), (ax_ar, "ar", "C")):
        for key, (color, marker, line_style, label) in styles.items():
            weight, max_dets = key[:3], key[3:]
            values = [100 * value for value in val128[f"{weight}_{metric}{max_dets}"]]
            ax.plot(
                EPOCHS,
                values,
                color=color,
                marker=marker,
                ls=line_style,
                lw=2.1 if weight == "ema" else 1.6,
                ms=5.2,
                alpha=1.0 if weight == "ema" else 0.88,
                label=label,
            )
        ax.axvline(29, color=COLORS["purple"], alpha=0.18, lw=1.0)
        ax.set_ylim(55, 100.5)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        metric_name = "AP-small" if metric == "ap" else "AR-small"
        ax.set_title(
            f"{panel}  Fixed val128: COCO {metric_name} (raw vs. EMA)",
            loc="left",
        )
        finish_axis(ax, f"{metric_name} (%)")
    ax_ap.legend(loc="lower right", fontsize=8.8)

    fig.suptitle(
        "SGQS AP-small inverted-U: full-val trajectory and fixed-subset replay",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "All panels use true COCO AP/AR. Full-val and val128 have different sample composition and must not be compared by absolute level. "
        "No SGQS full-val raw curve was saved.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["dark"],
    )
    fig.subplots_adjust(top=0.91, bottom=0.095, left=0.075, right=0.985)
    path = out_dir / "trajectory_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def stage_curve(
    rows: list[dict[str, str]], group: str, stage: str, iou: float = 0.75
) -> list[float]:
    mapping = {
        int(row["epoch"]): 100 * float(row["coverage"])
        for row in rows
        if row["group"] == group
        and row["stage"] == stage
        and abs(float(row["iou_threshold"]) - iou) < 1e-9
    }
    return require_epochs(mapping, f"{group}/{stage}/IoU={iou}")


def plot_stage_coverage(experiment: Path, out_dir: Path) -> Path:
    path = experiment / "data" / "val128_ema" / "stage_coverage.csv"
    rows = read_csv(path)
    styles = {
        "dense_all": (COLORS["light_gray"], "o", ":", "Dense candidates (all 8,084)"),
        "dense_top640": (COLORS["blue"], "^", "--", "Selected dense Top-640"),
        "decoder_all640": (COLORS["teal"], "s", "-", "Decoder all 640"),
        "decoder_top300": (COLORS["orange"], "o", "-", "Decoder Top-300 by score"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=200)
    panels = (
        (
            axes[0],
            "coco_small",
            "A  COCO-small (N=430)",
            "Original annotation area < 32²",
            (58, 101),
        ),
        (
            axes[1],
            "hybrid_small",
            "B  Local hybrid-small (N=7,314)",
            "After direct 640 resize: area < 32² OR short side < 16",
            (70, 101),
        ),
    )
    for ax, group, title, subtitle, ylim in panels:
        for stage, (color, marker, line_style, label) in styles.items():
            values = stage_curve(rows, group, stage)
            ax.plot(
                EPOCHS,
                values,
                color=color,
                marker=marker,
                ls=line_style,
                lw=2.35 if stage.startswith("decoder") else 1.7,
                ms=5.2,
                label=label,
                zorder=4 if stage == "decoder_top300" else 3,
            )
            if stage == "decoder_top300":
                for epoch, value in zip(EPOCHS, values):
                    ax.annotate(
                        f"{value:.1f}",
                        (epoch, value),
                        xytext=(0, 8 if group == "coco_small" else -13),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7.8,
                        color=color,
                    )
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=11.5)
        finish_axis(ax, "Independent GT coverage (%)")

    # The middle-right of the COCO panel is deliberately empty; keeping the
    # legend there avoids obscuring the diagnostic Top-300 trajectory.
    axes[0].legend(loc="center right", fontsize=8.7)
    fig.suptitle(
        "Where the signal appears: stage-wise best-IoU coverage (fixed val128, EMA, IoU ≥ 0.75)",
        fontsize=15.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Per-GT independent best-IoU coverage: the same query may cover multiple GTs. This is NOT COCO recall, TP, or FN. "
        "The two panels use different GT definitions and denominators.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["dark"],
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.075, right=0.985, wspace=0.22)
    output = out_dir / "stage_coverage.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def select_row(rows: Iterable[dict[str, str]], **conditions: str) -> dict[str, str]:
    matches = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def plot_rank_and_budget(experiment: Path, out_dir: Path) -> Path:
    analysis = experiment / "data" / "val128_ema"
    rank_rows = read_csv(analysis / "transition_rank_shifts.csv")
    subgroup_rows = read_csv(analysis / "exclusive_top300_transitions.csv")

    rise = select_row(
        rank_rows,
        start_epoch="28",
        end_epoch="29",
        group="coco_small",
        iou_threshold="0.75",
        transition="gain",
    )
    fall = select_row(
        rank_rows,
        start_epoch="30",
        end_epoch="34",
        group="coco_small",
        iou_threshold="0.75",
        transition="loss",
    )

    fig, (ax_rank, ax_budget) = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=200)

    ax_rank.axhspan(1, 300, color=COLORS["green_fill"], alpha=0.88, zorder=0)
    ax_rank.axhspan(300, 450, color=COLORS["red_fill"], alpha=0.88, zorder=0)
    ax_rank.axhline(300, color=COLORS["dark"], ls="--", lw=1.4, zorder=2)
    ax_rank.text(3.53, 294, "Top-300 cutoff", va="bottom", ha="right", fontsize=9)

    transitions = (
        (rise, (0.0, 1.0), COLORS["purple"], "e28→e29 gain", "49 GTs"),
        (fall, (2.35, 3.35), COLORS["orange"], "e30→e34 loss", "36 GTs"),
    )
    for row, (x0, x1), color, label, count in transitions:
        medians = [float(row["rank_before_median"]), float(row["rank_after_median"])]
        p10 = [float(row["rank_before_p10"]), float(row["rank_after_p10"])]
        p90 = [float(row["rank_before_p90"]), float(row["rank_after_p90"])]
        yerr = [[m - low for m, low in zip(medians, p10)], [high - m for m, high in zip(medians, p90)]]
        ax_rank.errorbar(
            [x0, x1],
            medians,
            yerr=yerr,
            color=color,
            marker="o",
            ms=7,
            lw=2.4,
            capsize=5,
            capthick=1.4,
            label=f"{label} ({count})",
            zorder=5,
        )
        for x, median in zip((x0, x1), medians):
            ax_rank.annotate(
                f"{median:g}",
                (x, median),
                xytext=(0, -16 if median < 300 else 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )
    ax_rank.set_xlim(-0.35, 3.7)
    ax_rank.set_ylim(450, 175)  # lower rank is better; invert the visual direction
    ax_rank.set_xticks([0, 1, 2.35, 3.35], ["e28\nbefore", "e29\nafter", "e30\nbefore", "e34\nafter"])
    ax_rank.set_ylabel("Rank of best decoder query with IoU ≥ 0.75\n(median; whiskers = P10–P90)")
    ax_rank.set_title("A  COCO-small GTs cross the rank-300 boundary", loc="left")
    ax_rank.legend(loc="lower left", fontsize=8.8)
    ax_rank.spines[["top", "right"]].set_visible(False)
    ax_rank.set_axisbelow(True)

    periods = [("28", "29", "e28→e29"), ("30", "34", "e30→e34")]
    groups = [
        ("coco_and_hybrid", "COCO ∩ hybrid", COLORS["purple"]),
        ("coco_only", "COCO-only", COLORS["gray"]),
        ("hybrid_only", "Hybrid-only", COLORS["teal"]),
    ]
    centers = [0.0, 1.0]
    width = 0.23
    for group_index, (group, label, color) in enumerate(groups):
        values: list[int] = []
        for start, end, _ in periods:
            row = select_row(
                subgroup_rows,
                start_epoch=start,
                end_epoch=end,
                group=group,
                iou_threshold="0.75",
            )
            values.append(int(row["net_gain"]))
        x_values = [center + (group_index - 1) * width for center in centers]
        bars = ax_budget.bar(x_values, values, width=width * 0.9, color=color, label=label)
        for bar, value in zip(bars, values):
            y_offset = 3 if value >= 0 else -4
            va = "bottom" if value >= 0 else "top"
            ax_budget.text(
                bar.get_x() + bar.get_width() / 2,
                value + y_offset,
                f"{value:+d}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
                color=color,
            )
    ax_budget.axhline(0, color=COLORS["dark"], lw=1.0)
    ax_budget.set_xticks(centers, [label for _, _, label in periods])
    ax_budget.set_ylim(-72, 79)
    ax_budget.yaxis.set_major_locator(MultipleLocator(20))
    ax_budget.set_ylabel("Net change in independently covered GTs\n(Decoder Top-300, IoU ≥ 0.75)")
    ax_budget.set_title("B  Top-300 budget shifts between small subgroups", loc="left")
    ax_budget.legend(loc="upper left", fontsize=8.8)
    ax_budget.spines[["top", "right"]].set_visible(False)
    ax_budget.set_axisbelow(True)

    fig.suptitle(
        "Direct carrier of the inverted-U signal: score/rank redistribution at maxDets=300",
        fontsize=15.3,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Fixed val128, EMA. Coverage transitions are independent per GT, so subgroup net changes are evidence of budget redistribution—not a one-to-one query swap.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["dark"],
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.87, bottom=0.18, left=0.09, right=0.985, wspace=0.25)
    output = out_dir / "rank_transition.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def qualitative_paths(checkpoint_dir: Path, family: str, image_number: int) -> list[Path]:
    display_name = f"image ({image_number})"
    native_name = f"{family}_image_{image_number}"
    return [
        checkpoint_dir / family / f"{display_name}_encoder_selection.png",
        checkpoint_dir / family / f"{display_name}_all_640_decoder_queries.png",
        checkpoint_dir / family / f"{display_name}_small_decoder_queries.png",
        checkpoint_dir
        / "native_scale_visualizations"
        / f"{native_name}_native_selected_encoder_geometry.png",
        checkpoint_dir
        / "native_scale_visualizations"
        / f"{native_name}_native_decoder_geometry.png",
    ]


def plot_contact_sheet(
    baseline_root: Path,
    out_dir: Path,
    family: str,
    image_number: int,
    filename: str,
) -> Path:
    column_titles = [
        "Encoder selection",
        "All 640 decoder queries",
        "Hybrid-small decoder",
        "Native selected encoder",
        "Native decoder",
    ]
    all_paths: list[list[Path]] = []
    for epoch in EPOCHS:
        checkpoint = baseline_root / (
            f"SGQS_probe_checkpoint{epoch:04d}_query_diagnostic_hybrid_small_640"
        )
        row_paths = qualitative_paths(checkpoint, family, image_number)
        missing = [str(path) for path in row_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing contact-sheet source(s):\n" + "\n".join(missing))
        all_paths.append(row_paths)

    # Preserve the source aspect ratio instead of forcing the much wider
    # wireless image into the wired sheet's tall grid (which would create
    # large, misleading bands of whitespace between epochs).
    with Image.open(all_paths[0][0]) as first_image:
        source_aspect = first_image.width / first_image.height
    figure_width = 18.0
    target_cell_width = figure_width * 0.96 / 5
    figure_height = 7 * target_cell_width / source_aspect + 1.65
    fig, axes = plt.subplots(7, 5, figsize=(figure_width, figure_height), dpi=160)
    for row_index, (epoch, row_paths) in enumerate(zip(EPOCHS, all_paths)):
        for col_index, path in enumerate(row_paths):
            ax = axes[row_index, col_index]
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#D1D5DB")
                spine.set_linewidth(0.55)
            if row_index == 0:
                ax.set_title(column_titles[col_index], fontsize=12, pad=7)
            if col_index == 0:
                ax.text(
                    -0.035,
                    0.5,
                    f"e{epoch}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=12,
                    fontweight="bold",
                    color=COLORS["dark"],
                )

    fig.suptitle(
        f"SGQS cross-epoch qualitative comparison — {family}/image ({image_number})\n"
        "QUALITATIVE ONLY: 13-image probe, no ground truth; NOT GT-attribution evidence",
        fontsize=17,
        fontweight="bold",
        color=COLORS["dark"],
        y=0.993,
    )
    fig.text(
        0.5,
        0.008,
        "Rows are checkpoints e28–e34. Identical source image and visualization settings are used across each row comparison.",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS["dark"],
    )
    top = 1.0 - 1.18 / figure_height
    bottom = 0.38 / figure_height
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=top,
        bottom=bottom,
        wspace=0.012,
        hspace=0.018,
    )
    output = out_dir / filename
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    outputs = [
        plot_metric_trajectories(experiment, out_dir),
        plot_stage_coverage(experiment, out_dir),
        plot_rank_and_budget(experiment, out_dir),
    ]
    if args.baseline_root is not None:
        baseline_root = args.baseline_root.resolve()
        outputs.extend(
            [
                plot_contact_sheet(
                    baseline_root,
                    out_dir,
                    family="wired",
                    image_number=12,
                    filename="qualitative_wired_image12.png",
                ),
                plot_contact_sheet(
                    baseline_root,
                    out_dir,
                    family="wireless",
                    image_number=5,
                    filename="qualitative_wireless_image5.png",
                ),
            ]
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
