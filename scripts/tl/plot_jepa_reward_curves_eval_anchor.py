#!/usr/bin/env python3
"""Plot averaged JEPA reward curves with a final eval-return anchor.

The script intentionally reads only local text logs and completed rows from
results.csv, so it does not depend on W&B sync state.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    pattern: re.Pattern[str]


EXPERIMENTS = [
    ExperimentSpec(
        "JEPA residual",
        re.compile(r"^dino_jamesbond_jepa_full_block3_weight_1_residual_\d{8}_\d{6}$"),
    ),
    ExperimentSpec(
        "JEPA residual + straight",
        re.compile(r"^dino_jamesbond_jepa_full_block3_weight_1_residual_straight_0p1_\d{8}_\d{6}$"),
    ),
    ExperimentSpec(
        "JEPA residual + SIGReg",
        re.compile(r"^dino_jamesbond_jepa_full_block3_weight_1_residual_sigreg_0p1_\d{8}_\d{6}$"),
    ),
    ExperimentSpec(
        "JEPA residual + straight + SIGReg",
        re.compile(r"^dino_jamesbond_jepa_full_block3_weight_1_residual_straight_0p1_sigreg_0p1_\d{8}_\d{6}$"),
    ),
    ExperimentSpec(
        "Straight",
        re.compile(r"^dino_jamesbond_jepa_full_block3_straight_0p1_\d{8}_\d{6}$"),
    ),
    ExperimentSpec(
        "Straight + SIGReg",
        re.compile(r"^dino_jamesbond_jepa_full_block3_straight_0p1_sigreg_0p1_\d{8}_\d{6}$"),
    ),
    ExperimentSpec(
        "SIGReg",
        re.compile(r"^dino_jamesbond_jepa_full_block3_sigreg_0p1_\d{8}_\d{6}$"),
    ),
]

STEP_RE = re.compile(r"Training:\s+\d+%.*?\|\s*(\d+)/(\d+)")
METRIC_RE = re.compile(r"train/([A-Za-z0-9_]+)=([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/jepa_comparison"))
    parser.add_argument("--metric", default="raw_reward")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/jepa_comparison/plots"))
    parser.add_argument("--output-name", default="jepa_reward_curves_eval_anchor")
    parser.add_argument("--eval-anchor-step", type=int, default=100_000)
    parser.add_argument("--eval-anchor-divisor", type=float, default=6.0)
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help="Drop earlier steps from the plot and CSV.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional maximum step for comparing only a shared training range.",
    )
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=50.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Rolling mean window over logged points. Use 1 to disable smoothing.",
    )
    return parser.parse_args()


def newest_matching_run(log_root: Path, spec: ExperimentSpec) -> Path | None:
    matches = [path for path in log_root.iterdir() if path.is_dir() and spec.pattern.match(path.name)]
    return sorted(matches)[-1] if matches else None


def completed_seed_logs(run_root: Path) -> list[Path]:
    results_path = run_root / "results.csv"
    if not results_path.exists():
        return []

    completed: list[Path] = []
    with results_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "0":
                continue
            log_file = Path(row.get("log_file", ""))
            if not log_file.is_absolute():
                log_file = Path.cwd() / log_file
            if log_file.exists():
                completed.append(log_file)
    return sorted(completed)


def completed_eval_mean(run_root: Path) -> float | None:
    results_path = run_root / "results.csv"
    if not results_path.exists():
        return None

    values: list[float] = []
    with results_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "0":
                continue
            value = row.get("eval_return_mean", "")
            if value:
                values.append(float(value))
    return sum(values) / len(values) if values else None


def parse_metric_points(log_file: Path, metric: str) -> dict[int, float]:
    points: dict[int, float] = {}
    text = log_file.read_text(errors="replace")
    for chunk in re.split(r"[\r\n]+", text):
        step_match = STEP_RE.search(chunk)
        if step_match is None:
            continue
        metrics = dict(METRIC_RE.findall(chunk))
        if metric not in metrics:
            continue
        points[int(step_match.group(1))] = float(metrics[metric])
    return points


def collect_curves(log_root: Path, metric: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for spec in EXPERIMENTS:
        run_root = newest_matching_run(log_root, spec)
        if run_root is None:
            print(f"missing: {spec.name}")
            continue

        seed_logs = completed_seed_logs(run_root)
        print(f"{spec.name}: {run_root} ({len(seed_logs)} completed seeds)")
        for log_file in seed_logs:
            seed_match = re.search(r"seed_(\d+)\.log$", log_file.name)
            seed = int(seed_match.group(1)) if seed_match else -1
            for step, value in parse_metric_points(log_file, metric).items():
                rows.append({
                    "experiment": spec.name,
                    "run_root": str(run_root),
                    "seed": seed,
                    "step": step,
                    "metric": metric,
                    "value": value,
                })
    if not rows:
        raise RuntimeError(f"No completed metric points found for metric={metric!r}.")
    return pd.DataFrame(rows)


def add_eval_anchor(
    summary: pd.DataFrame,
    log_root: Path,
    step: int,
    divisor: float,
) -> pd.DataFrame:
    rows = []
    for spec in EXPERIMENTS:
        run_root = newest_matching_run(log_root, spec)
        if run_root is None:
            continue
        eval_mean = completed_eval_mean(run_root)
        if eval_mean is None:
            continue
        rows.append({
            "experiment": spec.name,
            "step": step,
            "metric": "eval_return_mean_div_lives",
            "n": len(completed_seed_logs(run_root)),
            "mean": eval_mean / divisor,
            "std": 0.0,
        })
    if not rows:
        return summary
    summary = summary[summary["step"] != step]
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)


def summarize(curves: pd.DataFrame) -> pd.DataFrame:
    summary = (
        curves
        .groupby(["experiment", "step", "metric"], as_index=False)
        .agg(
            n=("value", "count"),
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .fillna({"std": 0.0})
    )
    return summary.sort_values(["experiment", "step"])


def smooth_summary(summary: pd.DataFrame, window: int) -> pd.DataFrame:
    summary = summary.copy()
    if window <= 1:
        summary["mean_smooth"] = summary["mean"]
        summary["std_smooth"] = summary["std"]
        return summary
    summary["mean_smooth"] = (
        summary
        .groupby("experiment")["mean"]
        .transform(lambda values: values.rolling(window=window, min_periods=1, center=True).mean())
    )
    summary["std_smooth"] = (
        summary
        .groupby("experiment")["std"]
        .transform(lambda values: values.rolling(window=window, min_periods=1, center=True).mean())
    )
    return summary


def restore_eval_anchor(summary: pd.DataFrame, step: int) -> pd.DataFrame:
    summary = summary.copy()
    mask = summary["step"] == step
    summary.loc[mask, "mean_smooth"] = summary.loc[mask, "mean"]
    summary.loc[mask, "std_smooth"] = summary.loc[mask, "std"]
    return summary


def plot_summary(
    summary: pd.DataFrame,
    output_path: Path,
    metric: str,
    ymin: float | None,
    ymax: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for spec in EXPERIMENTS:
        data = summary[summary["experiment"] == spec.name].sort_values("step")
        if data.empty:
            continue
        steps = data["step"].to_numpy()
        mean_column = "mean_smooth" if "mean_smooth" in data else "mean"
        mean = data[mean_column].to_numpy()
        ax.plot(steps, mean, label=f"{spec.name} (n={int(data['n'].max())})", linewidth=2)

    ax.set_title(f"Average {metric} Curve")
    ax.set_xlabel("Training frames")
    ax.set_ylabel(metric)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    curves = collect_curves(args.log_root, args.metric)
    curves = curves[curves["step"] >= args.min_step]
    if args.max_step is not None:
        curves = curves[curves["step"] <= args.max_step]
    summary = summarize(curves)
    summary = add_eval_anchor(
        summary,
        args.log_root,
        args.eval_anchor_step,
        args.eval_anchor_divisor,
    )
    summary = smooth_summary(summary, args.smooth_window)
    summary = restore_eval_anchor(summary, args.eval_anchor_step)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_name}_{args.metric}.csv"
    png_path = args.output_dir / f"{args.output_name}_{args.metric}.png"
    summary.to_csv(csv_path, index=False)
    plot_summary(summary, png_path, args.metric, args.ymin, args.ymax)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
