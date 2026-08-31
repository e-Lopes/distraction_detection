"""Compara representações robustas à pose usando somente treino/validação da G4.6."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

from ..data.splits import SplitBlock, window_subset
from ..data.windowing import build_windows
from ..features.pose_robustness import (
    apply_pose_correction,
    fit_pose_correction,
    pose_signal_correlations,
    rolling_perclos,
)
from ..preprocessing.missingness import expand_behavior_labels
from .dummy_baseline import CLASSES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPRESENTATIONS = {
    "baseline": ("ear", "mar", "pitch", "yaw", "roll"),
    "geometry_normalized": (
        "ear",
        "ear_left",
        "ear_right",
        "ear_asymmetry",
        "eye_quality_left",
        "eye_quality_right",
        "mar",
        "pitch",
        "yaw",
        "roll",
    ),
    "pose_corrected": (
        "ear_pose_corrected",
        "ear_left",
        "ear_right",
        "ear_asymmetry",
        "eye_quality_left",
        "eye_quality_right",
        "mar",
        "pitch",
        "yaw",
        "roll",
        "perclos_30s",
        "perclos_60s",
        "coverage_30s",
        "coverage_60s",
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para gravar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _training_rows(
    series: Mapping[str, pd.DataFrame], blocks: Sequence[SplitBlock], fold: int
) -> pd.DataFrame:
    selected = []
    for block in blocks:
        if block.fold == fold and block.subset == "train":
            selected.append(series[block.video_id].iloc[block.start_frame : block.end_frame + 1])
    if not selected:
        raise ValueError(f"Fold {fold} sem frames de treino")
    return pd.concat(selected, ignore_index=True)


def _window_features(
    frame: pd.DataFrame, start: int, end: int, columns: Sequence[str]
) -> list[float]:
    segment = frame.iloc[start : end + 1]
    output = []
    for column in columns:
        values = pd.to_numeric(segment[column], errors="coerce")
        output.extend(
            (
                float(values.mean()) if values.notna().any() else 0.0,
                float(values.std(ddof=0)) if values.notna().any() else 0.0,
            )
        )
    output.append(float(segment["face_detected"].eq(1).mean()))
    return output


def _make_dataset(
    enriched: Mapping[str, pd.DataFrame],
    labels: Mapping[str, Sequence[str | None]],
    blocks: Sequence[SplitBlock],
    *,
    fold: int,
    representation: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[object]]]:
    windows = build_windows(
        labels,
        size_frames=60,
        stride_frames=15,
        behavior_classes=set(CLASSES),
        minimum_proportion=0.60,
    )
    result = {}
    for subset in ("train", "validation"):
        chosen = [
            window
            for window in windows
            if window.label in CLASSES
            and window_subset(
                video_id=window.video_id,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                blocks=blocks,
                fold=fold,
            )
            == subset
        ]
        if not chosen:
            raise ValueError(f"Fold {fold}/{subset} sem janelas")
        values = np.asarray(
            [
                _window_features(
                    enriched[w.video_id],
                    w.start_frame,
                    w.end_frame,
                    REPRESENTATIONS[representation],
                )
                for w in chosen
            ]
        )
        if not np.isfinite(values).all():
            raise FloatingPointError("Features G4.6 contêm NaN/Inf")
        result[subset] = (values, np.asarray([w.label for w in chosen]), chosen)
    return result


def _false_fatigue_episodes(predicted: np.ndarray) -> int:
    active = predicted == "fatigue"
    return int(active[0]) + int(np.sum(active[1:] & ~active[:-1])) if len(active) else 0


def _plot_diagnostics(video_id: str, frame: pd.DataFrame, output: Path) -> None:
    time = pd.to_numeric(frame["timestamp_seconds"], errors="coerce") / 60
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    colors = {"alert": "#2ca02c", "fatigue": "#d62728", "distraction": "#ffbf00"}
    labels = frame["behavior_label"].fillna("unavailable").to_numpy()
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    for start, end in zip(starts, ends, strict=True):
        color = colors.get(str(labels[start]))
        if color and end > start:
            for axis in axes:
                axis.axvspan(
                    float(time.iloc[start]), float(time.iloc[end - 1]), color=color, alpha=0.055
                )
    axes[0].plot(time, frame["ear_left"], alpha=0.35, label="EAR esquerdo")
    axes[0].plot(time, frame["ear_right"], alpha=0.35, label="EAR direito")
    axes[0].plot(time, frame["ear_pose_corrected"], linewidth=1, label="EAR corrigido")
    axes[0].legend()
    axes[0].set_ylabel("EAR / razão")
    axes[0].grid(alpha=0.2)
    axes[1].plot(time, frame["pitch"], label="pitch")
    axes[1].plot(time, frame["yaw"], label="yaw")
    axes[1].legend()
    axes[1].set_ylabel("Pose (graus)")
    axes[1].grid(alpha=0.2)
    axes[2].plot(time, frame["perclos_30s"], label="PERCLOS 30 s")
    axes[2].plot(time, frame["perclos_60s"], label="PERCLOS 60 s")
    axes[2].axhline(7.5, color="goldenrod", linestyle="--")
    axes[2].axhline(15, color="red", linestyle="--")
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("PERCLOS (%)")
    axes[2].set_xlabel("Tempo (min)")
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    figure.suptitle(f"Robustez à pose — {video_id}")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)

    scatter, scatter_axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, pose in enumerate(("pitch", "yaw")):
        scatter_axes[row, 0].scatter(frame[pose], frame["ear"], s=2, alpha=0.12)
        scatter_axes[row, 0].set_title(f"EAR bruto × {pose}")
        scatter_axes[row, 1].scatter(frame[pose], frame["ear_pose_corrected"], s=2, alpha=0.12)
        scatter_axes[row, 1].set_title(f"EAR corrigido × {pose}")
        for axis in scatter_axes[row]:
            axis.set_xlabel(f"{pose} (graus)")
            axis.grid(alpha=0.2)
        scatter_axes[row, 0].set_ylabel("EAR")
        scatter_axes[row, 1].set_ylabel("EAR corrigido")
    scatter.suptitle(f"Dependência pose–sinal ocular — {video_id} (validação)")
    scatter.tight_layout()
    scatter.savefig(output.with_name(output.stem + "_scatter.png"), dpi=160, bbox_inches="tight")
    plt.close(scatter)


def _validation_hours(windows: Sequence[object], enriched: Mapping[str, pd.DataFrame]) -> float:
    seconds = 0.0
    for video_id in sorted({window.video_id for window in windows}):
        selected = [window for window in windows if window.video_id == video_id]
        start = min(window.start_frame for window in selected)
        end = max(window.end_frame for window in selected)
        timestamps = pd.to_numeric(enriched[video_id]["timestamp_seconds"], errors="raise")
        seconds += float(timestamps.iloc[end] - timestamps.iloc[start])
    return seconds / 3600


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="fase_2/data/interim/geometry_v2")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics/G46")
    parser.add_argument("--figure-dir", default="fase_2/outputs/figures/G46")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    folds = args.fold or [1, 2, 3, 4]
    matrix = [
        {"fold": fold, "representation": representation, "seed": 42, "subset": "validation"}
        for fold in folds
        for representation in REPRESENTATIONS
    ]
    print(json.dumps({"planned_runs": len(matrix), "test_loaded": False, "runs": matrix}, indent=2))
    if args.dry_run:
        return 0
    paths = sorted(Path(args.input_dir).glob("video_*.csv"))
    if len(paths) != 4:
        raise FileNotFoundError("G4.6 exige quatro séries geometry_v2 completas")
    series = {path.stem: pd.read_csv(path) for path in paths}
    intervals = _read_csv(Path("fase_2/data/manifests/annotation_frame_intervals.csv"))
    labels = expand_behavior_labels({key: len(value) for key, value in series.items()}, intervals)
    for video_id, frame in series.items():
        frame["behavior_label"] = labels[video_id]
    blocks = [
        SplitBlock(
            int(r["fold"]), r["subset"], r["video_id"], int(r["start_frame"]), int(r["end_frame"])
        )
        for r in _read_csv(Path("fase_2/data/manifests/temporal_splits.csv"))
    ]
    metrics = []
    correlations = []
    plotted = set()
    for fold in folds:
        correction = fit_pose_correction(_training_rows(series, blocks, fold))
        enriched = {}
        for video_id, frame in series.items():
            value = apply_pose_correction(frame, correction)
            value = rolling_perclos(value, window_seconds=30, minimum_coverage=0.5)
            value = rolling_perclos(value, window_seconds=60, minimum_coverage=0.5)
            enriched[video_id] = value
        validation_blocks = [b for b in blocks if b.fold == fold and b.subset == "validation"]
        validation_frames = pd.concat(
            [enriched[b.video_id].iloc[b.start_frame : b.end_frame + 1] for b in validation_blocks],
            ignore_index=True,
        )
        before = pose_signal_correlations(validation_frames, "ear")
        after = pose_signal_correlations(validation_frames, "ear_pose_corrected")
        correlations.append(
            {
                "fold": fold,
                "training_rows": correction.training_rows,
                "training_videos": ";".join(correction.training_videos),
                "ear_pitch_before": before["pitch"],
                "ear_pitch_after": after["pitch"],
                "ear_yaw_before": before["yaw"],
                "ear_yaw_after": after["yaw"],
            }
        )
        for representation in REPRESENTATIONS:
            dataset = _make_dataset(
                enriched, labels, blocks, fold=fold, representation=representation
            )
            x_train, y_train, _ = dataset["train"]
            x_val, y_val, windows = dataset["validation"]
            model = RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
            ).fit(x_train, y_train)
            predicted = model.predict(x_val)
            macro = f1_score(y_val, predicted, labels=CLASSES, average="macro", zero_division=0)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, predicted, labels=CLASSES, zero_division=0
            )
            duration_hours = _validation_hours(windows, enriched)
            row = {
                "fold": fold,
                "representation": representation,
                "seed": 42,
                "subset": "validation",
                "num_windows": len(y_val),
                "macro_f1": macro,
                "balanced_accuracy": balanced_accuracy_score(y_val, predicted),
                "false_fatigue_episodes_per_hour": _false_fatigue_episodes(predicted)
                / max(duration_hours, 1e-9),
            }
            for index, label in enumerate(CLASSES):
                row.update(
                    {
                        f"{label}_precision": precision[index],
                        f"{label}_recall": recall[index],
                        f"{label}_f1": f1[index],
                    }
                )
            metrics.append(row)
        for block in validation_blocks:
            if block.video_id not in plotted:
                _plot_diagnostics(
                    block.video_id,
                    enriched[block.video_id].iloc[block.start_frame : block.end_frame + 1],
                    Path(args.figure_dir) / f"{block.video_id}_pose_robustness.png",
                )
                plotted.add(block.video_id)
        print(
            f"fold {fold} concluído | correlação EAR×pitch {before['pitch']:.3f} -> {after['pitch']:.3f}",
            flush=True,
        )
    _write_csv(Path(args.output_dir) / "g46_runs.csv", metrics)
    _write_csv(Path(args.output_dir) / "g46_pose_correlations.csv", correlations)
    print("G4.6 concluída sem carregar o teste externo", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
