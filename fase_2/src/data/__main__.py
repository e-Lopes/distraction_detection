"""CLI de fundação dos dados da fase 2."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .annotations import (
    annotation_summary,
    intervals_to_frames,
    load_annotations,
    validate_annotations,
)
from .config import load_yaml, repository_path
from .manifest import build_manifest_rows, write_manifest
from .splits import SplitBlock, generate_leave_one_video_out, validate_split_blocks, window_subset
from .windowing import build_windows
from ..preprocessing.missingness import (
    GroupedMissingness,
    WindowMissingness,
    append_grouped_report,
    diagnose_file,
    expand_behavior_labels,
    load_detection_series,
    summarize_by_class,
    summarize_windows,
    write_dataclass_csv,
    write_missingness_report,
)


def _data_config(path: str) -> tuple[dict, dict]:
    config = load_yaml(path)
    dataset = config.get("dataset")
    windowing = config.get("windowing")
    if not isinstance(dataset, dict) or not isinstance(windowing, dict):
        raise TypeError("Configuração exige seções dataset e windowing")
    return dataset, windowing


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def command_manifest(args: argparse.Namespace) -> int:
    dataset, _ = _data_config(args.config)
    root = Path.cwd()
    rows = build_manifest_rows(
        dataset["videos"],
        videos_dir=repository_path(dataset["videos_dir"], root),
        repository_root=root,
    )
    output = repository_path(dataset["manifest"], root)
    write_manifest(rows, output)
    print(f"Manifesto gerado: {output} ({len(rows)} vídeos)")
    return 0


def command_validate_annotations(args: argparse.Namespace) -> int:
    dataset, _ = _data_config(args.config)
    annotation_path = repository_path(dataset["annotations"])
    intervals, load_issues = load_annotations(annotation_path)
    configured_ids = {video["video_id"] for video in dataset["videos"]}
    durations = None
    manifest_path = repository_path(dataset["manifest"])
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            durations = {
                row["video_id"]: float(row["duration_seconds"]) for row in csv.DictReader(stream)
            }
    issues = load_issues + validate_annotations(
        intervals,
        expected_video_ids=configured_ids,
        behavior_classes=set(dataset["classes"]),
        operational_states={"valid", *dataset["operational_states"]},
        video_duration_seconds=durations,
    )
    output_dir = repository_path(args.output_dir)
    issue_dicts = [asdict(issue) for issue in issues]
    _write_csv(
        output_dir / "annotation_validation.csv",
        issue_dicts,
        ["severity", "code", "message", "video_id", "source_row"],
    )
    summary = annotation_summary(intervals)
    _write_csv(
        output_dir / "annotation_duration_summary.csv",
        summary,
        ["video_id", "label", "duration_seconds"],
    )
    errors = sum(issue.severity == "error" for issue in issues)
    report = output_dir / "annotation_validation.md"
    report.write_text(
        "# Validação das anotações\n\n"
        f"- Arquivo: `{dataset['annotations']}`\n"
        f"- Intervalos lidos: {len(intervals)}\n"
        f"- Erros: {errors}\n"
        f"- Avisos: {sum(issue.severity == 'warning' for issue in issues)}\n\n"
        + (
            "A cobertura foi confrontada com o manifesto real.\n"
            if durations
            else "A cobertura final contra a duração depende do manifesto real.\n"
        ),
        encoding="utf-8",
    )
    print(f"Relatório gerado: {report}")
    return 1 if errors else 0


def command_convert_annotations(args: argparse.Namespace) -> int:
    dataset, _ = _data_config(args.config)
    intervals, load_issues = load_annotations(repository_path(dataset["annotations"]))
    if load_issues:
        raise ValueError("Anotações não puderam ser lidas; execute validate-annotations")
    with repository_path(dataset["manifest"]).open(newline="", encoding="utf-8") as stream:
        metadata = {
            row["video_id"]: {
                "fps": float(row["fps"]),
                "num_frames": int(row["num_frames"]),
            }
            for row in csv.DictReader(stream)
        }
    rows = intervals_to_frames(intervals, metadata)
    output = repository_path(args.output)
    _write_csv(
        output,
        rows,
        [
            "video_id",
            "start_frame",
            "end_frame",
            "behavior_label",
            "operational_state",
            "annotation_version",
            "conversion_fps",
            "conversion_note",
        ],
    )
    print(f"Intervalos em frames gerados: {output} ({len(rows)} intervalos)")
    return 0


def command_generate_splits(args: argparse.Namespace) -> int:
    split_config = load_yaml(args.split_config)
    manifest_path = repository_path(args.manifest)
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        video_frames = {row["video_id"]: int(row["num_frames"]) for row in csv.DictReader(stream)}
    blocks = generate_leave_one_video_out(
        video_frames,
        validation_fraction=float(split_config["validation_fraction"]),
        purge_gap_frames=int(split_config["purge_gap_frames"]),
        validation_start_frames=split_config.get("validation_start_frames"),
    )
    errors = validate_split_blocks(blocks, purge_gap_frames=int(split_config["purge_gap_frames"]))
    if errors:
        raise ValueError("Splits inválidos: " + "; ".join(errors))
    rows = [asdict(block) for block in blocks]
    output = repository_path(args.output)
    _write_csv(output, rows, ["fold", "subset", "video_id", "start_frame", "end_frame"])
    print(f"Splits gerados: {output} ({len(rows)} blocos)")
    return 0


def _load_frame_intervals(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def command_diagnose(args: argparse.Namespace) -> int:
    dataset, windowing = _data_config(args.config)
    with repository_path(dataset["manifest"]).open(newline="", encoding="utf-8") as stream:
        metadata = {row["video_id"]: row for row in csv.DictReader(stream)}
    frame_intervals = _load_frame_intervals(repository_path(args.frame_intervals))
    labels_by_video: dict[str, list[str | None]] = {
        video_id: [None] * int(row["num_frames"]) for video_id, row in metadata.items()
    }
    frame_counts: Counter[tuple[str, str, str]] = Counter()
    for interval in frame_intervals:
        video_id = interval["video_id"]
        start, end = int(interval["start_frame"]), int(interval["end_frame"])
        behavior = interval["behavior_label"] or None
        operational = interval["operational_state"]
        label = behavior or operational
        category = "behavior" if behavior else "operational"
        frame_counts[(video_id, category, label)] += end - start + 1
        labels_by_video[video_id][start : end + 1] = [behavior] * (end - start + 1)

    dataset_frames = sum(int(row["num_frames"]) for row in metadata.values())
    frame_rows: list[dict[str, object]] = []
    for (video_id, category, label), count in sorted(frame_counts.items()):
        video_frames = int(metadata[video_id]["num_frames"])
        fps = float(metadata[video_id]["fps"])
        frame_rows.append(
            {
                "video_id": video_id,
                "category": category,
                "label": label,
                "num_frames": count,
                "duration_seconds_estimated": count / fps,
                "percentage_video": 100 * count / video_frames,
                "percentage_dataset": 100 * count / dataset_frames,
            }
        )

    window_counts: Counter[tuple[str, int, str]] = Counter()
    total_windows: Counter[tuple[str, int]] = Counter()
    all_windows = []
    for size in windowing["sizes_frames"]:
        windows = build_windows(
            labels_by_video,
            size_frames=int(size),
            stride_frames=int(windowing["stride_frames"]),
            behavior_classes=set(dataset["classes"]),
            minimum_proportion=float(windowing["minimum_target_proportion"]),
        )
        for window in windows:
            all_windows.append(window)
            window_counts[(window.video_id, int(size), window.label)] += 1
            total_windows[(window.video_id, int(size))] += 1
    window_rows: list[dict[str, object]] = []
    for (video_id, size, label), count in sorted(window_counts.items()):
        total = total_windows[(video_id, size)]
        window_rows.append(
            {
                "video_id": video_id,
                "window_size_frames": size,
                "label": label,
                "num_windows": count,
                "percentage": 100 * count / total,
                "mixed_windows_in_group": window_counts[(video_id, size, "mixed")],
            }
        )
    output_dir = repository_path(args.output_dir)
    _write_csv(
        output_dir / "frame_distribution.csv",
        frame_rows,
        [
            "video_id",
            "category",
            "label",
            "num_frames",
            "duration_seconds_estimated",
            "percentage_video",
            "percentage_dataset",
        ],
    )
    _write_csv(
        output_dir / "window_distribution.csv",
        window_rows,
        [
            "video_id",
            "window_size_frames",
            "label",
            "num_windows",
            "percentage",
            "mixed_windows_in_group",
        ],
    )
    splits_path = repository_path(args.splits)
    if splits_path.exists():
        with splits_path.open(newline="", encoding="utf-8") as stream:
            blocks = [
                SplitBlock(
                    fold=int(row["fold"]),
                    subset=row["subset"],
                    video_id=row["video_id"],
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                )
                for row in csv.DictReader(stream)
            ]
        split_counts: Counter[tuple[int, str, int, str]] = Counter()
        for window in all_windows:
            for fold in sorted({block.fold for block in blocks}):
                subset = window_subset(
                    video_id=window.video_id,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    blocks=blocks,
                    fold=fold,
                )
                if subset:
                    split_counts[(fold, subset, window.size_frames, window.label)] += 1
        split_rows = [
            {
                "fold": fold,
                "subset": subset,
                "window_size_frames": size,
                "label": label,
                "num_windows": count,
            }
            for (fold, subset, size, label), count in sorted(split_counts.items())
        ]
        _write_csv(
            output_dir / "split_window_distribution.csv",
            split_rows,
            ["fold", "subset", "window_size_frames", "label", "num_windows"],
        )
    print(f"Diagnósticos gerados em: {output_dir}")
    return 0


def command_diagnose_missingness(args: argparse.Namespace) -> int:
    dataset, windowing = _data_config(args.config)
    input_dir = repository_path(args.input_dir)
    paths = sorted(input_dir.glob("video_*.csv"))
    if not paths:
        raise FileNotFoundError(f"Nenhuma série video_*.csv encontrada em {input_dir}")
    summaries = [
        diagnose_file(path, short_gap_max_frames=args.short_gap_max_frames) for path in paths
    ]
    output_dir = repository_path(args.output_dir)
    write_missingness_report(
        summaries,
        csv_path=output_dir / "facial_missingness.csv",
        markdown_path=output_dir / "facial_missingness.md",
        short_gap_max_frames=args.short_gap_max_frames,
    )
    detected_by_video = dict(load_detection_series(path) for path in paths)
    intervals = _load_frame_intervals(repository_path(args.frame_intervals))
    labels_by_video = expand_behavior_labels(
        {video_id: len(values) for video_id, values in detected_by_video.items()}, intervals
    )
    class_rows = summarize_by_class(
        labels_by_video,
        detected_by_video,
        behavior_classes=set(dataset["classes"]),
    )
    with repository_path(args.splits).open(newline="", encoding="utf-8") as stream:
        blocks = [
            SplitBlock(
                fold=int(row["fold"]),
                subset=row["subset"],
                video_id=row["video_id"],
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
            )
            for row in csv.DictReader(stream)
        ]
    video_window_rows, fold_window_rows = summarize_windows(
        labels_by_video,
        detected_by_video,
        sizes_frames=[int(size) for size in windowing["sizes_frames"]],
        stride_frames=int(windowing["stride_frames"]),
        behavior_classes=set(dataset["classes"]),
        minimum_proportion=float(windowing["minimum_target_proportion"]),
        split_blocks=blocks,
    )
    write_dataclass_csv(
        output_dir / "facial_missingness_by_class.csv",
        class_rows,
        list(GroupedMissingness.__dataclass_fields__),
    )
    window_fields = list(WindowMissingness.__dataclass_fields__)
    write_dataclass_csv(
        output_dir / "facial_missingness_by_window.csv", video_window_rows, window_fields
    )
    write_dataclass_csv(
        output_dir / "facial_missingness_by_fold.csv", fold_window_rows, window_fields
    )
    append_grouped_report(
        output_dir / "facial_missingness.md", class_rows, video_window_rows
    )
    print(f"Diagnóstico de missingness gerado em: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="Gera manifesto dos vídeos")
    manifest.add_argument("--config", default="fase_2/configs/data/base.yaml")
    manifest.set_defaults(func=command_manifest)

    annotations = commands.add_parser("validate-annotations", help="Valida intervalos manuais")
    annotations.add_argument("--config", default="fase_2/configs/data/base.yaml")
    annotations.add_argument("--output-dir", default="fase_2/outputs/metrics")
    annotations.set_defaults(func=command_validate_annotations)

    convert = commands.add_parser("convert-annotations", help="Converte segundos em frames")
    convert.add_argument("--config", default="fase_2/configs/data/base.yaml")
    convert.add_argument("--output", default="fase_2/data/manifests/annotation_frame_intervals.csv")
    convert.set_defaults(func=command_convert_annotations)

    splits = commands.add_parser("generate-splits", help="Gera folds temporais")
    splits.add_argument("--manifest", default="fase_2/data/manifests/videos.csv")
    splits.add_argument("--split-config", default="fase_2/configs/splits/leave_one_video_out.yaml")
    splits.add_argument("--output", default="fase_2/data/manifests/temporal_splits.csv")
    splits.set_defaults(func=command_generate_splits)

    diagnose = commands.add_parser("diagnose", help="Gera distribuições de frames e janelas")
    diagnose.add_argument("--config", default="fase_2/configs/data/base.yaml")
    diagnose.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    diagnose.add_argument("--output-dir", default="fase_2/outputs/metrics")
    diagnose.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    diagnose.set_defaults(func=command_diagnose)

    missingness = commands.add_parser(
        "diagnose-missingness", help="Audita séries faciais e gaps de detecção"
    )
    missingness.add_argument(
        "--input-dir", default="fase_2/data/interim/legacy_extraction"
    )
    missingness.add_argument("--config", default="fase_2/configs/data/base.yaml")
    missingness.add_argument(
        "--frame-intervals",
        default="fase_2/data/manifests/annotation_frame_intervals.csv",
    )
    missingness.add_argument("--splits", default="fase_2/data/manifests/temporal_splits.csv")
    missingness.add_argument("--output-dir", default="fase_2/outputs/metrics")
    missingness.add_argument("--short-gap-max-frames", type=int, default=15)
    missingness.set_defaults(func=command_diagnose_missingness)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
