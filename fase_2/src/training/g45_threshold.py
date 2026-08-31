"""G4.5A: ajuste cross-fit do threshold de Fatigue por sessão de validação."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from ..data.config import load_yaml, repository_path
from .dummy_baseline import CLASSES


PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in CLASSES)
REQUIRED_COLUMNS = {
    "video_id",
    "start_frame",
    "end_frame",
    "actual",
    *PROBABILITY_COLUMNS,
}


def threshold_grid(minimum: float, maximum: float, step: float) -> tuple[float, ...]:
    if not 0 < minimum <= maximum <= 1 or step <= 0:
        raise ValueError("Grade de threshold inválida")
    count = int(round((maximum - minimum) / step))
    values = tuple(round(minimum + index * step, 10) for index in range(count + 1))
    if values[-1] != round(maximum, 10):
        raise ValueError("A grade não termina no máximo configurado")
    return values


def _renormalize(probabilities: Sequence[float]) -> tuple[float, ...]:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(CLASSES),) or not np.isfinite(values).all():
        raise ValueError("Probabilidades inválidas")
    if np.any(values < 0):
        raise ValueError("Probabilidade negativa")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Soma de probabilidades deve ser positiva")
    return tuple(float(value) for value in values / total)


def load_validation_predictions(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Carrega somente validação e falha explicitamente diante de qualquer artefato de teste."""

    rows: list[dict[str, object]] = []
    if not paths:
        raise ValueError("Nenhum arquivo de validação fornecido")
    for path in paths:
        if not path.name.endswith("__validation.csv") or "__test" in path.name:
            raise ValueError(f"G4.5A aceita somente predições de validação: {path}")
        fold_match = re.search(r"__fold_(\d+)__", path.name)
        if not fold_match:
            raise ValueError(f"Fold ausente no nome do artefato: {path.name}")
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
                missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
                raise ValueError(f"Colunas ausentes em {path.name}: {missing}")
            for source_row in reader:
                probabilities = _renormalize(
                    [float(source_row[column]) for column in PROBABILITY_COLUMNS]
                )
                rows.append(
                    {
                        "run_id": path.stem.removesuffix("__validation"),
                        "fold": int(fold_match.group(1)),
                        "video_id": source_row["video_id"],
                        "start_frame": int(source_row["start_frame"]),
                        "end_frame": int(source_row["end_frame"]),
                        "actual": source_row["actual"],
                        **{
                            column: probabilities[index]
                            for index, column in enumerate(PROBABILITY_COLUMNS)
                        },
                    }
                )
    return rows


def window_id(row: Mapping[str, object]) -> str:
    return f"{row['video_id']}:{row['start_frame']}:{row['end_frame']}"


def deduplicate_predictions(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Calcula a média das probabilidades para janelas repetidas e renormaliza."""

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[window_id(row)].append(row)
    output: list[dict[str, object]] = []
    for identifier in sorted(grouped):
        selected = grouped[identifier]
        actual = {str(row["actual"]) for row in selected}
        if len(actual) != 1:
            raise ValueError(f"Rótulos divergentes para {identifier}: {sorted(actual)}")
        probabilities = _renormalize(
            [
                float(np.mean([float(row[column]) for row in selected]))
                for column in PROBABILITY_COLUMNS
            ]
        )
        first = selected[0]
        output.append(
            {
                "window_id": identifier,
                "video_id": first["video_id"],
                "start_frame": int(first["start_frame"]),
                "end_frame": int(first["end_frame"]),
                "actual": next(iter(actual)),
                "source_count": len(selected),
                "source_folds": json.dumps(
                    sorted({int(row["fold"]) for row in selected}), separators=(",", ":")
                ),
                **{
                    column: probabilities[index]
                    for index, column in enumerate(PROBABILITY_COLUMNS)
                },
            }
        )
    return output


def predict_with_threshold(row: Mapping[str, object], threshold: float) -> str:
    if float(row["prob_fatigue"]) >= threshold:
        return "fatigue"
    return (
        "alert"
        if float(row["prob_alert"]) >= float(row["prob_distraction"])
        else "distraction"
    )


def predict_argmax(row: Mapping[str, object]) -> str:
    probabilities = [float(row[column]) for column in PROBABILITY_COLUMNS]
    return CLASSES[int(np.argmax(probabilities))]


def _video_equal_metrics(
    rows: Sequence[Mapping[str, object]], threshold: float
) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    if not grouped:
        raise ValueError("Nenhuma sessão para avaliar threshold")
    macro_values: list[float] = []
    fatigue_f1_values: list[float] = []
    fatigue_precision_values: list[float] = []
    for selected in grouped.values():
        actual = [str(row["actual"]) for row in selected]
        predicted = [predict_with_threshold(row, threshold) for row in selected]
        macro_values.append(
            float(f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0))
        )
        precision, _, f1_values, _ = precision_recall_fscore_support(
            actual, predicted, labels=CLASSES, zero_division=0
        )
        fatigue_index = CLASSES.index("fatigue")
        fatigue_f1_values.append(float(f1_values[fatigue_index]))
        fatigue_precision_values.append(float(precision[fatigue_index]))
    return {
        "equal_video_mean_macro_f1": float(np.mean(macro_values)),
        "equal_video_mean_fatigue_f1": float(np.mean(fatigue_f1_values)),
        "equal_video_mean_fatigue_precision": float(np.mean(fatigue_precision_values)),
    }


def select_threshold_for_video(
    deduplicated: Sequence[Mapping[str, object]],
    *,
    target_video: str,
    grid: Sequence[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    calibration = [row for row in deduplicated if row["video_id"] != target_video]
    held_out = [row for row in deduplicated if row["video_id"] == target_video]
    if not calibration or not held_out:
        raise ValueError(f"Cross-fit impossível para {target_video}")
    calibration_videos = sorted({str(row["video_id"]) for row in calibration})
    if target_video in calibration_videos:
        raise AssertionError("Vídeo-alvo entrou na calibração")
    curve: list[dict[str, object]] = []
    for threshold in grid:
        metrics = _video_equal_metrics(calibration, float(threshold))
        curve.append(
            {
                "target_video": target_video,
                "calibration_videos": json.dumps(calibration_videos, separators=(",", ":")),
                "threshold": float(threshold),
                **metrics,
            }
        )

    def rank(row: Mapping[str, object]) -> tuple[float, ...]:
        return (
            round(float(row["equal_video_mean_macro_f1"]), 12),
            round(float(row["equal_video_mean_fatigue_f1"]), 12),
            round(float(row["equal_video_mean_fatigue_precision"]), 12),
            -abs(float(row["threshold"]) - 1 / 3),
            float(row["threshold"]),
        )

    best = max(curve, key=rank)
    selected_threshold = float(best["threshold"])
    held_out_metrics = _video_equal_metrics(held_out, selected_threshold)
    decision = {
        **best,
        "held_out_num_windows": len(held_out),
        **{f"held_out_{key}": value for key, value in held_out_metrics.items()},
    }
    return decision, curve


def crossfit_thresholds(
    deduplicated: Sequence[Mapping[str, object]], grid: Sequence[float]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    decisions: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for video_id in sorted({str(row["video_id"]) for row in deduplicated}):
        decision, video_curve = select_threshold_for_video(
            deduplicated, target_video=video_id, grid=grid
        )
        decisions.append(decision)
        curves.extend(video_curve)
    return decisions, curves


def reconstruct_predictions(
    rows: Sequence[Mapping[str, object]], decisions: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    thresholds = {
        str(row["target_video"]): float(row["threshold"]) for row in decisions
    }
    output: list[dict[str, object]] = []
    for row in rows:
        video_id = str(row["video_id"])
        if video_id not in thresholds:
            raise ValueError(f"Threshold ausente para {video_id}")
        threshold = thresholds[video_id]
        output.append(
            {
                **dict(row),
                "window_id": window_id(row),
                "threshold_fatigue": threshold,
                "predicted_argmax": predict_argmax(row),
                "predicted_crossfit": predict_with_threshold(row, threshold),
            }
        )
    return output


def summarize_reconstructed(
    rows: Sequence[Mapping[str, object]], *, model_key: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []
    for fold in sorted({int(row["fold"]) for row in rows}):
        selected = [row for row in rows if int(row["fold"]) == fold]
        actual = [str(row["actual"]) for row in selected]
        for method, column in (
            ("argmax", "predicted_argmax"),
            ("crossfit_threshold", "predicted_crossfit"),
        ):
            predicted = [str(row[column]) for row in selected]
            summary.append(
                {
                    "model_key": model_key,
                    "fold": fold,
                    "method": method,
                    "num_windows": len(selected),
                    "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                    "macro_f1_all_classes": float(
                        f1_score(
                            actual,
                            predicted,
                            labels=CLASSES,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
            precision, recall, f1_values, support = precision_recall_fscore_support(
                actual, predicted, labels=CLASSES, zero_division=0
            )
            for index, label in enumerate(CLASSES):
                per_class.append(
                    {
                        "model_key": model_key,
                        "fold": fold,
                        "method": method,
                        "label": label,
                        "precision": float(precision[index]),
                        "recall": float(recall[index]),
                        "f1": float(f1_values[index]),
                        "support": int(support[index]),
                    }
                )
    return summary, per_class


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Nenhuma linha para gravar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _report(
    path: Path,
    summary: Sequence[Mapping[str, object]],
    per_class: Sequence[Mapping[str, object]],
    decisions: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# G4.5A — threshold de Fatigue com cross-fit por sessão",
        "",
        "Somente predições de validação da G4 foram carregadas. Cada vídeo-alvo foi "
        "excluído da "
        "calibração de seu threshold; janelas repetidas foram deduplicadas por média das "
        "probabilidades e renormalização.",
        "",
        "## Thresholds escolhidos",
        "",
        "| Modelo | Vídeo-alvo | Threshold | Macro F1 de calibração | Macro F1 retido |",
        "|---|---|---:|---:|---:|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['model_key']} | {row['target_video']} | {float(row['threshold']):.2f} | "
            f"{float(row['equal_video_mean_macro_f1']):.4f} | "
            f"{float(row['held_out_equal_video_mean_macro_f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reconstrução por fold",
            "",
            "| Modelo | Método | Macro F1 médio | F1 Fatigue médio | Recall Fatigue médio |",
            "|---|---|---:|---:|---:|",
        ]
    )
    keys = sorted({(str(row["model_key"]), str(row["method"])) for row in summary})
    for model_key, method in keys:
        global_rows = [
            row for row in summary if row["model_key"] == model_key and row["method"] == method
        ]
        fatigue_rows = [
            row
            for row in per_class
            if row["model_key"] == model_key
            and row["method"] == method
            and row["label"] == "fatigue"
        ]
        lines.append(
            f"| {model_key} | {method} | "
            f"{np.mean([float(row['macro_f1_all_classes']) for row in global_rows]):.4f} | "
            f"{np.mean([float(row['f1']) for row in fatigue_rows]):.4f} | "
            f"{np.mean([float(row['recall']) for row in fatigue_rows]):.4f} |"
        )
    lines.extend(
        [
            "",
            "Os resultados são qualificatórios, usam seed 42 e não incluem o teste externo. "
            "Melhora em validação não autoriza promover configuração antes das demais etapas "
            "congeladas da G4.5.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="fase_2/configs/experiment/g45_threshold.yaml"
    )
    parser.add_argument(
        "--prediction-dir", default="fase_2/outputs/predictions/G4"
    )
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics/G45")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = repository_path(args.config)
    config = load_yaml(config_path)
    if tuple(config["classes"]) != CLASSES or config["subset"] != "validation":
        raise ValueError("Configuração G4.5A incompatível com classes/subset congelados")
    grid_config = config["threshold_grid"]
    grid = threshold_grid(
        float(grid_config["minimum"]),
        float(grid_config["maximum"]),
        float(grid_config["step"]),
    )
    prediction_dir = repository_path(args.prediction_dir)
    output_dir = repository_path(args.output_dir)
    all_decisions: list[dict[str, object]] = []
    all_curves: list[dict[str, object]] = []
    all_deduplicated: list[dict[str, object]] = []
    all_reconstructed: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    all_per_class: list[dict[str, object]] = []
    source_audit: list[dict[str, object]] = []
    for model_key, model_config in config["models"].items():
        prefix = str(model_config["file_prefix"])
        paths = sorted(prediction_dir.glob(f"{prefix}*__validation.csv"))
        if len(paths) != 4:
            raise ValueError(
                f"{model_key}: esperados 4 arquivos de validação, "
                f"encontrados {len(paths)}"
            )
        source_audit.extend(
            {
                "model_key": model_key,
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "subset": "validation",
            }
            for path in paths
        )
        raw = load_validation_predictions(paths)
        deduplicated = deduplicate_predictions(raw)
        decisions, curves = crossfit_thresholds(deduplicated, grid)
        for collection in (deduplicated, decisions, curves):
            for row in collection:
                row["model_key"] = model_key
        reconstructed = reconstruct_predictions(raw, decisions)
        for row in reconstructed:
            row["model_key"] = model_key
        summary, per_class = summarize_reconstructed(reconstructed, model_key=model_key)
        all_decisions.extend(decisions)
        all_curves.extend(curves)
        all_deduplicated.extend(deduplicated)
        all_reconstructed.extend(reconstructed)
        all_summary.extend(summary)
        all_per_class.extend(per_class)

    _write_csv(output_dir / "g45a_threshold_decisions.csv", all_decisions)
    _write_csv(output_dir / "g45a_threshold_curves.csv", all_curves)
    _write_csv(output_dir / "g45a_deduplicated_validation.csv", all_deduplicated)
    _write_csv(output_dir / "g45a_reconstructed_predictions.csv", all_reconstructed)
    _write_csv(output_dir / "g45a_summary.csv", all_summary)
    _write_csv(output_dir / "g45a_per_class.csv", all_per_class)
    _report(output_dir / "g45a_report.md", all_summary, all_per_class, all_decisions)
    audit = {
        "generation": "G45",
        "experiment": "g45a_session_crossfit_threshold",
        "config_path": config_path.as_posix(),
        "config_sha256": _sha256(config_path),
        "loaded_test_artifacts": False,
        "source_file_count": len(source_audit),
        "source_files": source_audit,
        "raw_prediction_rows": len(all_reconstructed),
        "deduplicated_rows": len(all_deduplicated),
        "threshold_decisions": len(all_decisions),
        "threshold_curve_rows": len(all_curves),
    }
    (output_dir / "g45a_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"G4.5A concluída sem carregar teste externo: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
