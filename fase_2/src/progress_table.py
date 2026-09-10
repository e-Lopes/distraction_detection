"""Gera um resumo legível dos treinamentos concluídos no screening."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from .pipeline import _read_csv, _write_csv_atomic, load_config


NAMES = {
    "logistic_regression": "Regressão logística",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "random_shapelet_ridge": "Shapelet + Ridge",
    "minirocket_ridge": "MiniRocket + Ridge",
}


def _fatigue_by_run(root: Path) -> dict[str, float]:
    values = {}
    for path in (root / "screening_details").glob("*__validation__per_class.csv"):
        for row in _read_csv(path):
            if row.get("label", "").lower() == "fatigue":
                value = row.get("f1", row.get("f1_score", ""))
                if value not in {"", None}:
                    values[row.get("run_id", path.name.split("__validation__")[0])] = float(value)
    return values


def generate_progress_table(config_path: str | Path) -> tuple[Path, Path]:
    config = load_config(config_path)
    root = Path(config["outputs"]["root"])
    metrics = [row for row in _read_csv(root / "screening_metrics.csv")
               if row.get("subset") == "validation"]
    fatigue = _fatigue_by_run(root)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        grouped[(row["model"], row.get("representation", row.get("ablation", "")),
                 row.get("window_size_frames", ""))].append(row)

    summary = []
    for (model, representation, window), rows in grouped.items():
        macro = [float(row["macro_f1_all_classes"]) for row in rows]
        balanced = [float(row["balanced_accuracy"]) for row in rows]
        accuracy = [float(row["accuracy"]) for row in rows]
        fatigue_values = [fatigue[row["run_id"]] for row in rows if row["run_id"] in fatigue]
        seconds = [float(row["train_seconds"]) for row in rows]
        summary.append({
            "model": NAMES.get(model, model), "model_id": model,
            "representation": representation, "window_size_frames": window,
            "completed_folds": len(rows), "macro_f1_mean": mean(macro),
            "macro_f1_std": stdev(macro) if len(macro) > 1 else 0.0,
            "balanced_accuracy_mean": mean(balanced), "accuracy_mean": mean(accuracy),
            "fatigue_f1_mean": mean(fatigue_values) if fatigue_values else float("nan"),
            "training_seconds_total": sum(seconds),
        })
    summary.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    for rank, row in enumerate(summary, 1):
        row["rank"] = rank

    csv_path = root / "training_progress_summary.csv"
    fields = ("rank", "model", "model_id", "representation", "window_size_frames",
              "completed_folds", "macro_f1_mean", "macro_f1_std",
              "balanced_accuracy_mean", "accuracy_mean", "fatigue_f1_mean",
              "training_seconds_total")
    _write_csv_atomic(csv_path, summary, fields)

    registry = _read_csv(root / "run_registry.csv")
    completed = sum(row.get("status") == "completed" and row.get("scope") == "screening"
                    for row in registry)
    dtw = [row for row in registry if row.get("model") == "knn_dtw"
           and row.get("scope") == "screening" and row.get("status") == "blocked"]
    lines = [
        "# Resumo dos treinamentos realizados",
        "",
        f"> Atualizado em {datetime.now().astimezone().strftime('%d/%m/%Y às %H:%M')}. ",
        f"> **{completed} treinamentos de comparação concluídos e salvos.**",
        "",
        "## Desempenho na validação",
        "",
        "A tabela usa somente a validação interna, que é a parte adequada para comparar e escolher modelos.",
        "",
        "| Posição | Modelo | Entrada | Janela | Folds | F1 geral | Variação | Acurácia balanceada | F1 de fadiga | Tempo total |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        fatigue_text = "—" if row["fatigue_f1_mean"] != row["fatigue_f1_mean"] else f'{row["fatigue_f1_mean"]:.3f}'
        seconds = row["training_seconds_total"]
        time_text = f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.1f} s"
        lines.append(
            f'| {row["rank"]} | **{row["model"]}** | {row["representation"]} | '
            f'{row["window_size_frames"]} | {row["completed_folds"]}/4 | '
            f'{row["macro_f1_mean"]:.3f} | ± {row["macro_f1_std"]:.3f} | '
            f'{row["balanced_accuracy_mean"]:.3f} | {fatigue_text} | {time_text} |')
    lines.extend([
        "",
        "## Leitura rápida",
        "",
        f'- Melhor F1 geral: **{summary[0]["model"]}**, janela de {summary[0]["window_size_frames"]} imagens '
        f'({summary[0]["macro_f1_mean"]:.3f}).',
        "- F1 geral considera igualmente alerta, fadiga e distração. Quanto mais perto de 1, melhor.",
        "- Acurácia balanceada reduz o efeito da grande diferença de quantidade entre as classes.",
        "- F1 de fadiga deve ser analisado separadamente; um valor próximo de zero indica que o comportamento quase não foi reconhecido.",
        "- Os resultados do vídeo externo permanecem guardados, mas não foram usados para ordenar esta tabela.",
        "",
        "## Situação das etapas",
        "",
        "| Etapa | Quantidade | Situação | Próxima ação |",
        "|---|---:|---|---|",
        f"| Comparação dos modelos | {completed} | **Concluída e salva** | Revisar esta tabela e escolher candidatos |",
        f"| KNN-DTW | {len(dtw)} | Bloqueado pelo custo | Executar somente após aceitar o custo elevado |",
        "| Confirmação SVM/LSTM | 24 | Aguardando escolha | Criar `screening_promotion.yaml` |",
        "",
        "## Observação",
        "",
        "A posição representa apenas o resultado médio da validação atual. Ela não significa que o primeiro colocado já esteja pronto para uso real. A escolha final também precisa considerar fadiga, falsos alarmes, estabilidade entre vídeos e custo.",
    ])
    report_path = Path(config["outputs"]["report"]).parent / "resumo_treinamentos_atuais.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, csv_path


def generate_binary_attention_table(config_path: str | Path) -> tuple[Path, Path]:
    """Reavalia as previsões salvas: alert=atenção; demais classes=não atenção."""
    config = load_config(config_path)
    root = Path(config["outputs"]["root"])
    validation = [row for row in _read_csv(root / "screening_metrics.csv")
                  if row.get("subset") == "validation"]
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for metric in validation:
        path = root / "predictions" / f'{metric["run_id"]}__validation.csv'
        if not path.is_file():
            continue
        pairs = [(row["actual"] != "alert", row["predicted"] != "alert")
                 for row in _read_csv(path)]
        tp = sum(actual and predicted for actual, predicted in pairs)
        tn = sum(not actual and not predicted for actual, predicted in pairs)
        fp = sum(not actual and predicted for actual, predicted in pairs)
        fn = sum(actual and not predicted for actual, predicted in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_non = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_attention = tn / (tn + fn) if tn + fn else 0.0
        recall_attention = tn / (tn + fp) if tn + fp else 0.0
        f1_attention = (2 * precision_attention * recall_attention /
                        (precision_attention + recall_attention)
                        if precision_attention + recall_attention else 0.0)
        key = (metric["model"], metric.get("representation", metric.get("ablation", "")),
               metric.get("window_size_frames", ""))
        grouped[key].append({"fold": int(metric["fold"]),
                             "macro_f1": (f1_non + f1_attention) / 2,
                             "balanced_accuracy": (recall + recall_attention) / 2,
                             "accuracy": (tp + tn) / len(pairs), "precision_non_attention": precision,
                             "recall_non_attention": recall, "f1_non_attention": f1_non})
    summary = []
    for (model, representation, window), rows in grouped.items():
        item = {"model": NAMES.get(model, model), "model_id": model,
                "representation": representation, "window_size_frames": window,
                "completed_folds": len(rows)}
        for field in ("macro_f1", "balanced_accuracy", "accuracy", "precision_non_attention",
                      "recall_non_attention", "f1_non_attention"):
            values = [row[field] for row in rows]
            item[field + "_mean"] = mean(values)
            if field == "macro_f1":
                item["macro_f1_std"] = stdev(values) if len(values) > 1 else 0.0
        best = max(rows, key=lambda row: row["macro_f1"])
        worst = min(rows, key=lambda row: row["macro_f1"])
        item.update(best_fold=int(best["fold"]), best_fold_macro_f1=best["macro_f1"],
                    worst_fold=int(worst["fold"]), worst_fold_macro_f1=worst["macro_f1"])
        summary.append(item)
    summary.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    for rank, row in enumerate(summary, 1):
        row["rank"] = rank
    fields = ("rank", "model", "model_id", "representation", "window_size_frames",
              "completed_folds", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean",
              "accuracy_mean", "precision_non_attention_mean", "recall_non_attention_mean",
              "f1_non_attention_mean", "best_fold", "best_fold_macro_f1",
              "worst_fold", "worst_fold_macro_f1")
    csv_path = root / "binary_attention_summary.csv"
    _write_csv_atomic(csv_path, summary, fields)
    lines = ["# Cenário binário: atenção e não atenção", "",
             "> Atenção = `alert`. Não atenção = `fatigue` + `distraction`.", "",
             "Esta é uma reavaliação das previsões dos modelos de três classes; não houve novo treinamento.", "",
             "| Posição | Modelo | Janela | F1 geral | Melhor fold | Pior fold | Acurácia balanceada | Precisão: não atenção | Recall: não atenção | F1: não atenção |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f'| {row["rank"]} | **{row["model"]}** | {row["window_size_frames"]} | '
                     f'{row["macro_f1_mean"]:.3f} | {row["best_fold"]} ({row["best_fold_macro_f1"]:.3f}) | '
                     f'{row["worst_fold"]} ({row["worst_fold_macro_f1"]:.3f}) | '
                     f'{row["balanced_accuracy_mean"]:.3f} | '
                     f'{row["precision_non_attention_mean"]:.3f} | '
                     f'{row["recall_non_attention_mean"]:.3f} | {row["f1_non_attention_mean"]:.3f} |')
    lines.extend(["", "## Interpretação", "",
                  "A precisão baixa de não atenção indica muitos falsos alarmes. O recall mostra quanto da não atenção real foi encontrado. Um treinamento diretamente binário pode produzir resultados diferentes e deve ser validado antes de qualquer conclusão final."])
    report_path = Path(config["outputs"]["report"]).parent / "resumo_binario_atencao.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path, csv_path
