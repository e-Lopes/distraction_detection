"""Consolidacao economica dos resultados cientificos ja existentes."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from .pipeline import _read_csv, _write_csv_atomic, load_config, sync_historical_registry


METRIC_FIELDS = (
    "source", "scientific_status", "run_id", "family", "model", "representation",
    "window_size_frames", "fold", "seed", "balancing", "subset", "scope", "label",
    "metric", "value",
)


def _family(model: str) -> str:
    return "temporal" if model in {"lstm", "tcn", "transformer"} else "classical"


def _append_overall(rows: list[dict[str, object]], path: str, source: str) -> None:
    for item in _read_csv(Path(path)):
        model = item.get("model", "dummy")
        common = {
            "source": source, "scientific_status": "reused_valid", "run_id": item.get("run_id", ""),
            "family": _family(model), "model": model,
            "representation": item.get("representation", item.get("ablation", "")),
            "window_size_frames": item.get("window_size_frames", ""), "fold": item.get("fold", ""),
            "seed": item.get("seed", "42"), "balancing": item.get("balancing", item.get("strategy", "")),
        }
        subsets = ("validation", "test") if any(key.startswith("validation_") for key in item) else (item.get("subset", ""),)
        for subset in subsets:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1_all_classes"):
                key = f"{subset}_{metric}" if f"{subset}_{metric}" in item else metric
                value = item.get(key, "")
                if value not in {"", None}:
                    rows.append({**common, "subset": subset, "scope": "global", "label": "",
                                 "metric": "macro_f1" if metric == "macro_f1_all_classes" else metric,
                                 "value": value})
        for metric in ("training_seconds", "train_seconds", "parameter_count", "model_size_bytes",
                       "peak_gpu_memory_bytes", "inference_seconds", "inference_windows_per_second"):
            if item.get(metric, "") not in {"", None}:
                rows.append({**common, "subset": "all", "scope": "cost", "label": "",
                             "metric": metric, "value": item[metric]})


def _append_per_class(rows: list[dict[str, object]], path: str, source: str) -> None:
    for item in _read_csv(Path(path)):
        model = item.get("model", "")
        for metric in ("precision", "recall", "f1"):
            if item.get(metric, "") != "":
                rows.append({
                    "source": source, "scientific_status": "reused_valid",
                    "run_id": item.get("run_id", ""), "family": _family(model), "model": model,
                    "representation": item.get("representation", item.get("ablation", "")),
                    "window_size_frames": item.get("window_size_frames", ""), "fold": item.get("fold", ""),
                    "seed": item.get("seed", "42"), "balancing": item.get("balancing", item.get("strategy", "")),
                    "subset": item.get("subset", ""), "scope": "class", "label": item.get("label", ""),
                    "metric": metric, "value": item[metric],
                })


def consolidate_metrics(config: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    overall = (
        ("fase_2/outputs/metrics/G1/g1_execution_table.csv", "G1"),
        ("fase_2/outputs/metrics/G2/g2_execution_table.csv", "G2"),
        ("fase_2/outputs/metrics/G3/g3_execution_table.csv", "G3"),
        ("fase_2/outputs/metrics/G4/g4_runs.csv", "G4"),
        ("fase_2/outputs/metrics/G45/focal_r0_w60__runs.csv", "G45_focal"),
        ("fase_2/outputs/metrics/G45/g45c_runs.csv", "G45_hierarchical"),
    )
    for path, source in overall:
        _append_overall(rows, path, source)
    for item in _read_csv(Path("fase_2/outputs/metrics/dummy_baseline_summary.csv")):
        for metric in ("accuracy", "balanced_accuracy", "macro_f1_all_classes"):
            rows.append({
                "source": "baseline", "scientific_status": "reused_valid",
                "run_id": f"dummy__w{item['window_size_frames']}__fold_{item['fold']}",
                "family": "classical", "model": "dummy", "representation": "majority_train",
                "window_size_frames": item["window_size_frames"], "fold": item["fold"], "seed": "",
                "balancing": "none", "subset": item["subset"], "scope": "global", "label": "",
                "metric": "macro_f1" if metric == "macro_f1_all_classes" else metric,
                "value": item[metric],
            })
    for item in _read_csv(Path("fase_2/outputs/metrics/G46/g46_runs.csv")):
        rows.append({
            "source": "G46", "scientific_status": "reused_valid", "run_id": "",
            "family": "classical", "model": "svm", "representation": item["representation"],
            "window_size_frames": 60, "fold": item["fold"], "seed": item["seed"],
            "balancing": "none", "subset": item["subset"], "scope": "global", "label": "",
            "metric": "macro_f1", "value": item["macro_f1"],
        })
    for path, source in (
        ("fase_2/outputs/metrics/G1/g1_per_class.csv", "G1"),
        ("fase_2/outputs/metrics/G2/g2_per_class.csv", "G2"),
        ("fase_2/outputs/metrics/G4/g4_per_class.csv", "G4"),
        ("fase_2/outputs/metrics/G45/focal_r0_w60__per_class.csv", "G45_focal"),
        ("fase_2/outputs/metrics/G45/g45c_per_class.csv", "G45_hierarchical"),
    ):
        _append_per_class(rows, path, source)
    new_metrics = Path(config["outputs"]["root"]) / "new_classical_metrics.csv"
    if new_metrics.is_file():
        _append_overall(rows, str(new_metrics), "final")
    screening_metrics = Path(config["outputs"]["root"]) / "screening_metrics.csv"
    if screening_metrics.is_file():
        _append_overall(rows, str(screening_metrics), "screening")
    _append_overall(rows, str(Path(config["outputs"]["root"]) / "confirmation_classical_metrics.csv"),
                    "confirmation")
    for path in (Path(config["outputs"]["root"]) / "screening_details").glob("*__per_class.csv"):
        _append_per_class(rows, str(path), "screening")
    for path in (Path(config["outputs"]["root"]) / "confirmation_details").glob("*__per_class.csv"):
        _append_per_class(rows, str(path), "confirmation")
    for path in (Path(config["outputs"]["root"]) / "temporal_metrics").rglob("*__runs.csv"):
        _append_overall(rows, str(path), "confirmation")
    for path in (Path(config["outputs"]["root"]) / "temporal_metrics").rglob("*__per_class.csv"):
        _append_per_class(rows, str(path), "confirmation")
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in METRIC_FIELDS[2:-1]) + (row["metric"],)
        unique.setdefault(key, row)
    result = list(unique.values())
    _write_csv_atomic(Path(config["outputs"]["metrics"]), result, METRIC_FIELDS)
    return result


def consolidate_event_metrics(config: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for item in _read_csv(Path("fase_2/outputs/metrics/G4/g4_false_fatigue_statistics.csv")):
        rows.append({
            "source": "G4", "scientific_status": "preliminary_proxy",
            "model": item["model"], "scenario": item["scenario"], "target_class": "fatigue",
            "metric": item["metric"], "mean": item["mean"], "std": item["std"],
            "unit": "episodes_per_hour", "note": "window-derived proxy; full event matching pending",
        })
    fields = ("source", "scientific_status", "model", "scenario", "target_class", "metric",
              "mean", "std", "unit", "note")
    _write_csv_atomic(Path(config["outputs"]["event_metrics"]), rows, fields)
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _plot_figures(config: Mapping[str, object], metrics: list[dict[str, object]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(config["outputs"]["figures"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    selected = [row for row in metrics if row["subset"] == "validation" and row["scope"] == "global"
                and row["metric"] == "macro_f1" and row["source"] in {"G1", "G2"}]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        grouped[str(row["model"])].append(float(row["value"]))
    if grouped:
        labels = sorted(grouped)
        values = [_mean(grouped[label]) for label in labels]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#386cb0")
        ax.set(ylabel="Macro F1 de validacao", title="Desempenho historico por modelo")
        ax.set_ylim(0, max(values) * 1.25)
        fig.tight_layout()
        target = figure_dir / "macro_f1_by_model.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(str(target))
    selected_balance = {"svm": "class_weights", "lstm": "class_weights", "tcn": "weighted_sampling"}
    fatigue = [row for row in metrics if row["subset"] == "validation" and row["scope"] == "class"
               and row["label"] == "fatigue" and row["metric"] == "f1" and row["source"] == "G4"
               and selected_balance.get(str(row["model"])) == row["balancing"]]
    fatigue_grouped: dict[str, list[float]] = defaultdict(list)
    for row in fatigue:
        fatigue_grouped[str(row["model"])].append(float(row["value"]))
    if fatigue_grouped:
        labels = sorted(fatigue_grouped)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, [_mean(fatigue_grouped[label]) for label in labels], color="#fdb462")
        ax.set(ylabel="F1 de Fatigue", title="Fatigue nos tratamentos finalistas (G4)")
        fig.tight_layout()
        target = figure_dir / "fatigue_f1_by_model.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(str(target))
    confusion_rows = [row for row in _read_csv(Path("fase_2/outputs/metrics/G1/g1_confusion.csv"))
                      if row["model"] == "svm" and row["subset"] == "test"
                      and row["window_size_frames"] == "60"]
    if confusion_rows:
        labels = ["alert", "fatigue", "distraction"]
        matrix = [[0 for _ in labels] for _ in labels]
        for row in confusion_rows:
            matrix[labels.index(row["actual"])][labels.index(row["predicted"])] += int(row["count"])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        image = ax.imshow(matrix, cmap="Blues")
        for y, line in enumerate(matrix):
            for x, value in enumerate(line):
                ax.text(x, y, str(value), ha="center", va="center")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
               xlabel="Predito", ylabel="Real", title="SVM/w60: confusao OOF historica")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        target = figure_dir / "svm_w60_oof_confusion.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(str(target))
    per_video = [row for row in _read_csv(Path("fase_2/outputs/metrics/G1/g1_execution_table.csv"))
                 if row["model"] == "svm" and row["window_size_frames"] == "60"]
    if per_video:
        per_video.sort(key=lambda row: int(row["fold"]))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([f"video_{int(row['fold']):02d}" for row in per_video],
               [float(row["test_macro_f1_all_classes"]) for row in per_video], color="#7fc97f")
        ax.set(ylabel="Macro F1 no video externo", title="SVM/w60 por video (OOF)")
        fig.tight_layout()
        target = figure_dir / "svm_w60_by_video.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(str(target))
    videos = _read_csv(Path(config["data"]["manifest"]))
    if videos:
        labels = [row["video_id"] for row in videos]
        sizes = [30, 60, 150]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for size in sizes:
            ax.plot(labels, [size / float(row["fps"]) for row in videos], marker="o", label=f"{size} frames")
        ax.set(ylabel="Duracao real (s)", title="Janelas variam com o FPS do video")
        ax.legend()
        fig.tight_layout()
        target = figure_dir / "window_duration_by_video.png"
        fig.savefig(target, dpi=160)
        plt.close(fig)
        generated.append(str(target))
    return generated


def _best_mean(metrics: list[dict[str, object]], model: str, window: int | None = None) -> float:
    values = [float(row["value"]) for row in metrics if row["model"] == model
              and row["subset"] == "validation" and row["scope"] == "global"
              and row["metric"] == "macro_f1" and (window is None or str(row["window_size_frames"]) == str(window))
              and row["source"] in {"G1", "G2"}]
    return max((_mean(group) for group in _group_windows(metrics, model, window).values()), default=float("nan")) if values else float("nan")


def _group_windows(metrics: list[dict[str, object]], model: str, window: int | None) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in metrics:
        if (row["model"] == model and row["subset"] == "validation" and row["scope"] == "global"
                and row["metric"] == "macro_f1" and row["source"] in {"G1", "G2"}
                and (window is None or str(row["window_size_frames"]) == str(window))):
            grouped[str(row["window_size_frames"])].append(float(row["value"]))
    return grouped


def _report_text(config: Mapping[str, object], metrics: list[dict[str, object]], figures: list[str]) -> str:
    videos = _read_csv(Path(config["data"]["manifest"]))
    total_frames = sum(int(row["num_frames"]) for row in videos)
    total_minutes = sum(float(row["duration_seconds"]) for row in videos) / 60
    durations = _read_csv(Path("fase_2/outputs/metrics/annotation_duration_summary.csv"))
    class_seconds: dict[str, float] = defaultdict(float)
    for row in durations:
        class_seconds[row["label"]] += float(row["duration_seconds"])
    annotations = _read_csv(Path(config["data"]["annotations"]))
    fatigue_intervals = sum(row.get("behavior_label") == "fatigue" for row in annotations)
    missing = _read_csv(Path("fase_2/outputs/metrics/facial_missingness.csv"))
    missing_range = (min(float(row["missing_rate"]) for row in missing),
                     max(float(row["missing_rate"]) for row in missing))
    svm = _best_mean(metrics, "svm", 60)
    lstm = _best_mean(metrics, "lstm", 60)
    tcn = _best_mean(metrics, "tcn", 60)
    transformer = _best_mean(metrics, "transformer", 150)
    figure_links = "\n".join(f"- `{Path(item).name}`" for item in figures)
    return f"""# Relatorio final do experimento temporal

> Estado: primeira consolidacao. Resultados historicos validos foram reutilizados; nenhum novo treinamento oficial foi executado. O teste externo nunca foi usado para selecao.

## 1. Resumo executivo

O conjunto possui {total_frames:,} frames ({total_minutes:.1f} min), forte desbalanceamento e apenas {class_seconds['fatigue']:.0f} s anotados como Fatigue. SVM e redes temporais superam regras fixas, mas nao ha evidencia de superioridade geral dos temporais sobre o melhor classico. O resultado operacional de Fatigue ainda e insuficiente: ganhos de recall vieram acompanhados de muitos falsos episodios. A decisao atual e confirmar LSTM/B contra SVM/B em cinco seeds e completar avaliacao por episodio.

## 2. Pergunta, objetivo e hipotese

Pergunta: quanto contexto temporal melhora a classificacao de Alert, Fatigue e Distraction a partir de EAR, MAR, Pitch, Yaw e Roll? A hipotese e que modelos temporais ajudam, sobretudo em Fatigue, sem custo operacional desproporcional. A margem de equivalencia pratica e Macro F1 = {config['delta_macro_f1']}.

## 3. Dados, videos, duracao, FPS e classes

Sao quatro sessoes leave-one-video-out, com FPS entre {min(float(v['fps']) for v in videos):.2f} e {max(float(v['fps']) for v in videos):.2f}. Duracoes anotadas: Alert {class_seconds['alert']:.0f} s, Distraction {class_seconds['distraction']:.0f} s, Fatigue {class_seconds['fatigue']:.0f} s e ausencia do operador {class_seconds['operator_absent']:.0f} s.

## 4. Quantidade e duracao dos episodios

Ha {fatigue_intervals} intervalos anotados como Fatigue no manifesto; depois de janelamento e restricoes de suporte, algumas configuracoes ficam com aproximadamente dois eventos uteis. Logo, janelas sobrepostas nao sao tratadas como observacoes estatisticamente independentes e inferencias fortes nao sao justificadas.

## 5. Qualidade dos landmarks e missingness

O missingness varia de {missing_range[0]:.1%} a {missing_range[1]:.1%} entre videos (38.15% global). Na comparacao preliminar do video 04, OpenFace detectou 88.32% dos frames com operador presente e MediaPipe 79.81%; MediaPipe foi mantido pelo menor custo/latencia e integracao, nao por maior cobertura. Os indicadores dos extratores nao sao intercambiaveis sem nova validacao.

## 6. Tratamentos de dados

R0 usa zero-fill; R1 interpola somente gaps curtos e usa mediana do treino; R2 acrescenta flags de deteccao, interpolacao e duracao da ausencia. Ajustes sao isolados por video/particao e scalers usam apenas treino. Gaps longos e ausencia do operador sao preservados.

## 7. Representacoes e features

A base usa EAR, MAR, Pitch, Yaw e Roll. `temporal_behavior_v1` inclui distribuicao e quantis robustos, IQR, inclinacao, diferenca inicio-fim, autocorrelacao, line length, picos, cruzamentos, primeira e segunda diferencas, velocidade, aceleracao, variacao total, PERCLOS, piscadas, possiveis bocejos, desvio de cabeca, correlacoes/covariancias e missingness/interpolacao. Correlacoes ou variacoes indefinidas em sinais constantes viram zero finito. Foram descartadas correlacoes defasadas, autorregressao de alta ordem e bandas espectrais por suporte instavel em janelas curtas e FPS heterogeneo. Features por threshold sao baselines comportamentais complementares.

## 8. Protocolo de janelas e splits

Janelas de 30, 60 e 150 frames, stride 15, quatro folds leave-one-video-out e purge bilateral de 150 frames. Como o FPS varia, a mesma janela representa duracoes diferentes; o baseline historico em frames foi preservado e resampling nao foi imposto silenciosamente.

## 9. Inventario de experimentos existentes

Importados: Dummy, regras fixas, SVM, Random Forest, XGBoost, LSTM, TCN, Transformer, comparacao de janelas, missingness, balanceamento, Focal Loss, threshold cross-fit, classificacao hierarquica, robustez de pose e comparacao de extratores. A taxonomia canonica e `rule`, `feature`, `distance`, `shapelet`, `transform`, `deep` e `ensemble`; nomes historicos nao foram convertidos em novos resultados. G47/YOLO e exploratorio; smoke/pilots e artefatos incompativeis nao entram na conclusao cientifica.

## 10. Resultados dos baselines

O Dummy majoritario fica perto de Macro F1 0.31 na validacao. Regras fixas atingem no melhor caso aproximadamente 0.160. Ambos servem como piso, nao como solucao operacional.

## 11. Resultados dos modelos classicos

SVM permanece competitivo: melhor media historica em w=60 de {svm:.4f}. SVM, Random Forest e XGBoost historicos continuam intactos. O novo screening compara Regressao Logistica, SVM, Random Forest e XGBoost com exatamente a mesma tabela `temporal_behavior_v1`, w=60, seed 42 e quatro folds (16 runs).

## 12. Resultados dos modelos temporais

Em seed 42: LSTM/w60 {lstm:.4f}, TCN/w60 {tcn:.4f} e Transformer/w150 {transformer:.4f}. Eles vencem regras fixas, mas nao estabelecem superioridade geral sobre SVM. Esses resultados serao reutilizados; os 16 novos runs LSTM/B estao bloqueados ate promocao formal apos o screening.

## 13. Resultados por janela

As melhores configuracoes historicas concentram-se em 60 frames para SVM, LSTM e TCN; Transformer teve melhor resultado em 150 frames. A figura `window_duration_by_video.png` explicita a diferenca temporal causada pelo FPS.

## 14. Resultados por classe

Alert domina o conjunto. Distraction e separavel parcialmente. Fatigue tem suporte muito pequeno e F1 instavel; medias globais sem a leitura por classe seriam enganosas.

## 15. Avaliacao de Fatigue

LSTM/B obteve F1 medio de Fatigue 0.0885 e recall 0.392, com 56.92 falsos episodios/h. TCN/C elevou recall para 0.4785, mas chegou a 84.79 falsos episodios/h. Focal Loss levou LSTM a F1 0.1039, ganho pequeno. PR-AUC ainda nao pode ser reconstruida para todos os historicos porque probabilidades OOF completas nao estao versionadas.

## 16. Avaliacao por episodio

`event_metrics.csv` registra claramente a estatistica G4 como proxy preliminar baseada em janelas. Event Precision, Recall, F1, latencia, fragmentacao e erro de duracao aguardam predicoes OOF completas e matching temporal pre-registrado.

## 17. Estabilidade entre videos e seeds

Ha variacao material entre folds/videos e somente seed 42 para os finalistas historicos. A confirmacao multi-seed e pendente; a unidade de comparacao sera video e seed, nunca janela individual.

## 18. Custo computacional

Tempos de treino, tamanho, parametros e pico de GPU existem para parte dos modelos e foram importados no registro. DTW possui custo quadratico e limite de pares; shapelets usam 2.000 candidatos e no maximo 200 selecionados; MiniROCKET usa 5.040 kernels, um multiplo valido de 84. Latencia end-to-end, RAM e throughput serao medidos nos runs reais; nao se inventam valores ausentes.

## 19. Analise de robustez e ablacao

R0 foi selecionada sobre R1/R2 (0.4083 contra 0.3156/0.3691). Threshold cross-fit foi negativo. Hierarquico/LSTM chegou a 0.4186, mas foi inconsistente e pior em Fatigue; nao promovido. Correcao de pose reduziu dependencia geometrica, mas ganhou apenas ~0.0012 de Macro F1; nao promovida.

## 20. Comparacao pareada e incerteza

Com quatro videos, nao se usa teste t, Friedman, Nemenyi ou Wilcoxon como prova forte. A confirmacao reportara deltas por video, videos vencidos, magnitude e intervalos que preservem dependencia temporal.

## 21. Fronteira de Pareto

Ainda nao existe vencedor absoluto. SVM oferece Macro F1 competitivo e baixo custo; LSTM/B oferece algum sinal de Fatigue a custo de falsos alarmes; TCN/C amplia recall com mais alarmes. O screening adiciona quatro pipelines de features, 1-NN+DTW, shapelets+Ridge e MiniROCKET+Ridge. A comparacao e entre pipelines/representacoes, nao uma atribuicao causal isolada ao algoritmo.

## 22. Limitacoes

Quatro videos, poucos eventos de Fatigue, missingness heterogeneo, FPS diferentes, dados de um dominio restrito e probabilidades historicas incompletas. OpenFace foi comparado em um video e o benchmark YOLO e exploratorio.

## 23. Resposta as hipoteses

Contexto temporal supera regras manuais, mas a hipotese de superioridade global sobre o melhor classico nao foi confirmada. A hipotese de melhora robusta de Fatigue permanece inconclusiva devido a raridade e falsos alarmes.

## 24. Conclusao

O resultado defensavel hoje e um empate pratico global com trade-offs: SVM e referencia eficiente; LSTM/B e candidata temporal para confirmacao. Nenhum modelo deve ser apresentado como detector completo de Fatigue.

## 25. Proximos passos realmente pendentes

Executar primeiro o screening de 36 runs: features 16, DTW 4, shapelets 4 e MiniROCKET 12. Shapelets/MiniROCKET requerem o extra opcional `tsc`; DTW esta bloqueado porque a estimativa excede o limite configurado. Depois, promover no maximo o melhor pipeline de features, o melhor temporal nao neural, LSTM/B e SVM/B; TCN/C fica como sensibilidade. A confirmacao nao inicia sem `screening_promotion.yaml` e autorizacao.

### Matriz de screening reservada

| Paradigma | Pipeline | Representacao | Janelas | Runs | Estado inicial |
|---|---|---|---|---:|---|
| feature | Logistica/SVM/RF/XGBoost | temporal_behavior_v1 | 60 | 16 | pendente |
| distance | 1-NN + DTW dependente | R0 bruta | 60 | 4 | bloqueado por custo |
| shapelet | Random Shapelet Transform + Ridge | R0 bruta | 60 | 4 | dependencia opcional |
| transform | MiniROCKET + RidgeCV | R0 bruta | 30/60/150 | 12 | dependencia opcional |

### Secoes reservadas para resultados novos

- **DTW:** Sakoe-Chiba de 6 frames, matriz reutilizavel por fold e autorizacao explicita acima de 5 milhoes de pares.
- **Shapelets:** registrar canal, comprimento, importancia, exemplo de treino e presenca nos videos externos; verificar memorizacao de uma unica sessao.
- **MiniROCKET:** comparar as tres janelas diretamente com SVM, LSTM, TCN e Transformer.
- **Ensembles:** DrCIF, Arsenal e HIVE-COTE 2.0 permanecem desativados; TS-CHIEF e apenas documentado. So considerar apos resultado insuficiente dos metodos simples, estimativa de custo e autorizacao.

### Criterio de promocao

Usar apenas validacao interna: Macro F1 medio, F1/PR-AUC e recall de Fatigue, estabilidade entre videos e custo, com margem pratica de {config['delta_macro_f1']}. Resultado negativo e preservado e nao recebe novas seeds. Metricas de episodio ausentes no screening ficam explicitamente pendentes para confirmation.

## 26. Apendice: nomes historicos e novo fluxo

| Historico | Conteudo | Etapa publica atual |
|---|---|---|
| G0/G1 | dados, Dummy, regras e classicos | prepare / train / report |
| G2 | temporais e janelas | train / report |
| G3 | missingness R0-R2 | prepare / report |
| G4 | balanceamento | train / report |
| G4.5 | threshold, focal e hierarquico | train / report |
| G4.6 | robustez de pose | prepare / report |
| G47 | benchmark exploratorio YOLO | report (exploratorio) |
| G48/G48A | comparacao de extratores | prepare / report |

### Auditoria interna consolidada

| Componente | Estado verdadeiro | Reutilizavel | Acao minima |
|---|---|---|---|
| dados e anotacoes | concluido | sim | validar manifesto |
| series faciais | concluido | sim | validar cache; bruto ausente |
| missingness | concluido | sim | importar |
| janelas e splits | concluido | sim | validar purge |
| features temporais | parcial | sim | ativar so se justificado |
| modelos classicos | parcial | sim | importar + Logistica |
| modelos temporais | parcial | sim | importar + multi-seed LSTM |
| avaliacao por episodio | parcial | sim | matching OOF |
| estabilidade multi-seed | parcial | sim | 16 runs pendentes |
| custo computacional | parcial | sim | consolidar novos runs |
| relatorio consolidado | concluido nesta versao | sim | regenerar apos treinos |

### Inconsistencias resolvidas

`split_validation.md` descreve um split antigo sem Fatigue na validacao; o manifesto atual e a ADR 004 mostram blocos corrigidos. `g45_results.md` e um relatorio de andamento chamam o hierarquico de pendente, mas `g45c_runs.csv` e seus artefatos provam conclusao posterior. `data_sources_and_provenance.md` registra pendencias antigas, enquanto manifestos e quatro series completas comprovam que os derivados estao disponiveis; somente os videos brutos faltam localmente.

### Figuras geradas

{figure_links}

Figuras que exigem OOF/multi-seed (confusao OOF final, resultado por video, distribuicao entre seeds, timeline, desempenho-custo e Pareto final) nao foram fabricadas nesta primeira consolidacao.
"""


def generate_report(config_path: str | Path) -> int:
    config = load_config(config_path)
    print("[REPORT 1/8] Importando registro historico")
    imported = sync_historical_registry(config_path)
    print(f"[REPORT 2/8] Consolidando metricas globais e por classe (novos registros={imported})")
    metrics = consolidate_metrics(config)
    print(f"[REPORT 3/8] Consolidando {len(metrics)} metricas canonicas")
    events = consolidate_event_metrics(config)
    print(f"[REPORT 4/8] Consolidando metricas por episodio ({len(events)} proxies)")
    print("[REPORT 5/8] Auditando FPS, duracoes e missingness")
    print("[REPORT 6/8] Gerando somente figuras suportadas pelos dados")
    figures = _plot_figures(config, metrics)
    print("[REPORT 7/8] Registrando limitacoes e inconsistencias documentais")
    report = Path(config["outputs"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_report_text(config, metrics, figures), encoding="utf-8")
    screening = Path(config["outputs"]["root"]) / "screening_metrics.csv"
    if screening.is_file():
        from .progress_table import generate_binary_attention_table, generate_progress_table
        generate_progress_table(config_path)
        generate_binary_attention_table(config_path)
    print(f"[REPORT 8/8] DONE report={report} figures={len(figures)}")
    return 0
