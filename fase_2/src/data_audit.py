"""Verificação reproduzível das entradas antes de qualquer treinamento."""

from collections import Counter
import json
import math
from pathlib import Path

from .data.splits import SplitBlock, validate_split_blocks
from .pipeline import _now, _read_csv, _validate_series, load_config


def audit(config_path, *, save=True):
    config = load_config(config_path)
    errors, warnings = [], []
    videos = _read_csv(Path(config["data"]["manifest"]))
    expected = config["data"]["videos"]
    sizes = config["windowing"]["sizes_frames"]
    counts = {}
    try:
        if Counter(row["video_id"] for row in videos) != Counter(expected):
            errors.append("A lista de vídeos está incompleta ou contém duplicatas.")
        for row in videos:
            fps, frames = float(row["fps"]), int(row["num_frames"])
            if not math.isfinite(fps) or fps <= 0 or frames <= 0:
                raise ValueError("FPS e quantidade de imagens devem ser positivos.")
            counts[row["video_id"]] = frames
        intervals = _read_csv(Path(config["data"]["annotations"]))
        if not intervals:
            errors.append("As anotações dos comportamentos estão ausentes.")
        previous = {}
        for row in sorted(intervals, key=lambda r: (r["video_id"], int(r["start_frame"]))):
            video, start, end = row["video_id"], int(row["start_frame"]), int(row["end_frame"])
            if video not in counts or not 0 <= start <= end < counts[video]:
                errors.append(f"Anotação fora dos limites: {video}, {start}–{end}.")
            if start <= previous.get(video, -1):
                errors.append(f"Anotações sobrepostas: {video}, imagem {start}.")
            previous[video] = end
            if row["behavior_label"] and row["behavior_label"] not in config["data"]["classes"]:
                errors.append(f"Comportamento desconhecido em {video}.")
        blocks = [SplitBlock(int(r["fold"]), r["subset"], r["video_id"],
                             int(r["start_frame"]), int(r["end_frame"]))
                  for r in _read_csv(Path(config["splits"]["manifest"]))]
        gap = int(config["splits"]["purge_gap_frames"])
        if not sizes or min(sizes) <= 0 or int(config["windowing"]["stride_frames"]) <= 0:
            errors.append("O tamanho dos trechos e o intervalo entre eles devem ser positivos.")
        elif gap < max(sizes):
            errors.append("O espaço entre treino e validação é menor que o maior trecho.")
        errors.extend(validate_split_blocks(blocks, purge_gap_frames=gap))
        if {b.fold for b in blocks} != set(config["splits"]["folds"]):
            errors.append("Faltam divisões de avaliação, ou há divisões não configuradas.")
        for b in blocks:
            if b.subset not in {"train", "validation", "test"}:
                errors.append(f"Divisão desconhecida: {b.subset}.")
            if b.video_id not in counts or b.end_frame >= counts[b.video_id]:
                errors.append(f"Divisão fora dos limites: {b.video_id}.")
        for fold in config["splits"]["folds"]:
            selected = [b for b in blocks if b.fold == fold]
            if {b.subset for b in selected} != {"train", "validation", "test"}:
                errors.append(f"Divisão {fold}: faltam treino, validação ou teste.")
            for b in selected:
                if b.subset == "test" and (b.start_frame != 0 or b.end_frame != counts.get(b.video_id, 0)-1):
                    errors.append(f"Divisão {fold}: o vídeo de teste não está completo.")
        test_videos = [b.video_id for b in blocks if b.subset == "test"]
        if Counter(test_videos) != Counter(expected):
            errors.append("Cada vídeo precisa aparecer uma única vez como teste externo.")
        for row in videos:
            series = Path(config["data"]["facial_series"]) / f"{row['video_id']}.csv"
            try:
                _validate_series(series, counts[row["video_id"]], float(row["fps"]))
            except (OSError, ValueError, KeyError) as error:
                errors.append(f"Série facial {row['video_id']}: {error}")
        # Esta regra histórica usa apenas imagens anotadas no denominador.
        warnings.append("A regra histórica de rótulo considera apenas imagens anotadas. "
                        "Trechos parcialmente sem rótulo não são automaticamente descartados.")
        warnings.append("Interpolação linear é retrospectiva dentro do trecho; "
                        "a configuração principal usa preenchimento com zero, sem interpolação.")
    except (ValueError, KeyError, TypeError) as error:
        errors.append(f"Configuração ou manifesto inválido: {error}")
    result = {"checked_at": _now(), "status": "failed" if errors else "passed",
              "errors": errors, "warnings": warnings,
              "note": "Verificação de integridade; não certifica a qualidade das anotações humanas."}
    if save:
        target = Path(config["outputs"]["root"]) / "data_audit.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
    return result


def main(config_path):
    result = audit(config_path)
    print("Dados verificados." if result["status"] == "passed" else "Há dados que precisam de correção.")
    for message in result["errors"] + result["warnings"]:
        print(f"- {message}")
    return 0 if result["status"] == "passed" else 2
