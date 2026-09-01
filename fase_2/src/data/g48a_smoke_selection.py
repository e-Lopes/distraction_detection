"""Seleção determinística dos 30 frames do smoke G48A."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def select_smoke_frames(candidates: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    required = {
        "sample_id", "video_id", "frame_index", "timestamp_seconds", "behavior_label",
        "face_detected_baseline", "pose_bin_baseline",
    }
    if not required <= set(candidates.columns):
        raise ValueError(f"Manifesto candidato não contém: {sorted(required - set(candidates.columns))}")
    rules = {
        "easy": (candidates.face_detected_baseline.eq(1) & candidates.pose_bin_baseline.isin(["near", "medium"])),
        "intermediate": (candidates.face_detected_baseline.eq(1) & candidates.pose_bin_baseline.eq("large")),
        "hard": candidates.face_detected_baseline.eq(0),
    }
    rng = random.Random(seed)
    selected_parts = []
    for difficulty, mask in rules.items():
        pool = candidates.loc[mask].copy()
        indices = list(pool.index)
        rng.shuffle(indices)
        chosen: list[int] = []
        # Primeiro garante dois exemplos por vídeo; depois completa por round-robin de classe/vídeo.
        for video_id in sorted(pool.video_id.unique()):
            available = [index for index in indices if pool.loc[index, "video_id"] == video_id]
            chosen.extend(available[:2])
        remaining = [index for index in indices if index not in chosen]
        labels_present = set(pool.loc[chosen, "behavior_label"])
        for label in ("fatigue", "distraction", "alert"):
            if label not in labels_present:
                match = next((index for index in remaining if pool.loc[index, "behavior_label"] == label), None)
                if match is not None:
                    chosen.append(match)
                    remaining.remove(match)
        chosen.extend(remaining[: 10 - len(chosen)])
        if len(chosen) != 10:
            raise ValueError(f"Candidatos insuficientes para dificuldade {difficulty}")
        part = pool.loc[chosen].copy()
        part["difficulty"] = difficulty
        reasons = {
            "easy": "Face Mesh válido; pose frontal/moderada",
            "intermediate": "Face Mesh válido; pose lateral ampla",
            "hard": "falha conhecida do Face Mesh",
        }
        part["selection_reason"] = reasons[difficulty]
        selected_parts.append(part)
    output = pd.concat(selected_parts, ignore_index=True)
    output = output.rename(
        columns={
            "frame_index": "frame_id",
            "behavior_label": "temporal_class",
            "face_detected_baseline": "mediapipe_previous_status",
        }
    )
    output["timestamp_ms"] = output["timestamp_seconds"] * 1000.0
    output["source_manifest"] = "fase_2/data/manifests/g48_annotation_sample.csv"
    fields = [
        "sample_id", "video_id", "frame_id", "timestamp_ms", "temporal_class", "difficulty",
        "selection_reason", "mediapipe_previous_status", "source_manifest",
    ]
    return output[fields].sort_values(["video_id", "frame_id"]).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("fase_2/data/manifests/g48_annotation_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("fase_2/data/manifests/g48a_smoke_30.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Saída existente: {args.output}. Use --overwrite.")
    frame = select_smoke_frames(pd.read_csv(args.input), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(pd.crosstab(frame.difficulty, frame.video_id).to_string())
    print(pd.crosstab(frame.difficulty, frame.temporal_class).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
