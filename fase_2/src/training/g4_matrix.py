"""Audita o congelamento e materializa o dry-run oficial da G4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..preprocessing.missingness import expand_behavior_labels
from .dummy_baseline import CLASSES, load_series
from .temporal_data import build_sequence_fold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def verify_manifest(root: Path, manifest: Path, run_ids: set[str]) -> int:
    verified = 0
    for row in read_csv(manifest):
        if row["run_id"] not in run_ids:
            continue
        artifact = root / row["relative_path"]
        if not artifact.is_file() or artifact.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Artefato ausente ou tamanho divergente: {artifact}")
        if sha256(artifact) != row["sha256"]:
            raise ValueError(f"SHA-256 divergente: {artifact}")
        verified += 1
    if not verified:
        raise ValueError(f"Nenhum artefato relevante em {manifest}")
    return verified


def _run_id(config: Mapping[str, object], model: str, fold: int) -> str:
    window = int(config["window_size_frames"])
    return (f"G4__{config['configuration_id']}__{model}__r0__w{window}__"
            f"fold_{fold}__seed_42")


def build_matrix(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    registry_path = root / "fase_2/configs/experiment/imbalance.yaml"
    registry = load_yaml(registry_path)
    config_paths = [
        root / f"fase_2/configs/experiment/g4_{scenario}_w{window}.yaml"
        for scenario in ("b", "c", "d") for window in (60, 150)
    ]
    rows: list[dict[str, object]] = []
    config_audit = []
    for path in config_paths:
        config = load_yaml(path)
        window = int(config["window_size_frames"])
        source = load_yaml(root / f"fase_2/configs/experiment/g2_temporal_r0_w{window}.yaml")
        expected_models = ["lstm", "tcn"] if window == 60 else ["transformer"]
        if config["models"] != expected_models or config["parameters"] != {
            name: source["parameters"][name] for name in expected_models
        }:
            raise ValueError(f"Arquitetura/hiperparametros descongelados em {path}")
        inherited = dict(source["training"]); inherited["balancing"] = config["training"]["balancing"]
        if config["training"] != inherited or config["seeds"] != [42]:
            raise ValueError(f"Orcamento/semente divergente em {path}")
        if config["training"]["balancing"] not in {"class_weights", "weighted_sampling", "augmentation"}:
            raise ValueError("G4 nao permite combinacao de estrategias")
        config_audit.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
        for fold in (1, 2, 3, 4):
            for model in expected_models:
                run_id = _run_id(config, model, fold)
                rows.append({"run_id": run_id, "model": model, "window_size_frames": window,
                    "scenario": config["scenario"], "strategy": config["training"]["balancing"],
                    "fold": fold, "seed": 42, "execution": "new",
                    "config_path": path.relative_to(root).as_posix(),
                    "checkpoint": f"fase_2/outputs/models/G4/{run_id}/best_macro_f1.pt"})

    g1_ids, g2_ids = set(), set()
    for fold in (1, 2, 3, 4):
        run_id = f"G1__qualification_r0_flat__svm__r0_flat__w60__fold_{fold}__seed_42"
        g1_ids.add(run_id)
        rows.append({"run_id": run_id, "model": "svm", "window_size_frames": 60,
            "scenario": "A", "strategy": "none", "fold": fold, "seed": 42,
            "execution": "reused_g1", "config_path": "fase_2/configs/experiment/g1_baselines.yaml",
            "checkpoint": f"fase_2/outputs/models/G1/{run_id}.joblib"})
        for scenario, strategy in (("B", "class_weights"), ("C", "weighted_sampling"),
                                   ("D", "augmentation")):
            new_id = f"G4__g4_{scenario.lower()}_{strategy}_w60__svm__r0__w60__fold_{fold}__seed_42"
            rows.append({"run_id": new_id, "model": "svm", "window_size_frames": 60,
                "scenario": scenario, "strategy": strategy, "fold": fold, "seed": 42,
                "execution": "new", "config_path": "fase_2/configs/experiment/imbalance.yaml",
                "checkpoint": f"fase_2/outputs/models/G4/{new_id}.joblib"})
        for model, window, cid in (("lstm",60,"qualification_r0_w60_b1024"),
                                   ("tcn",60,"qualification_r0_w60_b1024"),
                                   ("transformer",150,"qualification_r0_w150_b1024")):
            run_id = f"G2__{cid}__{model}__r0__w{window}__fold_{fold}__seed_42"
            g2_ids.add(run_id)
            rows.append({"run_id": run_id, "model": model, "window_size_frames": window,
                "scenario": "A", "strategy": "none", "fold": fold, "seed": 42,
                "execution": "reused_g2", "config_path": f"fase_2/configs/experiment/g2_temporal_r0_w{window}.yaml",
                "checkpoint": f"fase_2/outputs/models/G2/{run_id}/best_macro_f1.pt"})

    g1_verified = verify_manifest(root, root / "fase_2/outputs/metrics/G1/g1_artifact_manifest.csv", g1_ids)
    g2_verified = verify_manifest(root, root / "fase_2/outputs/metrics/G2/g2_artifact_manifest.csv", g2_ids)

    blocks = [SplitBlock(int(r["fold"]), r["subset"], r["video_id"], int(r["start_frame"]), int(r["end_frame"]))
              for r in read_csv(root / "fase_2/data/manifests/temporal_splits.csv")]
    series = load_series(root / "fase_2/data/interim/legacy_extraction")
    labels = expand_behavior_labels({k: len(v) for k, v in series.items()},
                                    read_csv(root / "fase_2/data/manifests/annotation_frame_intervals.csv"))
    preprocessing = load_yaml(root / "fase_2/configs/preprocessing/baseline_zero_fill.yaml")
    data_config = load_yaml(root / "fase_2/configs/data/base.yaml")
    distributions, equivalence = [], []
    for window in (60, 150):
        for fold in (1, 2, 3, 4):
            splits, _, _ = build_sequence_fold(series, labels, blocks, preprocessing, fold=fold,
                size_frames=window, stride_frames=int(data_config["windowing"]["stride_frames"]),
                minimum_proportion=float(data_config["windowing"]["minimum_target_proportion"]), representation="R0")
            for subset, split in splits.items():
                counts = Counter(CLASSES[int(x)] for x in split.labels)
                distributions.append({"window_size_frames": window, "fold": fold, "subset": subset,
                    **{name: counts.get(name, 0) for name in CLASSES}, "total": len(split.labels)})
            digest = hashlib.sha256(np.ascontiguousarray(splits["train"].values).tobytes()
                                     + np.ascontiguousarray(splits["train"].labels).tobytes()).hexdigest()
            flat = splits["train"].values.reshape(len(splits["train"].labels), -1)
            if not np.array_equal(flat.reshape(splits["train"].values.shape), splits["train"].values):
                raise ValueError("SVM flatten alterou exemplos R0")
            equivalence.append({"window_size_frames": window, "fold": fold, "train_sha256": digest,
                                "svm_temporal_examples_equal": True})
    write_csv(root / "fase_2/outputs/metrics/G4/g4_class_distribution.csv", distributions)
    write_csv(root / "fase_2/outputs/metrics/G4/g4_input_equivalence.csv", equivalence)
    rows.sort(key=lambda r: (str(r["scenario"]), str(r["model"]), int(r["fold"])))
    audit = {"generation": "G4", "registry": registry_path.relative_to(root).as_posix(),
        "registry_sha256": sha256(registry_path), "new_runs": sum(r["execution"] == "new" for r in rows),
        "reused_runs": sum(str(r["execution"]).startswith("reused") for r in rows),
        "total_comparisons": len(rows), "g1_artifacts_verified": g1_verified,
        "g2_artifacts_verified": g2_verified, "r0_numerical_equivalence": "verified",
        "configs": config_audit, "status": "ready_for_tests_and_smokes"}
    return rows, audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics/G4")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); output = repository_path(args.output_dir)
    rows, audit = build_matrix(root)
    write_csv(output / "g4_dry_run.csv", rows)
    (output / "g4_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
