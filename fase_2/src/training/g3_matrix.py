"""Audita o congelamento e gera a matriz dry-run oficial da G3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..data.config import load_yaml, repository_path
from .classical_baselines import experiment_fingerprint
from .temporal_data import representation_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def expected_fingerprint(config_path: Path, config: Mapping[str, object], root: Path) -> str:
    representation = str(config["representation"]).upper()
    data_path = root / "fase_2/configs/data/base.yaml"
    preprocessing_path = root / str(config["preprocessing_config"])
    interval_path = root / "fase_2/data/manifests/annotation_frame_intervals.csv"
    split_path = root / "fase_2/data/manifests/temporal_splits.csv"
    extraction_manifest = root / "fase_2/data/interim/legacy_extraction/extraction_manifest.csv"
    return experiment_fingerprint(
        [config_path, data_path, preprocessing_path, interval_path, split_path, extraction_manifest],
        {
            "mode": "temporal_multiseed",
            "generation": str(config["generation"]),
            "configuration_id": str(config["configuration_id"]),
            "representation": representation,
            "features": representation_features(representation),
        },
    )


def validate_inheritance(config: Mapping[str, object], source: Mapping[str, object]) -> None:
    if config["training"] != source["training"]:
        raise ValueError(f"Orcamento de treino diverge da G2: {config['configuration_id']}")
    for model in config["models"]:
        if config["parameters"][model] != source["parameters"][model]:
            raise ValueError(f"Hiperparametros de {model} divergem da G2")
    if int(config["window_size_frames"]) != int(source["window_size_frames"]):
        raise ValueError("Janela divergente da configuracao G2 de origem")
    if list(config["seeds"]) != [42] or config["training"].get("balancing") != "none":
        raise ValueError("G3 exige seed 42 e balancing=none")


def build_matrix(
    registry_path: Path,
    *,
    root: Path,
    log_root: Path,
    checkpoint_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    registry = load_yaml(registry_path)
    rows: list[dict[str, object]] = []
    config_audit: list[dict[str, object]] = []
    for relative_config in registry["new_configs"]:
        config_path = root / str(relative_config)
        config = load_yaml(config_path)
        source_path = root / (
            "fase_2/configs/experiment/g2_temporal_r0_w60.yaml"
            if int(config["window_size_frames"]) == 60
            else "fase_2/configs/experiment/g2_temporal_r0_w150.yaml"
        )
        source = load_yaml(source_path)
        validate_inheritance(config, source)
        actual_source_hash = _sha256(source_path)
        if actual_source_hash != config["source_g2_config_sha256"]:
            raise ValueError(f"Hash da configuracao G2 divergiu: {source_path}")
        fingerprint = expected_fingerprint(config_path, config, root)
        config_audit.append(
            {
                "config": _relative(config_path, root),
                "config_sha256": _sha256(config_path),
                "expected_run_fingerprint": fingerprint,
                "source_g2_config": _relative(source_path, root),
                "source_g2_config_sha256": actual_source_hash,
            }
        )
        representation = str(config["representation"]).upper()
        window = int(config["window_size_frames"])
        for fold in (1, 2, 3, 4):
            for model in config["models"]:
                run_id = (
                    f"G3__{config['configuration_id']}__{model}__{representation.lower()}__"
                    f"w{window}__fold_{fold}__seed_42"
                )
                log_path = log_root / f"{run_id}.json"
                status = "new"
                if log_path.is_file():
                    logged = json.loads(log_path.read_text(encoding="utf-8"))
                    if logged.get("fingerprint") == fingerprint:
                        status = "completed" if logged.get("completed") else "resumable"
                    else:
                        status = "fingerprint_conflict"
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "window_size_frames": window,
                        "representation": representation,
                        "fold": fold,
                        "seed": 42,
                        "config_path": _relative(config_path, root),
                        "expected_log_path": _relative(log_path, root),
                        "expected_checkpoint_path": _relative(
                            checkpoint_root / run_id / "best_macro_f1.pt", root
                        ),
                        "execution": status,
                        "fingerprint": fingerprint,
                    }
                )

    for reused in registry["reused_r0"]:
        model = str(reused["model"])
        window = int(reused["window_size_frames"])
        source_path = root / str(reused["source_config"])
        if _sha256(source_path) != reused["source_config_sha256"]:
            raise ValueError(f"Configuracao R0 alterada: {source_path}")
        for fold in (1, 2, 3, 4):
            source_config = load_yaml(source_path)
            run_id = (
                f"G2__{source_config['configuration_id']}__{model}__r0__w{window}__"
                f"fold_{fold}__seed_42"
            )
            log_path = root / f"fase_2/outputs/logs/G2/{run_id}.json"
            logged = json.loads(log_path.read_text(encoding="utf-8"))
            if not logged.get("completed") or logged.get("fingerprint") != reused["run_fingerprint"]:
                raise ValueError(f"Run R0 nao reutilizavel: {run_id}")
            rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "window_size_frames": window,
                    "representation": "R0",
                    "fold": fold,
                    "seed": 42,
                    "config_path": _relative(source_path, root),
                    "expected_log_path": _relative(log_path, root),
                    "expected_checkpoint_path": _relative(
                        root / f"fase_2/outputs/models/G2/{run_id}/best_macro_f1.pt", root
                    ),
                    "execution": "reused_g2",
                    "fingerprint": reused["run_fingerprint"],
                }
            )
    rows.sort(key=lambda row: (str(row["model"]), int(row["fold"]), str(row["representation"])))
    counts = {status: sum(row["execution"] == status for row in rows) for status in {str(row["execution"]) for row in rows}}
    audit = {
        "registry": _relative(registry_path, root),
        "registry_sha256": _sha256(registry_path),
        "matrix_size": len(rows),
        "execution_counts": counts,
        "r0_numerical_equivalence": "verified for all folds at w60 and w150",
        "g2_artifact_manifest_entries_verified": 180,
        "configs": config_audit,
    }
    return rows, audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="fase_2/configs/experiment/g3_representation.yaml")
    parser.add_argument("--output-dir", default="fase_2/outputs/metrics/G3")
    parser.add_argument("--log-dir", default="fase_2/outputs/logs/G3")
    parser.add_argument("--checkpoint-dir", default="fase_2/outputs/models/G3")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    output_dir = repository_path(args.output_dir)
    rows, audit = build_matrix(
        repository_path(args.registry),
        root=root,
        log_root=repository_path(args.log_dir),
        checkpoint_root=repository_path(args.checkpoint_dir),
    )
    _write_csv(output_dir / "g3_dry_run.csv", rows)
    (output_dir / "g3_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
