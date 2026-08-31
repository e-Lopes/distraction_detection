"""Treinamento rastreável dos modelos YOLO26-Pose faciais da G4.7."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..features.g47_schema import DEFAULT_FLIP_INDEX, load_schema

SCALES = ("n", "s", "m", "l", "x")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest() -> dict[str, object]:
    import torch

    manifest: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    for package in ("ultralytics", "mediapipe", "numpy", "opencv-python"):
        try:
            manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest[package] = None
    if torch.cuda.is_available():
        manifest.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        manifest["nvidia_driver"] = completed.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        manifest["nvidia_driver"] = None
    return manifest


def validate_environment(manifest: dict[str, object], *, require_cuda: bool) -> None:
    required = ("ultralytics", "mediapipe", "numpy")
    missing = [name for name in required if not manifest.get(name)]
    if missing:
        raise RuntimeError(f"Dependências G4.7 ausentes: {', '.join(missing)}")
    if require_cuda:
        if not manifest.get("cuda_available"):
            raise RuntimeError("CUDA não está disponível neste ambiente")
        capability = tuple(manifest.get("cuda_capability", ()))
        if capability != (6, 1):
            raise RuntimeError(f"Esperada GTX 1060/capability (6, 1); encontrada {capability}")


def validate_dataset_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Dataset YAML deve ser um objeto")
    if payload.get("kpt_shape") != [22, 3]:
        raise ValueError("Dataset exige kpt_shape: [22, 3]")
    if tuple(payload.get("flip_idx", ())) != DEFAULT_FLIP_INDEX:
        raise ValueError("flip_idx do dataset difere do contrato G4.7")
    if not all(key in payload for key in ("train", "val", "names")):
        raise ValueError("Dataset YAML exige train, val e names")
    return payload


def verify_flip_approval(path: Path, schema_path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            "fliplr bloqueado: crie o marcador após aprovar visualmente a auditoria de flip: "
            f"{path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("approved") is not True or payload.get("schema_sha256") != _sha256(schema_path):
        raise ValueError("Marcador de aprovação do flip é inválido ou pertence a outro schema")


def loss_grid() -> list[dict[str, float]]:
    return [
        {"pose": float(pose), "kobj": float(kobj), "rle": float(rle)}
        for pose, kobj, rle in product((12, 18), (1, 2), (1, 2))
    ]


def pose_loss_diverged(results_csv: Path) -> bool:
    frame = pd.read_csv(results_csv)
    columns = [column for column in frame if "pose_loss" in column and "train" in column]
    if not columns:
        raise ValueError(f"Coluna train pose_loss ausente em {results_csv}")
    values = pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return True
    baseline = float(np.median(values[: min(5, len(values))]))
    return bool(baseline > 0 and values[-1] > 5 * baseline)


def train_one(
    *,
    scale: str,
    fold: int,
    data_yaml: Path,
    output_root: Path,
    device: str,
    epochs: int,
    patience: int,
    losses: dict[str, float],
    fliplr: float,
    seed: int,
) -> Path:
    from ultralytics import YOLO

    if scale not in SCALES:
        raise ValueError(f"Escala inválida: {scale}")
    run_name = (
        f"fold_{fold}__yolo26{scale}__pose_{losses['pose']:g}__"
        f"kobj_{losses['kobj']:g}__rle_{losses['rle']:g}"
    )
    model = YOLO(f"yolo26{scale}-pose.pt")
    result = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=640,
        batch=-1,
        nbs=64,
        device=device,
        seed=seed,
        deterministic=True,
        fliplr=fliplr,
        pose=losses["pose"],
        kobj=losses["kobj"],
        rle=losses["rle"],
        project=str(output_root),
        name=run_name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )
    run_dir = Path(result.save_dir)
    results_csv = run_dir / "results.csv"
    if pose_loss_diverged(results_csv):
        (run_dir / "BLOCKED_POSE_LOSS_DIVERGENCE").write_text(
            "Treino bloqueado: revise flip_idx e anotações antes de usar os pesos.\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"pose_loss divergiu em {run_dir}")
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Peso best.pt ausente em {run_dir}")
    canonical = output_root / "weights" / f"yolo26{scale}-face-custom__fold_{fold}.pt"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, canonical)
    return canonical


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YAML; aceita {fold} no nome")
    parser.add_argument(
        "--schema", type=Path, default=Path("fase_2/configs/features/g47_face_landmarks.yaml")
    )
    parser.add_argument(
        "--flip-approval",
        type=Path,
        default=Path("fase_2/data/external/g47/flip_audit_approved.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("fase_2/outputs/models/G47"))
    parser.add_argument("--scale", choices=SCALES, action="append")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--pose", type=float, default=12)
    parser.add_argument("--kobj", type=float, default=1)
    parser.add_argument("--rle", type=float, default=1)
    parser.add_argument("--search-losses", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--environment-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = load_schema(args.schema)
    del schema
    manifest = environment_manifest()
    require_cuda = str(args.device) not in {"cpu", "mps"}
    validate_environment(manifest, require_cuda=require_cuda)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.environment_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.fliplr > 0:
        verify_flip_approval(args.flip_approval, args.schema)
    scales = args.scale or ["n", "s", "m"]
    folds = args.fold or [1, 2, 3, 4]
    if any(scale in {"l", "x"} for scale in scales):
        print("Aviso: l/x foram preparados para a máquina com GPU mais forte.")
    losses = loss_grid() if args.search_losses else [{"pose": args.pose, "kobj": args.kobj, "rle": args.rle}]
    if args.search_losses and scales != ["n"]:
        raise ValueError("A busca de perdas é permitida somente com --scale n")
    runs: list[dict[str, object]] = []
    for fold in folds:
        data_path = Path(str(args.data).format(fold=fold))
        validate_dataset_yaml(data_path)
        for scale in scales:
            for values in losses:
                runs.append({"fold": fold, "scale": scale, "data": str(data_path), **values})
    print(json.dumps(runs, indent=2))
    if args.dry_run:
        return 0
    for run in runs:
        train_one(
            scale=str(run["scale"]),
            fold=int(run["fold"]),
            data_yaml=Path(str(run["data"])),
            output_root=args.output_dir,
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            losses={name: float(run[name]) for name in ("pose", "kobj", "rle")},
            fliplr=args.fliplr,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
