"""Motor PyTorch reproduzível com AMP, early stopping e checkpoints retomáveis."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..models.temporal import build_temporal_model
from .dummy_baseline import CLASSES
from .temporal_data import SequenceSplit, WindowSequenceDataset


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    stopping_epoch: int
    best_validation_macro_f1: float
    training_seconds: float
    resumed: bool
    peak_gpu_memory_bytes: int


class FocalLoss(nn.Module):
    """Focal Loss multiclasse ponderada conforme o protocolo G4.5B."""

    def __init__(self, alpha: torch.Tensor, *, gamma: float) -> None:
        super().__init__()
        if alpha.ndim != 1 or len(alpha) != len(CLASSES):
            raise ValueError(f"alpha deve possuir {len(CLASSES)} valores")
        if not torch.isfinite(alpha).all() or torch.any(alpha <= 0):
            raise ValueError("alpha deve ser positivo e finito")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma deve ser não negativo e finito")
        self.register_buffer("alpha", alpha.detach().to(dtype=torch.float32))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # O cast explícito mantém log_softmax em float32 mesmo sob autocast/AMP.
        log_probabilities = F.log_softmax(logits.float(), dim=1)
        selected_log_probabilities = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        probabilities = selected_log_probabilities.exp()
        alpha_targets = self.alpha[targets]
        losses = -alpha_targets * (1.0 - probabilities).pow(self.gamma) * selected_log_probabilities
        denominator = alpha_targets.sum().clamp_min(torch.finfo(torch.float32).eps)
        return losses.sum() / denominator


def set_seed(seed: int, *, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def class_weights(labels: np.ndarray, *, num_classes: int | None = None) -> np.ndarray:
    """Pesos N/(K*N_c), falhando quando uma classe requerida nao existe."""
    class_count = len(CLASSES) if num_classes is None else int(num_classes)
    if class_count < 2:
        raise ValueError("num_classes deve ser pelo menos 2")
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0 or np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("Rotulos vazios ou fora do intervalo esperado")
    counts = np.bincount(labels, minlength=class_count).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Treino sem classe necessária: {counts.tolist()}")
    return len(labels) / (class_count * counts)


def sample_weights(labels: np.ndarray) -> np.ndarray:
    weights = class_weights(labels)
    return weights[np.asarray(labels, dtype=int)]


def weighted_sample_indices(labels: np.ndarray, *, seed: int) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(sample_weights(labels), dtype=torch.double)
    return torch.multinomial(weights, len(labels), replacement=True, generator=generator).numpy()


def make_loader(
    split: SequenceSplit,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    weighted_sampling: bool = False,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if weighted_sampling:
        if not shuffle:
            raise ValueError("weighted sampling e exclusivo do loader de treino")
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights(split.labels), dtype=torch.double),
            len(split.labels),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        WindowSequenceDataset(split),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    gradient_clip_norm: float,
    amp: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    expected: list[int] = []
    predicted: list[int] = []
    for values, labels in loader:
        values = values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(values)
                loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("NaN ou Inf detectado nos logits/loss")
            if optimizer is not None and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * len(labels)
        expected.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if not expected:
        raise ValueError("DataLoader vazio")
    return (
        total_loss / len(expected),
        np.asarray(expected, dtype=int),
        np.asarray(predicted, dtype=int),
    )


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    history: list[dict[str, float | int]],
    metadata: Mapping[str, object],
    train_generator_state: torch.Tensor | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_generator_state": train_generator_state,
            **metadata,
        },
        temporary,
    )
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def load_training_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    device: torch.device,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint if checkpoint.get("fingerprint") == fingerprint else None


def train_model(
    *,
    model_name: str,
    model_parameters: Mapping[str, object],
    splits: Mapping[str, SequenceSplit],
    training: Mapping[str, object],
    seed: int,
    fold: int,
    device: torch.device,
    checkpoint_dir: Path,
    fingerprint: str,
    resume: bool,
    feature_names: Sequence[str] | None = None,
    num_classes: int | None = None,
    progress_label: str | None = None,
) -> TrainingResult:
    if str(training.get("monitor", "val_macro_f1")) != "val_macro_f1":
        raise ValueError("O motor temporal suporta apenas monitor=val_macro_f1")
    if str(training.get("mode", "max")) != "max":
        raise ValueError("val_macro_f1 exige mode=max")
    deterministic = bool(training["deterministic"])
    set_seed(seed, deterministic=deterministic)
    output_classes = len(CLASSES) if num_classes is None else int(num_classes)
    # Valida antes de construir o modelo e garante que treino e validacao obedecem ao contrato.
    class_weights(splits["train"].labels, num_classes=output_classes)
    if np.any(splits["validation"].labels < 0) or np.any(
        splits["validation"].labels >= output_classes
    ):
        raise ValueError("Validacao contem rotulo fora do intervalo esperado")
    model = build_temporal_model(
        model_name,
        input_dim=splits["train"].values.shape[-1],
        num_classes=output_classes,
        parameters=model_parameters,
    ).to(device)
    balancing = str(training.get("balancing", "none"))
    if balancing in {"none", "weighted_sampling", "augmentation"}:
        weights = None
    elif balancing == "class_weights":
        weights = torch.tensor(
            class_weights(splits["train"].labels, num_classes=output_classes),
            dtype=torch.float32,
            device=device,
        )
    else:
        raise ValueError(
            f"Balanceamento nao suportado neste motor: {balancing}. "
            "Use none, class_weights, weighted_sampling ou augmentation"
        )
    loss_name = str(training.get("loss", "cross_entropy"))
    if loss_name == "cross_entropy":
        criterion: nn.Module = nn.CrossEntropyLoss(weight=weights)
    elif loss_name == "focal":
        if balancing != "none":
            raise ValueError("Focal Loss G4.5B não pode ser combinada com balanceamento")
        alpha = torch.tensor(
            class_weights(splits["train"].labels), dtype=torch.float32, device=device
        )
        criterion = FocalLoss(alpha, gamma=float(training.get("focal_gamma", 2.0)))
    else:
        raise ValueError(f"Loss temporal desconhecida: {loss_name}")
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    amp = bool(training["amp"]) and device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        splits["train"],
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training["num_workers"]),
        pin_memory=pin_memory,
        weighted_sampling=balancing == "weighted_sampling",
    )
    validation_loader = make_loader(
        splits["validation"],
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(training["num_workers"]),
        pin_memory=pin_memory,
    )
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best_macro_f1.pt"
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    best_epoch = 0
    best_metric = -1.0
    epochs_without_improvement = 0
    resumed = False
    elapsed_before_resume = 0.0
    checkpoint = (
        load_training_checkpoint(last_path, fingerprint=fingerprint, device=device)
        if resume
        else None
    )
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        gradient_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metric = float(checkpoint["best_metric"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        elapsed_before_resume = float(checkpoint.get("training_seconds", 0.0))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        if checkpoint.get("train_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["train_generator_state"].cpu())
        resumed = True

    metadata = {
        "fingerprint": fingerprint,
        "model_name": model_name,
        "fold": fold,
        "seed": seed,
        "model_parameters": dict(model_parameters),
        "training_config": dict(training),
        "feature_names": list(feature_names or ()),
    }
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopping_epoch = start_epoch - 1
    epochs = (
        range(0)
        if epochs_without_improvement >= int(training["patience"])
        else range(start_epoch, int(training["max_epochs"]) + 1)
    )
    for epoch in epochs:
        train_loss, train_expected, train_predicted = _epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=gradient_scaler,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            amp=amp,
        )
        validation_loss, validation_expected, validation_predicted = _epoch(
            model,
            validation_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            amp=amp,
        )
        validation_macro_f1 = f1_score(
            validation_expected,
            validation_predicted,
            labels=range(output_classes),
            average="macro",
            zero_division=0,
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": validation_loss,
            "training_macro_f1": f1_score(
                train_expected,
                train_predicted,
                labels=range(output_classes),
                average="macro",
                zero_division=0,
            ),
            "validation_macro_f1": validation_macro_f1,
            "training_accuracy": accuracy_score(train_expected, train_predicted),
            "validation_accuracy": accuracy_score(validation_expected, validation_predicted),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = validation_macro_f1 > best_metric + float(training["minimum_delta"])
        if improved:
            best_metric = validation_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(validation_macro_f1)
        training_seconds = elapsed_before_resume + time.perf_counter() - started
        save_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=gradient_scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_metric=best_metric,
            epochs_without_improvement=epochs_without_improvement,
            history=history,
            metadata={**metadata, "training_seconds": training_seconds},
            train_generator_state=train_loader.generator.get_state(),
        )
        if improved:
            save_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=gradient_scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                metadata={**metadata, "training_seconds": training_seconds},
                train_generator_state=train_loader.generator.get_state(),
            )
        checkpoint_every = int(training["checkpoint_every_epochs"])
        if checkpoint_every and epoch % checkpoint_every == 0:
            save_training_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=gradient_scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                metadata={**metadata, "training_seconds": training_seconds},
                train_generator_state=train_loader.generator.get_state(),
            )
        stopping_epoch = epoch
        progress = min(epoch / int(training["max_epochs"]), 1.0)
        filled = round(progress * 20)
        progress_bar = "█" * filled + "░" * (20 - filled)
        label = progress_label or f"{model_name.upper()} | fold {fold} | seed {seed}"
        print(
            f"[{progress_bar}] {epoch:03d}/{int(training['max_epochs']):03d} | {label} | "
            f"loss {train_loss:.4f}/{validation_loss:.4f} | "
            f"F1 val {validation_macro_f1:.4f} | melhor {best_metric:.4f} "
            f"(ép. {best_epoch}) | espera {epochs_without_improvement:02d}/"
            f"{int(training['patience']):02d}",
            flush=True,
        )
        if epochs_without_improvement >= int(training["patience"]):
            break
    best = load_training_checkpoint(best_path, fingerprint=fingerprint, device=device)
    if best is None:
        raise RuntimeError(f"Checkpoint best ausente: {best_path}")
    model.load_state_dict(best["model_state_dict"])
    total_seconds = elapsed_before_resume + time.perf_counter() - started
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        stopping_epoch=stopping_epoch,
        best_validation_macro_f1=best_metric,
        training_seconds=total_seconds,
        resumed=resumed,
        peak_gpu_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    )


def predict_split(
    model: nn.Module,
    split: SequenceSplit,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = make_loader(
        split,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    probabilities: list[list[float]] = []
    with torch.inference_mode():
        for values, labels in loader:
            values = values.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(values)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("NaN ou Inf detectado nas predicoes")
            expected.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
    result = np.asarray(probabilities, dtype=float)
    if not np.isfinite(result).all():
        raise FloatingPointError("NaN ou Inf detectado nas probabilidades")
    return np.asarray(expected), np.asarray(predicted), result


def save_run_result(path: Path, result: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
