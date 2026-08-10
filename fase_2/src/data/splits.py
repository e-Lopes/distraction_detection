"""Splits leave-one-video-out com blocos temporais e purge gap."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from math import floor


@dataclass(frozen=True)
class SplitBlock:
    fold: int
    subset: str
    video_id: str
    start_frame: int
    end_frame: int


def generate_leave_one_video_out(
    video_num_frames: Mapping[str, int],
    *,
    validation_fraction: float = 0.20,
    purge_gap_frames: int = 150,
) -> list[SplitBlock]:
    """Gera um fold por vídeo; validação é o bloco final dos vídeos de desenvolvimento."""
    if len(video_num_frames) < 2:
        raise ValueError("São necessários ao menos dois vídeos")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction deve estar em (0, 1)")
    if purge_gap_frames < 0:
        raise ValueError("purge_gap_frames não pode ser negativo")

    blocks: list[SplitBlock] = []
    for fold, test_video in enumerate(sorted(video_num_frames), start=1):
        test_frames = video_num_frames[test_video]
        if test_frames <= 0:
            raise ValueError(f"Número de frames inválido para {test_video}")
        blocks.append(SplitBlock(fold, "test", test_video, 0, test_frames - 1))
        for video_id in sorted(video_num_frames):
            if video_id == test_video:
                continue
            num_frames = video_num_frames[video_id]
            validation_start = floor(num_frames * (1 - validation_fraction))
            train_end = validation_start - purge_gap_frames - 1
            if train_end < 0 or validation_start >= num_frames:
                raise ValueError(
                    f"Vídeo {video_id} é curto para validação e purge gap configurados"
                )
            blocks.append(SplitBlock(fold, "train", video_id, 0, train_end))
            blocks.append(
                SplitBlock(fold, "validation", video_id, validation_start, num_frames - 1)
            )
    return blocks


def validate_split_blocks(blocks: Iterable[SplitBlock], *, purge_gap_frames: int) -> list[str]:
    """Retorna violações de isolamento, sobreposição e purge gap."""
    errors: list[str] = []
    by_fold: dict[int, list[SplitBlock]] = {}
    for block in blocks:
        by_fold.setdefault(block.fold, []).append(block)
        if block.start_frame < 0 or block.end_frame < block.start_frame:
            errors.append(f"fold {block.fold}: intervalo inválido em {block.video_id}")

    for fold, fold_blocks in sorted(by_fold.items()):
        test_videos = {block.video_id for block in fold_blocks if block.subset == "test"}
        development_videos = {
            block.video_id for block in fold_blocks if block.subset in {"train", "validation"}
        }
        if len(test_videos) != 1:
            errors.append(f"fold {fold}: deve existir exatamente um vídeo de teste")
        if test_videos & development_videos:
            errors.append(f"fold {fold}: vídeo de teste também aparece em desenvolvimento")
        by_video: dict[str, list[SplitBlock]] = {}
        for block in fold_blocks:
            by_video.setdefault(block.video_id, []).append(block)
        for video_id, video_blocks in by_video.items():
            ordered = sorted(video_blocks, key=lambda block: block.start_frame)
            for previous, current in pairwise(ordered):
                if current.start_frame <= previous.end_frame:
                    errors.append(f"fold {fold}: sobreposição temporal em {video_id}")
                actual_gap = current.start_frame - previous.end_frame - 1
                if {previous.subset, current.subset} == {"train", "validation"} and (
                    actual_gap < purge_gap_frames
                ):
                    errors.append(f"fold {fold}: purge gap insuficiente em {video_id}")
    return errors


def window_subset(
    *, video_id: str, start_frame: int, end_frame: int, blocks: Iterable[SplitBlock], fold: int
) -> str | None:
    """Atribui janela somente quando ela está integralmente contida em um bloco."""
    matches = [
        block.subset
        for block in blocks
        if block.fold == fold
        and block.video_id == video_id
        and start_frame >= block.start_frame
        and end_frame <= block.end_frame
    ]
    if len(matches) > 1:
        raise ValueError("Janela atribuída a múltiplos subconjuntos")
    return matches[0] if matches else None
