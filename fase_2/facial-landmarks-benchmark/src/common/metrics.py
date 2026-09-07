from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator, List, Tuple

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg")


def percentile(values: List[float], pct: float) -> float:
    """Percentil simples (interpolação linear), sem depender de numpy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@contextmanager
def timer_ms() -> Iterator[dict]:
    """Context manager que devolve um dict {'ms': ...} preenchido ao sair do bloco."""
    result = {"ms": 0.0}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - t0) * 1000.0


def discover_videos(input_path: str) -> List[Tuple[str, str]]:
    """
    Descobre vídeos a processar.

    Retorna lista de tuplas (video_id, caminho_absoluto).
    video_id é derivado do caminho relativo (sem extensão), preservando
    subpastas para não colidir nomes repetidos entre sujeitos/cenários.
    """
    videos = []
    if os.path.isfile(input_path):
        video_id = os.path.splitext(os.path.basename(input_path))[0]
        return [(video_id, input_path)]

    for root, _dirs, files in os.walk(input_path):
        for fname in sorted(files):
            if fname.lower().endswith(VIDEO_EXTENSIONS):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, input_path)
                video_id = os.path.splitext(rel_path)[0].replace(os.sep, "__")
                videos.append((video_id, full_path))
    return videos
