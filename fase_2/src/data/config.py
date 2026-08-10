"""Carregamento mínimo de configurações da fase 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Carrega um YAML e exige um objeto no nível raiz."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Configuração deve ser um objeto YAML: {config_path}")
    return value


def repository_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve um caminho relativo contra a raiz usada pelo comando."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Caminho absoluto não permitido na configuração: {path}")
    return (Path(root or Path.cwd()) / path).resolve()
