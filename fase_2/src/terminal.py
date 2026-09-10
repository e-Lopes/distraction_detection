"""Central de trabalho do protocolo, sem dependências de interface gráfica."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .pipeline import build_plan, load_config, pipeline_fingerprint


PROTOCOL = Path(__file__).resolve().parents[1] / "protocolo_final_validacao_series_temporais.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def overview(config_path: str | Path) -> str:
    """Consulta leve: não sincroniza registro nem certifica artefatos históricos."""
    path = Path(config_path)
    config = load_config(path)
    outputs = config["outputs"]
    missing = [video for video in config["data"]["videos"]
               if not (Path(config["data"]["facial_series"]) / f"{video}.csv").is_file()]
    state_path = Path(outputs["root"]) / "prepare_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        prepared = (state.get("status") == "completed"
                    and state.get("fingerprint") == pipeline_fingerprint(path, config))
    except (OSError, ValueError, AttributeError):
        prepared = False
    promotion = Path(config["training"]["confirmation"]["promotion_file"])
    events = config["event_evaluation"]
    undefined = [key for key in ("minimum_duration_seconds", "gap_tolerance_seconds", "minimum_iou")
                 if events.get(key) is None]
    lines = [
        "CENTRAL DO PROTOCOLO TEMPORAL", f"Configuração: {path}",
        f"Protocolo: {PROTOCOL}",
        "Consulta de arquivos e plano; presença de artefato não comprova validade científica.",
        "", "Etapas A–F do protocolo",
        f"A · Auditoria: séries ausentes = {', '.join(missing) if missing else 'nenhuma'}.",
        f"B · Preparação: {'cache compatível' if prepared else 'pendente ou cache desatualizado'}.",
        f"    Janelas {config['windowing']['sizes_frames']} frames; "
        f"stride {config['windowing']['stride_frames']}; "
        f"folds {config['splits']['folds']}.",
        "C/D · Screening: comparar representações/pipelines na validação interna.",
        "      Comparação isolada de algoritmos exige representação comum.",
        f"E · Promoção: {'arquivo presente; revisar conteúdo' if promotion.is_file() else 'pendente'} "
        f"({promotion}).",
        f"    Seeds de confirmação: {config['training']['confirmation']['seeds']}.",
        "F · Avaliação final: requer OOF completo, episódios, incerteza e custo.",
        f"    Parâmetros de episódio indefinidos: {', '.join(undefined) if undefined else 'nenhum'}.",
        "    A interface não certifica completude OOF ou conformidade metodológica.",
        "", "Plano atual",
    ]
    for scope in ("screening", "confirmation"):
        counts = Counter(run.status for run in build_plan(path, "all", scope, "all"))
        lines.append(f"  {scope}: " + (", ".join(f"{key}={value}" for key, value
                                               in sorted(counts.items())) or "sem candidatos"))
    lines.extend(["", "Próxima ação"])
    if missing:
        lines.append("Disponibilizar as séries faciais canônicas antes de preparar/treinar.")
    elif not prepared:
        lines.append("Executar prepare para validar os dados e atualizar o cache.")
    elif not promotion.is_file():
        lines.append("Revisar o plano de screening, executar pendências e decidir promoção na validação interna.")
    else:
        lines.append("Revisar o plano de confirmação e completar a avaliação final do protocolo.")
    lines.extend(["", "Artefatos"])
    for name in ("registry", "metrics", "event_metrics", "predictions", "report"):
        if name in outputs:
            artifact = Path(outputs[name])
            lines.append(f"  {name}: {artifact} [{'presente' if artifact.exists() else 'ausente'}]")
    return "\n".join(lines)


def run_action(config_path: str | Path, arguments: list[str]) -> int:
    """Executa a CLI em processo isolado, mantendo logs e Ctrl+C no terminal."""
    command = [sys.executable, "-m", "fase_2", *arguments, "--config", str(config_path)]
    print(f"\n$ {shlex.join(command)}", flush=True)
    try:
        code = subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except KeyboardInterrupt:
        print("\nExecução interrompida. Consulte o registro antes de retomar.")
        return 130
    print(f"\nProcesso encerrado com código {code}. Consulte o plano para runs bloqueados ou pendentes.")
    return code


def main(config_path: str | Path, *, summary: bool = False) -> int:
    # Os caminhos internos da configuração seguem a convenção da CLI: relativos à raiz.
    path = Path(config_path).resolve()
    if Path.cwd().resolve() != PROJECT_ROOT:
        raise ValueError(f"Execute a interface na raiz do repositório: {PROJECT_ROOT}")
    print(overview(path))
    if summary:
        return 0
    scope, paradigm = "screening", "all"
    while True:
        print(f"\nEscopo: {scope} | Paradigma: {paradigm}")
        print("1 Estado e pendências   2 Ler protocolo       3 Revisar plano")
        print("4 Preparar dados       5 Executar treinos    6 Gerar relatório")
        print("7 Ler relatório        8 Alterar escopo      9 Filtrar paradigma")
        print("0 Sair")
        try:
            choice = input("Opção: ").strip()
            if choice == "0":
                return 0
            if choice == "1":
                print(overview(path))
            elif choice in {"2", "7"}:
                document = (PROTOCOL if choice == "2" else
                            Path(load_config(path)["outputs"]["report"]))
                print(document.read_text(encoding="utf-8") if document.is_file()
                      else f"Documento ausente: {document}")
            elif choice == "3":
                run_action(path, ["train", "--plan", "--scope", scope, "--paradigm", paradigm])
            elif choice == "4":
                run_action(path, ["prepare"])
            elif choice == "5":
                run_action(path, ["train", "--scope", scope, "--paradigm", paradigm])
            elif choice == "6":
                run_action(path, ["report"])
            elif choice == "8":
                scope = "confirmation" if scope == "screening" else "screening"
            elif choice == "9":
                value = input("Paradigma (all, feature, distance, shapelet, transform, deep): ").strip()
                if value in {"all", "feature", "distance", "shapelet", "transform", "deep"}:
                    paradigm = value
                else:
                    print("Paradigma inválido; filtro mantido.")
            else:
                print("Opção inválida.")
        except (EOFError, KeyboardInterrupt):
            print("\nInterface encerrada.")
            return 0
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Não foi possível concluir a ação: {error}")
