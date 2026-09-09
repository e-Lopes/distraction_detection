"""Interface publica unica: prepare, train e report."""

from __future__ import annotations

import argparse

from .src.pipeline import DEFAULT_CONFIG, format_plan, prepare, status, train
from .src.reporting import generate_report


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="configuracao canonica")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fase_2", description="Pipeline temporal unificado")
    commands = parser.add_subparsers(dest="command", required=True)

    status_parser = commands.add_parser("status", help="mostra dados, artefatos e runs pendentes")
    _config_argument(status_parser)

    prepare_parser = commands.add_parser("prepare", help="valida e prepara entradas reutilizaveis")
    _config_argument(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true", help="ignora o cache de preparacao")

    train_parser = commands.add_parser("train", help="treina e avalia uma familia")
    train_parser.add_argument("--family", choices=("classical", "temporal", "all"), default="all")
    train_parser.add_argument("--scope", choices=("screening", "confirmation", "all"), default=None)
    train_parser.add_argument("--paradigm", choices=("rule", "feature", "distance", "shapelet",
                                                     "transform", "deep", "ensemble", "all"),
                              default="all")
    train_parser.add_argument("--plan", action="store_true", help="apenas expande e exibe o plano leve")
    train_parser.add_argument("--resume", action="store_true", default=True,
                              help="retoma runs compativeis (padrao)")
    train_parser.add_argument("--force", action="store_true", help="refaz runs; use explicitamente")
    train_parser.add_argument("--allow-expensive", action="store_true",
                              help="autoriza DTW acima do limite configurado")
    _config_argument(train_parser)

    report_parser = commands.add_parser("report", help="consolida resultados sem treinar")
    _config_argument(report_parser)

    all_parser = commands.add_parser("all", help="executa prepare, train e report")
    _config_argument(all_parser)
    all_parser.add_argument("--plan", action="store_true", help="exibe o plano sem executar etapas")
    all_parser.add_argument("--force", action="store_true", help="ignora caches e refaz runs")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.command == "status":
        print(status(args.config))
        return 0
    if args.command == "prepare":
        return prepare(args.config, force=args.force)
    if args.command == "train":
        scope = args.scope or ("confirmation" if args.family == "temporal" else "screening")
        if args.plan:
            print(format_plan(args.config, args.family, scope, args.paradigm))
            return 0
        return train(args.config, args.family, scope=scope, paradigm=args.paradigm,
                     force=args.force, allow_expensive=args.allow_expensive)
    if args.command == "report":
        return generate_report(args.config)
    if args.command == "all":
        if args.plan:
            print(format_plan(args.config, "all", "all", "all"))
            return 0
        prepare(args.config, force=args.force)
        train(args.config, "all", scope="all", paradigm="all", force=args.force)
        return generate_report(args.config)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
