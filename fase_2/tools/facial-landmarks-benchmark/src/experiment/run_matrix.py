"""Host-side sequential orchestrator for the complete benchmark matrix."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}
PUBLIC = ("drozy", "yawdd", "nitymed")

def run(command: list[str]):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Execute frameworks sequentially and reproducibly")
    parser.add_argument("--datasets", nargs="+", default=list(PUBLIC))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-cuda-check", action="store_true")
    parser.add_argument("--without-insightface-cpu", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions deve ser >= 1")
    root = Path(__file__).resolve().parents[2]
    summary = root / "results" / "aggregate_summary.csv"
    if summary.exists():
        with summary.open(newline="", encoding="utf-8") as source:
            if any(row.get("run_id") == args.run_id for row in csv.DictReader(source)):
                raise SystemExit(f"run_id {args.run_id!r} ja existe em {summary}; escolha outro")

    manifest = {"run_id": args.run_id, "datasets": {}, "parameters": vars(args)}
    for dataset in args.datasets:
        folder = root / "data" / dataset
        videos = sorted(str(path.relative_to(folder)).replace("\\", "/")
                        for path in folder.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
        if args.max_videos:
            videos = videos[:args.max_videos]
        if not videos:
            raise SystemExit(f"Nenhum video encontrado em {folder}")
        manifest["datasets"][dataset] = {"n_videos": len(videos), "videos": videos}
    manifests = root / "results" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{args.run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    compose = ["docker", "compose"]
    if not args.skip_build:
        run(compose + ["build", "mediapipe", "openface", "insightface", "analysis"])
    if not args.skip_cuda_check:
        run(compose + ["run", "--rm", "--no-deps", "--entrypoint", "python3", "insightface",
                       "-m", "src.diagnostics.check_cuda"])
    common_limits = []
    if args.max_frames:
        common_limits += ["--max-frames", str(args.max_frames)]
    if args.max_videos:
        common_limits += ["--max-videos", str(args.max_videos)]
    common_limits += ["--warmup-frames", str(args.warmup_frames)]
    for repetition in range(1, args.repetitions + 1):
        for dataset in args.datasets:
            base = ["--input", f"/data/{dataset}", "--dataset", dataset,
                    "--output-dir", "/results", "--run-id", args.run_id,
                    "--repetition", str(repetition)] + common_limits
            run(compose + ["run", "--rm", "--no-deps", "mediapipe"] + base)
            run(compose + ["run", "--rm", "--no-deps", "openface"] + base)
            if not args.without_insightface_cpu:
                run(compose + ["run", "--rm", "--no-deps", "insightface"] + base + ["--backend", "cpu"])
            run(compose + ["run", "--rm", "--no-deps", "insightface"] + base + ["--backend", "cuda"])
    run(compose + ["run", "--rm", "--no-deps", "analysis"])
    print(f"Benchmark concluido: run_id={args.run_id}")

if __name__ == "__main__":
    main()
