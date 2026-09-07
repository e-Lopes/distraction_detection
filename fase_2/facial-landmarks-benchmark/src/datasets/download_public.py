"""Download and unzip the three public video datasets with the Kaggle CLI."""
from __future__ import annotations

import argparse
import os
import subprocess

DATASETS = {
    "drozy": "ahmedfaroukksiu/drozy",
    "yawdd": "enider/yawdd-dataset",
    "nitymed": "nikospetrellis/nitymed",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit("Defina KAGGLE_USERNAME e KAGGLE_KEY antes do download.")
    for name in args.datasets:
        destination = os.path.join(args.data_dir, name)
        if os.path.isdir(destination) and any(os.scandir(destination)) and not args.force:
            print(f"[download] {name}: ja existe; use --force para baixar novamente")
            continue
        os.makedirs(destination, exist_ok=True)
        print(f"[download] {name}: {DATASETS[name]} -> {destination}")
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASETS[name],
                        "-p", destination, "--unzip", "--quiet"], check=True)

if __name__ == "__main__":
    main()
