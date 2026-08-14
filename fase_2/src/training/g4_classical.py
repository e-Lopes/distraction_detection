"""Executa o controle SVM da G4 nas estrategias B, C e D."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..data.config import load_yaml, repository_path
from ..data.splits import SplitBlock
from ..preprocessing.missingness import expand_behavior_labels
from .classical_baselines import (build_model, evaluate_predictions, experiment_fingerprint,
                                  load_checkpoint, predict_model, save_checkpoint)
from .dummy_baseline import CLASSES, load_series
from .g1_baselines import flatten_r0
from .temporal_data import build_sequence_fold
from .temporal_engine import class_weights, weighted_sample_indices


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def save_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("B","C","D"), action="append")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--max-samples-per-subset", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); scenarios = args.scenario or ["B","C","D"]
    folds = args.fold or [1,2,3,4]
    strategy = {"B":"class_weights", "C":"weighted_sampling", "D":"augmentation"}
    smoke_suffix = f"__smoke_{args.max_samples_per_subset}" if args.max_samples_per_subset else ""
    matrix = [{"run_id": f"G4__g4_{s.lower()}_{strategy[s]}_w60__svm__r0__w60__fold_{f}__seed_42{smoke_suffix}",
               "scenario": s, "fold": f} for s in scenarios for f in folds]
    print(json.dumps({"planned_run_count": len(matrix), "runs": matrix}, indent=2))
    if args.dry_run: return 0
    registry_path = root / "fase_2/configs/experiment/imbalance.yaml"
    g1_path = root / "fase_2/configs/experiment/g1_baselines.yaml"
    data_path = root / "fase_2/configs/data/base.yaml"
    preprocessing_path = root / "fase_2/configs/preprocessing/baseline_zero_fill.yaml"
    interval_path = root / "fase_2/data/manifests/annotation_frame_intervals.csv"
    split_path = root / "fase_2/data/manifests/temporal_splits.csv"
    input_dir = root / "fase_2/data/interim/legacy_extraction"
    registry, g1, data, preprocessing = map(load_yaml, (registry_path,g1_path,data_path,preprocessing_path))
    blocks = [SplitBlock(int(r["fold"]),r["subset"],r["video_id"],int(r["start_frame"]),int(r["end_frame"])) for r in read_csv(split_path)]
    series = load_series(input_dir)
    labels = expand_behavior_labels({k:len(v) for k,v in series.items()}, read_csv(interval_path))
    summaries: list[dict[str,object]]=[]; classes=[]; confusions=[]
    for scenario in scenarios:
        balancing = strategy[scenario]
        fingerprint = experiment_fingerprint(
            [registry_path,g1_path,data_path,preprocessing_path,interval_path,split_path,input_dir/"extraction_manifest.csv"],
            {"generation":"G4","model":"svm","scenario":scenario,"balancing":balancing,
             "window_size_frames":60,"seed":42,"max_samples":args.max_samples_per_subset})
        for fold in folds:
            audit: list[dict[str,object]]=[]
            splits, scaler, _ = build_sequence_fold(series,labels,blocks,preprocessing,fold=fold,size_frames=60,
                stride_frames=int(data["windowing"]["stride_frames"]),
                minimum_proportion=float(data["windowing"]["minimum_target_proportion"]), representation="R0",
                training_augmentation=(registry["augmentation"] if scenario=="D" else None),
                augmentation_seed=42, augmentation_audit=audit)
            if args.max_samples_per_subset:
                from .temporal_data import limit_sequence_split
                splits={k:limit_sequence_split(v,args.max_samples_per_subset,seed=42+fold) for k,v in splits.items()}
            x_train,y_train=flatten_r0(splits["train"]); original_counts=Counter(y_train)
            sampling_info=None
            if scenario=="C":
                indices=weighted_sample_indices(splits["train"].labels,seed=42)
                x_train=x_train[indices]; y_train=y_train[indices]
                sampling_info={"replacement":True,"num_samples":len(indices),"seed":42,
                    "indices_sha256":hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
                    "materialized_counts":dict(Counter(y_train))}
            run_id=f"G4__g4_{scenario.lower()}_{balancing}_w60__svm__r0__w60__fold_{fold}__seed_42{smoke_suffix}"
            checkpoint=root/f"fase_2/outputs/models/G4/{run_id}.joblib"
            loaded=None if args.no_resume else load_checkpoint(checkpoint,fingerprint)
            if loaded: model,meta=loaded; seconds=float(meta["training_seconds"]); resumed=True
            else:
                parameters=dict(g1["parameters"]["svm"])
                if scenario=="B":
                    weights=class_weights(splits["train"].labels)
                    parameters["class_weight"]={CLASSES[i]:float(weights[i]) for i in range(len(CLASSES))}
                model=build_model("svm",seed=42,xgb_device="cpu",parameters=parameters)
                started=time.perf_counter(); model.fit(x_train,y_train); seconds=time.perf_counter()-started; resumed=False
                save_checkpoint(model,checkpoint,{"fingerprint":fingerprint,"run_id":run_id,
                    "training_seconds":seconds,"scenario":scenario,"balancing":balancing})
            run_summary=[]; run_classes=[]; run_confusion=[]
            for subset in ("validation","test"):
                values,expected=flatten_r0(splits[subset]); predicted=predict_model("svm",model,values)
                summary,per_class,confusion=evaluate_predictions(model_name="svm",ablation="r0_flat",fold=fold,
                    subset=subset,expected=expected,predicted=predicted,train_seconds=seconds,resumed=resumed)
                context={"run_id":run_id,"generation":"G4","configuration_id":f"g4_{scenario.lower()}_{balancing}_w60",
                    "representation":"R0","window_size_frames":60,"scenario":scenario,"balancing":balancing,"seed":42}
                summary.update(context); [row.update(context) for row in per_class]; [row.update(context) for row in confusion]
                run_summary.append(summary); run_classes.extend(per_class); run_confusion.extend(confusion)
                write_csv(root/f"fase_2/outputs/predictions/G4/{run_id}__{subset}.csv",[
                    {"video_id":m.video_id,"start_frame":m.start_frame,"end_frame":m.end_frame,
                     "actual":a,"predicted":p} for m,a,p in zip(splits[subset].metadata,expected,predicted,strict=True)])
            if audit: write_csv(root/f"fase_2/outputs/metrics/G4/augmentation/g4_d_w60__fold_{fold}.csv",audit)
            log={"completed":True,"fingerprint":fingerprint,"run_id":run_id,"model":"svm","fold":fold,
                 "seed":42,"scenario":scenario,"balancing":balancing,"summary":run_summary,"per_class":run_classes,
                 "confusion":run_confusion,"train_class_counts":dict(original_counts),"sampling":sampling_info,
                 "augmentation_record_count":len(audit),"scaler":{"mean":scaler.mean,"scale":scaler.scale}}
            save_json(root/f"fase_2/outputs/logs/G4/{run_id}.json",log)
            summaries.extend(run_summary); classes.extend(run_classes); confusions.extend(run_confusion)
    write_csv(root/"fase_2/outputs/metrics/G4/g4_svm_runs.csv",summaries)
    write_csv(root/"fase_2/outputs/metrics/G4/g4_svm_per_class.csv",classes)
    write_csv(root/"fase_2/outputs/metrics/G4/g4_svm_confusion.csv",confusions)
    print(f"SVM G4 concluido: {len(matrix)} runs"); return 0


if __name__ == "__main__": raise SystemExit(main())
