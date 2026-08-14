"""Consolida as 64 comparacoes, figuras e rastreabilidade da G4."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ..data.config import load_yaml
from .dummy_baseline import CLASSES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
METRICS = ROOT / "fase_2/outputs/metrics/G4"
FIGURES = ROOT / "fase_2/outputs/figures/G4"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows: raise ValueError(f"Tabela vazia: {path}")
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n",extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def context(model: str, scenario: str, fold: int) -> dict[str, object]:
    return {"model":model,"scenario":scenario,"strategy":{"A":"none","B":"class_weights","C":"weighted_sampling","D":"augmentation"}[scenario],"fold":fold,"seed":42}


def load_all() -> tuple[list[dict[str,object]],list[dict[str,object]],list[dict[str,object]],list[dict[str,object]]]:
    summaries=[]; per_class=[]; confusion=[]; histories=[]
    g1_runs=read_csv(ROOT/"fase_2/outputs/metrics/G1/g1_runs.csv")
    g1_class=read_csv(ROOT/"fase_2/outputs/metrics/G1/g1_per_class.csv")
    g1_conf=read_csv(ROOT/"fase_2/outputs/metrics/G1/g1_confusion.csv")
    wanted=lambda r:r["model"]=="svm" and int(r["window_size_frames"])==60
    for target,source in ((summaries,g1_runs),(per_class,g1_class),(confusion,g1_conf)):
        for row in source:
            if wanted(row): target.append({**row,**context("svm","A",int(row["fold"]))})
    for model,window,cid in (("lstm",60,"qualification_r0_w60_b1024"),("tcn",60,"qualification_r0_w60_b1024"),("transformer",150,"qualification_r0_w150_b1024")):
        for fold in (1,2,3,4):
            run_id=f"G2__{cid}__{model}__r0__w{window}__fold_{fold}__seed_42"
            log=json.loads((ROOT/f"fase_2/outputs/logs/G2/{run_id}.json").read_text(encoding="utf-8"))
            ctx=context(model,"A",fold)
            summaries.extend({**r,**ctx} for r in log["summary"])
            per_class.extend({**r,**ctx} for r in log["per_class"])
            confusion.extend({**r,**ctx} for r in log["confusion"])
            histories.extend({**r,**ctx} for r in log["history"])
    for path in sorted((ROOT/"fase_2/outputs/logs/G4").glob("G4__*.json")):
        if "smoke" in path.name: continue
        log=json.loads(path.read_text(encoding="utf-8"))
        if not log.get("completed"): raise ValueError(f"Run incompleto: {path}")
        scenario=str(log["scenario"] if "scenario" in log else str(log["configuration_id"]).split("_")[1]).upper()
        ctx=context(str(log["model"]),scenario,int(log["fold"]))
        summaries.extend({**r,**ctx} for r in log["summary"])
        per_class.extend({**r,**ctx} for r in log["per_class"])
        confusion.extend({**r,**ctx} for r in log["confusion"])
        histories.extend({**r,**ctx} for r in log.get("history",[]))
    return summaries,per_class,confusion,histories


def aggregate(rows: Sequence[Mapping[str,object]], value: str, *, include_label: bool=False) -> list[dict[str,object]]:
    groups=defaultdict(list)
    for row in rows:
        key=(str(row["model"]),str(row["scenario"]),str(row.get("subset","validation")))
        if include_label: key += (str(row["label"]),)
        raw=row[value]
        groups[key].append(float(raw) if raw not in ("",None) else 0.0)
    output=[]
    for key,values in sorted(groups.items()):
        result={"model":key[0],"scenario":key[1],"strategy":{"A":"none","B":"class_weights","C":"weighted_sampling","D":"augmentation"}[key[1]],"subset":key[2]}
        if include_label: result["label"]=key[3]
        result.update({"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"median":float(np.median(values)),"minimum":min(values),"maximum":max(values),"fold_count":len(values)})
        output.append(result)
    return output


def paired_deltas(summaries: Sequence[Mapping[str,object]], classes: Sequence[Mapping[str,object]]) -> list[dict[str,object]]:
    base={(str(r["model"]),int(r["fold"]),str(r["subset"])):float(r["macro_f1_all_classes"]) for r in summaries if r["scenario"]=="A"}
    fatigue={(str(r["model"]),str(r["scenario"]),int(r["fold"]),str(r["subset"])):r for r in classes if r["label"]=="fatigue"}
    output=[]
    for row in summaries:
        if row["scenario"]=="A": continue
        key=(str(row["model"]),int(row["fold"]),str(row["subset"]))
        a=fatigue.get((key[0],"A",key[1],key[2]),{}); b=fatigue.get((key[0],str(row["scenario"]),key[1],key[2]),{})
        number=lambda value: float(value) if value not in ("",None) else 0.0
        output.append({"model":key[0],"scenario":row["scenario"],"strategy":row["strategy"],"fold":key[1],"subset":key[2],
            "macro_f1_a":base[key],"macro_f1_scenario":float(row["macro_f1_all_classes"]),"delta_macro_f1":float(row["macro_f1_all_classes"])-base[key],
            "fatigue_recall_a":number(a.get("recall",0)),"fatigue_recall_scenario":number(b.get("recall",0)),"delta_fatigue_recall":number(b.get("recall",0))-number(a.get("recall",0)),
            "fatigue_f1_a":number(a.get("f1",0)),"fatigue_f1_scenario":number(b.get("f1",0)),"delta_fatigue_f1":number(b.get("f1",0))-number(a.get("f1",0))})
    return output


def prediction_path(run_id: str, generation: str, subset: str) -> Path:
    return ROOT/f"fase_2/outputs/predictions/{generation}/{run_id}__{subset}.csv"


def prediction_analysis(summaries: Sequence[Mapping[str,object]]) -> tuple[list[dict[str,object]],list[dict[str,object]]]:
    videos={r["video_id"]:r for r in read_csv(ROOT/"fase_2/data/manifests/videos.csv")}
    distributions=[]; operational=[]
    seen=set()
    for summary in summaries:
        key=(str(summary["run_id"]),str(summary["subset"]));
        if key in seen: continue
        seen.add(key); generation=str(summary["generation"])
        rows=read_csv(prediction_path(key[0],generation,key[1])); counts=Counter(r["predicted"] for r in rows)
        distributions.append({"run_id":key[0],"model":summary["model"],"scenario":summary["scenario"],"strategy":summary["strategy"],"fold":summary["fold"],"subset":key[1],"total":len(rows),**{f"predicted_{c}":counts.get(c,0) for c in CLASSES}})
        by_video=defaultdict(list)
        for row in rows: by_video[row["video_id"]].append(row)
        for video_id,items in by_video.items():
            items.sort(key=lambda r:int(r["start_frame"])); false=[r for r in items if r["predicted"]=="fatigue" and r["actual"]!="fatigue"]
            episodes=0; previous_end=-10**9
            for row in false:
                start,end=int(row["start_frame"]),int(row["end_frame"])
                if start>previous_end+15: episodes+=1
                previous_end=max(previous_end,end)
            fps=float(videos[video_id]["fps"]); observed=(max(int(r["end_frame"]) for r in items)-min(int(r["start_frame"]) for r in items)+1)/fps/3600
            operational.append({"run_id":key[0],"model":summary["model"],"scenario":summary["scenario"],"strategy":summary["strategy"],"fold":summary["fold"],"subset":key[1],"video_id":video_id,"session_id":videos[video_id]["session_id"],"observed_hours":observed,"false_fatigue_windows":len(false),"false_fatigue_episodes":episodes,"false_windows_per_hour":len(false)/observed,"false_episodes_per_hour":episodes/observed})
    return distributions,operational


def plot_bars(stats,classes,operational,deltas,histories,confusion):
    FIGURES.mkdir(parents=True,exist_ok=True); models=("svm","lstm","tcn","transformer"); scenarios=("A","B","C","D"); colors=("#777777","#377eb8","#e41a1c","#4daf4a")
    for subset in ("validation","test"):
        fig,ax=plt.subplots(figsize=(11,5)); width=.18; x=np.arange(4)
        for i,s in enumerate(scenarios):
            vals=[next(r["mean"] for r in stats if r["model"]==m and r["scenario"]==s and r["subset"]==subset) for m in models]
            errs=[next(r["std"] for r in stats if r["model"]==m and r["scenario"]==s and r["subset"]==subset) for m in models]
            ax.bar(x+(i-1.5)*width,vals,width,yerr=errs,label=s,color=colors[i],capsize=3)
        ax.set_xticks(x,models); ax.set_ylabel("Macro F1 (media ± DP entre folds)"); ax.set_title(f"G4 - {subset}"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/f"macro_f1_{subset}.png",dpi=180); plt.close(fig)
    fatigue=[r for r in classes if r["label"]=="fatigue" and r["subset"]=="validation"]
    fstats=aggregate(fatigue,"recall")
    fig,ax=plt.subplots(figsize=(11,5)); x=np.arange(4); width=.18
    for i,s in enumerate(scenarios): ax.bar(x+(i-1.5)*width,[next(r["mean"] for r in fstats if r["model"]==m and r["scenario"]==s) for m in models],width,label=s,color=colors[i])
    ax.set_xticks(x,models); ax.set_ylabel("Recall de Fatigue"); ax.set_title("Recuperacao da classe rara na validacao"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"fatigue_recall_validation.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); grouped=[[float(r["delta_macro_f1"]) for r in deltas if r["model"]==m and r["scenario"]==s and r["subset"]=="validation"] for m in models for s in scenarios[1:]]
    ax.boxplot(grouped,tick_labels=[f"{m}\n{s}" for m in models for s in scenarios[1:]],showmeans=True); ax.axhline(0,color="black",lw=1); ax.set_ylabel("Delta Macro F1 vs A"); fig.tight_layout(); fig.savefig(FIGURES/"paired_delta_macro_f1.png",dpi=180); plt.close(fig)
    op=defaultdict(list)
    for r in operational:
        if r["subset"]=="validation": op[(r["model"],r["scenario"])].append(float(r["false_episodes_per_hour"]))
    fig,ax=plt.subplots(figsize=(11,5));
    for i,s in enumerate(scenarios): ax.bar(x+(i-1.5)*width,[np.mean(op[(m,s)]) for m in models],width,label=s,color=colors[i])
    ax.set_xticks(x,models); ax.set_ylabel("Falsos episodios Fatigue/hora"); ax.set_title("Custo operacional na validacao"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"false_fatigue_episodes_per_hour.png",dpi=180); plt.close(fig)
    temporal=[r for r in histories if r["scenario"] in scenarios]
    for model in ("lstm","tcn","transformer"):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for scenario,color in zip(scenarios,colors):
            selected=[r for r in temporal if r["model"]==model and r["scenario"]==scenario]
            for fold in (1,2,3,4):
                curve=sorted((r for r in selected if int(r["fold"])==fold),key=lambda r:int(r["epoch"]));
                if curve: axes[0].plot([r["epoch"] for r in curve],[r["validation_macro_f1"] for r in curve],color=color,alpha=.35)
            axes[0].plot([],[],color=color,label=scenario)
        axes[0].set_title(f"{model}: Macro F1 por epoch"); axes[0].legend(); axes[0].set_xlabel("epoch")
        for scenario,color in zip(scenarios,colors):
            selected=[r for r in temporal if r["model"]==model and r["scenario"]==scenario]
            for fold in (1,2,3,4):
                curve=sorted((r for r in selected if int(r["fold"])==fold),key=lambda r:int(r["epoch"]));
                if curve: axes[1].plot([r["epoch"] for r in curve],[r["validation_loss"] for r in curve],color=color,alpha=.35)
        axes[1].set_title("Validation loss"); axes[1].set_xlabel("epoch"); fig.tight_layout(); fig.savefig(FIGURES/f"training_curves_{model}.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(13,12))
    for i,m in enumerate(models):
        for j,s in enumerate(scenarios):
            matrix=np.zeros((3,3),dtype=int)
            for r in confusion:
                if r["model"]==m and r["scenario"]==s and r["subset"]=="validation": matrix[CLASSES.index(r["actual"]),CLASSES.index(r["predicted"])]+=int(r["count"])
            axes[i,j].imshow(matrix,cmap="Blues"); axes[i,j].set_title(f"{m} / {s}")
            for y in range(3):
                for z in range(3): axes[i,j].text(z,y,str(matrix[y,z]),ha="center",va="center",fontsize=7)
            axes[i,j].set_xticks(range(3),CLASSES,rotation=45,ha="right",fontsize=7); axes[i,j].set_yticks(range(3),CLASSES,fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES/"confusion_validation_all.png",dpi=180); plt.close(fig)


def manifest() -> list[dict[str,object]]:
    rows=[]
    roots=[ROOT/"fase_2/configs/experiment",ROOT/"fase_2/src/training",ROOT/"fase_2/docs",ROOT/"fase_2/outputs/models/G4",ROOT/"fase_2/outputs/logs/G4",ROOT/"fase_2/outputs/predictions/G4",METRICS,FIGURES,ROOT/"fase_2/reports"]
    for base in roots:
        for path in sorted(base.rglob("*")):
            if path.is_file() and "smoke" not in path.as_posix().lower() and path.name!="g4_artifact_manifest.csv":
                rows.append({"artifact_type":base.name,"relative_path":path.relative_to(ROOT).as_posix(),"size_bytes":path.stat().st_size,"sha256":sha256(path)})
    return rows


def main() -> int:
    summaries,classes,confusion,histories=load_all()
    if len({r["run_id"] for r in summaries})!=64: raise ValueError("G4 exige 64 comparacoes")
    stats=aggregate(summaries,"macro_f1_all_classes"); class_stats=aggregate(classes,"f1",include_label=True)
    class_metric_stats=[]
    for metric in ("precision","recall","f1"):
        class_metric_stats.extend({**row,"metric":metric} for row in aggregate(classes,metric,include_label=True))
    deltas=paired_deltas(summaries,classes); distribution,operational=prediction_analysis(summaries)
    stability=[]
    for model in ("lstm","tcn","transformer"):
        for scenario in ("A","B","C","D"):
            rows=[r for r in summaries if r["model"]==model and r["scenario"]==scenario and r["subset"]=="validation"]
            for metric in ("best_epoch","stopping_epoch","best_validation_macro_f1","training_seconds"):
                values=[float(r[metric]) for r in rows]
                stability.append({"model":model,"scenario":scenario,"metric":metric,"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"minimum":min(values),"maximum":max(values)})
    operational_stats=[]
    for model in ("svm","lstm","tcn","transformer"):
        for scenario in ("A","B","C","D"):
            values=[float(r["false_episodes_per_hour"]) for r in operational if r["model"]==model and r["scenario"]==scenario and r["subset"]=="validation"]
            operational_stats.append({"model":model,"scenario":scenario,"metric":"false_fatigue_episodes_per_hour","mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"median":float(np.median(values)),"minimum":min(values),"maximum":max(values),"session_fold_count":len(values)})
    treatment=[]
    for path in sorted((ROOT/"fase_2/outputs/logs/G4").glob("G4__*.json")):
        if "smoke" in path.name: continue
        log=json.loads(path.read_text(encoding="utf-8")); counts=log.get("train_class_counts",{})
        scenario=str(log["scenario"]).upper() if "scenario" in log else str(log["configuration_id"]).split("_")[1].upper()
        treatment.append({"run_id":log["run_id"],"model":log["model"],"scenario":scenario,"fold":log["fold"],"balancing":log["balancing"],
            "train_alert":counts.get("alert",0),"train_fatigue":counts.get("fatigue",0),"train_distraction":counts.get("distraction",0),
            "class_weights":json.dumps(log.get("class_weights"),separators=(",",":")),"sampling":json.dumps(log.get("sampling"),separators=(",",":")),"augmentation_record_count":log.get("augmentation_record_count",0)})
    for name,rows in (("g4_runs.csv",summaries),("g4_per_class.csv",classes),("g4_confusion.csv",confusion),("g4_history.csv",histories),("g4_fold_statistics.csv",stats),("g4_per_class_statistics.csv",class_stats),("g4_per_class_metric_statistics.csv",class_metric_stats),("g4_paired_deltas.csv",deltas),("g4_prediction_distribution.csv",distribution),("g4_false_fatigue_operational.csv",operational),("g4_false_fatigue_statistics.csv",operational_stats),("g4_training_stability.csv",stability),("g4_treatment_audit.csv",treatment)): write_csv(METRICS/name,rows)
    plot_bars(stats,classes,operational,deltas,histories,confusion)
    validation=[r for r in stats if r["subset"]=="validation"]
    ranking=sorted(validation,key=lambda r:float(r["mean"]),reverse=True); write_csv(METRICS/"g4_ranking.csv",ranking)
    fingerprints={}
    for path in sorted((ROOT/"fase_2/outputs/logs/G4").glob("G4__*.json")):
        if "smoke" not in path.name:
            log=json.loads(path.read_text(encoding="utf-8")); fingerprints[log["run_id"]]=log["fingerprint"]
    resolved={"registry":load_yaml(ROOT/"fase_2/configs/experiment/imbalance.yaml"),"official_new_runs":48,"reused_runs":16,"run_fingerprints":fingerprints,"determinism_limitation":"PyTorch memory-efficient Transformer attention on CUDA warned that backward is not bitwise deterministic; seed and all other deterministic controls remained fixed."}
    (METRICS/"g4_resolved_config.json").write_text(json.dumps(resolved,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    best=ranking[0]; fatigue_stats=aggregate([r for r in classes if r["label"]=="fatigue"],"f1",include_label=True)
    best_fat=next(r for r in fatigue_stats if r["model"]==best["model"] and r["scenario"]==best["scenario"] and r["subset"]=="validation")
    fatigue_recall={(r["model"],r["scenario"]):r for r in class_metric_stats if r["subset"]=="validation" and r["label"]=="fatigue" and r["metric"]=="recall"}
    macro={(r["model"],r["scenario"]):r for r in validation}
    fatigue_f1={(r["model"],r["scenario"]):r for r in class_stats if r["subset"]=="validation" and r["label"]=="fatigue"}
    false_rate={(r["model"],r["scenario"]):r for r in operational_stats}
    lines=["# G4 — Mitigação isolada do desbalanceamento","","## Resultado executivo","",f"Foram consolidados **48 runs novos** e **16 reutilizados**, totalizando **64 comparações**. O maior Macro F1 médio de validação foi **{float(best['mean']):.4f} ± {float(best['std']):.4f}**, em **{best['model']} / cenário {best['scenario']} ({best['strategy']})**. O F1 médio de Fatigue nessa configuração foi **{float(best_fat['mean']):.4f} ± {float(best_fat['std']):.4f}**.","",f"Esse máximo não resolve a classe rara: augmentation manteve Fatigue em zero nos temporais. O compromisso temporal mais equilibrado foi **LSTM/B**, com Macro F1 **{macro[('lstm','B')]['mean']:.4f} ± {macro[('lstm','B')]['std']:.4f}**, F1 de Fatigue **{fatigue_f1[('lstm','B')]['mean']:.4f} ± {fatigue_f1[('lstm','B')]['std']:.4f}**, recall **{fatigue_recall[('lstm','B')]['mean']:.4f}** e **{false_rate[('lstm','B')]['mean']:.2f}** falsos episódios/hora.","","Os resultados permanecem qualificatórios: há somente a seed 42. H3 não é confirmada antes das cinco seeds dos finalistas.","","## Desenho e controles","","- R0, folds, janelas, arquiteturas, hiperparâmetros e orçamento foram congelados.","- A foi reutilizado após verificação de hashes; B, C e D foram aplicados isoladamente e apenas no treino.","- SVM/60 atuou como controle clássico nas quatro condições.","- Validação e teste não receberam sampling ou augmentation.","","## Ranking de validação (média ± DP entre quatro folds)",""]
    for row in ranking: lines.append(f"- {row['model']} / {row['scenario']}: {float(row['mean']):.4f} ± {float(row['std']):.4f}")
    lines += ["","## Interpretação científica","",f"B e C recuperaram Fatigue nos temporais, mas reduziram o Macro F1. TCN/C obteve o maior recall médio de Fatigue (**{fatigue_recall[('tcn','C')]['mean']:.4f}**), ao custo de Macro F1 **{macro[('tcn','C')]['mean']:.4f}** e **{false_rate[('tcn','C')]['mean']:.2f}** falsos episódios/hora. D preservou o Macro F1, porém não recuperou Fatigue nos temporais. A comparação principal usa médias entre folds e deltas pareados contra A; nenhum melhor fold isolado determina a seleção.","","Para a comparação arquitetural justa, LSTM/B superou o controle SVM/B tanto em Macro F1 quanto em F1 de Fatigue, mas não superou o melhor SVM sem tratamento/augmentation. Portanto, há evidência preliminar favorável à modelagem temporal sob a mesma estratégia B, mas H3 continua aberta.","","## Reprodutibilidade e limitação","","Cada temporal possui checkpoints `last.pt` e `best_macro_f1.pt`, histórico por epoch, configuração resolvida, fingerprint e predições. O PyTorch advertiu que a atenção memory-efficient do Transformer em CUDA não é bit a bit determinística; esta limitação está registrada e deve ser considerada ao interpretar/repetir Transformer.","","## Decisão","","A seleção Pareto recomendada para a futura etapa de cinco seeds é **LSTM/B** como candidato primário, acompanhado de **SVM/B** como controle clássico pareado. **TCN/C** e **SVM/C** ficam como análise secundária de alta sensibilidade a Fatigue, devido ao custo operacional elevado. SVM/D é preservado como melhor Macro F1 observado, mas não é tratado como solução do desbalanceamento porque praticamente reproduziu A. A G5 não foi iniciada.",""]
    report=ROOT/"fase_2/reports/g4_imbalance_results.md"; report.write_text("\n".join(lines),encoding="utf-8")
    write_csv(METRICS/"g4_artifact_manifest.csv",manifest())
    print(f"G4 consolidada: 64 comparacoes; melhor={best['model']}/{best['scenario']} {float(best['mean']):.4f}"); return 0


if __name__=="__main__": raise SystemExit(main())
