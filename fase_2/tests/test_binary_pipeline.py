from pathlib import Path
import json
import numpy as np
import pytest
import yaml

from fase_2.src import pipeline
from fase_2.src.data.targets import ATTENTION_CLASSES, map_target_labels, target_classes
from fase_2.src.data.windowing import label_window


@pytest.fixture
def binary_config(tmp_path):
    config=pipeline.load_config('fase_2/configs/binary_experiment.yaml')
    series=tmp_path/'series';series.mkdir()
    videos=[];annotations=[];blocks=[]
    ids=[f'video_{i:02d}' for i in range(1,5)]
    for video in ids:
        videos.append(dict(video_id=video,fps=10,num_frames=360,width=100,height=100,
                           duration_seconds=36,relative_path=f'missing/{video}.mp4'))
        rows=[]
        for frame in range(360):
            label=['alert','fatigue','distraction'][(frame//60)%3]
            signal=.3 if label=='alert' else .1
            rows.append(dict(video_id=video,frame_index=frame,timestamp_seconds=frame/10,
                             ear=signal,mar=.2,pitch=frame%3,yaw=0,roll=0,face_detected=1))
        pipeline._write_csv_atomic(series/f'{video}.csv',rows)
        for start in range(0,360,60):
            annotations.append(dict(video_id=video,start_frame=start,end_frame=start+59,
                                    behavior_label=['alert','fatigue','distraction'][(start//60)%3],
                                    operational_state='valid'))
    for fold,test in enumerate(ids,1):
        blocks.append(dict(fold=fold,subset='test',video_id=test,start_frame=0,end_frame=359))
        for video in ids:
            if video==test:continue
            blocks.extend([dict(fold=fold,subset='train',video_id=video,start_frame=0,end_frame=149),
                           dict(fold=fold,subset='validation',video_id=video,start_frame=180,end_frame=329)])
    for name,rows in [('videos',videos),('annotations',annotations),('splits',blocks),('extraction',videos)]:
        pipeline._write_csv_atomic(tmp_path/f'{name}.csv',rows)
    config['data'].update(manifest=str(tmp_path/'videos.csv'),annotations=str(tmp_path/'annotations.csv'),
                          facial_series=str(series),extraction_manifest=str(tmp_path/'extraction.csv'))
    config['splits'].update(manifest=str(tmp_path/'splits.csv'),purge_gap_frames=30)
    config['windowing']['sizes_frames']=[30]
    for section in config['models']['screening'].values():
        if section.get('enabled'):section['windows']=[30]
    config['outputs']['root']=str(tmp_path/'binary_output')
    config['training'].update(device='cpu',max_epochs=2,batch_size=8)
    config['training']['early_stopping']['patience']=1
    path=tmp_path/'binary.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
    return path


def test_map_before_majority_and_exclude_unavailable():
    config=pipeline.load_config('fase_2/configs/binary_experiment.yaml')
    original={'v':['alert']*4+['fatigue']*3+['distraction']*3}
    binary=map_target_labels(original,config)
    assert label_window(original['v'],behavior_classes=set(config['data']['classes']))[0]=='mixed'
    assert label_window(binary['v'],behavior_classes=set(ATTENTION_CLASSES))[0]=='distraction'
    assert original['v'][4]=='fatigue'
    assert map_target_labels({'v':[None,'operator_absent','face_missing','occlusion','mixed']},config)['v']==[None]*5
    with pytest.raises(ValueError):map_target_labels({'v':['typo']},config)


def test_binary_metric_has_two_classes():
    from fase_2.src.training.classical_baselines import evaluate_predictions
    labels=np.array(['attention','distraction'])
    summary,per_class,confusion=evaluate_predictions(model_name='test',ablation='test',fold=1,
        subset='validation',expected=labels,predicted=labels,train_seconds=0,resumed=False,classes=ATTENTION_CLASSES)
    assert summary['macro_f1_all_classes']==1.0
    assert len(per_class)==2 and len(confusion)==4


def test_prepare_and_classical_fits_are_binary_and_resume(binary_config):
    from fase_2.src.reporting import generate_report
    import joblib
    assert pipeline.prepare(binary_config)==0
    config=pipeline.load_config(binary_config)
    counts=pipeline._read_csv(Path(config['outputs']['root'])/'window_counts.csv')
    assert set(r['label'] for r in counts)<={'attention','distraction','mixed'}
    runs=[r for r in pipeline.build_plan(binary_config,scope='screening')
          if r.fold==1 and r.model in {'dummy','svm','xgboost','fixed_rules'}]
    pipeline._train_screening(config,runs,force=False,allow_expensive=False)
    current={r.run_id:r for r in pipeline.build_plan(binary_config,scope='screening')}
    assert all(current[r.run_id].status=='completed' for r in runs)
    for run in runs:
        rows=pipeline._read_csv(Path(config['outputs']['predictions'])/f'{run.run_id}__validation.csv')
        assert set(r['actual'] for r in rows)==set(ATTENTION_CLASSES)
        assert not any('fatigue' in k for k in rows[0])
        assert not (Path(config['outputs']['predictions'])/f'{run.run_id}__test.csv').exists()
        if run.model=='xgboost':
            model=joblib.load(current[run.run_id].artifact)
            assert set(model.classes_)==set(ATTENTION_CLASSES)
            assert set(r['predicted'] for r in rows)<=set(ATTENTION_CLASSES)
    before={r.run_id:Path(current[r.run_id].artifact).stat().st_mtime_ns for r in runs}
    pipeline._train_screening(config,[current[r.run_id] for r in runs],force=False,allow_expensive=False)
    assert before=={r.run_id:Path(current[r.run_id].artifact).stat().st_mtime_ns for r in runs}
    generate_report(binary_config)
    metrics=pipeline._read_csv(Path(config['outputs']['metrics']))
    assert metrics and {r['source'] for r in metrics}=={'binary_training'}
    assert {r['label'] for r in metrics if r['scope']=='class'}==set(ATTENTION_CLASSES)
    assert pipeline.sync_historical_registry(binary_config)==0


@pytest.mark.parametrize('balancing,loss',[('class_weights','cross_entropy'),('weighted_sampling','cross_entropy'),('none','focal')])
def test_neural_binary_outputs_loss_weights_and_persistence(binary_config,balancing,loss):
    import torch
    torch.set_num_threads(1)
    config=pipeline.load_config(binary_config)
    config['models']['screening']['deep']['balancing']=balancing
    config['training']['loss']=loss
    binary_config.write_text(yaml.safe_dump(config))
    config=pipeline.load_config(binary_config)
    run=next(r for r in pipeline.build_plan(binary_config,scope='screening') if r.model=='lstm' and r.fold==1)
    pipeline._train_temporal(binary_config,config,[run],False)
    current=next(r for r in pipeline.build_plan(binary_config,scope='screening') if r.run_id==run.run_id)
    assert current.status=='completed'
    data=json.loads(next((Path(config['outputs']['root'])/'runs').glob('*.json')).read_text())
    assert data['classes']==list(ATTENTION_CLASSES)
    assert set(data['train_class_counts'])==set(ATTENTION_CLASSES)
    for key in ['class_weights','focal_alpha']:
        if data[key] is not None: assert len(data[key])==2
    rows=pipeline._read_csv(next(Path(config['outputs']['predictions']).glob('*.csv')))
    assert 'prob_fatigue' not in rows[0]
    assert all(abs(float(r['prob_attention'])+float(r['prob_distraction'])-1)<1e-5 for r in rows)


def test_suite_ui_and_fresh_round(tmp_path,monkeypatch):
    from fase_2.src.desktop import execution_command
    from fase_2.__main__ import build_parser
    from fase_2.src.binary_suite import new_round
    path=Path('fase_2/configs/binary_suite.yaml').resolve()
    command=execution_command(path,'chain','Comparar modelos','Todos',allow_expensive=True)
    args=build_parser().parse_args(command[4:])
    assert args.command=='suite' and args.action=='chain' and args.allow_expensive
    old=pipeline.load_config(path)
    # Resolve child paths before changing cwd; fresh-round writes only within the temp project.
    old['suite']['profiles']=[dict(item,config=str(Path(item['config']).resolve())) for item in old['suite']['profiles']]
    file=tmp_path/'suite.yaml';file.write_text(yaml.safe_dump(old))
    monkeypatch.chdir(tmp_path)
    fresh=new_round(file)
    new=pipeline.load_config(fresh)
    assert len(new['suite']['profiles'])==len(old['suite']['profiles'])
    for entry in new['suite']['profiles']:
        child=pipeline.load_config(entry['config'])
        assert 'rounds' in child['outputs']['root']
        assert not Path(child['outputs']['registry']).exists()
        assert target_classes(child)==ATTENTION_CLASSES
