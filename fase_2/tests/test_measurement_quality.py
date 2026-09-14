import copy
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fase_2.src.features.measurement_raw import letterbox, original_points, OperatorLock
from fase_2.src.features.extract_facial_series import compute_ear
from fase_2.src.preprocessing.measurement_quality import treat_block, region_valid, legacy_angles, relative_rotation, rotation_from_angles
from fase_2.src.pipeline import load_config, build_plan, pipeline_fingerprint


def raw(n=12, video='v1'):
    result=[]
    for i in range(n):
        r=dict(schema_version='measurement_v1', video_id=video, frame_index=str(i),timestamp_seconds=str(i*.05),
            ear='.25',ear_left='.2',ear_right='.3',mar='.1',pitch='0',yaw='0',roll='0',
            legacy_pitch='0',legacy_yaw='0',legacy_roll='0',
            valid_left='1',valid_right='1',valid_mouth='1',valid_pose='1',
            face_detected='1',tracking_status='tracked',track_id='operator:0',
            eye_width_left='12',eye_width_right='20',rotation_matrix=json.dumps(np.eye(3).tolist()))
        result.append(r)
    return result


def test_letterbox_original_pixels_and_ear_aspect():
    frame=np.zeros((1080,1920,3),np.uint8)
    _,g=letterbox(frame,[1100,400,500,300],[640,640])
    pts=np.array([[1200,500],[1205,497],[1215,497],[1220,500],[1215,503],[1205,503]],float)
    x,y,_,_=g['roi'];px,py=g['padding']
    landmarks=[SimpleNamespace(x=((a-x)*g['scale_x']+px)/640,y=((b-y)*g['scale_y']+py)/640) for a,b in pts]
    np.testing.assert_allclose(original_points(landmarks,g),pts)
    original=[SimpleNamespace(x=a/1920,y=b/1080) for a,b in pts]
    assert compute_ear(original,list(range(6)),1920,1080)==pytest.approx(.3,abs=1e-6)


def test_tracker_does_not_reacquire_other_person_after_loss():
    person=np.array([[10,10],[20,20]])
    tracker=OperatorLock([10,10,10,10],.25)
    assert tracker.select([person])[1]=='tracked'
    assert tracker.select([])[0] is None
    assert tracker.select([person])[1]=='lost_requires_review'
    tracker=OperatorLock([10,10,10,10],.25)
    assert tracker.select([person,person+1])[0] is None
    assert OperatorLock(None,.25).select([person])[0] is None


def test_partial_missing_and_no_value_based_rejection():
    rows=raw(3); before=copy.deepcopy(rows)
    rows[1]['valid_left']='0'; rows[1]['ear_left']='.9'
    rows[1]['ear_right']='0'; rows[1]['mar']='4'
    r=treat_block(rows,{'eye':'mean','reject_quality':True})
    assert float(r[1]['ear'])==0 and float(r[1]['mar'])==4
    assert r[1]['ear_source']=='right'
    rows[1]['valid_right']='0'
    r=treat_block(rows,{'eye':'mean'})
    assert np.isnan(float(r[1]['ear'])) and np.isfinite(float(r[1]['mar']))
    assert before[0]==rows[0]


@pytest.mark.parametrize('limit,expected',[(0,False),(.099,False),(.1,True),(.2,True)])
def test_gap_duration_is_timestamp_anchor_interval(limit,expected):
    rows=raw(5); rows[2]['mar']='nan'
    before=copy.deepcopy(rows)
    r=treat_block(rows,{'interpolation_seconds':limit})
    assert (r[2]['interpolated_mar']=='1')==expected
    assert float(r[2]['gap_seconds_mar'])==pytest.approx(.1)
    assert rows==before


def test_no_interpolation_across_identity_loss_or_long_gap_or_edge():
    rows=raw(10)
    for i in (0,3,4,5,9): rows[i]['mar']='nan'
    r=treat_block(rows,{'interpolation_seconds':.1})
    assert all(r[i]['interpolated_mar']=='0' for i in (0,3,4,5,9))
    rows=raw(5); rows[2]['mar']='nan'; rows[2]['tracking_status']='lost_requires_review'
    r=treat_block(rows,{'interpolation_seconds':.2})
    assert r[2]['interpolated_mar']=='0'
    rows=raw(5); rows[2]['mar']='nan'; rows[3]['track_id']='operator:3'
    assert treat_block(rows,{'interpolation_seconds':.2})[2]['interpolated_mar']=='0'


def test_rotation_composition_and_slerp_crossing_180():
    r0=Rotation.from_euler('xyz',[20,30,40],degrees=True).as_matrix()
    delta=Rotation.from_euler('xyz',[10,0,0],degrees=True).as_matrix()
    np.testing.assert_allclose(relative_rotation(r0@delta,r0),delta,atol=1e-7)
    rows=raw(3)
    rows[0]['roll']='179'; rows[2]['roll']='-179'
    rows[1]['valid_pose']='0'
    r=treat_block(rows,{'interpolation_seconds':.1})
    assert abs(abs(float(r[1]['roll']))-180)<1e-6
    assert r[1]['valid_roll']=='0' and r[1]['interpolated_roll']=='1'
    with pytest.raises(ValueError): relative_rotation(np.eye(3),np.zeros((3,3)))


def test_filter_does_not_cross_gap_or_future():
    rows=raw(5); rows[1]['mar']='nan'; rows[0]['mar']='100'; rows[2]['mar']='.1'; rows[3]['mar']='.2'; rows[4]['mar']='200'
    r=treat_block(rows,{'smoothing_seconds':.1})
    assert float(r[2]['mar'])==pytest.approx(.1)
    assert float(r[3]['mar'])==pytest.approx(.15)
    assert np.isnan(float(r[1]['mar']))


def test_forbid_old_schema_and_cross_session_or_frame_boundary():
    rows=raw(3);rows[1]['video_id']='v2'
    with pytest.raises(ValueError): treat_block(rows,{})
    rows=raw(3);rows[1]['frame_index']='20'
    with pytest.raises(ValueError): treat_block(rows,{})
    rows=raw(3); rows[1]['schema_version']='v1'
    with pytest.raises(ValueError): treat_block(rows,{})


def test_fold_observed_only_scaler_common_windows_and_serialization(tmp_path):
    from fase_2.src.data.splits import SplitBlock
    from fase_2.src.training.temporal_data import build_sequence_fold
    series={v:raw(12,v) for v in ('v1','v2')}
    for r in series['v1'][6:]: r['ear_left']='200';r['ear_right']='200'
    for r in series['v2']: r['mar']='500'
    for r in series['v1'][:3]: r['mar']='nan'
    blocks=[SplitBlock(1,'train','v1',0,5),SplitBlock(1,'validation','v1',6,11),SplitBlock(1,'test','v2',0,11)]
    labels={v:['alert']*12 for v in series}
    def build(rep):
        return build_sequence_fold(series,labels,blocks,{'measurement_policy':{'reject_quality':True,'eye':'mean'}},
            fold=1,size_frames=3,stride_frames=3,minimum_proportion=.6,representation=rep)
    splits,scaler,_=build('QB')
    assert scaler.mean[0]==pytest.approx(.25) and scaler.mean[1]==pytest.approx(.1)
    assert np.isfinite(splits['train'].values).all()
    assert np.all(splits['train'].values[0,:,1]==0)
    assert set(np.unique(splits['train'].values[:,:,5:]))<={0,1}
    assert splits['validation'].values.shape[-1]==15
    baseline,_,_=build('QA')
    assert [(m.video_id,m.start_frame,m.end_frame) for m in baseline['validation'].metadata]==[(m.video_id,m.start_frame,m.end_frame) for m in splits['validation'].metadata]
    joblib.dump(scaler,tmp_path/'scaler.joblib')
    assert joblib.load(tmp_path/'scaler.joblib')==scaler


def test_plan_budget_fingerprint_and_report(tmp_path):
    from fase_2.src.measurement_reporting import measurement_report
    p=Path('fase_2/configs/measurement_experiment.yaml');c=load_config(p)
    plan=build_plan(p,scope='screening')
    assert len(plan)==60
    assert {r.model for r in plan}=={'svm','lstm','missingness_logistic'}
    a=pipeline_fingerprint(p,c);c['measurement_protocol']['quality']['min_eye_width_px']+=1
    assert pipeline_fingerprint(p,c)!=a
    assert 'nenhuma cobertura científica' in measurement_report(c,[])


def test_existing_screening_pipeline_synthetic_predictions_and_resume(tmp_path):
    from fase_2.src.pipeline import _write_csv_atomic, _train_screening, PlannedRun, _read_csv
    config=load_config('fase_2/configs/measurement_experiment.yaml')
    for key in config['outputs']:
        config['outputs'][key]=str(tmp_path/key)
    config['outputs']['root']=str(tmp_path)
    series_path=tmp_path/'raw';series_path.mkdir()
    annotations=[]
    for video in ('v1','v2','v3'):
        rows=raw(12,video)
        for i,r in enumerate(rows):
            r['ear_left']=str(.1+(i//3)%3*.1)
            r['ear_right']=r['ear_left']
            annotations.append(dict(video_id=video,start_frame=i,end_frame=i,
                                    behavior_label=('alert','fatigue','distraction')[(i//3)%3]))
        _write_csv_atomic(series_path/f'video_{video}.csv',rows)
    _write_csv_atomic(tmp_path/'annotations.csv',annotations)
    _write_csv_atomic(tmp_path/'videos.csv',[{'video_id':v,'fps':20} for v in ('v1','v2','v3')])
    _write_csv_atomic(tmp_path/'splits.csv',[dict(fold=1,subset=s,video_id=v,start_frame=0,end_frame=11)
                     for s,v in zip(('train','validation','test'),('v1','v2','v3'))])
    config['data'].update(facial_series=str(series_path),annotations=str(tmp_path/'annotations.csv'),manifest=str(tmp_path/'videos.csv'))
    config['splits']['manifest']=str(tmp_path/'splits.csv')
    config['windowing']['stride_frames']=3
    runs=[PlannedRun(f'synthetic_{m}','screening','feature','classical','screening',m,'QB',3,1,42,
                     'class_weights','synthetic_hash','pending') for m in ('svm','missingness_logistic')]
    _train_screening(config,runs,force=False,allow_expensive=False)
    for r in runs:
        assert (Path(config['outputs']['predictions'])/f'{r.run_id}__validation.csv').exists()
        assert not (Path(config['outputs']['predictions'])/f'{r.run_id}__test.csv').exists()
    model=joblib.load(Path(config['outputs']['checkpoints'])/'synthetic_svm.joblib')
    assert model.n_features_in_==45
    _train_screening(config,runs,force=False,allow_expensive=False)
    assert len(_read_csv(tmp_path/'screening_metrics.csv'))==2


def test_lstm_fifteen_channels_and_synthetic_report(tmp_path):
    import torch
    from fase_2.src.models.temporal import build_temporal_model
    from fase_2.src.measurement_reporting import measurement_figures
    m=build_temporal_model('lstm',input_dim=15,num_classes=3,parameters={'hidden_dim':4,'num_layers':1,'dropout':0})
    x=torch.zeros((3,6,15));x[:,:,5:10]=1
    logits=m(x)
    torch.nn.functional.cross_entropy(logits,torch.arange(3)).backward()
    assert logits.shape==(3,3) and torch.isfinite(logits).all()
    c=load_config('fase_2/configs/measurement_experiment.yaml')
    c['outputs']['root']=str(tmp_path);c['outputs']['figures']=str(tmp_path/'figures')
    metrics=[dict(model='svm',representation=rep,fold=1,seed=42,metric='macro_f1',label='',subset='validation',value=v)
             for rep,v in [('QA',.2),('QB',.3)]]
    assert len(measurement_figures(c,metrics))==1
    assert (tmp_path/'quality_paired_deltas.csv').exists()
