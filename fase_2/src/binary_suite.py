"""Sequential, resumable binary experiment suite used by CLI and desktop."""
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def profiles(path):
    from .pipeline import load_config
    config = load_config(path)
    return config.get('suite', {}).get('profiles', [{'label': config['name'], 'config': str(path)}])


def plan(path, family='all', scope='all', paradigm='all'):
    from .pipeline import build_plan
    result=[]
    for item in profiles(path):
        for run in build_plan(item['config'],family,scope,paradigm):
            result.append(replace(run,run_id=f"{Path(item['config']).stem}::{run.run_id}",
                                  reason=f"{item['label']}: {run.reason}"))
    return result


def new_round(path):
    """Clone resolved configurations to a fresh root; preserve every prior artifact."""
    from .pipeline import load_config
    config=load_config(path)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root=Path('fase_2/outputs/binary_v1/rounds')/stamp
    directory=root/'configs';directory.mkdir(parents=True)
    entries=config.get('suite',{}).get('profiles',[{'label':config['name'],'config':str(path)}])
    items=[]
    for index,item in enumerate(entries):
        child=load_config(item['config'])
        child.pop('suite',None)
        child['outputs']['root']=str(root/f"{index:02d}_{Path(item['config']).stem}")
        child['training']['confirmation']['candidates']=[]
        target=directory/f'{index:02d}.yaml'
        target.write_text(yaml.safe_dump(child,sort_keys=False,allow_unicode=True),encoding='utf-8')
        items.append({'label':item['label'],'config':str(target)})
    config['suite']={'profiles':items}
    config['outputs']['root']=str(root/'summary')
    config['name']=f'Rodada binária {stamp}'
    target=directory/'suite.yaml'
    target.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return target


def run(path, *, action='chain', scope='screening', paradigm='all', allow_expensive=False):
    from .pipeline import load_config, _write_csv_atomic, build_plan
    from .workflow import save_state
    config=load_config(path)
    root=Path(config['outputs']['root']);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    state={'status':'running','profiles':[]}
    state_path=root/'suite_state.json'
    entries=profiles(path)
    env=os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','2');env.setdefault('MKL_NUM_THREADS','2')
    for index,item in enumerate(entries,1):
        print(f"\n[SUITE {index}/{len(entries)}] {item['label']}",flush=True)
        commands= ['prepare','train','report'] if action=='chain' else [action]
        record={'config':item['config'],'label':item['label'],'status':'running'}
        state['profiles'].append(record);save_state(state_path,state)
        for stage in commands:
            arguments=[sys.executable,'-u','-m','fase_2',stage,'--config',item['config']]
            if stage=='train':
                arguments += ['--scope',scope,'--paradigm',paradigm,'--resume']
                if allow_expensive:arguments.append('--allow-expensive')
            record['stage']=stage;save_state(state_path,state)
            with (logs/f'{index:02d}_{stage}.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                         text=True,env=env)
                try:
                    for line in process.stdout:
                        print(line,end='',flush=True);log.write(line);log.flush()
                    code=process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate();process.wait()
            if code:
                record.update(status='failed',exit_code=code)
                break
        if record['status']=='running':
            pending=[r for r in build_plan(item['config'],scope=scope,paradigm=paradigm)
                     if r.status!='completed']
            record['status']='pending' if pending and action=='chain' else 'completed'
            record['remaining']=len(pending)
        save_state(state_path,state)
    state['status']='completed' if all(r['status']=='completed' for r in state['profiles']) else 'incomplete'
    save_state(state_path,state)
    _write_csv_atomic(root/'suite_status.csv',state['profiles'])
    text=['# Bateria binária','',f"Estado: {state['status']}",'']
    for item in entries:
        child=load_config(item['config'])
        report=Path(child['outputs']['report']).resolve()
        text.append(f"- [{item['label']}]({os.path.relpath(report,root.resolve())})")
    Path(config['outputs']['report']).write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(f"[SUITE] {state['status']}: {state_path}",flush=True)
    return 0 if state['status']=='completed' else 2


def create_confirmation(path, run_id):
    """Freeze the selected development candidate before external evaluation."""
    from .pipeline import load_config, build_plan, pipeline_fingerprint
    if '::' in run_id:
        prefix, run_id = run_id.split('::',1)
        path = next(item['config'] for item in profiles(path) if Path(item['config']).stem == prefix)
    config=load_config(path)
    if not config.get('target') or config['target']['evaluation']!='development':
        raise ValueError('Selecione um candidato binário de desenvolvimento.')
    runs=build_plan(path,scope='screening')
    selected=next(r for r in runs if r.run_id==run_id)
    matching=[r for r in runs if (r.model,r.representation,r.window_size_frames,r.balancing)==
              (selected.model,selected.representation,selected.window_size_frames,selected.balancing)]
    if len(matching)!=len(config['splits']['folds']) or any(r.status!='completed' for r in matching):
        raise ValueError('Conclua todos os folds desse candidato antes de confirmar.')
    from .pipeline import _screening_candidate
    parameters=_screening_candidate(config,selected)['parameters']
    candidate=dict(family=selected.family,model=selected.model,representation=selected.representation,
                   window=selected.window_size_frames,balancing=selected.balancing,parameters=parameters,
                   enabled=True,deterministic=selected.family!='temporal',paradigm=selected.paradigm)
    lock={'source_config':str(Path(path).resolve()),
          'source_fingerprint':pipeline_fingerprint(Path(path),config),'candidate':candidate,
          'target':config['target']['mapping'],'selection_source':'validation_only'}
    destination=Path(config['outputs']['root'])/'confirmations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    config['outputs']['root']=str(destination)
    config.pop('suite',None)
    config['target']['evaluation']='external'
    config['target']['selection_lock']=lock
    config['name']=f'Confirmação binária · {selected.model} · {selected.representation} · w{selected.window_size_frames}'
    for section in config['models']['screening'].values():section['enabled']=False
    config['training']['confirmation']['candidates']=[candidate]
    file=destination/'config.yaml'
    file.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    (destination/'selection.yaml').write_text(yaml.safe_dump(lock,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return file


def valid_selection(config, candidate):
    from .pipeline import load_config, pipeline_fingerprint
    lock=config.get('target',{}).get('selection_lock',{})
    try:
        source=Path(lock['source_config'])
        return (lock['target']==config['target']['mapping'] and lock['candidate']==candidate
                and lock['selection_source']=='validation_only'
                and pipeline_fingerprint(source,load_config(source))==lock['source_fingerprint'])
    except (KeyError,OSError,ValueError):
        return False
