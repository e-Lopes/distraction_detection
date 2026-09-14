import joblib
import numpy as np
import pytest
import torch
from fase_2.src.training.modern_adapters import ModernRidgeAdapter, panel
from fase_2.src.models.temporal import build_temporal_model
from fase_2.src.pipeline import load_config, build_plan, pipeline_fingerprint
from pathlib import Path


@pytest.mark.parametrize('length', [30, 60, 150])
@pytest.mark.parametrize('name', ['multirocket_ridge', 'hydra_multirocket_ridge'])
def test_transform_multivariate_persistence_and_train_only(tmp_path, length, name):
    pytest.importorskip('aeon')
    rng = np.random.default_rng(42)
    x = rng.normal(size=(9, length, 5)).astype('float32')
    y = np.array(['alert', 'fatigue', 'distraction'] * 3)
    model = ModernRidgeAdapter(name, {'n_kernels': 84, 'n_groups': 2, 'n_jobs': 1}).fit(x, y)
    mean = model.scaler_.mean_.copy()
    query = x[:3] + 100
    prediction = model.predict(query)
    np.testing.assert_array_equal(model.scaler_.mean_, mean)
    path = tmp_path / 'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(query), prediction)
    assert model.decision_function(query).shape == (3, 3)


@pytest.mark.parametrize('length', [30, 60, 150])
def test_inception_backward_and_reload(length, tmp_path):
    torch.set_num_threads(1)
    model = build_temporal_model('inception_individual', input_dim=5, num_classes=3,
                                parameters={'filters': 4, 'depth': 3})
    x = torch.randn(3, length, 5)
    optimizer = torch.optim.Adam(model.parameters())
    loss = torch.nn.functional.cross_entropy(model(x), torch.arange(3))
    loss.backward(); optimizer.step()
    model.eval()
    p = tmp_path / 'net.pt'; torch.save(model.state_dict(), p)
    model.load_state_dict(torch.load(p, weights_only=True))
    assert model(x).shape == (3, 3)
    assert torch.isfinite(model(x)).all()


def test_plan_and_fingerprint():
    path = Path('fase_2/configs/modern_experiment.yaml')
    config = load_config(path)
    plan = build_plan(path, scope='screening')
    assert len(plan) == 32
    assert {r.window_size_frames for r in plan} == {60}
    assert sum(r.model == 'inception_individual' for r in plan) == 4
    fingerprint = pipeline_fingerprint(path, config)
    config['models']['screening']['transform']['candidates'][0]['parameters']['alpha'] = 10
    assert pipeline_fingerprint(path, config) != fingerprint


def test_panel_rejects_concatenated_channels():
    with pytest.raises(ValueError):
        panel(np.zeros((4, 300, 1)))
def test_modern_report_without_scientific_results():
    from fase_2.src.modern_reporting import modern_report
    from fase_2.src.pipeline import load_config
    config = load_config('fase_2/configs/modern_experiment.yaml')
    report = modern_report(config, [])
    assert 'Sem resultados científicos novos' in report
    assert 'mantis_frozen_ridge' in report
