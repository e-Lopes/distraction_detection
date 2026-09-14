"""Optional multivariate adapters; no local fit outside the supplied training split."""
from __future__ import annotations

import importlib.util
import time
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

MODELS = {'multirocket_ridge', 'hydra_multirocket_ridge', 'mantis_frozen_ridge'}


def available(name):
    packages = ('mantis', 'torch') if name == 'mantis_frozen_ridge' else ('aeon', 'torch')
    return all(importlib.util.find_spec(p) is not None for p in packages)


def panel(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 5 or x.shape[1] < 10 or not np.isfinite(x).all():
        raise ValueError('Expected finite [samples, time>=10, five channels], after R0 preprocessing')
    return np.ascontiguousarray(x.transpose(0, 2, 1))


class ModernRidgeAdapter:
    def __getstate__(self):
        state = self.__dict__.copy()
        encoder = state.pop('encoder_', None)
        if encoder is not None:
            state['_encoder_weights'] = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        return state

    def __setstate__(self, state):
        weights = state.pop('_encoder_weights', None)
        self.__dict__.update(state)
        if weights is not None:
            from mantis.architecture import MantisV1
            # Persist tensors, not upstream local lambda functions; no network on resume.
            self.encoder_ = MantisV1(device=self.device_)
            self.encoder_.load_state_dict(weights)
            self.encoder_.eval().requires_grad_(False)

    def __init__(self, name, parameters, *, seed=42, balancing='class_weights'):
        if name not in MODELS or balancing not in ('none', 'class_weights'):
            raise ValueError('Unsupported model or balancing')
        self.name, self.parameters, self.seed = name, dict(parameters), seed
        self.classifier = RidgeClassifier(alpha=float(parameters.get('alpha', 1.0)),
                                         class_weight='balanced' if balancing == 'class_weights' else None)

    def _batches(self, transformer, x):
        size = int(self.parameters.get('batch_size', 64))
        pieces = []
        for start in range(0, len(x), size):
            pieces.append(np.asarray(transformer.transform(x[start:start + size])))
            print(f'[features] {min(start + size, len(x))}/{len(x)}', flush=True)
        return np.concatenate(pieces)

    def _mantis(self, x):
        import torch
        from torch.nn.functional import interpolate
        parts = []
        batch = int(self.parameters.get('batch_size', 16))
        for start in range(0, len(x), batch):
            block = torch.as_tensor(x[start:start + batch], device=self.device_)
            block = interpolate(block, size=512, mode='linear', align_corners=False)
            # Independent channel encodings, concatenated in feature space, never time.
            with torch.inference_mode():
                encoded = self.encoder_(block.reshape(-1, 1, 512))
            parts.append(encoded.cpu().numpy().reshape(len(block), -1))
            print(f'[embeddings] {min(start + batch, len(x))}/{len(x)}', flush=True)
        return np.concatenate(parts)

    def fit(self, values, labels):
        x = panel(values)
        started = time.perf_counter()
        if self.name == 'mantis_frozen_ridge':
            import torch
            from mantis.architecture import MantisV1
            revision = self.parameters.get('revision', '')
            if len(revision) != 40:
                raise ValueError('Mantis requires an immutable 40-character checkpoint revision')
            requested = self.parameters.get('device', 'cpu')
            if requested == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable in PyTorch')
            self.device_ = requested
            self.encoder_ = MantisV1(device=requested).from_pretrained(
                self.parameters['checkpoint'], revision=revision)
            self.encoder_.eval()
            self.encoder_.requires_grad_(False)
            z = self._mantis(x)
        else:
            from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer
            self.rocket_ = MultiRocket(n_kernels=int(self.parameters.get('n_kernels', 840)),
                normalise=False, n_jobs=int(self.parameters.get('n_jobs', 2)), random_state=self.seed)
            self.rocket_.fit(x)
            z = self._batches(self.rocket_, x)
        self.scaler_ = StandardScaler()
        z = self.scaler_.fit_transform(z)
        if self.name == 'hydra_multirocket_ridge':
            from aeon.classification.convolution_based._hydra import _SparseScaler
            self.hydra_ = HydraTransformer(n_groups=int(self.parameters.get('n_groups', 16)),
                n_jobs=int(self.parameters.get('n_jobs', 2)), random_state=self.seed)
            self.hydra_.fit(x)
            self.hydra_scaler_ = _SparseScaler()
            import torch
            h = torch.as_tensor(self._batches(self.hydra_, x))
            z = np.concatenate((z, np.asarray(self.hydra_scaler_.fit_transform(h))), axis=1)
        self.transform_fit_seconds_ = time.perf_counter() - started
        started = time.perf_counter()
        self.classifier.fit(z, labels)
        self.classifier_fit_seconds_ = time.perf_counter() - started
        self.classes_ = self.classifier.classes_
        self.feature_bytes_ = z.nbytes
        return self

    def transform(self, values):
        x = panel(values)
        z = self._mantis(x) if self.name == 'mantis_frozen_ridge' else self._batches(self.rocket_, x)
        z = self.scaler_.transform(z)
        if self.name == 'hydra_multirocket_ridge':
            import torch
            h = self.hydra_scaler_.transform(torch.as_tensor(self._batches(self.hydra_, x)))
            z = np.concatenate((z, np.asarray(h)), axis=1)
        return z

    def predict(self, values):
        return self.classifier.predict(self.transform(values))

    def decision_function(self, values):
        return self.classifier.decision_function(self.transform(values))
