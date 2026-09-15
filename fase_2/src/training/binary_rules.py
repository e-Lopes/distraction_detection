"""Persistable fixed-rule reference; binary votes are combined at frame level."""
from collections import Counter
import numpy as np
from ..data.targets import ATTENTION_MAPPING


class FixedRules:
    def __init__(self, parameters, scaler, *, binary=False):
        self.parameters, self.scaler, self.binary = parameters, scaler, binary

    def fit(self, values, labels):
        return self

    def predict(self, values):
        p = self.parameters
        raw = np.asarray(values) * np.asarray(self.scaler.scale) + np.asarray(self.scaler.mean)
        results = []
        for window in raw:
            states, streak = [], 0
            for ear, mar, pitch, yaw, roll in window:
                streak = streak + 1 if ear < p['ear_threshold'] else 0
                label = ('fatigue' if streak >= p['ear_consecutive_frames'] or mar > p['mar_threshold']
                         else 'distraction' if pitch < p['pitch_threshold'] else 'alert')
                states.append(ATTENTION_MAPPING[label] if self.binary else label)
            priority = ('distraction', 'attention') if self.binary else ('fatigue', 'distraction', 'alert')
            counts = Counter(states)
            results.append(max(priority, key=lambda label: counts[label]))
        return np.asarray(results)
