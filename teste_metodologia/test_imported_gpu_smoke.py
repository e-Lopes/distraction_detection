import unittest

import numpy as np
import pandas as pd

from teste_metodologia.run_imported_gpu_smoke import (
    FEATURES, join_labels, strict_windows, select_episode, support_normalize,
)


class ImportTests(unittest.TestCase):
    def frames(self, n=12):
        frame = pd.DataFrame(dict(video_id=['video_01']*n, frame_index=range(n),
                                  state=['alert']*n, all_indicators_valid=[1]*n))
        for name in FEATURES:
            frame[name] = 1.0
        return frame

    def test_join_does_not_use_old_features(self):
        frame = self.frames(3).drop(columns='state')
        labels = self.frames(2)
        labels['ear'] = 99
        merged, missing = join_labels(frame, labels)
        self.assertEqual(missing, 1)
        self.assertEqual(merged.state.iloc[2], 'unknown')
        self.assertTrue(merged.ear.eq(1).all())
        with self.assertRaises(ValueError):
            join_labels(frame, pd.concat([labels, labels]))

    def test_nan_missing_and_absent_are_not_imputed(self):
        frame = self.frames()
        frame.loc[1, 'ear'] = np.nan
        frame.loc[4, 'state'] = 'absent'
        frame = frame.drop(index=7)
        features, windows, excluded = strict_windows(frame, 3)
        self.assertEqual(features.shape, (1, 3, 5))
        self.assertEqual(windows.start.tolist(), [9])
        self.assertEqual(len(excluded), 3)

    def test_support_query_are_disjoint(self):
        frame = self.frames()
        frame.loc[6:, 'state'] = 'distraction'
        _, windows, _ = strict_windows(frame, 3)
        support, query = select_episode(windows, 'video_01', 1, 1, 42)
        self.assertFalse(set(support) & set(query))
        self.assertEqual(len(support), 2)
        broken = windows.copy()
        broken['start'], broken['end_exclusive'] = 0, 3
        with self.assertRaises(ValueError):
            select_episode(broken, 'video_01', 1, 1, 42)

    def test_query_does_not_change_normalization(self):
        support = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
        query = np.full((2, 3, 5), 1e6, dtype=np.float32)
        sx, _, mean, std = support_normalize(support, query)
        np.testing.assert_allclose(mean, support.mean(axis=(0, 1), keepdims=True))
        np.testing.assert_allclose(std, support.std(axis=(0, 1), keepdims=True))
        np.testing.assert_allclose(sx.mean(axis=(0, 1)), 0, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
