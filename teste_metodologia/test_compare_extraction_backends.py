import unittest

import numpy as np
import pandas as pd

from teste_metodologia.compare_extraction_backends import difference, select_clips, summarize, FEATURES


class ComparisonTests(unittest.TestCase):
    def test_selection_uses_contiguous_runs_and_center(self):
        labels = pd.DataFrame(dict(video_id=['video_01']*8,
                                   frame_index=[0, 1, 5, 6, 7, 8, 9, 10],
                                   state=['alert']*7 + ['absent']))
        clips = select_clips(labels, 3)
        self.assertEqual(clips[0], dict(video_id='video_01', state='alert', start=6, count=3))
        self.assertEqual(clips[1]['count'], 1)

    def test_duplicate_frames_rejected(self):
        labels = pd.DataFrame(dict(video_id=['v', 'v'], frame_index=[0, 0], state=['alert']*2))
        with self.assertRaises(ValueError):
            select_clips(labels, 3)

    def test_angles_wrap(self):
        np.testing.assert_allclose(difference([179, -179], [-179, 179], True), [2, 2])

    def test_missing_indicators_not_zero_filled(self):
        rows = []
        for backend in ('torch_cpu', 'torch_cuda', 'mediapipe_tracking', 'mediapipe_static'):
            for index in range(2):
                values = {f: float(index) for f in FEATURES}
                if backend == 'torch_cuda' and index == 1:
                    values['ear'] = np.nan
                rows.append(dict(backend=backend, video_id='v', frame_index=index,
                                 state='alert', face_mesh_detected=1, elapsed_ms=1,
                                 all_indicators_valid=int(np.isfinite(list(values.values())).all()), **values))
        _, coverage, pairs = summarize(rows)
        self.assertEqual(pairs[0]['ear_paired_n'], 1)
        self.assertEqual(pairs[0]['ear_mae'], 0)
        self.assertEqual(pairs[0]['common_valid'], 1)
        self.assertEqual(coverage.loc[coverage.backend.eq('torch_cuda'), 'valid_fraction'].iloc[0], .5)


if __name__ == '__main__':
    unittest.main()
