"""Regression checks for detection instrumentation (no model inference)."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import visualizar_cinco_series as viewer


class InstrumentationTests(unittest.TestCase):
    def test_available_gpu_never_asks_for_cpu(self):
        gpu = object()
        with patch.object(viewer, 'CudaBackend', return_value=gpu), \
             patch.object(viewer, 'CpuBackend') as cpu, patch('builtins.input') as prompt:
            self.assertIs(viewer.select_backend(), gpu)
            cpu.assert_not_called()
            prompt.assert_not_called()

    def test_cpu_requires_explicit_consent(self):
        with patch.object(viewer, 'CudaBackend', side_effect=RuntimeError('CUDA unavailable')), \
             patch.object(viewer, 'CpuBackend') as cpu, patch('builtins.input', return_value='n'):
            with self.assertRaises(SystemExit):
                viewer.select_backend()
            cpu.assert_not_called()
        fallback = SimpleNamespace(metadata={})
        with patch.object(viewer, 'CudaBackend', side_effect=RuntimeError('CUDA unavailable')), \
             patch.object(viewer, 'CpuBackend', return_value=fallback), \
             patch('builtins.input', return_value='s'):
            self.assertIs(viewer.select_backend(), fallback)
            self.assertTrue(fallback.metadata['cpu_fallback_authorized'])

    def test_no_interactive_answer_and_required_gpu_never_fall_back(self):
        with patch.object(viewer, 'CudaBackend', side_effect=RuntimeError('CUDA unavailable')), \
             patch.object(viewer, 'CpuBackend') as cpu, patch('builtins.input', side_effect=EOFError):
            with self.assertRaises(RuntimeError):
                viewer.select_backend()
            with self.assertRaises(RuntimeError):
                viewer.select_backend(require_gpu=True)
            cpu.assert_not_called()

    def test_detector_combinations_preserve_acceptance(self):
        for pose, face, expected in ((False, False, 'both_missing'),
                                     (False, True, 'pose_missing'),
                                     (True, False, 'face_mesh_missing'),
                                     (True, True, 'both_detected')):
            with self.subTest(pose=pose, face=face), \
                 patch.object(viewer, 'image_left_eye', return_value=viewer.RIGHT_EYE), \
                 patch.object(viewer, 'compute_ear', return_value=.15) as ear, \
                 patch.object(viewer, 'compute_mar', return_value=.01), \
                 patch.object(viewer, 'compute_head_pose', return_value=(-30., -20., -150.)):
                values, raw, _, flags = viewer.measure_frame(pose, [] if face else None, 538, 491)
                self.assertEqual(flags['detection_status'], expected)
                self.assertEqual(flags['face_mesh_detected'], int(face))
                self.assertEqual(flags['indicators_accepted'], int(pose and face))
                self.assertEqual(flags['all_indicators_valid'], int(pose and face))
                self.assertEqual(ear.call_count, int(pose and face))
                if pose and face:
                    np.testing.assert_allclose(values, [.15, .01, -30., -20., -150.])
                else:
                    self.assertTrue(np.isnan(values).all())

    def test_independent_calculation_failures(self):
        with patch.object(viewer, 'image_left_eye', return_value=viewer.RIGHT_EYE), \
             patch.object(viewer, 'compute_ear', return_value=np.nan), \
             patch.object(viewer, 'compute_mar', return_value=.02), \
             patch.object(viewer, 'compute_head_pose', side_effect=RuntimeError('PnP failed')):
            values, _, _, flags = viewer.measure_frame(True, [], 538, 491)
            self.assertEqual(flags['indicators_accepted'], 1)
            self.assertEqual(flags['all_indicators_valid'], 0)
            self.assertEqual(flags['ear_status'], 'nonfinite_result')
            self.assertEqual(flags['head_pose_status'], 'calculation_error:RuntimeError')
            self.assertEqual(flags['mar_valid'], 1)
            self.assertEqual(values[1], .02)
            self.assertTrue(np.isnan(values[2:]).all())


if __name__ == '__main__':
    unittest.main()
