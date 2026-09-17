import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from investigate_angles import solve, HEAD_POSE_3D


class GeometryTests(unittest.TestCase):
    def test_known_projection_has_positive_depth_and_small_error(self):
        camera = np.array([[538., 0, 269], [0, 538., 245.5], [0, 0, 1]])
        rv = np.array([2.8, .1, .05])
        tv = np.array([.02, .01, 1.])
        points = cv2.projectPoints(HEAD_POSE_3D, rv, tv, camera, np.zeros(4))[0].reshape(-1, 2)
        values, rotation, projected = solve(points, camera, cv2.SOLVEPNP_SQPNP)
        self.assertTrue(values['all_positive_depth'])
        self.assertLess(values['rmse_px'], .001)
        np.testing.assert_allclose(projected, points, atol=.001)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)

    def test_finite_angles_do_not_guarantee_positive_depth(self):
        camera = np.array([[538., 0, 269], [0, 538., 245.5], [0, 0, 1]])
        rv = np.array([.2, .1, .05])
        tv = np.array([.02, .01, -1.])
        points = cv2.projectPoints(HEAD_POSE_3D, rv, tv, camera, np.zeros(4))[0].reshape(-1, 2)
        values, _, _ = solve(points, camera, cv2.SOLVEPNP_ITERATIVE)
        self.assertTrue(np.isfinite([values['pitch'], values['yaw'], values['roll']]).all())
        self.assertFalse(values['all_positive_depth'])


if __name__ == '__main__':
    unittest.main()
