import numpy as np
import unittest
import py.mvgcv_ch11

class EssentialMatTests(unittest.TestCase):
    def test_valid_decomposition(self):
        ess_mat = np.array([[0, -3.30582568, 0], [1.73692763, 0, 3.60320446], [0, -2.2520028, 0]])  # sing_vals=(4,4,0)

        u2, vt2 = py.mvgcv_ch11.EssentialMatSvd(ess_mat, check_post_cond=True)
        self.assertTrue(True)

    def test_valid_decomposition_unity_singular_values(self):
        ess_mat = np.array([[0, -0.82645642,  0], [0.43423191, 0, 0.90080112], [0, -0.5630007, 0]]) # sing_vals=(1,1,0)

        u2, vt2 = py.mvgcv_ch11.EssentialMatSvd(ess_mat, check_post_cond=True)
        self.assertTrue(True)
