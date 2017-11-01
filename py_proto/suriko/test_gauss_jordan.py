import numpy as np
import unittest
import suriko.la_utils

class GaussJordanEliminationTests(unittest.TestCase):
    def test1(self):
        # http://web.mit.edu/10.001/Web/Course_Notes/GaussElimPivoting.html
        m = np.array([[0.02,0.01,0,0], [1,2,1,0],[0,1,2,1],[0,0,100,200]],np.float)
        b = np.array([0.02,1,4,800],np.float)
        m_augment = np.hstack((m,b.reshape((4,1))))
        suc = suriko.la_utils.GaussJordanElimination(m_augment)
        self.assertTrue(suc)
        self.assertTrue(np.allclose([1,0,0,4],m_augment[:,4]))

    def test_singular(self):
        # https://en.wikipedia.org/wiki/Gaussian_elimination
        m_augment = np.array([[1,3,1,9], [1,1,-1,1],[3,11,5,35]],np.float)
        suc = suriko.la_utils.GaussJordanElimination(m_augment)
        self.assertFalse(suc)
