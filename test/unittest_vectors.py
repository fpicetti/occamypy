import unittest
import numpy as np
from occamypy.vector import superVector, VectorIC


class testVectorIC(unittest.TestCase):
    def test_create_from_tuple(self):
        vec = VectorIC((100,100))
        self.assertIsInstance(vec.arr, np.ndarray)
        self.assertIsInstance(vec.ndims, int)
        self.assertIsInstance(vec.shape, tuple)
        self.assertIsInstance(vec.size, int)
        
    def test_create_from_numpy(self):
        vec = VectorIC(np.zeros((100,100)))
        self.assertIsInstance(vec.arr, np.ndarray)
        self.assertIsInstance(vec.ndims, int)
        self.assertIsInstance(vec.shape, tuple)
        self.assertIsInstance(vec.size, int)
        
    def test_create_from_file(self):
        vecOC = '../testdata/'  # TODO finish
        vec = VectorIC(vecOC)
        self.assertIsInstance(vec.arr, np.ndarray)
        self.assertIsInstance(vec.ndims, int)
        self.assertIsInstance(vec.shape, tuple)
        self.assertIsInstance(vec.size, int)
        
    def test_getNdArray(self):
        vec = VectorIC(np.zeros((100, 100)))
        self.assertEqual(vec.getNdArray(), np.zeros((100, 100)))
    
    def test_norm(self):
        vec = VectorIC(np.zeros((100, 100)))
        vec.getNdArray()[0,0] = 1
        self.assertAlmostEqual(vec.norm(N=0), 1.)
        self.assertAlmostEqual(vec.norm(N=1), 1.)
        self.assertAlmostEqual(vec.norm(N=2), 1.)
    
    def test_min(self):
        vec = VectorIC(np.arange(25).reshape((5,5)))
        self.assertEqual(vec.min(), 0.)
        
    def test_max(self):
        vec = VectorIC(np.arange(25).reshape((5,5)))
        self.assertEqual(vec.max(), 24.)
    
    def test_set(self):
        vec = VectorIC(np.zeros((100, 100)))
        vec.set(1.)
        self.assertEqual(vec.getNdArray(), np.ones((100, 100)))
    
    def test_scale(self):
        vec = VectorIC(np.ones((100, 100)))
        vec.scale(2.)
        self.assertEqual(vec.getNdArray(), 2*np.ones((100, 100)))
    
    def test_bias(self):
        vec = VectorIC(np.zeros((100, 100)))
        vec.addbias(1.)
        self.assertEqual(vec.getNdArray(), np.ones((100, 100)))
    
    def test_rand(self):
        raise NotImplementedError
    
    def test_clone(self):
        raise NotImplementedError
    
    def test_cloneSpace(self):
        raise NotImplementedError
    
    def test_checkSpace(self):
        raise NotImplementedError
    
    def test_abs(self):
        raise NotImplementedError
    
    def test_sign(self):
        raise NotImplementedError
    
    def test_reciprocal(self):
        raise NotImplementedError
    
    def test_maximum(self):
        raise NotImplementedError
    
    def test_conj(self):
        raise NotImplementedError
    
    def test_pow(self):
        raise NotImplementedError
    
    def test_real(self):
        raise NotImplementedError
    
    def test_imag(self):
        raise NotImplementedError
    
    def test_copy(self):
        raise NotImplementedError
    
    def test_scaleAdd(self):
        raise NotImplementedError

    def test_dot(self):
        raise NotImplementedError
    
    def test_multiply(self):
        raise NotImplementedError
    
    def test_isDifferent(self):
        raise NotImplementedError
    
    def clipVector(self):
        raise NotImplementedError
