import unittest
import numpy as np
from orhsurf.simplify import normal_codes, grid_keys, representatives, select_points


class SimplifyTests(unittest.TestCase):
    def test_opposite_and_corner_normals_stay_separate(self):
        xyz = np.array([[0,0,0]]*4, np.float32)
        normal = np.array([[0,0,1],[0,0,-1],[1,0,0],[0,0,1]], np.float32)
        codes, span = normal_codes(normal)
        keys = grid_keys(xyz,codes,span,np.zeros(3),.01)
        idx,_ = representatives(keys,np.array([2,4,5,9],np.int16))
        self.assertEqual(set(idx),{1,2,3})

    def test_exact_budget_deterministic_source_indices(self):
        rng=np.random.default_rng(4)
        xyz=rng.random((3000,3),dtype=np.float32)
        normal=np.tile(np.array([0,0,1],np.float32),(len(xyz),1))
        sup=np.ones(len(xyz),np.int16)
        a,info=select_points(xyz,normal,sup,1000,log=lambda *a,**kw:None)
        b,_=select_points(xyz,normal,sup,1000,log=lambda *a,**kw:None)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(np.unique(a)),1000)
        self.assertGreaterEqual(info['clusters'],1000)

    def test_normal_bin_angle_bound(self):
        rng=np.random.default_rng(3)
        n=rng.normal(size=(2000,3)).astype(np.float32)
        n/=np.linalg.norm(n,axis=1)[:,None]
        codes,_=normal_codes(n,30)
        for code in np.unique(codes):
            group=n[codes==code]
            self.assertGreaterEqual(float((group @ group.T).min()),np.cos(np.radians(30))-1e-6)

    def test_zero_normals_do_not_merge_with_valid_direction(self):
        codes,_=normal_codes(np.array([[0,0,0],[0,0,1],[0,0,0]],np.float32))
        self.assertEqual(codes[0],codes[2])
        self.assertNotEqual(codes[0],codes[1])
        with self.assertRaises(ValueError):
            normal_codes(np.array([[np.nan,0,0]],np.float32))

if __name__=='__main__': unittest.main()
