import unittest
import numpy as np
from orhsurf.simplify import normal_codes, grid_keys, representatives, select_points, stratified_points


class SimplifyTests(unittest.TestCase):
    def test_random_default_save_resume_and_reject_changed_settings(self):
        import tempfile, json
        from pathlib import Path
        from orhsurf import atomicio
        from orhsurf.simplify import main, parse_targets
        self.assertEqual(parse_targets('10M,5M,1M'), [10000000,5000000,1000000])
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'original/00000'; out=Path(tmp)/'derived'
            arrays={k:np.ones((30,width) if width else (30,),dtype=dtype)
                    for k,(dtype,rank,width) in atomicio.SURFACE_ARRAYS.items()}
            arrays['xyz']=np.arange(90,dtype=np.float32).reshape(30,3)
            with atomicio.FrameStage(source) as stage:
                atomicio.atomic_savez(stage.path/'surface.npz',**arrays)
                atomicio.atomic_write_json(stage.path/'metadata.json',{})
            original=(source/'surface.npz').read_bytes()
            args=['--source',str(source),'--out',str(out),'--targets','10,5,1','--cpus','1','--resume']
            main(args)
            for count in (10,5,1):
                dest=out/str(count)/'00000'
                self.assertTrue(atomicio.verify_frame(dest)['ok'])
                with np.load(dest/'surface.npz') as z:
                    idx=np.random.default_rng(0).choice(30,count,replace=False);idx.sort()
                    for key in arrays: np.testing.assert_array_equal(z[key],arrays[key][idx])
            before=(out/'5/00000/surface.npz').stat().st_mtime_ns
            main(args)
            self.assertEqual(before,(out/'5/00000/surface.npz').stat().st_mtime_ns)
            self.assertEqual(original,(source/'surface.npz').read_bytes())
            with self.assertRaises(FileExistsError): main(args+['--seed','1'])

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

    def test_stratified_preserves_density_proportions_exactly_when_integral(self):
        xyz=np.zeros((1110,3),np.float32)
        xyz[1000:1100,0]=2;xyz[1100:,0]=4
        idx,info=stratified_points(xyz,222,voxel_m=1)
        self.assertEqual([int(np.count_nonzero(xyz[idx,0]==v)) for v in (0,2,4)],[200,20,2])
        self.assertEqual(len(np.unique(idx)),222)
        self.assertEqual(info['max_quota_error_points'],0)

    def test_stratified_rounding_is_bounded_and_seeded(self):
        xyz=np.zeros((37,3),np.float32);xyz[:,0]=np.repeat([0,2,4],[20,10,7])
        a,info=stratified_points(xyz,9,voxel_m=1,seed=4)
        b,_=stratified_points(xyz,9,voxel_m=1,seed=4)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(a),9)
        self.assertLess(info['max_quota_error_points'],1)

    def test_stratified_does_not_claim_minimum_one(self):
        xyz=np.arange(30,dtype=np.float32).reshape(10,3)
        idx,info=stratified_points(xyz,3,voxel_m=.1)
        self.assertEqual(len(idx),3)
        self.assertEqual(info['dropped_strata'],7)

if __name__=='__main__': unittest.main()
