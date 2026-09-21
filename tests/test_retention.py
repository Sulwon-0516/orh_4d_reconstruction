import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from orhsurf import atomicio, retention
from orhsurf.simplify import main as simplify


class RetentionTests(unittest.TestCase):
    def setup_clip(self, root):
        out=root/'out/C001'; source=out/'00000'; prepared=root/'data/C001_first150_prepared'
        prepared.mkdir(parents=True)
        rgb=prepared/'rgb/cam/00000.png'; mask=prepared/'masks_all_foreground/cam/00000.png'
        for p in (rgb,mask): p.parent.mkdir(parents=True); p.write_bytes(b'png')
        manifest=prepared/'manifest.json'
        manifest.write_text(json.dumps(dict(generator='orhsurf.convert (HuggingFace HEVC archive)',
            window={'n_timestamps':1},decoded_frames=[0],valid_serials=['cam'],mask_policy={'mode':'all_foreground'},
            cameras={'cam':{'frames':{'0':dict(index=0,frame_path=str(rgb),mask_path=str(mask))}}})))
        arrays={k:np.ones((10,w) if w else (10,),dtype=d) for k,(d,r,w) in atomicio.SURFACE_ARRAYS.items()}
        with atomicio.FrameStage(source) as stage:
            atomicio.atomic_savez(stage.path/'surface.npz',**arrays)
            atomicio.atomic_write_json(stage.path/'metadata.json',{})
        simplify(['--source',str(source),'--out',str(out.parent/'_simplified/C001/random'),
                  '--targets','5,1','--cpus','1'])
        spec=retention.identity(manifest,'recipe',[0],[5,1],'random',True,True)
        return out,manifest,rgb,mask,spec

    def test_keep_only_verified_derivatives_cleanup_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,manifest,rgb,mask,spec=self.setup_clip(Path(tmp))
            original=manifest.read_bytes()
            retention.finish(out,spec)
            self.assertFalse((out/'00000/surface.npz').exists())
            self.assertTrue((out/'00000/_ORIGINAL_DONE.json').exists())
            self.assertFalse(rgb.exists()); self.assertFalse(mask.exists())
            self.assertEqual(manifest.read_bytes(),original)
            retention.finish(out,spec)
            self.assertEqual(retention.verify(out,json.loads((out/retention.RECEIPT).read_text()))['n_ok'],1)

    def test_process_resume_skips_prepare_and_gpu_and_verify_follows_retention(self):
        from orhsurf import cli, process
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out,manifest,rgb,mask,spec=self.setup_clip(root)
            retention.finish(out,spec)
            args=cli.build_parser().parse_args(['process','--clips','C001','--gpus','1',
                '--cpus-per-job','1','--out-root',str(out.parent),'--simplify','5,1',
                '--simplify-only','--cleanup-decoded'])
            with patch('orhsurf.alloc.resolve_gpus',return_value=[0]), patch('orhsurf.cli.cmd_doctor',return_value=0), \
                 patch('orhsurf.paths.data_root',return_value=root/'data'), \
                 patch('orhsurf.cli.recipe_from_args',return_value=Mock(hash=Mock(return_value='recipe'))), \
                 patch('orhsurf.process.prepare') as prepare, patch('orhsurf.cli.cmd_run') as run:
                self.assertEqual(process.run(args),0)
                prepare.assert_not_called(); run.assert_not_called()
            self.assertEqual(cli.cmd_verify(cli.build_parser().parse_args(['verify','--out',str(out)])),0)

    def test_sbatch_script_executes_retained_process_with_forwarded_options(self):
        import os,sys,shlex,subprocess
        from orhsurf import cli
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out,manifest,rgb,mask,spec=self.setup_clip(root)
            spec['recipe_hash']=cli.recipe_from_args(cli.build_parser().parse_args(['process','--clips','C001'])).hash()
            retention.finish(out,spec)
            fake=root/'checkout';(fake/'orhsurf').mkdir(parents=True);(fake/'bin').mkdir()
            (fake/'orhsurf/cli.py').touch();(fake/'env.sh').write_text(':\n')
            helper=fake/'invoke.py'
            helper.write_text("""import sys,os
from pathlib import Path
from unittest.mock import patch
from orhsurf import cli,process
args=cli.build_parser().parse_args(sys.argv[1:])
assert args.simplify=='5,1' and args.simplify_only and args.cleanup_decoded
args.cpus_per_job=1
with patch('orhsurf.paths.data_root',return_value=Path(os.environ['TEST_DATA'])), patch('orhsurf.alloc.resolve_gpus',return_value=[0]), patch('orhsurf.cli.cmd_doctor',return_value=0), patch('orhsurf.cli.cmd_run',side_effect=AssertionError('GPU should not run')):
    sys.exit(process.run(args))
""")
            binary=fake/'bin/orhsurf'
            binary.write_text('#!/bin/bash\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(helper))+' "$@"\n');binary.chmod(0o755)
            srun=fake/'bin/srun';srun.write_text('#!/bin/bash\nshift\nexec "$@"\n');srun.chmod(0o755)
            env=os.environ.copy();env.update(ORHSURF_REPO=str(fake),MODEL_OUTPUT_DIR=str(out.parent),
                TEST_DATA=str(root/'data'),SLURM_ARRAY_TASK_ID='0',
                PYTHONPATH=str(Path(__file__).resolve().parents[1]),PATH=str(fake/'bin')+os.pathsep+env['PATH'])
            script=Path(__file__).resolve().parents[1]/'slurm/process_clips.sbatch'
            result=subprocess.run(['bash',str(script),'--gpus','1','--simplify','5,1','--simplify-only','--cleanup-decoded','C001'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('[retention] complete',result.stdout)

    def test_failed_derivative_blocks_all_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,manifest,rgb,mask,spec=self.setup_clip(Path(tmp))
            (retention.derived_root(out,'random',1)/'00000/surface.npz').unlink()
            with self.assertRaises(RuntimeError): retention.finish(out,spec)
            self.assertTrue((out/'00000/surface.npz').exists()); self.assertTrue(rgb.exists())
            self.assertFalse((out/retention.RECEIPT).exists())

    def test_interrupted_cleanup_is_restartable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,manifest,rgb,mask,spec=self.setup_clip(Path(tmp))
            unlink=Path.unlink
            def interrupted(p,*a,**kw):
                if p==rgb: raise OSError('simulated interruption')
                return unlink(p,*a,**kw)
            with patch.object(Path,'unlink',interrupted):
                with self.assertRaises(OSError): retention.finish(out,spec)
            self.assertEqual(json.loads((out/retention.RECEIPT).read_text())['state'],'verified_pending_cleanup')
            retention.finish(out,spec)
            self.assertFalse(rgb.exists())

    def test_external_paths_rejected_and_provided_masks_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,manifest,rgb,mask,spec=self.setup_clip(Path(tmp))
            external=Path(tmp)/'external.png'; external.write_bytes(b'keep')
            man=json.loads(manifest.read_text()); man['cameras']['cam']['frames']['0']['frame_path']=str(external)
            manifest.write_text(json.dumps(man))
            with self.assertRaises(ValueError): retention.cleanup_paths(manifest)
            man['cameras']['cam']['frames']['0']['frame_path']=str(rgb)
            man['cameras']['cam']['frames']['0']['mask_path']=str(external)
            man['mask_policy']['mode']='provided';manifest.write_text(json.dumps(man))
            self.assertEqual(retention.cleanup_paths(manifest),[rgb])
            self.assertTrue(external.exists())
