import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from orhsurf import process

SCRIPT = Path(__file__).resolve().parents[1]/'slurm/process_clips.sbatch'

class ClipArrayTests(unittest.TestCase):
    def call(self, clips, task='1', options=()):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'orhsurf').mkdir(); (root/'orhsurf/cli.py').touch()
            (root/'env.sh').write_text(':\n')
            stub=root/'srun'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n'); stub.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_') and k!='ORHSURF_REPO'}
            env.update(SLURM_SUBMIT_DIR=str(root),PATH=str(root)+os.pathsep+env['PATH'])
            if task is not None: env['SLURM_ARRAY_TASK_ID']=task
            return subprocess.run(['bash',str(SCRIPT),*options,*clips],env=env,capture_output=True,text=True)

    def test_different_tasks_select_different_whole_clips(self):
        for i,clip in enumerate(['C001','C002','C003']):
            r=self.call(['C001','C002','C003'],str(i))
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('--clips\n'+clip+'\n',r.stdout)
            self.assertNotIn('--shard',r.stdout)

    def test_smoke_and_gpu_options_forwarded(self):
        r=self.call(['C001','C002'],options=['--smoke','--gpus','2'])
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--gpus\n2\n--smoke',r.stdout)

    def test_invalid_or_duplicate_selection_stops(self):
        for clips,task in [(['C001','C001'],'0'),(['C001'],'2'),(['C001','C002'],None)]:
            self.assertNotEqual(self.call(clips,task).returncode,0)

    def test_smoke_preparation_does_not_touch_full_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'_hevc/C001'; raw.mkdir(parents=True); (raw/'video_manifest.json').touch()
            full=root/'C001_full_prepared/manifest.json'; full.parent.mkdir();full.write_text('unchanged')
            def convert(src,dst,frames):
                self.assertEqual(frames,'0'); self.assertEqual(dst,root/'C001_smoke_prepared')
                dst.mkdir(); (dst/'manifest.json').write_text(json.dumps(dict(window={'n_timestamps':225},decoded_frames=[0],valid_serials=['cam'],cameras={'cam':{'frames':{'0':{'index':0}}}})))
                return 0
            with patch('orhsurf.process.fetch.convert_clip',side_effect=convert):
                p=process.prepare('C001',root,smoke=True)
            self.assertEqual(process.full_frames(p,True),'0-0')
            self.assertEqual(full.read_text(),'unchanged')

if __name__=='__main__': unittest.main()
