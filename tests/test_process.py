import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from orhsurf import process


def manifest(path, n=225):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(window={'n_timestamps':n},decoded_frames=list(range(n)),
                                    valid_serials=['cam'],cameras={'cam':{'frames':{str(i):{'index':i} for i in range(n)}}})))


class ProcessTests(unittest.TestCase):
    def test_full_range_not_150_and_partial_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'; manifest(p)
            self.assertEqual(process.full_frames(p),'0-224')
            m=json.loads(p.read_text()); m['decoded_frames']=[0,1,2,3,4]; p.write_text(json.dumps(m))
            with self.assertRaises(ValueError): process.full_frames(p)

    def test_resume_does_not_rewrite_manifest_or_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'C001_full_prepared/manifest.json'; manifest(p)
            before=p.stat().st_mtime_ns
            with patch('orhsurf.process.fetch.fetch_clip') as fetch, patch('orhsurf.process.fetch.convert_clip') as convert:
                self.assertEqual(process.prepare('C001',root),p)
                fetch.assert_not_called(); convert.assert_not_called()
            self.assertEqual(p.stat().st_mtime_ns,before)

    def exercise(self, failure):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); events=[]
            def prep(clip,data):
                events.append(('prepare',clip)); p=root/clip/'manifest.json'; manifest(p); return p
            def run(a):
                events.append(('run',Path(a.clip).parent.name))
                self.assertEqual(a.frames,'0-224'); self.assertEqual(a.gpus,1)
                return 7 if failure=='run' else 0
            def verify(a):
                events.append(('verify',Path(a.out).name)); return 8 if failure=='verify' else 0
            args=argparse.Namespace(clips=['C001','C002'],cpus_per_job=1,out_root=str(root/'out'))
            with patch('orhsurf.process.fetch.fetch_weights',return_value=0), patch('orhsurf.cli.cmd_doctor',return_value=0), patch('orhsurf.process.prepare',side_effect=prep), patch('orhsurf.cli.cmd_run',side_effect=run), patch('orhsurf.cli.cmd_verify',side_effect=verify):
                rc=process.run(args)
            return rc,events

    def test_clips_sequential(self):
        rc,events=self.exercise(None)
        self.assertEqual(rc,0)
        self.assertEqual(events,[(s,c) for c in ['C001','C002'] for s in ['prepare','run','verify']])

    def test_stop_on_reconstruction_or_verification_failure(self):
        for failure,expected in [('run',7),('verify',8)]:
            rc,events=self.exercise(failure)
            self.assertEqual(rc,expected)
            self.assertTrue(all(c=='C001' for _,c in events))

if __name__=='__main__': unittest.main()
