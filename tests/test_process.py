import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from orhsurf import process, alloc, cli


def manifest(path, n=225):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(window={'n_timestamps':n},decoded_frames=list(range(n)),
                                    valid_serials=['cam'],cameras={'cam':{'frames':{str(i):{'index':i} for i in range(n)}}})))


class ProcessTests(unittest.TestCase):
    def test_full_range_not_150_and_partial_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'; manifest(p)
            self.assertEqual(process.full_frames(p, all_frames=True),'0-224')
            m=json.loads(p.read_text()); m['decoded_frames']=[0,1,2,3,4]; p.write_text(json.dumps(m))
            with self.assertRaises(ValueError): process.full_frames(p)

    def test_resume_does_not_rewrite_manifest_or_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'C001_full_prepared/manifest.json'; manifest(p)
            before=p.stat().st_mtime_ns
            with patch('orhsurf.process.fetch.fetch_clip') as fetch, patch('orhsurf.process.fetch.convert_clip') as convert:
                self.assertEqual(process.prepare('C001',root,all_frames=True),p)
                fetch.assert_not_called(); convert.assert_not_called()
            self.assertEqual(p.stat().st_mtime_ns,before)

    def test_default_prepares_first_150_and_preserves_full_inputs(self):
        for n in (225, 80):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); raw=root/'_hevc/C001'; raw.mkdir(parents=True)
                (raw/'video_manifest.json').write_text(json.dumps({'window':{'n_timestamps':n}}))
                full=root/'C001_full_prepared/manifest.json'; manifest(full,n)
                original=full.read_bytes()
                def convert(src,dst,frames):
                    self.assertEqual(frames,f'0-{min(n,150)-1}')
                    m=dst/'manifest.json'; manifest(m,min(n,150))
                    content=json.loads(m.read_text()); content['window']['n_timestamps']=n
                    m.write_text(json.dumps(content)); return 0
                with patch('orhsurf.process.fetch.convert_clip',side_effect=convert) as convert_mock:
                    selected=process.prepare('C001',root)
                    self.assertEqual(process.full_frames(selected),f'0-{min(n,150)-1}')
                    process.prepare('C001',root)
                    self.assertEqual(convert_mock.call_count,1)
                self.assertEqual(full.read_bytes(),original)

    def test_fresh_download_requests_hevc_before_selecting_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def download(clip,data,**kwargs):
                self.assertEqual(kwargs,dict(convert=True,download_only=True))
                raw=data/'_hevc'/clip; raw.mkdir(parents=True)
                (raw/'video_manifest.json').write_text(json.dumps({'window':{'n_timestamps':225}}))
                return 0
            def convert(raw,dst,frames):
                self.assertEqual(raw,root/'_hevc/C001')
                self.assertEqual(frames,'0-149')
                manifest(dst/'manifest.json',150); return 0
            with patch('orhsurf.process.fetch.fetch_clip',side_effect=download), patch('orhsurf.process.fetch.convert_clip',side_effect=convert):
                self.assertEqual(process.prepare('C001',root),root/'C001_first150_prepared/manifest.json')

    def test_duration_versions_have_separate_roots_and_frame_limits(self):
        args=cli.build_parser().parse_args(['process','--clips','C001','C002','--durations','10,15',
            '--out-root','/tmp/durations-test','--simplify','5M','--simplify-only','--cleanup-decoded'])
        original=process.run
        with patch('orhsurf.process.run',return_value=0) as child:
            self.assertEqual(original(args),0)
        calls=[c.args[0] for c in child.call_args_list]
        self.assertEqual([(a.clips,a._frame_limit,a.out_root) for a in calls],
            [([c],d*15,f'/tmp/durations-test/{d}s') for c in ['C001','C002'] for d in [10,15]])
        self.assertTrue(all(a.simplify_only and a.cleanup_decoded and a.simplify=='5M' for a in calls))

    def test_15_second_preparation_decodes_225_not_150(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'_hevc/C001';raw.mkdir(parents=True)
            raw_manifest=raw/'video_manifest.json'
            raw_manifest.write_text(json.dumps(dict(window={'n_timestamps':225},conventions={'output_fps':15})))
            def convert(src,dst,frames):
                self.assertEqual(frames,'0-224');manifest(dst/'manifest.json',225);return 0
            with patch('orhsurf.process.fetch.convert_clip',side_effect=convert):
                p=process.prepare('C001',root,frame_limit=225,require_fps=15)
            self.assertEqual(p,root/'C001_first225_prepared/manifest.json')
            self.assertEqual(process.full_frames(p,frame_limit=225),'0-224')
            raw_manifest.write_text(json.dumps(dict(window={'n_timestamps':225},conventions={'output_fps':30})))
            with self.assertRaisesRegex(ValueError,'output_fps=15'):
                process.prepare('C001',root,frame_limit=150,require_fps=15)

    def exercise(self, failure, gpus=1, cpus=1, total_cpus=20, **options):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); events=[]
            def prep(clip,data,**kwargs):
                events.append(('prepare',clip)); p=root/clip/'manifest.json'; manifest(p,150); return p
            def run(a):
                events.append(('run',Path(a.clip).parent.name))
                self.assertEqual(a.frames,'0-149'); self.assertEqual(a.gpus,gpus)
                self.assertEqual(a.cpus_per_job, cpus if cpus is not None else total_cpus//gpus)
                return 7 if failure=='run' else 0
            def verify(a):
                events.append(('verify',Path(a.out).name)); return 8 if failure=='verify' else 0
            args=argparse.Namespace(clips=['C001','C002'],gpus=gpus,cpus_per_job=cpus,out_root=str(root/'out'),**options)
            with patch.dict(os.environ, {k:v for k,v in os.environ.items() if k != 'ORHSURF_CPUS_PER_JOB'}, clear=True), patch('orhsurf.alloc.visible_gpus',return_value=list(range(gpus))), patch('orhsurf.cpubudget.allocation_cpus',return_value=total_cpus), patch('orhsurf.process.fetch.fetch_weights',return_value=0), patch('orhsurf.cli.cmd_doctor',return_value=0), patch('orhsurf.process.prepare',side_effect=prep), patch('orhsurf.cli.cmd_run',side_effect=run), patch('orhsurf.cli.cmd_verify',side_effect=verify):
                rc=process.run(args)
            return rc,events

    def test_simplify_preserves_originals_and_only_defaults_to_1m_5m(self):
        for only in (False, True):
            options={'simplify_only':True} if only else {'simplify':'1M,5M'}
            with patch('orhsurf.simplify.main') as simplify, patch('orhsurf.retention.finish') as finish:
                rc,_=self.exercise(None,**options)
                self.assertEqual(rc,0)
                self.assertEqual(simplify.call_count,300)
                cmd=simplify.call_args.args[0]
                self.assertEqual(cmd[cmd.index('--targets')+1],'1M,5M')
                if only:
                    self.assertEqual(finish.call_count,2)
                    spec=finish.call_args.args[1]
                    self.assertTrue(spec['simplify_only'])
                    self.assertEqual(spec['targets'],[1000000,5000000])
                else:
                    finish.assert_not_called()

    def test_empty_simplify_budget_cannot_delete_originals(self):
        args=cli.build_parser().parse_args(['process','--clips','C001','--simplify','','--simplify-only'])
        with patch('orhsurf.alloc.resolve_gpus') as gpu:
            with self.assertRaises(ValueError): process.run(args)
            gpu.assert_not_called()

    def test_clips_sequential(self):
        rc,events=self.exercise(None)
        self.assertEqual(rc,0)
        self.assertEqual(events,[(s,c) for c in ['C001','C002'] for s in ['prepare','run','verify']])

    def test_stop_on_reconstruction_or_verification_failure(self):
        for failure,expected in [('run',7),('verify',8)]:
            rc,events=self.exercise(failure)
            self.assertEqual(rc,expected)
            self.assertTrue(all(c=='C001' for _,c in events))

    def test_multiple_gpus_divide_cpu_budget_and_keep_clip_order(self):
        rc,events=self.exercise(None,gpus=4,cpus=None,total_cpus=20)
        self.assertEqual(rc,0)
        self.assertEqual(events,[(s,c) for c in ['C001','C002'] for s in ['prepare','run','verify']])

    def test_oversubscription_fails_before_download(self):
        args=argparse.Namespace(clips=['C001'],gpus=4,cpus_per_job=8,out_root=None)
        with patch('orhsurf.alloc.visible_gpus',return_value=[0,1,2,3]), patch('orhsurf.cpubudget.allocation_cpus',return_value=20), patch('orhsurf.process.fetch.fetch_weights') as download:
            with self.assertRaisesRegex(SystemExit, 'exceeds 20 allocated'):
                process.run(args)
            download.assert_not_called()

    def test_insufficient_gpus_fails_before_download(self):
        args=argparse.Namespace(clips=['C001'],gpus=4,cpus_per_job=None,out_root=None)
        with patch('orhsurf.alloc.visible_gpus',return_value=[0]), patch('orhsurf.process.fetch.fetch_weights') as download:
            with self.assertRaisesRegex(SystemExit, 'only 1 are allocated'):
                process.run(args)
            download.assert_not_called()

    def test_dispatch_partitions_frames_and_preserves_device_mapping(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            env={'CUDA_VISIBLE_DEVICES':'3,5'}
            args=argparse.Namespace(keep_work=False,force=False)
            with patch.dict(os.environ,env), patch('orhsurf.cli._recipe_json',return_value={}), patch('orhsurf.cli.subprocess.Popen') as popen:
                popen.side_effect=[Mock(wait=Mock(return_value=0)),Mock(wait=Mock(return_value=0))]
                self.assertEqual(cli._dispatch(list(range(225)),[0,1],root/'manifest.json',root/'out',root/'work',None,args,10),0)
                batches=[]
                for call,dev in zip(popen.call_args_list,['3','5']):
                    cmd=call.args[0]
                    batches.append(cli.parse_frames(cmd[cmd.index('--frames')+1]))
                    self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'],dev)
                    self.assertEqual(cmd[cmd.index('--cpus')+1],'10')
                    call.kwargs['stdout'].close()
                self.assertEqual(batches[0],list(range(113)))
                self.assertEqual(batches[1],list(range(113,225)))

if __name__=='__main__': unittest.main()


class RecipeDispatchTests(unittest.TestCase):
    """`process` must reconstruct the recipe it was asked for, and say so in the receipt.

    It used to build the inner `run` namespace from a fixed argv carrying only paths, frames and
    allocation, so every recipe option stopped at the `process` boundary. That was invisible while
    `run` and `process` shared a default, and became "every batch run is secretly -r 4" the moment
    they diverged. These tests fail if the two ever drift apart again.
    """

    def inner_for(self, *argv):
        p = cli.build_parser()
        outer = p.parse_args(['process', '--clips', 'C001'] + list(argv))
        recipe = process.resolve_recipe(outer)
        inner = p.parse_args(['run', '--clip', 'x', '--frames', '0-0', '--gpus', '1',
                              '--cpus-per-job', '8', '--out', 'o', '--shard', '0', '--shards', '1'])
        process._forward_recipe(recipe, inner, cli)
        return recipe, cli.recipe_from_args(inner)

    def test_every_recipe_option_survives_dispatch(self):
        for argv in ([], ['--preset', 'quality'], ['--preset', 'draft'],
                     ['--preset', 'economy', '--resolution', '1'],
                     ['--iterations', '2500'], ['--nn-max-mm', '7.5'], ['--group-size', '12'],
                     ['--densify-from-iter', '300', '--densification-interval', '50']):
            with self.subTest(argv=argv):
                requested, executed = self.inner_for(*argv)
                self.assertEqual(requested.hash(), executed.hash())
                self.assertEqual(requested.resolution, executed.resolution)
                self.assertEqual(requested.iterations, executed.iterations)
                self.assertEqual(requested.nn_max_mm, executed.nn_max_mm)

    def test_fast_default_moves_resolution_and_threshold_together(self):
        requested, executed = self.inner_for()
        self.assertEqual((executed.resolution, executed.nn_max_mm), (4, 10.0))
        self.assertEqual(requested.hash(), executed.hash())

    def test_quality_is_not_silently_downgraded(self):
        _, executed = self.inner_for('--preset', 'quality')
        self.assertEqual((executed.iterations, executed.resolution, executed.nn_max_mm),
                         (7000, 2, 5.0))

    def test_every_settable_field_is_forwardable(self):
        # A field added to RECIPE_FLAGS without a matching `run` option would be dropped silently.
        run_args = cli.build_parser().parse_args(['run', '--clip', 'x'])
        for field in cli.RECIPE_FLAGS:
            self.assertTrue(hasattr(run_args, field), f'orhsurf run cannot set {field}')

    def test_smoke_forces_quality_and_refuses_conflicts(self):
        p = cli.build_parser()
        self.assertEqual(process.resolve_recipe(
            p.parse_args(['process', '--clips', 'C001', '--smoke'])).iterations, 7000)
        for argv in (['--preset', 'fast'], ['--preset', 'quality'], ['--iterations', '500'],
                     ['--nn-max-mm', '8'], ['--resolution', '4']):
            with self.subTest(argv=argv):
                a = p.parse_args(['process', '--clips', 'C001', '--smoke'] + argv)
                a.gpus = 0  # stop at the next check; the smoke gate must fire before any work
                with self.assertRaises(SystemExit) as cm:
                    process.run(a)
                self.assertIn('--smoke always runs the full quality recipe', str(cm.exception))

    def test_run_actually_forwards_the_recipe_to_cmd_run(self):
        """The end-to-end guard: the other tests call _forward_recipe themselves, so they would
        still pass if `run` stopped calling it. Capture what cmd_run really receives."""
        seen = {}
        def capture(args):
            seen['recipe'] = cli.recipe_from_args(args)
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = root/'C001_first150_prepared/manifest.json'; manifest(p, 150)
            for argv, expect in ((['--preset', 'quality'], (7000, 2, 5.0)),
                                 ([], (2000, 4, 10.0)),
                                 (['--preset', 'economy', '--nn-max-mm', '3.5'], (2000, 2, 3.5))):
                with self.subTest(argv=argv):
                    seen.clear()
                    a = cli.build_parser().parse_args(
                        ['process', '--clips', 'C001', '--gpus', '1', '--cpus-per-job', '1',
                         '--out-root', str(root/'out')] + argv)
                    with patch('orhsurf.alloc.resolve_gpus', return_value=[0]), \
                         patch('orhsurf.cli.cmd_doctor', return_value=0), \
                         patch('orhsurf.paths.data_root', return_value=root), \
                         patch('orhsurf.process.fetch.fetch_weights', return_value=0), \
                         patch('orhsurf.cli.cmd_verify', return_value=0), \
                         patch('orhsurf.cli.cmd_run', side_effect=capture):
                        self.assertEqual(process.run(a), 0)
                    r = seen['recipe']
                    self.assertEqual((r.iterations, r.resolution, r.nn_max_mm), expect)
