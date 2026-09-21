"""Regression checks for video selection, portable paths and sparse frame indices."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from orhsurf.convert import parse_frames, decode_views, build_manifest
from orhsurf.paths import read_manifest
from orhsurf.stages.prep import frame_of


class ConvertTests(unittest.TestCase):
    def test_invalid_selection(self):
        for spec in ('4-2', ',', ' '):
            with self.assertRaises(ValueError):
                parse_frames(spec, 5)
        with self.assertRaises(AssertionError):
            parse_frames('5', 5)
        self.assertEqual(parse_frames('3,1,3', 5), [1, 3])

    def test_sparse_paths_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera = dict(width=2048, height=1536, valid=True)
            vm = dict(window={'n_timestamps': 225}, valid_serials=['cam'],
                      cameras={'cam': camera})
            (root / 'video_manifest.json').write_text(json.dumps(vm))
            out = root / 'prepared/manifest.json'
            build_manifest(root, root/'prepared/rgb', None, out, frames=[40, 41], log=lambda _: None)
            original = out.read_bytes()
            m = read_manifest(out)
            f = frame_of(m['cameras']['cam'], 40)
            self.assertEqual(f['index'], 40)
            self.assertEqual(f['frame_path'], str(root/'prepared/rgb/cam/00040.png'))
            self.assertIsNone(f['mask_path'])
            self.assertEqual(out.read_bytes(), original)
            with self.assertRaisesRegex(AssertionError, "not in this manifest"):
                frame_of(m['cameras']['cam'], 0)

    def test_relative_list_and_dict_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'manifest.json'
            for frames in ([dict(index=0,frame_path='rgb/0.png',mask_path=None)],
                           {'0': dict(index=0,frame_path='rgb/0.png',mask_path=None)}):
                p.write_text(json.dumps({'cameras': {'cam': {'frames': frames}}}))
                f = frame_of(read_manifest(p)['cameras']['cam'],0)
                self.assertEqual(f['frame_path'],str(p.parent/'rgb/0.png'))

    def test_default_masks_and_explicit_override(self):
        from PIL import Image
        from orhsurf.fetch import convert_clip
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vm = dict(window={'n_timestamps': 1, 'duration_s': 1}, valid_serials=['cam'],
                      cameras={'cam': dict(width=2048, height=1536, valid=True)})
            (root/'video_manifest.json').write_text(json.dumps(vm))
            rgb = root/'rgb'
            (rgb/'cam').mkdir(parents=True)
            (rgb/'cam/00000.png').touch()  # decoding is mocked in this policy test
            out = root/'prepared'
            with patch('orhsurf.convert.decode_views', return_value=rgb):
                self.assertEqual(convert_clip(root, out, frames='0'), 0)
                m = read_manifest(out/'manifest.json')
                self.assertEqual(m['mask_policy']['mode'], 'all_foreground')
                with Image.open(frame_of(m['cameras']['cam'],0)['mask_path']) as im:
                    self.assertEqual(im.mode, 'RGBA')
                    self.assertEqual(im.size, (2048,1536))
                    self.assertEqual(im.getchannel('A').getextrema(), (255,255))
                supplied = root/'provided/cam'
                supplied.mkdir(parents=True)
                mask = supplied/'00000.png'
                Image.new('RGBA',(2048,1536),(0,0,0,128)).save(mask)
                before = mask.read_bytes()
                self.assertEqual(convert_clip(root,out,masks=str(supplied.parent),frames='0'),0)
                m = read_manifest(out/'manifest.json')
                self.assertEqual(m['mask_policy']['mode'],'provided')
                self.assertEqual(mask.read_bytes(),before)
                self.assertEqual(frame_of(m['cameras']['cam'],0)['mask_path'],str(mask))
                self.assertEqual(convert_clip(root,out,masks=str(root/'missing'),frames='0'),3)
                self.assertIsNone(frame_of(read_manifest(out/'manifest.json')['cameras']['cam'],0)['mask_path'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
    def test_sparse_video_decode_matches_source_indices(self):
        from PIL import Image
        import numpy as np
        os.environ['ORHSURF_CPUS_PER_JOB'] = '1'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'videos').mkdir()
            (p/'reference').mkdir()
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                            'testsrc=size=32x24:rate=5:duration=1','-threads','1',
                            '-c:v','libx264','-pix_fmt','yuv444p',str(p/'videos/cam.mp4')],check=True)
            subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(p/'videos/cam.mp4'),
                            '-threads','1','-start_number','0',str(p/'reference/%05d.png')],check=True)
            decode_views(p,p/'out',['cam'],5,frames=[1,3],log=lambda _:None)
            for i in [1,3]:
                with Image.open(p/f'reference/{i:05d}.png') as a, Image.open(p/f'out/rgb/cam/{i:05d}.png') as b:
                    np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            self.assertFalse((p/'out/rgb/cam/00000.png').exists())

if __name__ == '__main__':
    unittest.main()
