import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from orhsurf import viewer


def cloud(path,n):
    xyz=np.arange(n*3,dtype=np.float32).reshape(n,3)
    np.savez(path,xyz=xyz,normal=np.tile([0.,0.,1.],(n,1)).astype(np.float32),
             rgb=np.full((n,3),128,np.uint8),support=np.ones(n,np.int16))

class Handle:
    def __init__(self,value=None): self.value=value; self.callbacks=[]; self.content=''
    def on_update(self,fn): self.callbacks.append(fn)
    def update(self,value):
        self.value=value
        for fn in self.callbacks: fn(None)

class Server:
    def __init__(self): self.gui=self; self.scene=self; self.handles={}; self.pc=Handle()
    def add_folder(self,*a): return contextlib.nullcontext()
    def add_dropdown(self,name,options,initial_value):
        h=Handle(initial_value); self.handles[name]=h; return h
    def add_slider(self,name,*args):
        h=Handle(args[-1]); self.handles[name]=h; return h
    def add_markdown(self,*a): return Handle()
    def add_point_cloud(self,*a,**kw): return self.pc
    def atomic(self): return contextlib.nullcontext()

class ViewerTests(unittest.TestCase):
    def test_display_sample_is_repeatable_nested_and_reports_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'surface.npz';cloud(p,20)
            a=viewer.load_cloud(p,None,5);b=viewer.load_cloud(p,None,10)
            np.testing.assert_array_equal(a['xyz'],b['xyz'][:5])
            self.assertEqual(a['n'],5);self.assertEqual(a['n_total'],20)

    def test_version_callback_changes_cloud_without_camera_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'original.npz';q=Path(tmp)/'small.npz';cloud(p,20);cloud(q,7)
            server=Server()
            def switch(_):
                self.assertEqual(len(server.pc.points),20)
                server.handles['version'].update('Small')
                self.assertEqual(len(server.pc.points),7)
                server.handles['version'].update('Original')
                self.assertEqual(len(server.pc.points),20)
                raise KeyboardInterrupt
            with patch('orhsurf.viewer._require_viser') as lib, patch('orhsurf.viewer.time.sleep',side_effect=switch):
                lib.return_value.ViserServer.return_value=server
                self.assertEqual(viewer.serve(p,variants={'Small':q}),0)

if __name__=='__main__':unittest.main()
