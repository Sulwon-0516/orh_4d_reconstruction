"""Same inputs, same group, two different python envs -> are the DA3 depths numerically identical?

Isolates exactly one variable: torch 2.6.0+cu124 (envs/da3) vs 2.7.1+cu128 (envs/ambisur).
Uses lane R's OWN inputs (native/undist/*.png, all 2048x1536 -> uniform 756x1008), not the
AmbiSuR scene images, so the group has no center-crop and matches lane R's contract.
Group of 8, not 18, so it fits beside other users' jobs on a shared GPU.
"""
import json, os, sys
import numpy as np, torch
import os
ROOT = os.environ["ORHSURF_REF_ROOT"]   # reference project root; set it explicitly
os.environ.setdefault("HF_HOME", ROOT + "/cache/hf")
tag = sys.argv[1]
out = f"/tmp/claude-0/-/1e452415-f336-4195-a400-80b51304ebc3/scratchpad/da3_{tag}.npz"
from depth_anything_3.api import DepthAnything3
man = json.load(open(ROOT + "/data/orh/20260807_cand1_quant06__20260807_161358/manifest.json"))
cams = man["cameras"]
# deterministic group: lane R orders by azimuth; we just take the first 8 valid serials sorted.
serials = sorted(man["valid_serials"])[:8]
imgs, exts, ixts = [], [], []
for s in serials:
    p = ROOT + f"/lanes_out_orh/R/00037/native/undist/{s}.png"
    assert os.path.exists(p), p
    imgs.append(p)
    T = np.array(cams[s]["T_cam_from_world"], float)
    exts.append(T.astype(np.float32))
    ixts.append(np.array(cams[s]["K_undistort"], np.float32))
exts = np.asarray(exts, np.float32); ixts = np.asarray(ixts, np.float32)
m = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").to("cuda").eval()
torch.cuda.reset_peak_memory_stats()
pred = m.inference(image=imgs, extrinsics=exts, intrinsics=ixts,
                   align_to_input_ext_scale=True, use_ray_pose=True,
                   process_res=1008, process_res_method="upper_bound_resize")
d = np.asarray(pred.depth, np.float32); c = np.asarray(pred.conf, np.float32)
K = np.asarray(pred.intrinsics, np.float64); E = np.asarray(pred.extrinsics, np.float64)
np.savez_compressed(out, depth=d, conf=c, K=K, E=E, serials=np.array(serials))
print(tag, "torch", torch.__version__, "depth", d.shape,
      "peakMiB", int(torch.cuda.max_memory_allocated()/2**20),
      "ext_maxerr", float(np.abs(E - exts[:, :3, :]).max()), flush=True)
