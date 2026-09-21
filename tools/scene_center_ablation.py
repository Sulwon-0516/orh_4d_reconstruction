"""Does the DA3 grouping centre actually change the prior, or only the group membership?

The masks in this pipeline are needed in exactly ONE place: visual_hull_center() supplies a
per-frame scene centre, which orders the 47 cameras by azimuth into 4 overlapping groups of 18.
Training does not use masks (its loss is on unmasked RGB) and export validity is opacity/depth.

A camera-only centre gives DIFFERENT groups (0/4 identical, measured). "Different" is not "worse".
This measures whether the resulting DEPTHS differ by more than the same-env run-to-run
nondeterminism floor (~1.3e-06 m mean, measured earlier on this machine).

  python tools/scene_center_ablation.py <manifest.json> <frame> <workdir> <gpu>
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orhsurf.stages.prep import prep_views, cam_arrays          # noqa: E402
from orhsurf.stages.da3_prior import (azimuth_groups, visual_hull_center,  # noqa: E402
                                      DA3_MODEL, DA3_REVISION, PROCESS_RES,
                                      GROUP_SIZE, GROUP_OVERLAP)

manifest_path, frame, workdir, gpu = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
man = json.loads(Path(manifest_path).read_text())
serials = sorted(s for s in man["valid_serials"] if man["cameras"][s].get("valid", True))
print(f"[exp] {len(serials)} cameras, frame {frame}", flush=True)

info = prep_views(man, serials, frame, workdir / "native", workers=8)
centers = {s: np.asarray(man["cameras"][s]["camera_center_world"], float) for s in serials}


def optical_axis_centre():
    """Least-squares point closest to every camera's optical axis. Camera geometry only."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for s in serials:
        _, _, _, T = cam_arrays(man["cameras"][s])
        d = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # optical axis in world
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)                   # projector onto the plane normal to d
        A += P; b += P @ centers[s]
    return np.linalg.solve(A, b)


hull = visual_hull_center(man, serials, {s: info[s]["mask"] for s in serials})
optic = optical_axis_centre()
print(f"[exp] hull centre  {np.round(hull,3)}")
print(f"[exp] optic centre {np.round(optic,3)}   distance {np.linalg.norm(hull-optic):.3f} m",
      flush=True)

g_hull, o_hull = azimuth_groups(centers, serials, hull)
g_opt, o_opt = azimuth_groups(centers, serials, optic)
same = sum(a == b for a, b in zip(g_hull, g_opt))
print(f"[exp] groups identical: {same}/{len(g_hull)}", flush=True)

import torch
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained(DA3_MODEL, revision=DA3_REVISION).to("cuda").eval()


def run(groups, owner, tag):
    out = {}
    for gi, g in enumerate(groups):
        keep = [s for s in g if owner[s] == gi]
        imgs = [str(info[s]["undist"]) for s in g]
        exts = np.asarray([np.array(man["cameras"][s]["T_cam_from_world"], float) for s in g], np.float32)
        ixts = np.asarray([np.array(man["cameras"][s]["K_undistort"], float) for s in g], np.float32)
        pred = model.inference(image=imgs, extrinsics=exts, intrinsics=ixts,
                               align_to_input_ext_scale=True, use_ray_pose=True,
                               process_res=PROCESS_RES, process_res_method="upper_bound_resize")
        d = np.asarray(pred.depth, np.float32); c = np.asarray(pred.conf, np.float32)
        for k, s in enumerate(g):
            if s in keep:
                out[s] = (d[k].copy(), c[k].copy())
        del pred; torch.cuda.empty_cache()
    assert set(out) == set(serials), sorted(set(serials) ^ set(out))
    print(f"[exp] {tag}: {len(out)} views", flush=True)
    return out


t0 = time.time(); A = run(g_hull, o_hull, "hull"); B = run(g_opt, o_opt, "optic")
print(f"[exp] both DA3 passes in {time.time()-t0:.0f}s", flush=True)

dd, cc, per = [], [], {}
for s in serials:
    da, ca = A[s]; db, cb = B[s]
    d = np.abs(da - db); dd.append(d.ravel()); cc.append(np.abs(ca - cb).ravel())
    per[s] = dict(mean=float(d.mean()), p99=float(np.percentile(d, 99)), max=float(d.max()),
                  same_group=bool(o_hull[s] == o_opt[s]))
dd = np.concatenate(dd); cc = np.concatenate(cc)
res = dict(frame=frame, hull_centre=hull.tolist(), optic_centre=optic.tolist(),
           centre_distance_m=float(np.linalg.norm(hull - optic)),
           groups_identical=f"{same}/{len(g_hull)}",
           n_views_same_owner=sum(v["same_group"] for v in per.values()),
           depth_mean_abs_diff_m=float(dd.mean()), depth_p99_m=float(np.percentile(dd, 99)),
           depth_p999_m=float(np.percentile(dd, 99.9)), depth_max_m=float(dd.max()),
           conf_max_abs_diff=float(cc.max()),
           nondeterminism_floor_mean_m=1.3e-06, per_view=per)
(workdir / "centre_ablation.json").write_text(json.dumps(res, indent=1))
print(f"\n[RESULT] centre distance {res['centre_distance_m']:.3f} m, groups {res['groups_identical']}, "
      f"{res['n_views_same_owner']}/{len(serials)} views kept the same owning group")
print(f"[RESULT] depth |diff|: mean {res['depth_mean_abs_diff_m']:.4e} m  "
      f"p99 {res['depth_p99_m']:.4e}  p99.9 {res['depth_p999_m']:.4e}  max {res['depth_max_m']:.4e}")
print(f"[RESULT] same-env nondeterminism floor was mean 1.3e-06 m  -> ratio "
      f"{res['depth_mean_abs_diff_m']/1.3e-06:.0f}x the floor")
