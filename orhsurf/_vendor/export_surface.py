#!/usr/bin/env python
"""Lane S0 surface export -- THE SURFACE, not the Gaussian centres.

Gaussian centres are not a surface: they are anisotropic blobs that sit wherever the optimiser needed density,
including well inside and outside the true surface.  What AmbiSuR actually constrains is the *rendered plane
depth* per view.  So the surface is read out the way the renderer defines it:

  for every TRAINING camera
      render -> plane_depth (metric z-depth along the camera axis) + rendered_alpha
      keep pixels with alpha > --alpha-min and depth > 0
      backproject with the CALIBRATED K and pose:  X_cam = d * [(u-Cx)/Fx, (v-Cy)/Fy, 1],  X_w = R (X_cam - T)
      (both are asserted against the manifest / the scene's own COLMAP model before anything is exported)
  multi-view agreement
      project every point into every OTHER camera, compare its z-depth there with that camera's rendered depth;
      the camera agrees if |z - d_other| <= --consistency-mm.  A point is kept when at least --min-views
      cameras agree (its own camera counts as one).  `support` = number of agreeing cameras.
  de-duplication
      the same surface point is produced by every camera that sees it; a --voxel-mm voxel grid keeps the
      highest-support representative per voxel (this is what COLMAP's stereo_fusion does internally too).

Normals are computed from the backprojected world points by cross products of the 4-neighbourhood and flipped to
face the producing camera -- NOT taken from `rendered_normal`, whose frame convention is not documented upstream.
RGB is the GROUND-TRUTH photo at the render resolution (so the viewer shows the real appearance, not the model's).

Run by lanes_src/S0/run.py; standalone:
  PYTHONPATH=$ROOT/repos/AmbiSuR $ROOT/envs/ambisur/bin/python $ROOT/lanes_src/S0/export_surface.py \
      -s <scene> -m <model> --iteration 1500 --out <lane out dir> --manifest <manifest.json> --dataset-json <...>
"""
import json, os, sys, time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

# orhsurf: paths resolved relative to this package, never to the machine it was written on.
from orhsurf.paths import ambisur_repo
sys.path.insert(0, str(ambisur_repo()))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # lanes_io lives beside us now

from scene import Scene                                    # noqa: E402
from gaussian_renderer import render, GaussianModel        # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from utils.general_utils import safe_state                 # noqa: E402
from lanes_io import write_frame                           # noqa: E402
from colmap_dataset import read_model_txt           # noqa: E402


def normals_from_grid(Xw, valid):
    """World-frame normals from the backprojected (H,W,3) grid via central differences; unit length, 0 where
    the 4-neighbourhood is not fully valid."""
    H, W, _ = Xw.shape
    n = torch.zeros_like(Xw)
    dx = Xw[1:-1, 2:] - Xw[1:-1, :-2]
    dy = Xw[2:, 1:-1] - Xw[:-2, 1:-1]
    c = torch.cross(dx, dy, dim=-1)
    ok = (valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 1:-1])
    c = c / (c.norm(dim=-1, keepdim=True) + 1e-12)
    n[1:-1, 1:-1] = c * ok[..., None]
    return n


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--consistency-mm", type=float, default=5.0)
    parser.add_argument("--min-views", type=int, default=2)
    # Isolation gate.  Measured on mv0/frame 37 (viz4d/floater_prep.py, bench/nn_outlier.py):
    # `min-views 2 + NN k5 > 5mm` keeps 81.69% at hole 9.275%.  NN alone removes 79.12% of the
    # support-1 population at 5.78% of the corroborated cloud -- selectivity 13.7 -- WITHOUT any
    # reference surface, which is why it replaces the SELF carve family: those judge a point
    # against `support >= 8`, so a weakly reconstructed surface is convicted of floating in front
    # of the background that stands in for it, and the back of the person loses half its points.
    # --nn-max-mm 0 disables the gate and reproduces the pre-2026-09-20 export exactly.
    parser.add_argument("--nn-k", type=int, default=5)
    parser.add_argument("--nn-max-mm", type=float, default=5.0)
    parser.add_argument("--alpha-min", type=float, default=0.5)
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    nn_on = args.nn_max_mm > 0
    # Both gates then run together after de-duplication, on the k-NN distances of the full cloud.
    gate_min = 0 if nn_on else args.min_views
    dev = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(args.manifest))
    ds = json.load(open(args.dataset_json))
    scene_cams, scene_imgs = read_model_txt(Path(args.source_path) / "sparse_txt")

    with torch.no_grad():
        gaussians = GaussianModel(model.extract(args).sh_degree)
        sc = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
        views = sc.getTrainCameras()
        bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=dev)
        pipe = pipeline.extract(args)

        # ---------------- render + calibration check ----------------
        per_view, checks = [], {}
        for v in views:
            serial = v.image_name
            assert serial in man["cameras"], serial
            sc_img = scene_imgs[f"{serial}.png"]; sc_cam = scene_cams[sc_img["camera_id"]]
            ratio = v.image_width / sc_cam["width"]
            fx, fy, cx, cy = sc_cam["params"]
            # AmbiSuR keeps only the FoV and re-derives Fx/Cx at the render resolution -- verify that this
            # equals the calibrated K scaled by `ratio`, i.e. that nothing about the intrinsics was lost.
            e_fx, e_fy = fx * ratio, fy * (v.image_height / sc_cam["height"])
            e_cx, e_cy = cx * ratio, cy * (v.image_height / sc_cam["height"])
            dK = [abs(v.Fx - e_fx), abs(v.Fy - e_fy), abs(v.Cx - e_cx), abs(v.Cy - e_cy)]
            assert max(dK) < 0.51, (serial, v.Fx, e_fx, v.Cx, e_cx, v.Cy, e_cy, dK)
            # pose: R = R_world_from_cam, T = t_cam_from_world  (scene/dataset_readers.py)
            R_cw = np.array(v.R).T; t_cw = np.array(v.T)
            T_wc = np.array(man["cameras"][serial]["T_world_from_camera"])
            dR = float(np.abs(R_cw - T_wc[:3, :3].T).max())
            dT = float(np.abs(t_cw + T_wc[:3, :3].T @ T_wc[:3, 3]).max())
            assert dR < 1e-6 and dT < 1e-6, (serial, dR, dT)
            checks[serial] = dict(render_wh=[v.image_width, v.image_height], scene_wh=[sc_cam["width"], sc_cam["height"]],
                                  K_render=[float(v.Fx), float(v.Fy), float(v.Cx), float(v.Cy)],
                                  max_abs_K_err_px=float(max(dK)), max_abs_R_err=dR, max_abs_t_err_m=dT)

            o = render(v, gaussians, pipe, bg)
            d = o["plane_depth"].squeeze().float()
            alpha = o["rendered_alpha"].squeeze().float()
            gt, _ = v.get_image()
            H, W = d.shape
            vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                    torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
            Xc = torch.stack([(uu - v.Cx) / v.Fx * d, (vv - v.Cy) / v.Fy * d, d], -1)
            Rt = torch.tensor(np.array(v.R), dtype=torch.float32, device=dev)          # R_world_from_cam
            Tt = torch.tensor(np.array(v.T), dtype=torch.float32, device=dev)
            Xw = (Xc - Tt) @ Rt.T
            valid = (d > 0) & (alpha > args.alpha_min) & torch.isfinite(d)
            per_view.append(dict(serial=serial, d=d, valid=valid, Xw=Xw, rgb=gt.permute(1, 2, 0).float(),
                                 R=Rt, T=Tt, Fx=float(v.Fx), Fy=float(v.Fy), Cx=float(v.Cx), Cy=float(v.Cy),
                                 H=H, W=W, n=normals_from_grid(Xw, valid),
                                 centre=torch.tensor(np.array(v.R) @ -np.array(v.T), dtype=torch.float32, device=dev)))
            print(f"  {serial}: {int(valid.sum()):,}/{H*W:,} valid px, depth "
                  f"{float(d[valid].min()):.3f}..{float(d[valid].max()):.3f} m", flush=True)

        # ---------------- multi-view agreement ----------------
        thr = args.consistency_mm / 1000.0
        keep_xyz, keep_rgb, keep_n, keep_sup = [], [], [], []
        for i, A in enumerate(per_view):
            m = A["valid"]
            X = A["Xw"][m]; rgb = A["rgb"][m]; nn = A["n"][m]
            sup = torch.ones(len(X), dtype=torch.int16, device=dev)
            for j, B in enumerate(per_view):
                if i == j:
                    continue
                Xc = (X @ B["R"]) + B["T"]                       # R_cam_from_world = R_world_from_cam^T
                z = Xc[:, 2]
                u = Xc[:, 0] / z * B["Fx"] + B["Cx"]
                vpx = Xc[:, 1] / z * B["Fy"] + B["Cy"]
                iu = u.round().long(); iv = vpx.round().long()
                ok = (z > 0) & (iu >= 0) & (iu < B["W"]) & (iv >= 0) & (iv < B["H"])
                iu = iu.clamp(0, B["W"] - 1); iv = iv.clamp(0, B["H"] - 1)
                dB = B["d"][iv, iu]; vB = B["valid"][iv, iu]
                agree = ok & vB & ((z - dB).abs() <= thr)
                sup += agree.to(torch.int16)
            # With the isolation gate on, the min-views gate moves AFTER de-duplication: the k-NN
            # distances are the ones measured on the full cloud, where support-1 points still count
            # as neighbours.  Gating first would change the distances and so change the filter.
            # For a max-support de-dup the two orders are otherwise equivalent (the voxel winner is
            # the highest-support point either way), so nothing else shifts.
            k = sup >= gate_min
            keep_xyz.append(X[k]); keep_rgb.append(rgb[k]); keep_n.append(nn[k]); keep_sup.append(sup[k])
            print(f"  {A['serial']}: kept {int(k.sum()):,}/{len(X):,} "
                  f"(support>={gate_min} within {args.consistency_mm} mm)", flush=True)

        xyz = torch.cat(keep_xyz); rgbv = torch.cat(keep_rgb); nrm = torch.cat(keep_n); sup = torch.cat(keep_sup)
        n_before_dedup = int(len(xyz))

        # ---------------- voxel de-duplication (highest support wins) ----------------
        # The key/lin/order buffers are int64 and N-sized: at -r 1 (47 cams x 1972x1472) N is
        # ~1.4e8, so key alone is 3.3 GB and the argsort workspace pushes the total past 24 GB.
        # Above CPU_DEDUP_N the dedup runs in numpy instead; the result is moved to CPU on the
        # next line either way, so nothing downstream changes.
        CPU_DEDUP_N = 40_000_000
        vox = args.voxel_mm / 1000.0
        if len(xyz) > CPU_DEDUP_N:
            print(f"  [dedup] {len(xyz):,} points > {CPU_DEDUP_N:,}; de-duplicating on CPU", flush=True)
            xyz = xyz.cpu(); rgbv = rgbv.cpu(); nrm = nrm.cpu(); sup = sup.cpu()
            torch.cuda.empty_cache()
        key = torch.floor(xyz / vox).to(torch.int64)
        key = key - key.min(0).values
        span = key.max(0).values + 1
        lin = key[:, 0] * (span[1] * span[2]) + key[:, 1] * span[2] + key[:, 2]
        del key
        order = torch.argsort(lin * 64 - sup.to(torch.int64))        # same voxel -> highest support first
        lin_s = lin[order]
        first = torch.ones_like(lin_s, dtype=torch.bool)
        first[1:] = lin_s[1:] != lin_s[:-1]
        sel = order[first]
        del lin, lin_s, first, order
        xyz = xyz[sel]; rgbv = rgbv[sel]; nrm = nrm[sel]; sup = sup[sel]

    xyz_n = xyz.cpu().numpy().astype(np.float32)
    rgb_n = (rgbv.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    nrm_n = nrm.cpu().numpy().astype(np.float32)
    sup_n = sup.cpu().numpy().astype(np.int16)

    # ---------------- isolation + min-views gate ----------------
    n_iso = n_mv = 0
    if nn_on:
        from scipy.spatial import cKDTree
        t_nn = time.time()
        dist, _ = cKDTree(xyz_n).query(xyz_n, k=args.nn_k + 1, workers=8)
        dk_mm = dist[:, args.nn_k] * 1000.0            # column 0 is the point itself
        iso = dk_mm > args.nn_max_mm
        mv = sup_n < args.min_views
        n_iso, n_mv = int((iso & ~mv).sum()), int(mv.sum())
        keep = ~iso & ~mv
        assert keep.sum() > 0.5 * len(keep), (
            f"the isolation gate would drop {100*(1-keep.mean()):.1f}% of the cloud; "
            f"nn-k={args.nn_k} nn-max-mm={args.nn_max_mm} is wrong for this point density "
            f"(k={args.nn_k} distance median {np.median(dk_mm):.3f} mm)")
        print(f"  [nn] k={args.nn_k} median {np.median(dk_mm):.3f} mm; dropped {n_mv:,} for "
              f"support<{args.min_views} and a further {n_iso:,} as isolated "
              f"(> {args.nn_max_mm} mm), {time.time()-t_nn:.0f}s", flush=True)
        xyz_n, rgb_n, nrm_n, sup_n = xyz_n[keep], rgb_n[keep], nrm_n[keep], sup_n[keep]
    conf = np.clip((sup_n.astype(np.float32) - args.min_views) / 6.0, 0, 1)
    observed = np.ones(len(xyz_n), bool)
    assert np.isfinite(xyz_n).all() and len(xyz_n) > 0

    rep = dict(n_points=int(len(xyz_n)), n_before_dedup=n_before_dedup, voxel_mm=args.voxel_mm,
               consistency_mm=args.consistency_mm, min_views=args.min_views, alpha_min=args.alpha_min,
               nn_k=args.nn_k if nn_on else None, nn_max_mm=args.nn_max_mm if nn_on else None,
               n_dropped_min_views=n_mv, n_dropped_isolated=n_iso,
               iteration=sc.loaded_iter, n_gaussians=int(gaussians.get_xyz.shape[0]),
               n_train_cameras=len(views),
               bbox_min=xyz_n.min(0).tolist(), bbox_max=xyz_n.max(0).tolist(),
               median=np.median(xyz_n, 0).tolist(),
               p01=np.percentile(xyz_n, 1, axis=0).tolist(), p99=np.percentile(xyz_n, 99, axis=0).tolist(),
               support_mean=float(sup_n.mean()), support_max=int(sup_n.max()),
               calibration_check=checks,
               note="surface = rendered plane_depth backprojected with the calibrated K/pose and filtered by "
                    "multi-view agreement; Gaussian centres are NOT used")
    (out / "surface_export.json").write_text(json.dumps(rep, indent=1))
    meta = dict(timestamp=0.0, pose_version="pending", method="AmbiSuR rendered-depth surface",
                config_hash="pending", source_manifest=str(args.manifest))
    write_frame(out, xyz_n, nrm_n, rgb_n, conf, observed, sup_n, meta)
    print(f"[export] {len(xyz_n):,} points (from {n_before_dedup:,} before {args.voxel_mm} mm dedup), "
          f"{int(gaussians.get_xyz.shape[0]):,} gaussians, support mean {sup_n.mean():.2f}")


if __name__ == "__main__":
    main()
