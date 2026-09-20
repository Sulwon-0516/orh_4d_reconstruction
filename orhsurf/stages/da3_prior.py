"""DA3 depth prior at process_res 1008 with OUR calibrated poses passed in, then rewarped into
the AmbiSuR scene pinhole.

This is the `S0_da3hi` recipe.  In the source project it was two lanes run by hand and it had been
executed on exactly ONE frame (`lanes_out_orh/S0_da3hi/00037`); everything here is that chain made
into a single automatic per-frame stage.

Provenance of each piece:
  run_da3_1008()        <- lanes_src/R/run.py:199-260 (`run_da3`) + :170-198 (`azimuth_groups`)
  visual_hull_center()  <- lanes_src/R/run.py:108-165 (`visual_hull_box`)
  rewarp()              <- bench/da3hi_prior_for_ambisur.py

WHY 1008 AND NOT THE 504 DEFAULT.  The alternative prior (`estimate_colmap_local.py`) loads every
view at 504, takes the aspect from a RANDOMLY shuffled image_path_list[0] so every depth map comes
out 350x504 regardless of that view's true aspect, lets DA3 estimate its OWN poses, scales the
translations by a hardcoded 2.5, and then RANSAC-fits a similarity onto the rig -- median camera
centre residual 33.7 mm, only 8/47 below 10 mm.  Here we pass our calibrated extrinsics and
intrinsics IN (`align_to_input_ext_scale=True`), assert DA3 hands them back unchanged, and the
depths are metric by construction, so `trans.json` scale is exactly 1.0 and no alignment runs.

HONESTY NOTE, recorded because it affects nobody's code but everybody's expectations: on frame 37,
the only frame where both priors were ever run, 1008 did not measurably beat 504 (PSNR 23.492 vs
23.538; point counts and floater-audit hole fractions within noise).  1008 is the default because
the user judged the 3D reconstructions better by eye.  The cost of that choice is the 24 GB VRAM
floor below -- it is the DA3 step, not training, that sets the GPU requirement for this package.

VRAM.  Measured on the reference RTX 4090: a group of 18 views at 1008 peaks at 23,353 MiB
torch-allocated / 24,090 MiB device.  `group_size` is exposed so a smaller GPU can trade batch
size for cross-view context, but note that DA3's prediction depends on the whole group, so
changing it changes the output -- it is not a free knob.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The recipe's constants. Hardcoded and asserted, per project policy: if the data is not what this
# was written for, stop -- do not adapt and silently produce something else.
DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
# Pinned commit.  fetch_ckpts.sh downloads exactly this; loading without it would consult the
# `main` ref, which a commit-only snapshot does not create -- so an offline node with a correctly
# pre-staged cache would fail, and an online node could silently load a different revision.
DA3_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
PROCESS_RES = 1008
GROUP_SIZE = 18
GROUP_OVERLAP = 6
CONF_PERCENT = 20        # percentile used to threshold the initial point cloud
MAX_POINTS = 50_000      # == S0's --max-points-for-colmap
SEED = 42


# --------------------------------------------------------------------------- grouping --
def azimuth_groups(centers: dict[str, np.ndarray], serials: list[str], center: np.ndarray,
                   size: int = GROUP_SIZE, overlap: int = GROUP_OVERLAP):
    """Order cameras by azimuth about `center` and cut overlapping windows.

    From lanes_src/R/run.py:170-198, unchanged in behaviour.  Returns (groups, owner) where
    owner[serial] is the group in which that camera sits most centrally -- that group's prediction
    is the one kept, so every view is written exactly once despite the overlap.
    """
    ang = {s: float(np.arctan2(*(np.asarray(centers[s], float) - np.asarray(center, float))[[1, 0]]))
           for s in serials}
    order = sorted(serials, key=lambda s: ang[s])
    n = len(order)
    if size >= n:
        return [list(order)], {s: 0 for s in order}
    step = max(1, size - overlap)
    groups = [[order[(st + k) % n] for k in range(size)] for st in range(0, n, step)]
    owner, best = {}, {}
    for gi, g in enumerate(groups):
        for k, s in enumerate(g):
            d = abs(k - (size - 1) / 2.0)
            if s not in best or d < best[s]:
                best[s] = d
                owner[s] = gi
    assert set(owner) == set(serials), sorted(set(serials) ^ set(owner))
    return groups, owner


def visual_hull_center(manifest: dict, serials: list[str], mask_png: dict[str, Path],
                       *, spacing: float = 0.02, agree: float = 0.85, min_vis: int = 8,
                       margin: float = 0.10, coarse_grow: float = 0.40, log=print) -> np.ndarray:
    """Subject centre from the observation-derived masks alone. No human prior anywhere.

    Ported from lanes_src/R/run.py:108-165 (`visual_hull_box`), behaviour unchanged.

    WHY THIS IS NOT REPLACED BY THE RIG CENTROID.  Only the azimuth ORDER of the cameras is
    consumed downstream, so the rig centroid looks like a free simplification that would make this
    stage mask-free.  It is not: measured on the ORH clip the rig centroid sits 1.667 m from the
    visual-hull centre and produces DIFFERENT groups (all four of them -- see
    tools/check_azimuth_order.py, which exits non-zero on this clip).  DA3's prediction depends on
    the whole group, so different groups mean a different prior.  Masks cost us nothing anyway:
    the AmbiSuR scene build already requires them for each image's alpha channel.

    Coarse-to-fine: 5 cm over the camera-centre bounds, then `spacing` inside the survivor bbox. A
    voxel survives when it projects inside the mask in >= `agree` of the cameras that see it, with
    at least `min_vis` seeing it at all.
    """
    import cv2

    cams = manifest["cameras"]
    masks, Ku, Tcw, WH = {}, {}, {}, {}
    for s in serials:
        m = cv2.imread(str(mask_png[s]), cv2.IMREAD_GRAYSCALE)
        assert m is not None, f"cannot read mask for {s}: {mask_png[s]}"
        masks[s] = m >= 128
        Ku[s] = np.array(cams[s]["K_undistort"], float)
        Tcw[s] = np.array(cams[s]["T_cam_from_world"], float)
        WH[s] = (cams[s]["width"], cams[s]["height"])
    C = np.array([cams[s]["camera_center_world"] for s in serials], float)
    lo = np.array([C[:, 0].min() - 1, C[:, 1].min() - 1, -0.05])
    hi = np.array([C[:, 0].max() + 1, C[:, 1].max() + 1, 2.2])

    def carve(lo, hi, step):
        g = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
        P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        inside = np.zeros(len(P), np.int32)
        vis = np.zeros(len(P), np.int32)
        Ph = np.concatenate([P, np.ones((len(P), 1))], 1)
        for s in serials:
            W, H = WH[s]
            Xc = Ph @ Tcw[s].T
            z = Xc[:, 2]
            uv = Xc[:, :3] @ Ku[s].T
            ok = z > 1e-3
            u = np.full(len(P), -1.0)
            v = np.full(len(P), -1.0)
            u[ok] = uv[ok, 0] / z[ok]
            v[ok] = uv[ok, 1] / z[ok]
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            vis += ok
            ui = np.clip(u.astype(np.int32), 0, W - 1)
            vi = np.clip(v.astype(np.int32), 0, H - 1)
            inside += ok & masks[s][vi, ui]
        keep = (vis >= min_vis) & (inside >= agree * np.maximum(vis, 1))
        return P[keep], int(keep.sum())

    P5, n5 = carve(lo, hi, 0.05)
    assert n5 > 0, "coarse visual hull is empty: check the masks and the extrinsics convention"
    # The 5 cm carve keeps only the thick torso; limbs thinner than a voxel are carved away, so the
    # fine search box is the coarse bbox grown by `coarse_grow` (and down to the floor).
    lo2, hi2 = P5.min(0) - coarse_grow, P5.max(0) + coarse_grow
    lo2[2] = max(lo[2], min(lo2[2], -0.05))
    hi2[2] = min(hi[2], hi2[2])
    P2, n2 = carve(lo2, hi2, spacing)
    assert n2 > 0, "fine visual hull is empty"
    mn, mx = P2.min(0) - margin, P2.max(0) + margin
    c = (mn + mx) / 2
    log(f"[da3] visual hull: {n5} coarse / {n2} fine voxels, centre "
        f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    return c


# ------------------------------------------------------------------------ DA3 inference --
def run_da3_1008(manifest: dict, serials: list[str], undist_png: dict[str, Path],
                 mask_png: dict[str, Path], out_dir: Path,
                 *, model_id: str = DA3_MODEL, process_res: int = PROCESS_RES,
                 group_size: int = GROUP_SIZE, group_overlap: int = GROUP_OVERLAP,
                 revision: str = DA3_REVISION, log=print) -> dict:
    """Run DA3 on the undistorted frames with our poses; write <out_dir>/<serial>.npz per view.

    `undist_png[serial]` must be the UNDISTORTED pinhole image matching `K_undistort` -- all the
    same size across views.  DA3 centre-crops a batch to its smallest member, so feeding images of
    differing size (e.g. the AmbiSuR scene images, which are cropped per view by the principal-point
    recentring) would silently crop the batch and break the K_proc assertion below.
    """
    import torch
    from depth_anything_3.api import DepthAnything3

    cams = manifest["cameras"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = {(cams[s]["width"], cams[s]["height"]) for s in serials}
    assert len(sizes) == 1, (
        f"DA3 needs one image size for the whole batch, got {sorted(sizes)}. "
        f"Feed the undistorted frames, not the AmbiSuR scene images.")

    centers = {s: np.asarray(cams[s]["camera_center_world"], float) for s in serials}
    center = visual_hull_center(manifest, serials, mask_png, log=log)
    groups, owner = azimuth_groups(centers, serials, center, group_size, group_overlap)
    log(f"[da3] {len(serials)} views -> {len(groups)} group(s) of {len(groups[0])} "
        f"(overlap {group_overlap}), process_res {process_res}")

    try:
        model = DepthAnything3.from_pretrained(model_id, revision=revision).to("cuda").eval()
    except TypeError:                 # older DA3 wrapper without a revision kwarg
        log(f"[da3] WARNING: this DA3 build ignores `revision`; loading {model_id} unpinned")
        model = DepthAnything3.from_pretrained(model_id).to("cuda").eval()
    written, gmeta = {}, []
    for gi, g in enumerate(groups):
        keep = [s for s in g if owner[s] == gi]
        imgs = [str(undist_png[s]) for s in g]
        exts = np.asarray([np.array(cams[s]["T_cam_from_world"], float) for s in g], np.float32)
        ixts = np.asarray([np.array(cams[s]["K_undistort"], float) for s in g], np.float32)
        torch.cuda.reset_peak_memory_stats()
        pred = model.inference(image=imgs, extrinsics=exts, intrinsics=ixts,
                               align_to_input_ext_scale=True, use_ray_pose=True,
                               process_res=process_res,
                               process_res_method="upper_bound_resize")
        depth = np.asarray(pred.depth, np.float32)
        conf = np.asarray(pred.conf, np.float32)
        Kp = np.asarray(pred.intrinsics, np.float64)
        Ep = np.asarray(pred.extrinsics, np.float64)
        # The whole point of this recipe: DA3 must NOT have moved our cameras.
        assert np.allclose(Ep, exts[:, :3, :], atol=1e-4), (
            f"group {gi}: DA3 did not hand our extrinsics back "
            f"(max err {np.abs(Ep - exts[:, :3, :]).max():.3e}); "
            f"align_to_input_ext_scale is not behaving as this recipe requires")
        h, w = depth.shape[1:]
        peak = int(torch.cuda.max_memory_allocated() / 2 ** 20)
        log(f"[da3]   group {gi}: {len(g)} views, keeping {len(keep)}, "
            f"proc {h}x{w}, torch peak {peak} MiB")
        for k, s in enumerate(g):
            if s not in keep:
                continue
            W, H = cams[s]["width"], cams[s]["height"]
            Ku = np.array(cams[s]["K_undistort"], float)
            Kexp = Ku.copy()
            Kexp[0] *= w / W
            Kexp[1] *= h / H
            assert np.abs(Kp[k] - Kexp).max() < 1.0, (
                f"{s}: DA3's processed K is not the manifest K scaled by the resize factor")
            np.savez_compressed(out_dir / f"{s}.npz", depth=depth[k], conf=conf[k],
                                K_proc=Kp[k], K_undistort=Ku,
                                T_cam_from_world=np.array(cams[s]["T_cam_from_world"], float),
                                proc_hw=np.array([h, w]), full_hw=np.array([H, W]),
                                group=np.array([gi]))
            written[s] = dict(group=gi, depth_med=float(np.median(depth[k])),
                              conf_med=float(np.median(conf[k])))
        gmeta.append(dict(group=gi, n_views=len(g), n_kept=len(keep), proc_hw=[h, w],
                          torch_peak_mib=peak))
        del pred, depth, conf
        torch.cuda.empty_cache()
    assert set(written) == set(serials), sorted(set(serials) - set(written))
    return dict(model=model_id, revision=revision, process_res=process_res, group_size=group_size,
                group_overlap=group_overlap, groups=gmeta, per_view=written)


# --------------------------------------------------------------------------- rewarp --
def _q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _read_scene_model(scene: Path) -> dict:
    """serial -> dict(image_id, K, W, H, T) from the scene's own COLMAP TXT model."""
    cams = {}
    for line in open(scene / "sparse_txt/cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        cid, model, W, H = int(f[0]), f[1], int(f[2]), int(f[3])
        assert model == "PINHOLE", f"scene camera {cid} is {model}, expected PINHOLE"
        fx, fy, cx, cy = (float(x) for x in f[4:8])
        # AmbiSuR renders a symmetric frustum and hardcodes the principal point to the exact
        # centre (scene/cameras.py:76-77), so the scene build must already have centred it.
        assert abs(cx - W / 2.0) < 1e-9 and abs(cy - H / 2.0) < 1e-9, (
            f"scene camera {cid} principal point ({cx},{cy}) is not the exact centre of {W}x{H}")
        cams[cid] = dict(K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), W=W, H=H)
    out = {}
    for line in open(scene / "sparse_txt/images.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 10:
            continue
        T = np.eye(4)
        T[:3, :3] = _q2R([float(x) for x in f[1:5]])
        T[:3, 3] = [float(x) for x in f[5:8]]
        out[f[9].split(".")[0]] = dict(image_id=int(f[0]), name=f[9], T=T, **cams[int(f[8])])
    return out


def _write_points3D_bin(path: Path, xyz, rgb, image_ids) -> None:
    """COLMAP points3D.bin. AmbiSuR reads xyz/rgb/error and skips the track."""
    import struct
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i in range(len(xyz)):
            f.write(struct.pack("<QdddBBBd", i + 1,
                                float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                                int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), 0.0))
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ii", int(image_ids[i]), 0))


def rewarp(scene: Path, da3_dir: Path, *, target_w: int = PROCESS_RES,
           conf_percent: int = CONF_PERCENT, max_points: int = MAX_POINTS,
           seed: int = SEED, log=print) -> dict:
    """Rewarp the DA3 depths onto the AmbiSuR scene pinhole and write the prior into `scene`.

    THE GEOMETRY.  DA3 predicted on the manifest's K_undistort pinhole; AmbiSuR's scene uses a
    different pinhole (COLMAP's fx/fy with the principal point forced to the exact centre and the
    frame cropped).  Both share the SAME camera pose, so the two images are related by the pure
    homography H = K_proc @ inv(K_target), independent of depth -- and camera z-depth is INVARIANT
    under it, because X_cam = z * K^-1 [u,v,1]: changing K changes the ray, not z.  So the depth
    VALUES transfer unchanged and only the sampling grid moves.  Asserted, not assumed: every
    target pixel must land inside the source image.

    Writes exactly what repos/AmbiSuR/scene/dataset_readers.py reads:
        estimated_depths/<serial>.png.npy    note the ".png.npy": basename(image_path) + ".npy"
        estimated_confs/<serial>.png.npy
        sparse_da3_aligned/0/trans.json      only the "scale" key is read (line 126-128)
        sparse_da3_aligned/0/points3D.bin    the initial point cloud (line 219-226)
    Poses and intrinsics come from <scene>/sparse/0, NOT from sparse_da3_aligned.
    """
    import cv2

    scene, da3_dir = Path(scene), Path(da3_dir)
    rng = np.random.default_rng(seed)
    for d in ("estimated_depths", "estimated_confs", "sparse_da3_aligned/0"):
        (scene / d).mkdir(parents=True, exist_ok=True)

    model = _read_scene_model(scene)
    serials = sorted(model)
    have = sorted(p.stem for p in da3_dir.glob("*.npz"))
    assert have == serials, (
        f"DA3 outputs do not match the scene: missing {sorted(set(serials) - set(have))}, "
        f"extra {sorted(set(have) - set(serials))}")

    rep, all_conf, per_view = {}, [], {}
    worst_cov = 1.0
    for s in serials:
        c = model[s]
        z = np.load(da3_dir / f"{s}.npz")
        assert np.abs(np.array(z["T_cam_from_world"], float) - c["T"]).max() < 1e-9, (
            f"{s}: the DA3 pose and the AmbiSuR scene pose differ; the rewarp homography is only "
            f"valid when both share the same camera pose")
        Kp = np.array(z["K_proc"], float)
        dep, cnf = z["depth"].astype(np.float32), z["conf"].astype(np.float32)
        hs, ws = dep.shape

        Ht = int(round(target_w * c["H"] / c["W"] / 2.0) * 2)      # even, true aspect per view
        Kt = c["K"].copy()
        Kt[0] *= target_w / c["W"]
        Kt[1] *= Ht / c["H"]

        u, v = np.meshgrid(np.arange(target_w, dtype=np.float32),
                           np.arange(Ht, dtype=np.float32))
        d_cam = np.linalg.inv(Kt) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
        p = Kp @ d_cam                                  # d_cam[2] == 1 exactly, so no divide
        assert np.abs(p[2] - 1.0).max() < 1e-9
        mx = p[0].reshape(Ht, target_w).astype(np.float32)
        my = p[1].reshape(Ht, target_w).astype(np.float32)
        inside = (mx >= 0) & (mx <= ws - 1) & (my >= 0) & (my <= hs - 1)
        cov = float(inside.mean())
        worst_cov = min(worst_cov, cov)
        assert cov > 0.995, f"{s}: only {cov:.4f} of the scene frame is covered by the DA3 depth map"

        dw = cv2.remap(dep, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cw = cv2.remap(cnf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cw[~inside] = 0.0                               # extrapolated pixels never vote
        np.save(scene / f"estimated_depths/{s}.png.npy", dw)
        np.save(scene / f"estimated_confs/{s}.png.npy", cw)
        all_conf.append(cw.ravel())
        per_view[s] = dict(target_hw=[Ht, target_w], coverage=round(cov, 6),
                           depth_med=float(np.median(dw)), conf_med=float(np.median(cw)))
        rep[s] = (dw, cw, Kt, Ht)

    log(f"[da3] rewarped {len(serials)} views, worst scene-frame coverage {worst_cov:.5f}")

    # initial point cloud, in the scene's own frame
    thr = float(np.percentile(np.concatenate(all_conf), conf_percent))
    P, C, I = [], [], []
    for s in serials:
        dw, cw, Kt, Ht = rep[s]
        c = model[s]
        m = (cw >= thr) & (dw > 1e-3)
        vi, ui = np.nonzero(m)
        zz = dw[vi, ui].astype(np.float64)
        Xc = (np.linalg.inv(Kt) @ np.stack([ui, vi, np.ones(len(ui))])) * zz
        Twc = np.linalg.inv(c["T"])
        P.append((Twc[:3, :3] @ Xc).T + Twc[:3, 3])
        img = cv2.cvtColor(cv2.imread(str(scene / f"images/{s}.png"), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        C.append(cv2.resize(img, (target_w, Ht), interpolation=cv2.INTER_AREA)[vi, ui])
        I.append(np.full(len(ui), c["image_id"]))
    P, C, I = np.concatenate(P), np.concatenate(C), np.concatenate(I)
    if len(P) > max_points:
        k = rng.choice(len(P), max_points, replace=False)
        P, C, I = P[k], C[k], I[k]
    _write_points3D_bin(scene / "sparse_da3_aligned/0/points3D.bin", P, C, I)

    # scale 1.0 is the whole difference from the 504 path: our poses went IN, so the depths are
    # already metric in the scene frame and AmbiSuR must not rescale them.
    (scene / "sparse_da3_aligned/0/trans.json").write_text(json.dumps(
        {"scale": 1.0,
         "note": "identity: DA3 was conditioned on our calibrated poses, so the depths are already "
                 "metric in the scene frame. No similarity was fitted, so there is no pose_align "
                 "residual for this prior."}, indent=4))
    import shutil
    for f in ("cameras.bin", "images.bin"):
        shutil.copy(scene / f"sparse/0/{f}", scene / f"sparse_da3_aligned/0/{f}")

    log(f"[da3] conf threshold {thr:.3f}, initial cloud {len(P):,} points")
    return dict(target_w=target_w, conf_percent=conf_percent, conf_threshold=thr,
                n_points=int(len(P)), worst_coverage=worst_cov, trans_scale=1.0,
                per_view=per_view)
