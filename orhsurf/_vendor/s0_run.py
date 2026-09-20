#!/usr/bin/env python
"""Lane S0 -- AmbiSuR (Gaussian-surface optimisation with a Depth-Anything-3 multi-view prior).

  $ROOT/envs/ambisur/bin/python $ROOT/lanes_src/S0/run.py \
      --manifest $ROOT/data/hrdex_apple_take0/manifest.json --frame-index 0 \
      --out $ROOT/lanes_out/S0/00000 [--heldout 23022632,23280286] [--gpu 3] [--force]

Stages (all timed in timing.json):
  1. lanes_src/common/colmap_dataset.py -> the shared COLMAP known-pose workspace; S0 eats its UNDISTORTED
     PINHOLE product (undist/images + undist/sparse + undist/masks).
  2. scene build -> AmbiSuR's expected layout:
        native/scene/images/<serial>.png    RGBA: RGB = undistorted photo, ALPHA = foreground mask
                                            (AmbiSuR reads a per-view mask ONLY from the image's alpha channel,
                                             scene/cameras.py::process_image; the DTU `mask/` dir is unused by
                                             this code path)
        native/scene/sparse/0/*.bin         PINHOLE, one camera per image, poses = manifest T_cam_from_world
     ** PRINCIPAL POINT **: AmbiSuR keeps only FovX/FovY from the COLMAP camera and then hardcodes
        Cx = 0.5*W, Cy = 0.5*H (scene/cameras.py:76-77) and renders with a SYMMETRIC frustum
        (getProjectionMatrix(fovX, fovY)); `getProjectionMatrixCenterShift` exists in the repo but is never
        called.  DA3 likewise predicts its own centred intrinsics.  On DTU that is harmless (cx=777=W/2,
        cy=581=H/2).  Our undistorted HRdex cameras are off-centre by up to (39, 85) px, which at -r 4 is a
        (10, 21) px shift between the rendered geometry and the ground-truth image -- the optimiser would
        absorb it as wrong geometry.  This runner therefore RE-CENTRES the principal point: each view is
        resampled straight from the raw distorted image onto a virtual pinhole with the COLMAP-undistorted
        fx/fy but cx = W/2, cy = H/2 exactly, with W and H multiples of 4 so that AmbiSuR's `-r 4` downscale
        keeps the principal point exactly centred.  Cost: the off-centre border is cropped away (per-camera
        numbers in metadata.json["scene"]["recentre"]).  Nothing else about the calibration is changed.
  3. DA3 prior: smoke/ambisur/codex_smoke/single_batch/estimate_colmap_local.py with the upstream chunk 450
     (= all views in ONE inference batch).  ambisur.report.md measured that the chunked path (chunk 8) leaves
     the per-chunk DA3 frames without a common transform: only 9/49 camera centres aligned below 0.01 vs 47/49
     for the single batch.  --shared_camera is NOT passed: unlike DTU our 20 cameras have different intrinsics
     (fx 1790..7383).
  4. multi_view_priors/pose_align.py  (RANSAC similarity, thresh 0.01) -> sparse_da3_aligned/0 + trans.json;
     the runner then measures the camera-centre residual against the calibrated poses itself.
  5. train.py with the 1500-step budget from the smoke (recorded in metadata.json["budget"]).
  6. lanes_src/S0/export_surface.py -- THE SURFACE, not the Gaussian centres: render depth + normal from every
     training camera, backproject with the calibrated K/pose, and keep only points that at least --min-views
     cameras agree on within --consistency-mm.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import cv2

# orhsurf: only build_scene() is used from this file; the lane driver below is kept for reference
# but is not invoked. ROOT is resolved from the package, never from the machine it was written on.
from orhsurf.paths import REPO_ROOT as ROOT, ambisur_repo, python_bin
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_dataset import (build_dataset, load_manifest, select_serials, sha256,  # noqa: E402
                                   read_model_txt, quat_to_rot, rot_to_quat_wxyz, intrinsics, COLMAP_BIN)

REPO = ambisur_repo()
PY = python_bin()
DA3_SCRIPT = None   # orhsurf uses the 1008 prior in orhsurf/stages/da3_prior.py
LANE = "S0"
METHOD = "AmbiSuR (Gaussian surface) + DepthAnything3 multi-view prior, calibrated poses fixed"


def repo_env(gpu):
    # bin/lane_exec has already set CUDA_VISIBLE_DEVICES (and the 8-core taskset / thread caps) for this slot.
    # Overwriting it with the PHYSICAL id here would point the child at a device that is not visible to it.
    e = dict(os.environ)
    e.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    e.update(CUDA_HOME="/usr/local/cuda-12.8", TORCH_CUDA_ARCH_LIST="8.9",
             MAX_JOBS="8", PIP_CACHE_DIR=str(ROOT / "cache/pip"), HF_HOME=str(ROOT / "cache/hf"),
             TORCH_HOME=str(ROOT / "cache/torch"),
             TORCH_EXTENSIONS_DIR=str(ROOT / "cache/torch_ext/ambisur"),
             TRITON_CACHE_DIR=str(ROOT / "cache/triton/ambisur"),
             PYTHONPATH=f"{REPO/'multi_view_priors'}:{REPO}",
             PATH=f"{ROOT/'envs/ambisur/bin'}:/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", ""))
    return e


# ------------------------------------------------------------- scene build ---
def build_scene(ds, man, serials, scene_dir, init_points_txt=None):
    """Write AmbiSuR's scene layout with a re-centred principal point.  Returns a report dict."""
    scene_dir = Path(scene_dir)
    img_dir = scene_dir / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    txt = scene_dir / "sparse_txt"; txt.mkdir(parents=True, exist_ok=True)
    rep = {}
    with open(txt / "cameras.txt", "w") as fc, open(txt / "images.txt", "w") as fi, open(txt / "points3D.txt", "w") as fp:
        fc.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fi.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        fp.write("# empty: the initial point cloud comes from the aligned DA3 model\n")
        for i, s in enumerate(serials, start=1):
            cam = ds["cameras"][s]; u = cam["undist"]
            Ku = np.array(u["K"]); Wu, Hu = u["width"], u["height"]
            fx, fy, cx, cy = Ku[0, 0], Ku[1, 1], Ku[0, 2], Ku[1, 2]
            # largest centred window that still fits inside the undistorted image, rounded DOWN to a multiple of 4
            Wt = int(4 * ((2 * min(cx, Wu - 1 - cx)) // 4))
            Ht = int(4 * ((2 * min(cy, Hu - 1 - cy)) // 4))
            assert Wt >= 512 and Ht >= 384, (s, Wt, Ht)
            Kt = np.array([[fx, 0.0, Wt / 2.0], [0.0, fy, Ht / 2.0], [0.0, 0.0, 1.0]])
            K0, dist = intrinsics(man["cameras"][s])
            mx, my = cv2.initUndistortRectifyMap(K0, dist, None, Kt, (Wt, Ht), cv2.CV_32FC1)
            raw = cv2.imread(cam["image"], cv2.IMREAD_COLOR)
            rgb = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
            # AmbiSuR takes a per-view mask ONLY from the image alpha channel; with no mask (the ORH clip has
            # none) the image is written as plain RGB and `Camera.mask` stays None, i.e. full-scene training.
            mpath = Path(ds["workspace"]) / "masks_distorted" / f"{s}.png"
            alpha = None
            if mpath.exists():
                alpha = cv2.remap(cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE), mx, my, cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cv2.imwrite(str(img_dir / f"{s}.png"), rgb if alpha is None else np.dstack([rgb, alpha]))
            # float(): numpy>=2 makes repr(np.float64(x)) == "np.float64(x)", which COLMAP's
            # ReadCamerasText rejects with std::invalid_argument -> SIGABRT.  Keep full precision.
            fc.write(f"{i} PINHOLE {Wt} {Ht} {float(fx)!r} {float(fy)!r} "
                     f"{float(Wt)/2.0!r} {float(Ht)/2.0!r}\n")
            R = np.array(u["R_cam_from_world"]); t = np.array(u["t_cam_from_world"])
            q = rot_to_quat_wxyz(R)
            v = [float(x) for x in (*q, *t)]
            fi.write(f"{i} " + " ".join(repr(x) for x in v) + f" {i} {s}.png\n\n")
            rep[s] = dict(index=i, W=Wt, H=Ht, K=Kt.tolist(),
                          pp_offset_removed_px=[float(cx - Wu / 2.0), float(cy - Hu / 2.0)],
                          pp_shift_at_r4_px=[float((cx - Wu / 2.0) / 4.0), float((cy - Hu / 2.0) / 4.0)],
                          cropped_px=[Wu - Wt, Hu - Ht],
                          fg_fraction=None if alpha is None else float((alpha > 0).mean()))
    if init_points_txt is not None:                       # READ-ONLY copy of another lane's triangulation
        src = Path(init_points_txt) / "points3D.txt"
        assert src.exists(), src
        keep, n = [], 0
        for line in open(src):
            if line.startswith("#") or not line.strip():
                continue
            f = line.split(); n += 1
            # POINT3D_ID X Y Z R G B ERROR TRACK[]  -> keep xyz+rgb, drop the track (its image ids are that
            # model's, not ours; AmbiSuR/readColmapSceneInfo only reads xyz and rgb)
            keep.append(" ".join(f[:8]))
        (txt / "points3D.txt").write_text("\n".join(keep) + "\n")
        rep_points = dict(source=str(src), n_points=n)
    else:
        rep_points = None
    sp = scene_dir / "sparse/0"; sp.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(COLMAP_BIN), "model_converter", "--input_path", str(txt), "--output_path", str(sp),
                    "--output_type", "BIN"], check=True, capture_output=True)
    for f in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (sp / f).exists(), sp / f
    (scene_dir / "index_to_serial.json").write_text(json.dumps({v["index"]: k for k, v in rep.items()}, indent=1))
    return rep, rep_points


def multi_view_stats(man, serials, max_angle, min_dis, max_dis):
    """How many multi-view partners AmbiSuR's own selection rule (scene/__init__.py) gives each camera.
    The defaults (30 deg, 0.01..1.5 scene units) are tuned for the DTU arc; our rig is a 20-camera dome with
    optical-axis angles up to 172 deg, so this is worth recording with every run."""
    C = np.array([np.array(man["cameras"][s]["T_world_from_camera"])[:3, 3] for s in serials])
    A = np.array([np.array(man["cameras"][s]["T_world_from_camera"])[:3, 2] for s in serials])
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(A @ A.T, -1, 1)))
    dis = np.linalg.norm(C[:, None] - C[None], axis=-1)
    m = (ang < max_angle) & (dis > min_dis) & (dis < max_dis)
    np.fill_diagonal(m, False)
    cnt = m.sum(1)
    return dict(rule=dict(max_angle_deg=max_angle, min_dis_m=min_dis, max_dis_m=max_dis),
                per_camera={s: int(c) for s, c in zip(serials, cnt)}, min=int(cnt.min()),
                median=int(np.median(cnt)), max=int(cnt.max()), n_zero=int((cnt == 0).sum()),
                n_at_least_3=int((cnt >= 3).sum()))


def camera_centre_residuals(aligned_model_dir, man, serials):
    """Euclidean distance between the aligned DA3 camera centres and the calibrated ones (metres)."""
    subprocess.run([str(COLMAP_BIN), "model_converter", "--input_path", str(aligned_model_dir),
                    "--output_path", str(aligned_model_dir), "--output_type", "TXT"], check=True, capture_output=True)
    _, imgs = read_model_txt(aligned_model_dir)
    res = {}
    for s in serials:
        im = imgs[f"{s}.png"]
        R = quat_to_rot(im["qvec"]); C = -R.T @ im["tvec"]
        C0 = np.array(man["cameras"][s]["T_world_from_camera"])[:3, 3]
        res[s] = float(np.linalg.norm(C - C0))
    v = np.array(list(res.values()))
    return dict(per_camera_m=res, median_m=float(np.median(v)), p95_m=float(np.percentile(v, 95)),
                max_m=float(v.max()), n_below_10mm=int((v < 0.01).sum()), n=len(v))


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    # MASK POLICY: the reconstruction input must not contain a human prior. $ROOT/data/hrdex_apple_take0/
    # fg_masks is the P0 lane's alpha UNION projected MANO mesh, so it is NOT used as an input by default;
    # the default is the manifest mask read from its ALPHA channel. --masks fg is an explicit opt-in for
    # ablations, and fg_masks stays available for evaluation overlays (lanes_src/common/inspect_lane.py).
    ap.add_argument("--masks", choices=("alpha", "fg"), default="alpha",
                    help="alpha = manifest mask PNG alpha channel (default, no human prior); "
                         "fg = $ROOT/data/hrdex_apple_take0/fg_masks (alpha u projected MANO, ablation only)")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--resolution", type=int, default=4)
    ap.add_argument("--warmup-from-iter", type=int, default=700)
    ap.add_argument("--ncc-scale", type=float, default=0.5)
    ap.add_argument("--depth-weight", type=float, default=0.1)
    ap.add_argument("--sh-unc-lower-max", type=float, default=0.2)
    ap.add_argument("--ransac-thresh", type=float, default=0.01)
    ap.add_argument("--multi-view-num", type=int, default=8)
    # NOTE: AmbiSuR derives each CLI arg's type from its default, and multi_view_max_angle defaults to the
    # INT 30 -- passing "30.0" makes train.py exit 2 with "invalid int value".
    ap.add_argument("--multi-view-max-angle", type=int, default=30, help="AmbiSuR default (tuned for the DTU arc)")
    ap.add_argument("--multi-view-max-dis", type=float, default=1.5, help="AmbiSuR default, SCENE UNITS = metres here")
    ap.add_argument("--multi-view-min-dis", type=float, default=0.01)
    ap.add_argument("--max-points-for-colmap", type=int, default=50000)
    ap.add_argument("--init-points-from", default=None,
                    help="COLMAP TXT model dir whose points3D.txt seeds the scene (READ-ONLY, e.g. the C "
                         "lane's native/sparse_tri_txt). Only used as a fallback init: when the DA3 "
                         "alignment succeeds AmbiSuR reads its points from sparse_da3_aligned/0.")
    ap.add_argument("--consistency-mm", type=float, default=5.0)
    ap.add_argument("--min-views", type=int, default=2)
    a = ap.parse_args()

    # bin/lane_exec already exported CUDA_VISIBLE_DEVICES for this slot; --gpu is the PHYSICAL id used for the
    # nvidia-smi VRAM sampling only.  Never overwrite an existing CUDA_VISIBLE_DEVICES here.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    cv2.setNumThreads(int(os.environ.get("LANE_MAX_WORKERS", "8")))
    from common.stage import Timeline

    # absolute paths everywhere: the repo subprocesses run with cwd=$ROOT/repos/AmbiSuR
    a.manifest = str(Path(a.manifest).resolve())
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    native = out / "native"; native.mkdir(exist_ok=True)
    heldout = [s for s in a.heldout.split(",") if s]

    budget = dict(multi_view_num=a.multi_view_num, multi_view_max_angle=a.multi_view_max_angle,
                  multi_view_max_dis=a.multi_view_max_dis, multi_view_min_dis=a.multi_view_min_dis,
                  iterations=a.iterations, resolution_divisor=a.resolution, ncc_scale=a.ncc_scale,
                  depth_weight=a.depth_weight, sh_unc_lower_max=a.sh_unc_lower_max,
                  single_view_weight_from_iter=a.warmup_from_iter, multi_view_weight_from_iter=a.warmup_from_iter,
                  unc_from_iter=a.warmup_from_iter,
                  source="the reduced budget validated in envs/ambisur.report.md (DTU scan24 single_batch): "
                         "1500 steps instead of the official 30000, warmups 700 instead of 7000, -r 4 instead of -r 2. "
                         "This exercises every loss but is NOT converged.")
    cfg = dict(lane=LANE, method=METHOD, manifest=str(a.manifest), frame_index=a.frame_index,
               heldout=sorted(heldout), budget=budget, ransac_thresh=a.ransac_thresh,
               masks=a.masks, init_points_from=a.init_points_from,
               max_points_for_colmap=a.max_points_for_colmap, da3_chunk=450, shared_camera=False,
               consistency_mm=a.consistency_mm, min_views=a.min_views,
               repo_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip())
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    if (out / "surface.npz").exists() and (out / "metadata.json").exists() and not a.force:
        if json.load(open(out / "metadata.json")).get("config_hash") == config_hash:
            print(f"[S0] {out} already done with config_hash {config_hash}; nothing to do (--force to redo)")
            return 0

    tl = Timeline(a.gpu, ROOT / "logs/lanes" / f"S0_{a.frame_index:05d}")
    man = load_manifest(a.manifest); serials, heldout = select_serials(man, heldout)
    print(f"[S0] {len(serials)} cameras, held out {heldout or 'none'}, gpu {a.gpu}, config_hash {config_hash}")

    # ---- 1. shared dataset ---------------------------------------------------
    fg = ROOT / "data/hrdex_apple_take0/fg_masks" if a.masks == "fg" else None
    with tl.stage("dataset_build"):
        ds = build_dataset(a.manifest, a.frame_index, native / "dataset", heldout, force=a.force,
                           fg_mask_dir=fg if (fg and fg.is_dir()) else None)
    print(f"[S0] dataset: {ds['n_cams']} cams, masks from {ds['mask_source']}")

    # ---- 2. AmbiSuR scene ----------------------------------------------------
    scene = native / "scene"
    if scene.exists() and a.force:
        shutil.rmtree(scene)
    with tl.stage("scene_build"):
        rec, init_points = build_scene(ds, man, serials, scene, a.init_points_from)
    off = np.array([rec[s]["pp_offset_removed_px"] for s in serials])
    print(f"[S0] scene: principal-point offsets removed, |dx| max {np.abs(off[:,0]).max():.1f} px, "
          f"|dy| max {np.abs(off[:,1]).max():.1f} px; sizes "
          f"{min(rec[s]['W'] for s in serials)}..{max(rec[s]['W'] for s in serials)} x "
          f"{min(rec[s]['H'] for s in serials)}..{max(rec[s]['H'] for s in serials)}")

    env = repo_env(a.gpu)
    # ---- 3. DA3 prior --------------------------------------------------------
    tl.run("da3_prior", [PY, DA3_SCRIPT, "--scene_dir", scene, "--post_fix", "_da3", "--save_depth",
                         "--max_points_for_colmap", a.max_points_for_colmap], env=env, cwd=REPO,
           script_sha256=sha256(DA3_SCRIPT), chunk=450, shared_camera=False)
    n_depth = len(list((scene / "estimated_depths").glob("*.npy")))
    assert n_depth == len(serials), (n_depth, len(serials))

    # ---- 4. alignment --------------------------------------------------------
    tl.run("pose_align", [PY, REPO / "multi_view_priors/pose_align.py", "--scene1", scene / "sparse_da3/0",
                          "--scene2", scene / "sparse", "--out", scene / "sparse_da3_aligned/0",
                          "--ransac_thresh", a.ransac_thresh], env=env, cwd=REPO)
    align = camera_centre_residuals(scene / "sparse_da3_aligned/0", man, serials)
    align["scale"] = json.load(open(scene / "sparse_da3_aligned/0/trans.json"))["scale"]
    print(f"[S0] DA3 alignment: median residual {align['median_m']*1000:.1f} mm, max {align['max_m']*1000:.1f} mm, "
          f"{align['n_below_10mm']}/{align['n']} below 10 mm, scale {align['scale']:.4f}")

    # ---- 5. training ---------------------------------------------------------
    model = native / "model"
    tl.run("train", [PY, REPO / "train.py", "-s", scene, "-m", model, "-r", a.resolution,
                     "--ncc_scale", a.ncc_scale, "--depth_weight", a.depth_weight,
                     "--sh_unc_lower_max", a.sh_unc_lower_max, "--iterations", a.iterations,
                     "--single_view_weight_from_iter", a.warmup_from_iter,
                     "--multi_view_weight_from_iter", a.warmup_from_iter,
                     "--unc_from_iter", a.warmup_from_iter,
                     "--multi_view_num", a.multi_view_num,
                     "--multi_view_max_angle", a.multi_view_max_angle,
                     "--multi_view_max_dis", a.multi_view_max_dis,
                     "--multi_view_min_dis", a.multi_view_min_dis,
                     "--test_iterations", a.iterations, "--save_iterations", a.iterations],
           env=env, cwd=REPO, budget=budget)
    mv = multi_view_stats(man, serials, a.multi_view_max_angle, a.multi_view_min_dis, a.multi_view_max_dis)
    print(f"[S0] AmbiSuR multi-view neighbours (max_angle {a.multi_view_max_angle} deg, dis "
          f"{a.multi_view_min_dis}..{a.multi_view_max_dis} m): per camera min {mv['min']} median {mv['median']} "
          f"max {mv['max']}; {mv['n_zero']} camera(s) get none")

    # ---- 6. surface export ---------------------------------------------------
    ex = dict(consistency_mm=a.consistency_mm, min_views=a.min_views)
    tl.run("export_surface", [PY, ROOT / "lanes_src/S0/export_surface.py", "-s", scene, "-m", model,
                              "--iteration", a.iterations, "--out", out,
                              "--consistency-mm", a.consistency_mm, "--min-views", a.min_views,
                              "--manifest", a.manifest, "--dataset-json", native / "dataset/dataset.json"],
           env=env, cwd=REPO, **ex)
    surf = json.load(open(out / "surface_export.json"))

    meta = dict(timestamp=float(man.get("timestamp_index", a.frame_index)), frame_index=a.frame_index, lane=LANE,
                pose_version=man.get("extrinsics_direction", {}).get("verdict", "unknown"), method=METHOD,
                config_hash=config_hash, source_manifest=str(a.manifest), config=cfg, budget=budget,
                heldout=heldout, cameras_used=serials, n_cams=len(serials),
                world_frame="manifest T_world_from_camera frame, metres",
                native=dict(scene=str(scene), model=str(model), dataset=str(native / "dataset")),
                mask_source=ds["mask_source"],
                mask_note="AmbiSuR reads the per-view mask ONLY from the image alpha channel "
                          "(scene/cameras.py::process_image); it is used by the TSDF mesh path, NOT by the "
                          "training loss, so the background is still supervised.",
                scene=dict(recentre=rec, init_points=init_points,
                           principal_point_note="AmbiSuR forces Cx=0.5*W, Cy=0.5*H and renders a symmetric "
                                                "frustum; DA3 predicts centred intrinsics. Views were resampled "
                                                "onto a centred virtual pinhole (same fx/fy as the COLMAP "
                                                "undistortion) so this assumption holds exactly.",
                           undistorted_source=dict(
                               note="image_undistorter's own K/size, NOT the manifest K_undistort",
                               K={s: ds["cameras"][s]["undist"]["K"] for s in serials},
                               size={s: [ds["cameras"][s]["undist"]["width"], ds["cameras"][s]["undist"]["height"]]
                                     for s in serials})),
                da3=dict(script=str(DA3_SCRIPT), script_sha256=sha256(DA3_SCRIPT), chunk=450, shared_camera=False,
                         n_depth_maps=n_depth,
                         checkpoint=str(ROOT / "cache/ckpt/ambisur/DA3NESTED-GIANT-LARGE/model.safetensors")),
                alignment=align, multi_view_neighbours=mv, surface=surf)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    tl.write(out / "timing.json", n_cams=len(serials), frame_index=a.frame_index, lane=LANE,
             resolution=[int(np.median([rec[s]["W"] for s in serials]) // a.resolution),
                         int(np.median([rec[s]["H"] for s in serials]) // a.resolution)],
             config_hash=config_hash)
    print(f"[S0] DONE {surf['n_points']:,} points -> {out}/surface.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
