#!/usr/bin/env python
"""Build the COLMAP known-pose workspace that BOTH lane C (COLMAP dense) and lane S0 (AmbiSuR) consume.

From a common manifest (schema of $ROOT/data/hrdex_apple_take0/manifest.json) + a frame index
(+ an optional held-out serial list, which is excluded entirely) this writes

  <ws>/images/<serial>.png              copy of the ORIGINAL DISTORTED raw image (sha256-verified; a copy and
                                        not a symlink -- see the note in build_dataset)
  <ws>/masks_distorted/<serial>.png     uint8 0/255 foreground, read from the ALPHA channel of the manifest mask
  <ws>/sparse_known/{cameras,images,points3D}.txt
                                        OPENCV camera model (fx fy cx cy k1 k2 p1 p2), ONE CAMERA PER IMAGE,
                                        images.txt holds T_cam_from_world directly (q_wxyz, t), points3D.txt empty
  <ws>/database.db                      COLMAP DB with the cameras/images rows pre-seeded with the calibrated
                                        OPENCV intrinsics (so geometric verification uses them)
  <ws>/undist/{images,sparse}           `colmap image_undistorter` output: PINHOLE images + sparse model
  <ws>/undist/masks/<serial>.png        the same masks warped into the undistorted frame with
                                        cv2.initUndistortRectifyMap(K_original, dist, None, K_undist_colmap, size)
  <ws>/dataset.json                     everything measured, incl. the undistorted W/H and K per camera and how
                                        far they are from the manifest's own K_undistort

WHY the undistorted K is recorded: `image_undistorter` picks its own output size and its own K (it enlarges the
image so no valid pixel is lost).  $ROOT/envs/colmap.report.md §6 warns explicitly that this does NOT match the
manifest's `K_undistort`.  Anything downstream (S0's AmbiSuR dataset, any backprojection) must use the K in
dataset.json["cameras"][serial]["undist"], never the manifest K_undistort.

Reuses, in adapted form (generalised from 20 hardcoded cams to the manifest's camera set):
  $ROOT/smoke/colmap/make_known_pose_model.py   (quaternion writer, OPENCV camera lines, known-pose text model)
  $ROOT/smoke/colmap/db_set_intrinsics.py       (sqlite3 seeding of the DB cameras; pycolmap's Database binding
                                                 raises 'pure virtual function' in 3.13, see the report §5)

CLI:
  $ROOT/envs/colmap/bin/python lanes_src/common/colmap_dataset.py \
      --manifest $ROOT/data/hrdex_apple_take0/manifest.json --frame-index 0 --out <ws> [--heldout a,b] [--gpu 3]
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sqlite3, subprocess, sys
import os
from pathlib import Path

import numpy as np
import cv2

# orhsurf: ROOT removed. COLMAP is located by orhsurf.paths (env ORHSURF_COLMAP_BIN, then
# the package env, then $PATH) so nothing here is tied to the machine it was written on.
from orhsurf.paths import colmap_bin as _colmap_bin, read_manifest
COLMAP_BIN = _colmap_bin()
CAMERA_MODEL_OPENCV = 4           # COLMAP camera model id for OPENCV
CAMERA_MODEL_PINHOLE = 1


# ---------------------------------------------------------------- manifest ---
def load_manifest(path):
    man = read_manifest(path)
    for k in ("cameras", "calibrated_serials", "conventions"):
        assert k in man, f"{path}: manifest has no '{k}'"
    c = man["conventions"]
    assert c["pose"].startswith("T_world_from_camera"), c["pose"]
    assert c["units"] == "metres", c["units"]
    assert c["distortion_order"].startswith("OpenCV"), c["distortion_order"]
    return man


def select_serials(man, heldout):
    heldout = set(heldout or [])
    allser = [s_ for s_ in man["calibrated_serials"]
              if man["cameras"][s_].get("valid", True) and man["cameras"][s_].get("n_frames_in_window", 1) > 0]
    dropped = sorted(set(man["calibrated_serials"]) - set(allser))
    if dropped: print(f"[dataset] dropping invalid cameras (valid=false / no frames): {dropped}")
    unknown = heldout - set(allser)
    assert not unknown, f"--heldout names serials that are not in the manifest: {sorted(unknown)}"
    keep = [s for s in allser if s not in heldout]
    assert len(keep) >= 3, f"only {len(keep)} cameras left after holding out {sorted(heldout)}"
    return keep, sorted(heldout)


def frame_mask_path(cam, frame_index):
    """Per-frame mask, when the manifest carries one.

    The HRdex fixtures had ONE static mask per camera in cam['mask_path'], so that is what the
    dataset loop originally read.  An ORH foreground manifest (bench/make_fg_manifest.py) instead
    puts a mask on every frame entry, because the foreground moves.  Prefer the frame's own mask
    and fall back to the camera-level one, so both manifest shapes work.
    """
    fr = cam.get("frames")
    if fr:
        ent = fr[str(frame_index)] if isinstance(fr, dict) else fr[frame_index]
        if isinstance(ent, dict) and ent.get("mask_path"):
            return ent["mask_path"]
    cm = cam.get("mask_path")
    # a directory here means the manifest points at a per-frame mask FOLDER; that is not a mask
    return cm if (cm and not os.path.isdir(cm)) else None


def frame_image_path(cam, frame_index, man):
    """Resolve the image for `frame_index`.  The fixture manifest is a SINGLE timestamp: it carries one
    'image_path' per camera and manifest['timestamp_index'] is the only valid frame.  Multi-frame ORH
    manifests will carry 'frames' (list or dict keyed by frame index) -- handled, but assert loudly."""
    if "frames" in cam:
        fr = cam["frames"]
        ent = fr[str(frame_index)] if isinstance(fr, dict) else fr[frame_index]
        if isinstance(ent, dict):
            key = "image_path" if "image_path" in ent else "frame_path"   # HRdex fixture vs ORH manifest
            assert key in ent, f"frame entry has neither image_path nor frame_path: {list(ent)}"
            assert ent.get("index", frame_index) == frame_index, (ent.get("index"), frame_index)
            return ent[key]
        return ent
    assert "image_path" in cam, f"camera {cam.get('camera_id')} has neither 'frames' nor 'image_path'"
    ti = man.get("timestamp_index", 0)
    assert frame_index == ti, (f"manifest {man.get('sequence_id')} is a single-timestamp take "
                               f"(timestamp_index={ti}); --frame-index {frame_index} does not exist")
    return cam["image_path"]


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------ known poses ---
def rot_to_quat_wxyz(R):
    """Shepperd's method -> (w,x,y,z).  Copied from smoke/colmap/make_known_pose_model.py."""
    m = R; tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cam_from_world(cam):
    """manifest T_world_from_camera (4x4, X_world = T X_cam)  ->  (R, t) with X_cam = R X_world + t."""
    T = np.array(cam["T_world_from_camera"], dtype=np.float64)
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1]), T
    R_wc, t_wc = T[:3, :3], T[:3, 3]
    assert np.allclose(R_wc @ R_wc.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R_wc) - 1) < 1e-6
    return R_wc.T, -R_wc.T @ t_wc


def intrinsics(cam):
    K = np.array(cam["K_original"], dtype=np.float64)
    assert K.shape == (3, 3) and K[0, 1] == 0 and K[2, 2] == 1 and K[1, 0] == 0, K
    d = np.array(cam["dist_params"], dtype=np.float64)
    if d.shape == (5,):
        assert d[4] == 0.0, f"k3 must be exactly 0 to use COLMAP OPENCV (4-param) model, got {d[4]}"
        d = d[:4]
    assert d.shape == (4,), f"expected 4 OpenCV dist params (k1 k2 p1 p2), got {d.shape}"
    return K, d


def write_known_pose_model(out_dir, serials, cams):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cameras.txt", "w") as fc, open(out_dir / "images.txt", "w") as fi, \
         open(out_dir / "points3D.txt", "w") as fp:
        fc.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fi.write("# Image list with two lines of data per image:\n"
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fp.write("# 3D point list (empty: known poses only, points come from point_triangulator)\n")
        for i, s in enumerate(serials, start=1):
            cam = cams[s]; K, d = intrinsics(cam); W, H = int(cam["width"]), int(cam["height"])
            # NOTE: repr(np.float64) is 'np.float64(x)' under numpy 2.x and COLMAP's text reader aborts on it
            # (std::__throw_invalid_argument in ReadCamerasText) -- always cast to a python float first.
            p = [float(v) for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3])]
            fc.write(f"{i} OPENCV {W} {H} " + " ".join(repr(v) for v in p) + "\n")
            R, t = cam_from_world(cam); q = rot_to_quat_wxyz(R)
            assert np.allclose(quat_to_rot(q), R, atol=1e-8), s
            qt = [float(v) for v in (*q, *t)]
            fi.write(f"{i} " + " ".join(repr(v) for v in qt) + f" {i} {s}.png\n\n")
    return out_dir


# -------------------------------------------------------------- database ---
def create_seeded_db(db_path, serials, cams, colmap_bin=COLMAP_BIN):
    """`colmap database_creator` + insert one OPENCV camera and one image row per serial, with the
    CALIBRATED params already in place.  camera_id == image_id == index in `serials` (1-based), which is
    exactly the numbering used by write_known_pose_model, so the text model and the DB agree."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    subprocess.run([str(colmap_bin), "database_creator", "--database_path", str(db_path)],
                   check=True, capture_output=True)
    c = sqlite3.connect(db_path)
    for i, s in enumerate(serials, start=1):
        cam = cams[s]; K, d = intrinsics(cam)
        params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3]], dtype=np.float64)
        c.execute("insert into cameras(camera_id, model, width, height, params, prior_focal_length) values (?,?,?,?,?,1)",
                  (i, CAMERA_MODEL_OPENCV, int(cam["width"]), int(cam["height"]), params.tobytes()))
        c.execute("insert into images(image_id, name, camera_id) values (?,?,?)", (i, f"{s}.png", i))
    c.commit()
    n = c.execute("select count(*) from cameras").fetchone()[0]
    assert n == len(serials), n
    c.close()
    return db_path


def set_db_intrinsics(db_path, cams, expect_model=CAMERA_MODEL_OPENCV):
    """Re-write (or verify) the calibrated OPENCV params of every camera in an existing DB, matched by image
    name.  Adapted from smoke/colmap/db_set_intrinsics.py.  Returns the number of rows touched."""
    c = sqlite3.connect(db_path)
    rows = c.execute("select image_id, name, camera_id from images").fetchall()
    assert rows, f"{db_path} has no images"
    for image_id, name, cam_id in rows:
        s = name[:-4]
        assert s in cams, f"{db_path}: image {name} is not a manifest camera"
        model, w, h = c.execute("select model, width, height from cameras where camera_id=?", (cam_id,)).fetchone()
        assert (model, w, h) == (expect_model, int(cams[s]["width"]), int(cams[s]["height"])), (name, model, w, h)
        K, d = intrinsics(cams[s])
        params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3]], dtype=np.float64)
        c.execute("update cameras set params=?, prior_focal_length=1 where camera_id=?", (params.tobytes(), cam_id))
    c.commit(); c.close()
    return len(rows)


# ------------------------------------------------------- undistorted model ---
def read_model_txt(model_dir):
    """Minimal COLMAP TXT model reader -> ({camera_id: cam}, {image_name: img}).  No pycolmap dependency."""
    model_dir = Path(model_dir)
    cameras = {}
    for line in open(model_dir / "cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        cameras[int(f[0])] = dict(model=f[1], width=int(f[2]), height=int(f[3]),
                                  params=[float(x) for x in f[4:]])
    images = {}
    lines = [l for l in open(model_dir / "images.txt") if not l.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1; continue
        f = lines[i].split()
        q = np.array([float(x) for x in f[1:5]]); t = np.array([float(x) for x in f[5:8]])
        images[f[9]] = dict(image_id=int(f[0]), qvec=q, tvec=t, camera_id=int(f[8]), name=f[9])
        i += 2
    return cameras, images


def model_to_txt(colmap_bin, in_dir, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(colmap_bin), "model_converter", "--input_path", str(in_dir),
                    "--output_path", str(out_dir), "--output_type", "TXT"], check=True, capture_output=True)
    return out_dir


# ------------------------------------------------------------------ masks ---
def read_alpha_mask(path):
    """HRdex mask PNGs are RGBA with the selection in the ALPHA channel; the RGB channels are all 255, so a
    default cv2.imread() returns an all-white image (CONVENTIONS addendum).  Read UNCHANGED, take alpha."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert m is not None, f"cannot read mask {path}"
    if m.ndim == 3 and m.shape[2] == 4:
        a = m[..., 3]
        src = "alpha"
    elif m.ndim == 2:
        a = m; src = "gray"
    else:
        a = m[..., 0]; src = "rgb[0]"
    return (a > 0).astype(np.uint8) * 255, src


# ------------------------------------------------------------------ build ---

_NUM_THREADS_SUPPORT = {}


def _colmap_supports_num_threads(colmap_bin) -> bool:
    """Does THIS colmap's image_undistorter accept --num_threads? Probed once, then cached."""
    key = str(colmap_bin)
    if key not in _NUM_THREADS_SUPPORT:
        try:
            h = subprocess.run([key, "image_undistorter", "-h"], capture_output=True, text=True,
                               timeout=60)
            _NUM_THREADS_SUPPORT[key] = "--num_threads" in (h.stdout + h.stderr)
        except Exception:
            _NUM_THREADS_SUPPORT[key] = False
    return _NUM_THREADS_SUPPORT[key]


def _affinity_limiter(n):
    """preexec_fn restricting the child to `n` CPUs taken from our own affinity mask.

    Returns None where that is impossible or inadvisable:
      * non-Linux (no sched_setaffinity),
      * inside a Slurm allocation, where the cpuset cgroup is already the real bound and narrowing
        further duplicates -- or fights -- the scheduler's own binding.
    """
    if not hasattr(os, "sched_setaffinity") or os.environ.get("SLURM_JOB_ID"):
        return None
    try:
        mask = sorted(os.sched_getaffinity(0))
    except OSError:
        return None
    if n >= len(mask):
        return None
    keep = set(mask[:n])

    def _set():
        try:
            os.sched_setaffinity(0, keep)
        except OSError:
            pass
    return _set


def build_dataset(manifest_path, frame_index, ws, heldout=(), colmap_bin=COLMAP_BIN, timeline=None,
                  force=False, fg_mask_dir=None):
    ws = Path(ws); ws.mkdir(parents=True, exist_ok=True)
    man = load_manifest(manifest_path)
    serials, heldout = select_serials(man, heldout)
    cams = man["cameras"]

    # ---- 1. images (original distorted) + masks -----------------------------
    img_dir = ws / "images"; img_dir.mkdir(exist_ok=True)
    mdist_dir = ws / "masks_distorted"; mdist_dir.mkdir(exist_ok=True)
    per_cam = {}
    mask_source = None
    for s in serials:
        cam = cams[s]
        src = Path(frame_image_path(cam, frame_index, man))
        assert src.exists(), f"missing image {src}"
        got = sha256(src)
        if cam.get("image_sha256"):
            assert got == cam["image_sha256"], f"{src}: sha256 {got} != manifest {cam['image_sha256']}"
        dst = img_dir / f"{s}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # COPY, do not symlink: colmap's feature_extractor resolves symlinks and stores the RESOLVED relative
        # path as the DB image name, which no longer matches the <serial>.png names in the known-pose model
        # (it then inserts a second set of image+camera rows and ignores the seeded intrinsics).
        shutil.copyfile(src, dst)
        # mask: prefer an explicit foreground dir if one exists, else the manifest mask's ALPHA channel
        mpath, msrc = None, None
        if fg_mask_dir is not None and (Path(fg_mask_dir) / f"{s}.png").exists():
            mpath, msrc = Path(fg_mask_dir) / f"{s}.png", "fg_masks_dir"
        else:
            fmp = frame_mask_path(cam, frame_index)
            if fmp and Path(fmp).exists():
                mpath, msrc = Path(fmp), "manifest_mask_alpha"
        if mpath is not None:
            m, chan = read_alpha_mask(mpath)
            assert m.shape == (int(cam["height"]), int(cam["width"])), (s, m.shape)
            cv2.imwrite(str(mdist_dir / f"{s}.png"), m)
            msrc = f"{msrc}:{chan}"
            mask_source = msrc if mask_source is None else mask_source
        K, d = intrinsics(cam)
        per_cam[s] = dict(serial=s, image=str(src), image_sha256=got,
                          width=int(cam["width"]), height=int(cam["height"]),
                          K_original=K.tolist(), dist=d.tolist(),
                          K_undistort_manifest=np.array(cam["K_undistort"]).tolist(),
                          mask=str(mpath) if mpath else None, mask_source=msrc,
                          mask_sha256=sha256(mpath) if mpath else None,
                          mask_fg_fraction=float((m > 0).mean()) if mpath else None)

    # ---- 2. known-pose text model ------------------------------------------
    model_dir = write_known_pose_model(ws / "sparse_known", serials, cams)

    # ---- 3. database with seeded intrinsics --------------------------------
    db = create_seeded_db(ws / "database.db", serials, cams, colmap_bin)

    # ---- 4. image_undistorter ----------------------------------------------
    und = ws / "undist"
    if und.exists() and force:
        shutil.rmtree(und)
    cmd = [str(colmap_bin), "image_undistorter",
           "--image_path", str(img_dir), "--input_path", str(model_dir),
           "--output_path", str(und), "--output_type", "COLMAP", "--max_image_size", "-1"]
    # orhsurf: bound COLMAP to the one CPU budget.
    #
    # COLMAP sizes its thread pool from hardware concurrency, so on a 64-core shared box it will
    # happily use all of it.  MEASURED on COLMAP 3.13.0: `image_undistorter` does NOT accept
    # `--num_threads` ("Failed to parse options - unrecognised option: --num_threads"), so the
    # flag is PROBED rather than assumed, and when it is absent we restrict the CHILD'S CPU
    # AFFINITY instead.  Affinity is the stronger guarantee anyway: however many threads COLMAP
    # starts, it can only occupy the cores we granted it.
    _nt = int(os.environ.get("ORHSURF_CPUS_PER_JOB", "8"))
    if _colmap_supports_num_threads(colmap_bin):
        cmd = cmd[:2] + ["--num_threads", str(_nt)] + cmd[2:]
        _pre = None
    else:
        _pre = _affinity_limiter(_nt)
    if timeline is not None:
        timeline.run("dataset_image_undistorter", cmd)
    else:
        subprocess.run(cmd, check=True, capture_output=True, preexec_fn=_pre)
    assert (und / "sparse").exists(), f"image_undistorter produced no sparse model in {und}"
    txt = model_to_txt(colmap_bin, und / "sparse", und / "sparse_txt")
    ucams, uimgs = read_model_txt(txt)
    assert len(uimgs) == len(serials), (len(uimgs), len(serials))

    # ---- 5. undistorted masks + a check that our warp == COLMAP's ----------
    umask_dir = und / "masks"; umask_dir.mkdir(exist_ok=True)
    warp_check = {}
    for s in serials:
        im = uimgs[f"{s}.png"]; uc = ucams[im["camera_id"]]
        assert uc["model"] == "PINHOLE", uc["model"]
        fx, fy, cx, cy = uc["params"]
        Ku = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        Wu, Hu = uc["width"], uc["height"]
        K, d = intrinsics(cams[s])
        mx, my = cv2.initUndistortRectifyMap(K, d, None, Ku, (Wu, Hu), cv2.CV_32FC1)
        # verify the warp reproduces COLMAP's own undistorted image
        raw = cv2.imread(str(per_cam[s]["image"]), cv2.IMREAD_COLOR)
        mine = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
        theirs = cv2.imread(str(und / "images" / f"{s}.png"), cv2.IMREAD_COLOR)
        assert theirs is not None and theirs.shape == mine.shape, (s, None if theirs is None else theirs.shape, mine.shape)
        valid = (mx >= 0) & (mx < per_cam[s]["width"] - 1) & (my >= 0) & (my < per_cam[s]["height"] - 1)
        diff = np.abs(mine.astype(np.float32) - theirs.astype(np.float32)).mean(2)
        warp_check[s] = dict(mean_abs_diff=float(diff[valid].mean()), p99_abs_diff=float(np.percentile(diff[valid], 99)),
                             valid_fraction=float(valid.mean()))
        if per_cam[s]["mask"]:
            m = cv2.imread(str(mdist_dir / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
            mu = cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cv2.imwrite(str(umask_dir / f"{s}.png"), mu)
            per_cam[s]["undist_mask_fg_fraction"] = float((mu > 0).mean())
        Km = np.array(cams[s]["K_undistort"], dtype=np.float64)
        per_cam[s]["undist"] = dict(
            width=Wu, height=Hu, K=Ku.tolist(),
            principal_point_offset_from_centre_px=[float(cx - Wu / 2.0), float(cy - Hu / 2.0)],
            delta_vs_manifest_K_undistort=dict(
                d_fx=float(fx - Km[0, 0]), d_fy=float(fy - Km[1, 1]),
                d_cx=float(cx - Km[0, 2]), d_cy=float(cy - Km[1, 2]),
                d_width=Wu - int(cams[s]["width"]), d_height=Hu - int(cams[s]["height"])))
        q, t = im["qvec"], im["tvec"]
        per_cam[s]["undist"]["R_cam_from_world"] = quat_to_rot(q).tolist()
        per_cam[s]["undist"]["t_cam_from_world"] = t.tolist()

    info = dict(
        manifest=str(manifest_path), sequence_id=man.get("sequence_id"), frame_index=frame_index,
        serials=serials, heldout=heldout, n_cams=len(serials),
        workspace=str(ws), images=str(img_dir), known_pose_model=str(model_dir), database=str(db),
        undistorted=dict(root=str(und), images=str(und / "images"), sparse=str(und / "sparse"),
                         sparse_txt=str(txt), masks=str(umask_dir)),
        mask_source=mask_source, fg_mask_dir=str(fg_mask_dir) if fg_mask_dir else None,
        undistort_warp_check=dict(
            method="cv2.initUndistortRectifyMap(K_original, dist, None, K_colmap_undist, (Wu,Hu)) applied to the raw "
                   "image, compared with colmap image_undistorter's own output over the valid-source region",
            mean_abs_diff_over_cams=float(np.mean([v["mean_abs_diff"] for v in warp_check.values()])),
            max_mean_abs_diff=float(np.max([v["mean_abs_diff"] for v in warp_check.values()])),
            per_cam=warp_check),
        cameras=per_cam)
    (ws / "dataset.json").write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frame-index", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default="")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    heldout = [s for s in a.heldout.split(",") if s]
    fg = Path(os.environ.get("ORHSURF_FG_MASKS", "/nonexistent"))
    info = build_dataset(a.manifest, a.frame_index, a.out, heldout, force=a.force,
                         fg_mask_dir=fg if fg.is_dir() else None)
    print(json.dumps({k: v for k, v in info.items() if k != "cameras"}, indent=1))
    s0 = info["serials"][0]
    print("example undistorted camera", s0, json.dumps(info["cameras"][s0]["undist"]["delta_vs_manifest_K_undistort"]))


if __name__ == "__main__":
    main()
