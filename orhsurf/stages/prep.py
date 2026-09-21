"""Undistort each view onto its manifest K_undistort pinhole, and its alpha mask with it.

Ported from lanes_src/R/run.py:78-105 (`prep`) and :59-68 (`cam_arrays`).

WHY THIS IS SEPARATE FROM THE COLMAP UNDISTORT.  There are two different undistorted pinholes in
this pipeline and mixing them silently produces wrong depths:

  * THIS one -- cv2.undistort(K_original, dist, None, K_undistort) at the full 2048x1536, principal
    point left wherever calibration put it.  This is what DA3 is fed, because the DA3 stage asserts
    that DA3's returned K_proc equals the manifest K_undistort scaled by the resize factor.
  * The AmbiSuR SCENE pinhole -- COLMAP image_undistorter's fx/fy with the principal point forced
    to the exact centre and the frame cropped to a multiple of 4.  Sizes differ per view.

They share the same camera POSE, which is exactly what makes the rewarp in da3_prior.rewarp() a
pure, depth-independent homography.  Feeding the scene images to DA3 instead would also trip DA3's
centre-crop-to-smallest behaviour, because the scene images differ in size across views.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def cam_arrays(cam: dict):
    """(K_original, dist5, K_undistort, T_cam_from_world), with the conventions asserted."""
    K0 = np.array(cam["K_original"], float)
    Ku = np.array(cam["K_undistort"], float)
    d = np.array(cam["dist_params"], float)
    assert d.shape == (5,) and d[4] == 0.0, (
        f"expected OpenCV radtan5 with k3 == 0 (the capture software pads it), got {d.tolist()}")
    T = np.array(cam["T_cam_from_world"], float)
    assert np.allclose(T[3], [0, 0, 0, 1]), "T_cam_from_world is not a homogeneous 4x4"
    R = T[:3, :3]
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9, "T_cam_from_world rotation is not orthonormal"
    return K0, d, Ku, T


def frame_of(cam: dict, i: int) -> dict:
    frames = cam["frames"]
    f = frames[str(i)] if isinstance(frames, dict) else frames[i]
    assert f["index"] == i, f"frame index mismatch: entry says {f['index']}, asked for {i}"
    return f


def prep_views(manifest: dict, serials: list[str], frame_index: int, out_dir: Path,
               *, workers: int = 8, log=print) -> dict:
    """Write <out_dir>/undist/<serial>.png and <out_dir>/mask/<serial>.png. Returns per-view info.

    Existing files are reused, so re-running a partially-finished frame is cheap.  The mask is the
    ALPHA channel of the manifest's mask PNG, undistorted with the same map and re-thresholded at
    128 (undistortion interpolates, so a binary mask comes back soft).
    """
    import cv2

    ud, md = Path(out_dir) / "undist", Path(out_dir) / "mask"
    ud.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)

    def _readable(p: Path) -> bool:
        """A file EXISTING does not mean it is usable.

        The reuse guard below used to test `exists()` only, so a zero-byte or truncated PNG left by
        an interrupted run (or by a full filesystem) was silently trusted, and the failure surfaced
        later as `TypeError: '>=' not supported between NoneType and int` -- which names neither
        the file nor the cause. Same class of bug as trusting surface.npz's existence for resume.
        """
        try:
            return p.is_file() and p.stat().st_size > 0 and \
                cv2.imread(str(p), cv2.IMREAD_UNCHANGED) is not None
        except Exception:
            return False

    def one(s: str):
        cam = manifest["cameras"][s]
        K0, d, Ku, _ = cam_arrays(cam)
        f = frame_of(cam, frame_index)
        oi, om = ud / f"{s}.png", md / f"{s}.png"
        if not (_readable(oi) and _readable(om)):
            img = cv2.imread(f["frame_path"], cv2.IMREAD_COLOR)
            assert img is not None, f"cannot read frame {f['frame_path']}"
            assert img.shape[:2] == (cam["height"], cam["width"]), (
                f"{s}: frame is {img.shape[1]}x{img.shape[0]}, manifest says "
                f"{cam['width']}x{cam['height']}")
            assert f.get("mask_path"), (
                f"{s} frame {frame_index} has no mask_path; this pipeline needs per-view "
                f"foreground masks (see docs/DATA_CONTRACT.md)")
            raw = cv2.imread(f["mask_path"], cv2.IMREAD_UNCHANGED)
            assert raw is not None and raw.ndim == 3 and raw.shape[2] == 4, (
                f"{f['mask_path']}: expected a 4-channel RGBA PNG whose ALPHA is the mask")
            a = raw[..., 3]
            # cv2.imwrite returns False rather than raising -- most often a full filesystem.
            if not cv2.imwrite(str(oi), cv2.undistort(img, K0, d, None, Ku)):
                raise RuntimeError(f"{s}: could not write {oi} (is the filesystem full?)")
            if not cv2.imwrite(str(om),
                               (cv2.undistort(a, K0, d, None, Ku) >= 128).astype(np.uint8) * 255):
                raise RuntimeError(f"{s}: could not write {om} (is the filesystem full?)")
        raw_m = cv2.imread(str(om), cv2.IMREAD_GRAYSCALE)
        if raw_m is None:
            raise RuntimeError(
                f"{s}: wrote {om} but cannot read it back. The file is "
                f"{om.stat().st_size if om.is_file() else 'missing'} bytes. This usually means the "
                f"filesystem is full or the write was interrupted; check `df -h` on that path.")
        m = raw_m >= 128
        return s, dict(undist=oi, mask=om, mask_px=int(m.sum()))

    info = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s, r in ex.map(one, serials):
            info[s] = r
    # An all-zero mask is normal, not an error: with 47 cameras ringing a room the subject is
    # genuinely outside some of them (measured on the ORH clip, frame 40: 4/47 empty).  What must
    # hold is that ENOUGH views see the subject for the visual hull, whose own gate is
    # min_vis = 8 cameras.  Assert that, not per-view non-emptiness.
    empty = sorted(s for s, r in info.items() if r["mask_px"] == 0)
    n_seen = len(serials) - len(empty)
    assert n_seen >= 8, (
        f"only {n_seen} of {len(serials)} views have a non-empty foreground mask; the visual hull "
        f"needs at least 8. Empty: {empty}")
    log(f"[prep] undistorted {len(serials)} views + masks -> {out_dir}"
        + (f" ({len(empty)} view(s) see no subject: {', '.join(empty)})" if empty else ""))
    return info
