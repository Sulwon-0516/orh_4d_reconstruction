"""Headless debug renders. No viser, no GPU, no display -- numpy + OpenCV, and ffmpeg for video.

Adapted from bench/make_lane_compare.py and bench/make_sweep_video.py in the source project,
reduced to what a cluster user needs: "did this run produce something sane?"

HOUSE CONVENTION, preserved: render order is always SHADING -> NORMAL -> RGB.

Two levels:
  static_check_render()  ALWAYS produced by `orhsurf run`.  One PNG of the first finished frame,
                         from TWO viewpoints with a real baseline between them.  Two views because
                         a single view hides error along its own optical axis completely -- the
                         cheapest way to be fooled by a broken reconstruction is to look at it from
                         one side.
  debug_renders()        The three motions, each individually flag-gated and off by default:
                         orbit (static frame, moving camera), time (all frames, static camera),
                         both (all frames, moving camera).

A splat renderer would be prettier; this is a z-buffered point rasteriser, which is the honest
thing to show, because it draws exactly the points that are in the file and nothing else.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
MODES = ("shaded", "normal", "rgb")          # house order: shading -> normal -> rgb


def _load(frame_dir: Path, max_points: int = 6_000_000, sup_min: int = 0, seed: int = 0):
    """xyz, normal, rgb for one frame, subsampled deterministically."""
    with np.load(Path(frame_dir) / "surface.npz") as z:
        xyz, nrm, rgb = z["xyz"], z["normal"], z["rgb"]
        sup = z["support"]
    if sup_min > 0:
        k = sup >= sup_min
        xyz, nrm, rgb = xyz[k], nrm[k], rgb[k]
    if len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        idx.sort()
        xyz, nrm, rgb = xyz[idx], nrm[idx], rgb[idx]
    return xyz, nrm, rgb


def _look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """World->camera 4x4, OpenCV convention (+X right, +Y down, +Z forward)."""
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(float(f @ (up / np.linalg.norm(up)))) > 0.999:      # degenerate: pick another up
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


def _render(xyz, nrm, rgb, T, K, W, H, mode="shaded", splat=1):
    """Z-buffered point rasterisation. Returns HxWx3 uint8 BGR."""
    Xc = xyz @ T[:3, :3].T + T[:3, 3]
    z = Xc[:, 2]
    ok = z > 1e-6
    u = np.full(len(xyz), -1.0)
    v = np.full(len(xyz), -1.0)
    u[ok] = Xc[ok, 0] / z[ok] * K[0, 0] + K[0, 2]
    v[ok] = Xc[ok, 1] / z[ok] * K[1, 1] + K[1, 2]
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not ok.any():
        return np.zeros((H, W, 3), np.uint8)
    ui, vi, zz = u[ok].astype(np.int32), v[ok].astype(np.int32), z[ok]
    order = np.argsort(-zz)                      # far first, so near overwrites
    ui, vi, zz, sel = ui[order], vi[order], zz[order], np.nonzero(ok)[0][order]

    img = np.zeros((H, W, 3), np.float32)
    if mode == "rgb":
        col = rgb[sel].astype(np.float32) / 255.0
    elif mode == "normal":
        n = nrm[sel]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        col = (n * 0.5 + 0.5).astype(np.float32)
    else:                                         # shaded: Lambertian from the camera direction
        n = nrm[sel]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        nc = n @ T[:3, :3].T
        lam = np.abs(nc[:, 2]).astype(np.float32)
        col = np.repeat((0.15 + 0.85 * lam)[:, None], 3, axis=1)
    for dy in range(splat):
        for dx in range(splat):
            yy = np.clip(vi + dy, 0, H - 1)
            xx = np.clip(ui + dx, 0, W - 1)
            img[yy, xx] = col
    return (np.clip(img, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)


def _frame_cameras(xyz, W, H, fov_deg=70.0):
    """A K and a scene centre/radius that frame the cloud."""
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    c = (lo + hi) / 2
    r = float(np.linalg.norm(hi - lo)) / 2 + 1e-6
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return K, c, r


def static_check_render(out_root: Path, frame: int, dest: Path, *, gpu=None,
                        width: int = 720, height: int = 540) -> Path:
    """The always-on sanity artifact: one frame, TWO viewpoints, all three modes.

    Layout: 2 rows (viewpoints) x 3 columns (shaded, normal, rgb).  Two viewpoints separated by 50
    degrees of azimuth, because a single view cannot show error along its own optical axis -- the
    classic way a broken reconstruction looks fine.
    """
    import cv2

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fd = Path(out_root) / f"{frame:05d}"
    xyz, nrm, rgb = _load(fd, max_points=3_000_000)
    K, c, r = _frame_cameras(xyz, width, height)

    rows = []
    for az in (35.0, 85.0):                     # 50 degrees of baseline between the two views
        a = np.radians(az)
        eye = c + np.array([np.cos(a), np.sin(a), 0.35]) * (r * 2.2)
        T = _look_at(eye, c)
        rows.append(np.hstack([_render(xyz, nrm, rgb, T, K, width, height, m, splat=2)
                               for m in MODES]))
    sheet = np.vstack(rows)
    for i, m in enumerate(MODES):
        cv2.putText(sheet, m, (12 + i * width, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sheet, f"frame {frame:05d}  {len(xyz):,} pts (subsampled)", (12, height * 2 - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    p = dest / f"check_{frame:05d}.png"
    cv2.imwrite(str(p), sheet)
    return p


def _encode(pattern: str, out: Path, fps: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", str(out)], check=True)
    return out


def _video(out_root: Path, frames: list[int], dest: Path, name: str, *, motion: str,
           width=900, height=680, views=180, fps=30, sup_min=0, log=print) -> list[Path]:
    import cv2
    import tempfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ref, nrm0, rgb0 = _load(Path(out_root) / f"{frames[0]:05d}", max_points=2_000_000)
    K, c, r = _frame_cameras(ref, width, height)

    if motion == "orbit":
        steps = [(frames[0], k / views) for k in range(views)]
    elif motion == "time":
        steps = [(f, 0.12) for f in frames]
    else:
        steps = [(f, k / max(1, len(frames))) for k, f in enumerate(frames)]

    made = []
    for mode in MODES:                                   # shading -> normal -> rgb
        with tempfile.TemporaryDirectory(dir=str(dest)) as td:
            cache_f, cache = None, None
            for i, (f, t) in enumerate(steps):
                if f != cache_f:
                    cache = _load(Path(out_root) / f"{f:05d}", max_points=2_000_000,
                                  sup_min=sup_min)
                    cache_f = f
                a = 2 * np.pi * t
                eye = c + np.array([np.cos(a), np.sin(a), 0.35]) * (r * 2.2)
                img = _render(*cache, _look_at(eye, c), K, width, height, mode, splat=2)
                cv2.imwrite(f"{td}/{i:05d}.png", img)
            out = dest / f"{name}_{motion}_{mode}.mp4"
            _encode(f"{td}/%05d.png", out, fps)
            made.append(out)
            log(f"[render] {out}")
    return made


def debug_renders(out_root: Path, frames: list[int], dest: Path, *, orbit=False, time_=False,
                  both=False, log=print) -> list[Path]:
    """The three motions, each individually gated. Off by default; the caller passes the flags."""
    if not frames:
        log("[render] no finished frames; nothing to render")
        return []
    made = []
    if orbit:
        made += _video(out_root, frames, dest, "debug", motion="orbit", log=log)
    if time_ and len(frames) > 1:
        made += _video(out_root, frames, dest, "debug", motion="time", fps=15, log=log)
    elif time_:
        log("[render] --render-time needs more than one frame; skipped")
    if both and len(frames) > 1:
        made += _video(out_root, frames, dest, "debug", motion="both", log=log)
    elif both:
        log("[render] --render-both needs more than one frame; skipped")
    return made
