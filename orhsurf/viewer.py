"""Interactive viewer for one finished frame. Deliberately basic.

The job here is "open a frame and confirm it looks like a room", not "compare filters" -- nothing
is precomputed and nothing is cached, so it opens in seconds.

WHY `facing` EXISTS, and why it is not decoration.  `shaded` lights the cloud with |n.L| and is
therefore BLIND to the sign of the normal. That blindness is not hypothetical: it is exactly why
this project's always-on sanity render could not see that every exported normal was back-facing,
for weeks. `facing` colours green where a normal points at the nearest camera and red where it
points away, so the sign is readable at a glance.

Expect ~25% red even on a correct export: "nearest camera" is a proxy for "the camera that
produced this point", and they disagree for about a quarter of points. A MOSTLY RED cloud means
the normal-orientation bug is back.

viser is an OPTIONAL dependency -- `install.sh` does not install it, because the reconstruction
path does not need it. `pip install viser` in the AmbiSuR env.

Adapted from viz4d/npz_viewer.py in the source project.
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from threading import RLock
from pathlib import Path

import numpy as np

#: How many points to actually stream to the browser. A 25 M-point cloud will not render at full
#: density interactively; the subsample is a random permutation prefix, so it is unbiased.
BUDGETS = {"300 k": 300_000, "1 M": 1_000_000, "1.2 M": 1_200_000, "3 M": 3_000_000,
           "5 M": 5_000_000, "8 M": 8_000_000, "10 M": 10_000_000, "ALL (slow)": 10 ** 9}

_SEED = 20260921


def _require_viser():
    try:
        import viser
        return viser
    except ImportError as e:
        raise SystemExit(
            "viser is not installed.\n"
            "  It is an optional extra: the reconstruction pipeline does not need it, so\n"
            "  install.sh deliberately leaves it out.\n"
            "\n"
            f"  Install it into the env you run orhsurf with:\n"
            f"      pip install viser\n"
            f"\n  (original error: {e})")


def find_frame(out_root: Path, frame: int | None = None) -> Path:
    """Path to a frame's surface.npz. Defaults to the first COMPLETED frame."""
    from . import atomicio
    out_root = Path(out_root)
    if not out_root.is_dir():
        raise SystemExit(f"{out_root} does not exist. Has this clip been reconstructed?")
    frames = sorted(p for p in out_root.iterdir() if p.is_dir() and p.name.isdigit())
    if frame is not None:
        d = out_root / f"{frame:05d}"
        if not (d / "surface.npz").is_file():
            raise SystemExit(f"no surface.npz at {d}")
        return d / "surface.npz"
    for d in frames:
        if atomicio.is_done(d) and (d / "surface.npz").is_file():
            return d / "surface.npz"
    raise SystemExit(
        f"no completed frame under {out_root}.\n"
        f"  {len(frames)} frame director(ies) exist; none has a _DONE.json marker.\n"
        f"  Run `orhsurf verify` to see what is wrong with them.")


def camera_centres(manifest_path: Path | None) -> np.ndarray | None:
    if manifest_path is None:
        return None
    try:
        man = json.loads(Path(manifest_path).read_text())
    except Exception:
        return None
    serials = man.get("valid_serials") or sorted(man.get("cameras", {}))
    try:
        return np.array([man["cameras"][s]["camera_center_world"] for s in serials], np.float32)
    except Exception:
        return None


def load_cloud(npz_path: Path, cams: np.ndarray | None, max_points: int | None = None) -> dict:
    with np.load(npz_path, allow_pickle=False) as d:
        all_xyz = d["xyz"]
        n_total = len(all_xyz)
        if not n_total:
            raise ValueError(f"empty cloud: {npz_path}")
        # A fixed permutation prefix makes increasing the display budget nested/reproducible.
        perm = np.random.default_rng(_SEED).permutation(n_total)
        if max_points is not None:
            perm = perm[:max_points]
        xyz = all_xyz[perm].astype(np.float32)
        del all_xyz
        nrm = d["normal"][perm].astype(np.float32)
        rgb = d["rgb"][perm]
        sup = d["support"][perm]
    n = len(xyz)
    ln = np.linalg.norm(nrm, axis=1)
    good = ln > 0.5
    nrm[good] /= ln[good, None]

    if cams is not None and len(cams):
        # Chunked so a 25 M x 47 distance matrix never materialises.
        dot = np.empty(n, np.float32)
        for lo in range(0, n, 250_000):
            hi = min(lo + 250_000, n)
            j = np.argmin(((xyz[lo:hi, None, :] - cams[None, :, :]) ** 2).sum(-1), axis=1)
            t = cams[j] - xyz[lo:hi]
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            dot[lo:hi] = (nrm[lo:hi] * t).sum(1)
    else:
        dot = np.zeros(n, np.float32)
    return dict(xyz=xyz, nrm=nrm, rgb=rgb, sup=sup, dot=dot, good=good, n=n, n_total=n_total,
                has_cams=cams is not None and len(cams) > 0)


def serve(npz_path: Path, manifest_path: Path | None = None, *, port: int = 8080,
          host: str = "0.0.0.0", point_size: float = 0.004, budget: str = "3 M",
          variants: dict[str, Path] | None = None) -> int:
    viser = _require_viser()
    npz_path = Path(npz_path)
    print(f"[view] loading {npz_path}", flush=True)
    cams = camera_centres(manifest_path)
    choices = {"Original": npz_path}
    for label, path in (variants or {}).items():
        if label in choices or not Path(path).is_file():
            raise ValueError(f"duplicate variant label or missing NPZ: {label}: {path}")
        choices[label] = Path(path)
    cache = OrderedDict()
    lock = RLock()

    def get_cloud(label, limit):
        key = (label, limit)
        if key not in cache:
            print(f"[view] loading variant {label}, display limit {limit:,}", flush=True)
            cloud = load_cloud(choices[label], cams, max_points=limit)
            cache[key] = cloud
            while len(cache) > 2:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    c = get_cloud("Original", BUDGETS[budget])
    facing_pct = (100 * (c["dot"][c["good"]] > 0).mean()) if c["has_cams"] else float("nan")
    print(f"[view] {c['n_total']:,} stored points, {c['n']:,} loaded,  support {int(c['sup'].min())}..{int(c['sup'].max())}"
          + (f", {facing_pct:.1f}% of usable normals face the nearest camera" if c["has_cams"]
             else ", no manifest given so `facing` is unavailable"), flush=True)

    server = viser.ViserServer(host=host, port=port)
    pc = server.scene.add_point_cloud("/cloud", points=np.zeros((1, 3), np.float32),
                                      colors=np.zeros((1, 3), np.uint8),
                                      point_size=point_size, point_shape="circle")
    modes = ("rgb", "shaded", "normal", "facing", "support")
    with server.gui.add_folder("cloud"):
        g_variant = server.gui.add_dropdown("version", tuple(choices), initial_value="Original")
        g_colour = server.gui.add_dropdown("colour", modes, initial_value="rgb")
        g_budget = server.gui.add_dropdown("points drawn", tuple(BUDGETS), initial_value=budget)
        g_sup = server.gui.add_slider("support >=", 1, 20, 1, 1)
        g_size = server.gui.add_slider("point size [m]", 0.0005, 0.02, 0.0005, point_size)
        g_info = server.gui.add_markdown("")

    server.gui.add_markdown(
        f"**Frame {npz_path.parent.name}** — switch versions without changing the camera.\n\n"
        "The display budget subsamples large clouds for browser responsiveness. It does not change "
        "the saved file. Use the same display budget to compare methods, or ALL to compare full density.\n\n"
        "`shaded` uses **|n·L|** and is blind to normal sign *on purpose* — that blindness is why "
        "this project's always-on sanity render could not see back-facing normals for weeks.\n\n"
        "Use **facing** to read the sign: green = the normal points at the nearest camera, "
        "red = away. A correct export is mostly green. **~25% red is expected** (nearest camera is "
        "a proxy for the producing camera); a **mostly red** cloud means the normal-orientation "
        "bug is back.\n\n"
        "**support ≥** re-filters after the fact: `support` is how many cameras' rendered depth "
        "agreed on a point, so you can tighten the filter here without re-exporting.")

    def redraw(_=None):
        with lock:
            draw_current()

    def draw_current():
        nonlocal c
        label = g_variant.value
        g_info.content = f"Loading **{label}**…"
        c = get_cloud(label, BUDGETS[g_budget.value])
        budget_n = min(BUDGETS[g_budget.value], c["n"])
        keep = np.flatnonzero(c["sup"][:budget_n] >= g_sup.value)
        xyz, nrm, dot = c["xyz"][keep], c["nrm"][keep], c["dot"][keep]
        mode = g_colour.value
        if mode == "rgb":
            col = c["rgb"][keep]
        elif mode == "normal":
            col = ((nrm * 0.5 + 0.5) * 255).astype(np.uint8)
        elif mode == "shaded":
            L = np.array([0.3, 0.4, 0.86], np.float32)
            col = np.repeat((60 + 180 * np.abs(nrm @ L)).astype(np.uint8)[:, None], 3, 1)
        elif mode == "facing":
            col = np.zeros((len(keep), 3), np.uint8)
            col[dot > 0] = (60, 200, 110)
            col[dot <= 0] = (210, 60, 50)
        else:
            t = np.clip(c["sup"][keep].astype(np.float32) / 12.0, 0, 1)
            col = np.stack([255 * (1 - t), 255 * t, np.full(len(t), 70.0)], 1).astype(np.uint8)
        with server.atomic():
            pc.points, pc.colors = xyz, col
        msg = f"**{label}** — **{len(keep):,}** drawn / **{c['n_total']:,}** stored\n\n"
        if c["has_cams"] and len(keep):
            msg += (f"normals facing nearest camera: **{100 * (dot > 0).mean():.1f}%** "
                    f"(mean dot {dot.mean():+.3f})")
        elif not c["has_cams"]:
            msg += "_no manifest given, so `facing` is flat_"
        g_info.content = msg

    for h in (g_variant, g_colour, g_budget, g_sup):
        h.on_update(redraw)
    g_size.on_update(lambda _: setattr(pc, "point_size", g_size.value))
    redraw()

    print(f"\n[view] open  http://localhost:{port}", flush=True)
    print(f"[view] on a cluster the compute node is not reachable directly; forward the port:")
    print(f"[view]     ssh -J <login-alias> -N -L {port}:127.0.0.1:{port} <user>@<compute-node>"
          if host == "127.0.0.1" else f"[view]     ssh -L {port}:<compute-node>:{port} <user>@<login-node>")
    print(f"[view] then open http://localhost:{port} on your own machine. Ctrl-C to stop.\n",
          flush=True)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("[view] stopped")
    return 0
