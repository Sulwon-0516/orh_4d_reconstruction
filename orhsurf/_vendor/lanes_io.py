"""Surface-lane contract I/O (plan v2 / CONVENTIONS.md addendum).

$ROOT/lanes/<LANE_ID>/<frame:05d>/surface.npz   xyz float32 [m] (N,3), normal float32 (N,3), rgb uint8 (N,3),
                                                confidence float32 [0,1] (N,), observed bool (N,), support int16 (N,)
$ROOT/lanes/<LANE_ID>/<frame:05d>/surface.ply   binary_little_endian: x y z nx ny nz red green blue
$ROOT/lanes/<LANE_ID>/<frame:05d>/metadata.json timestamp, pose_version, method, config_hash, source_manifest
World frame = manifest T_world_from_camera frame, metres.
"""
import json, os, re
from pathlib import Path
import numpy as np

FRAME_RE = re.compile(r"^\d{5}$")
NPZ_KEYS = ("xyz", "normal", "rgb", "confidence", "observed", "support")


def discover_lanes(lanes_root):
    """-> {lane_id: {frame_index: frame_dir}} for dirs holding surface.npz or surface.ply."""
    lanes = {}
    for lane_dir in sorted(p for p in Path(lanes_root).iterdir() if p.is_dir()):
        frames = {}
        for fd in sorted(lane_dir.iterdir()):
            if fd.is_dir() and FRAME_RE.match(fd.name) and ((fd / "surface.npz").exists() or (fd / "surface.ply").exists()):
                frames[int(fd.name)] = fd
        if frames:
            lanes[lane_dir.name] = frames
    return lanes


def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        h = hdr.decode()
        assert "binary_little_endian" in h, "only binary_little_endian PLY supported"
        n = int([l for l in h.splitlines() if l.startswith("element vertex")][0].split()[-1])
        props = [l.split() for l in h.splitlines() if l.startswith("property") and l.split()[1] != "list"]
        tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4", "short": "<i2", "int16": "<i2"}
        v = np.frombuffer(f.read(n * np.dtype([(p[2], tmap[p[1]]) for p in props]).itemsize),
                          dtype=np.dtype([(p[2], tmap[p[1]]) for p in props]))
    names = v.dtype.names
    out = {"xyz": np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)}
    if all(k in names for k in ("nx", "ny", "nz")):
        out["normal"] = np.stack([v["nx"], v["ny"], v["nz"]], 1).astype(np.float32)
    if all(k in names for k in ("red", "green", "blue")):
        out["rgb"] = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    return out


def load_frame(frame_dir):
    """Load surface.npz (preferred) else surface.ply. Missing optional fields are filled with defaults.
    Returns dict with the 6 contract arrays + 'meta' (dict, {} if absent)."""
    frame_dir = Path(frame_dir)
    if (frame_dir / "surface.npz").exists():
        z = np.load(frame_dir / "surface.npz")
        d = {k: z[k] for k in z.files if k in NPZ_KEYS}
    else:
        d = read_ply(frame_dir / "surface.ply")
    n = len(d["xyz"])
    d["xyz"] = d["xyz"].astype(np.float32).reshape(n, 3)
    d.setdefault("normal", np.zeros((n, 3), np.float32))
    d.setdefault("rgb", np.full((n, 3), 180, np.uint8))
    d.setdefault("confidence", np.ones(n, np.float32))
    d.setdefault("observed", np.ones(n, bool))
    d.setdefault("support", np.ones(n, np.int16))
    for k in NPZ_KEYS:
        assert len(d[k]) == n, f"{frame_dir}: {k} has {len(d[k])} rows, xyz has {n}"
    d["meta"] = json.load(open(frame_dir / "metadata.json")) if (frame_dir / "metadata.json").exists() else {}
    return d


def write_frame(frame_dir, xyz, normal, rgb, confidence, observed, support, meta):
    frame_dir = Path(frame_dir); frame_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, np.float32); normal = np.asarray(normal, np.float32); rgb = np.asarray(rgb, np.uint8)
    confidence = np.asarray(confidence, np.float32); observed = np.asarray(observed, bool); support = np.asarray(support, np.int16)
    n = len(xyz)
    assert xyz.shape == (n, 3) and normal.shape == (n, 3) and rgb.shape == (n, 3) and np.isfinite(xyz).all()
    assert confidence.shape == (n,) and observed.shape == (n,) and support.shape == (n,)
    assert confidence.min() >= 0 and confidence.max() <= 1
    np.savez_compressed(frame_dir / "surface.npz", xyz=xyz, normal=normal, rgb=rgb, confidence=confidence, observed=observed, support=support)
    # orhsurf: the .ply is opt-in (ORHSURF_WRITE_PLY=1). It is ~688 MB/frame against ~618 MB for
    # the .npz, nothing downstream reads it, and on a 150-frame clip it is ~103 GB of pure cost.
    if os.environ.get("ORHSURF_WRITE_PLY", "0") != "1":
        for k in ("timestamp", "pose_version", "method", "config_hash", "source_manifest"):
            assert k in meta, f"metadata missing {k}"
        json.dump(meta, open(frame_dir / "metadata.json", "w"), indent=1)
        return
    rec = np.empty(n, dtype=[("xyz", "<f4", 3), ("n", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"], rec["n"], rec["rgb"] = xyz, normal, rgb
    with open(frame_dir / "surface.ply", "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\ncomment orh_surface lane frame, world frame (manifest), metres\n"
                 f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())
    for k in ("timestamp", "pose_version", "method", "config_hash", "source_manifest"):
        assert k in meta, f"metadata missing {k}"
    json.dump(meta, open(frame_dir / "metadata.json", "w"), indent=1)
