"""Outputs that survive a preemption, and a verify that actually opens them.

WHY THIS FILE EXISTS -- two real incidents in the source project:

1. `lanes_out_orh/S0_15fps_mv0/00002/surface.npz` and `.../00003` are 494,864,266 B where a healthy
   frame is ~757,423,573 B.  `np.load` on them raises `BadZipFile: File is not a zip file`.  The
   zip magic bytes at the head are intact, so any cheap header check passes and the failure only
   appears much later, inside a render.
2. Worse: in `00002` the truncated `surface.npz` (mtime 02:51) is NEWER than its own sidecars
   `surface_export.json` (02:50) and `metadata.json` (02:49).  A re-run overwrote a GOOD output in
   place and then died.  Existence-based resume (`if surface.npz exists: skip`) then treats the
   wreckage as finished forever.

So, three rules, all enforced here:

  * ATOMIC.  Write to `<name>.tmp.<pid>`, flush, fsync the file, rename into place, fsync the
    directory.  rename(2) within a filesystem is atomic, so a reader sees either the old file or
    the new one, never a half-written one.
  * NEVER OVERWRITE IN PLACE.  A frame is built in a sibling staging directory and swapped in only
    once complete, so a crashed re-run cannot destroy a good frame.
  * DONE LAST, CHECKED FIRST.  `_DONE.json` is written after every payload file is durable, and it
    records each file's size and each array's shape.  Resume checks the marker, not existence.

`verify` opens every array of every frame.  On a preemptible cluster this is the difference between
finding corruption now and finding it in a render three hours later.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

DONE = "_DONE.json"
SCHEMA_VERSION = 1

# Exactly the arrays lanes_io.write_frame produces.  Hardcoded and asserted: if a frame does not
# have precisely these, something upstream changed and we stop rather than adapt.
SURFACE_ARRAYS = {
    "xyz":        ("float32", 2, 3),
    "normal":     ("float32", 2, 3),
    "rgb":        ("uint8",   2, 3),
    "confidence": ("float32", 1, None),
    "observed":   ("bool",    1, None),
    "support":    ("int16",   1, None),
}


def _fsync_dir(d: Path) -> None:
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def atomic_write_json(path: Path, obj) -> Path:
    return atomic_write_bytes(path, json.dumps(obj, indent=1).encode())


def atomic_savez(path: Path, **arrays) -> Path:
    """np.savez_compressed, made atomic.

    np.savez_compressed writes straight to the destination and, interrupted, leaves the wreckage
    described at the top of this file.  We write to a temp name in the SAME directory (so the
    rename stays within one filesystem) and fsync before renaming.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


class FrameStage:
    """Build a frame's outputs in a staging dir, then swap it in whole.

        with FrameStage(out_root / "00037") as st:
            atomic_savez(st.path / "surface.npz", **arrays)
            st.record("surface.npz", n_points=len(xyz))
        # -> the real directory now exists, complete, with _DONE.json

    On any exception the staging directory is removed and the previous good frame (if any) is left
    exactly as it was.
    """

    def __init__(self, final: Path, keep_previous: bool = False):
        self.final = Path(final)
        self.keep_previous = keep_previous
        self.path: Path | None = None
        self._meta: dict = {}
        self._tmpdir: Path | None = None

    def __enter__(self) -> "FrameStage":
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self._tmpdir = Path(tempfile.mkdtemp(prefix=f".{self.final.name}.staging.",
                                             dir=str(self.final.parent)))
        self.path = self._tmpdir
        return self

    def record(self, filename: str, **info) -> None:
        self._meta[filename] = info

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._tmpdir is not None
        if exc_type is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            return False
        files = {}
        for p in sorted(self._tmpdir.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(self._tmpdir))
                files[rel] = dict(bytes=p.stat().st_size, **self._meta.get(rel, {}))
        marker = dict(schema_version=SCHEMA_VERSION, complete=True, files=files)
        # DONE is written INSIDE the staging dir, last, before the swap: the directory that appears
        # at `final` is therefore complete-with-marker the instant it becomes visible.
        atomic_write_json(self._tmpdir / DONE, marker)
        if self.final.exists():
            if self.keep_previous:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
                raise FileExistsError(f"{self.final} exists and keep_previous=True")
            # Swap, do not overwrite: move the old one aside, put the new one in, then delete.
            old = self.final.with_name(f".{self.final.name}.old.{os.getpid()}")
            os.replace(self.final, old)
            try:
                os.replace(self._tmpdir, self.final)
            except BaseException:
                os.replace(old, self.final)     # put the good one back
                raise
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(self._tmpdir, self.final)
        _fsync_dir(self.final.parent)
        return False


def is_done(frame_dir: Path) -> bool:
    """Cheap resume check: the marker exists, is valid JSON, and says complete.

    Deliberately does NOT open the arrays -- that is `verify`'s job and costs seconds per frame.
    The marker is only ever written after fsync, so its presence means the payload is durable.
    """
    m = Path(frame_dir) / DONE
    if not m.is_file():
        return False
    try:
        return json.loads(m.read_text()).get("complete") is True
    except Exception:
        return False


def verify_frame(frame_dir: Path, deep: bool = True) -> dict:
    """Open every array and check dtype/rank/width and mutual length. Returns a report dict."""
    frame_dir = Path(frame_dir)
    rep = {"frame": frame_dir.name, "path": str(frame_dir), "ok": False, "problems": []}
    bad = rep["problems"].append

    marker = frame_dir / DONE
    if not marker.is_file():
        bad(f"missing {DONE} (frame never completed, or was written by an older tool)")
    else:
        try:
            m = json.loads(marker.read_text())
            if m.get("complete") is not True:
                bad(f"{DONE} does not say complete")
            for name, info in (m.get("files") or {}).items():
                p = frame_dir / name
                if not p.is_file():
                    bad(f"{name}: listed in {DONE} but missing")
                elif "bytes" in info and p.stat().st_size != info["bytes"]:
                    bad(f"{name}: {p.stat().st_size} bytes on disk, {info['bytes']} in {DONE} "
                        f"(truncated or replaced)")
        except json.JSONDecodeError as e:
            bad(f"{DONE} is not valid JSON: {e}")

    npz = frame_dir / "surface.npz"
    if not npz.is_file():
        bad("surface.npz missing")
    elif deep:
        try:
            with np.load(npz) as z:
                missing = set(SURFACE_ARRAYS) - set(z.files)
                if missing:
                    bad(f"surface.npz missing arrays: {sorted(missing)}")
                n = None
                for name, (dtype, rank, width) in SURFACE_ARRAYS.items():
                    if name not in z.files:
                        continue
                    a = z[name]                       # forces full decompression: the real check
                    if a.dtype != np.dtype(dtype):
                        bad(f"{name}: dtype {a.dtype}, expected {dtype}")
                    if a.ndim != rank:
                        bad(f"{name}: ndim {a.ndim}, expected {rank}")
                    elif width is not None and a.shape[1] != width:
                        bad(f"{name}: shape {a.shape}, expected (N,{width})")
                    if n is None:
                        n = a.shape[0]
                    elif a.shape[0] != n:
                        bad(f"{name}: length {a.shape[0]} != xyz length {n}")
                    if name == "xyz" and a.size and not np.isfinite(a).all():
                        bad("xyz contains non-finite values")
                rep["n_points"] = int(n or 0)
                if n == 0:
                    bad("surface.npz has zero points")
        except Exception as e:
            # This is the BadZipFile case.  It is the whole point of --deep.
            bad(f"surface.npz unreadable: {type(e).__name__}: {e}")

    rep["ok"] = not rep["problems"]
    return rep


def verify_tree(out_root: Path, deep: bool = True) -> dict:
    frames = sorted(p for p in Path(out_root).iterdir() if p.is_dir() and p.name.isdigit())
    reports = [verify_frame(f, deep=deep) for f in frames]
    ok = [r for r in reports if r["ok"]]
    return dict(root=str(out_root), n_frames=len(reports), n_ok=len(ok),
                n_bad=len(reports) - len(ok),
                total_points=sum(r.get("n_points", 0) for r in ok),
                bad=[r for r in reports if not r["ok"]])
