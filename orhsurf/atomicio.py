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

# NOTE: numpy is imported LAZILY inside the functions that need it, never at module scope.
# orhsurf.cli imports this module early, and a module-scope `import numpy` here would bind
# OpenMP's thread pool before cpubudget.apply() has set OMP_NUM_THREADS -- at which point the
# budget is a string in os.environ and nothing more.  See orhsurf/cpubudget.py.

DONE = "_DONE.json"
PREV_PREFIX = ".prev."          # a frame displaced mid-publication; recover_interrupted() rolls it back
SCHEMA_VERSION = 2              # 2 adds `fingerprint` and fsynced payloads

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
    import numpy as np
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

    def __init__(self, final: Path, keep_previous: bool = False, fingerprint: dict | None = None):
        self.final = Path(final)
        self.keep_previous = keep_previous
        self.fingerprint = fingerprint
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
        # FSYNC EVERY PAYLOAD BEFORE THE MARKER.
        # The production writer is lanes_io.write_frame(), which uses an ordinary
        # np.savez_compressed -- NOT our atomic_savez().  So staging alone protected against an
        # interrupted process but not against a node or filesystem crash: the marker could reach
        # disk while the payload it vouches for was still in page cache.  Durability was claimed
        # and only the helper, not the writer, actually provided it.  We now force every staged
        # file (and the staging directory) to stable storage before writing the marker, which
        # makes the guarantee hold regardless of which writer produced the bytes.
        files = {}
        for p in sorted(self._tmpdir.rglob("*")):
            if p.is_file():
                fd = os.open(str(p), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                rel = str(p.relative_to(self._tmpdir))
                files[rel] = dict(bytes=p.stat().st_size, **self._meta.get(rel, {}))
        _fsync_dir(self._tmpdir)
        marker = dict(schema_version=SCHEMA_VERSION, complete=True, files=files,
                      fingerprint=self.fingerprint)
        # DONE is written INSIDE the staging dir, last, before the swap: the directory that appears
        # at `final` is therefore complete-with-marker the instant it becomes visible.
        atomic_write_json(self._tmpdir / DONE, marker)
        # PUBLISH WITH A SINGLE RENAME.
        # Replacement used to be two renames (final -> .old, staging -> final).  A SIGKILL between
        # them left NO final frame at all, and the .old copy was not part of resume, so the frame
        # looked never-built.  rename(2) cannot atomically swap two directories portably, so we
        # instead publish through a directory that is *created* by one rename and keep the previous
        # version aside under a name that `recover_interrupted()` knows how to roll back.
        if self.final.exists():
            if self.keep_previous:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
                raise FileExistsError(f"{self.final} exists and keep_previous=True")
            old = self.final.with_name(f"{PREV_PREFIX}{self.final.name}")
            shutil.rmtree(old, ignore_errors=True)
            os.replace(self.final, old)          # crash here -> recover_interrupted() restores it
            try:
                os.replace(self._tmpdir, self.final)
            except BaseException:
                os.replace(old, self.final)
                raise
            _fsync_dir(self.final.parent)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(self._tmpdir, self.final)
        _fsync_dir(self.final.parent)
        return False


def recover_interrupted(out_root: Path, log=print) -> int:
    """Roll back any frame whose publication was killed between the two renames.

    A `.prev.<name>` directory with no corresponding `<name>` means we died after moving the good
    frame aside and before the new one landed.  The good frame is put back.  Called at the start of
    every run, before the resume scan, so an interrupted publication is never mistaken for a frame
    that was never built.
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        return 0
    n = 0
    for prev in out_root.glob(f"{PREV_PREFIX}*"):
        if not prev.is_dir():
            continue
        final = prev.with_name(prev.name[len(PREV_PREFIX):])
        if final.exists():
            shutil.rmtree(prev, ignore_errors=True)      # publication completed; drop the old copy
        else:
            os.replace(prev, final)
            log(f"[recover] restored {final.name} after an interrupted publication")
            n += 1
    return n


def frame_state(frame_dir: Path, fingerprint: dict | None = None) -> str:
    """'missing' | 'mismatch' | 'done'.

    'mismatch' means the frame is complete but was built from a different recipe or manifest than
    this run is asking for -- previously accepted silently, which let one clip contain mutually
    incompatible frames.  Also rejects a marker whose payload has vanished or changed size.
    """
    frame_dir = Path(frame_dir)
    m = frame_dir / DONE
    if not m.is_file():
        return "missing"
    try:
        marker = json.loads(m.read_text())
    except Exception:
        return "missing"
    if marker.get("complete") is not True:
        return "missing"
    for name, info in (marker.get("files") or {}).items():
        p = frame_dir / name
        if not p.is_file():
            return "missing"
        if "bytes" in info and p.stat().st_size != info["bytes"]:
            return "missing"
    if fingerprint:
        have = marker.get("fingerprint")
        if have is None:
            return "mismatch"          # written by an older tool; cannot prove compatibility
        for k in ("recipe_hash", "manifest", "manifest_size", "manifest_mtime_ns"):
            if k in fingerprint and have.get(k) != fingerprint[k]:
                return "mismatch"
    return "done"


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

    import numpy as np
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
                # Cross-check against what the exporter committed, so a payload that is readable
                # but truncated to a different length cannot pass.
                se = frame_dir / "surface_export.json"
                if se.is_file():
                    try:
                        want = json.loads(se.read_text()).get("n_points")
                        if want is not None and n is not None and int(want) != int(n):
                            bad(f"surface.npz holds {n:,} points, surface_export.json "
                                f"recorded {int(want):,}")
                    except json.JSONDecodeError:
                        bad("surface_export.json is not valid JSON")
        except Exception as e:
            # This is the BadZipFile case.  It is the whole point of --deep.
            bad(f"surface.npz unreadable: {type(e).__name__}: {e}")

    rep["ok"] = not rep["problems"]
    return rep


def verify_tree(out_root: Path, deep: bool = True) -> dict:
    """Verify a clip. Uses the EXPECTED frame list written at run start when one is present.

    Discovery alone is not verification: frames that never reached publication simply do not exist
    on disk, so a scan of existing directories cannot see them, and an empty directory used to
    report `0/0 ok` and exit 0 -- a clean bill of health for a clip that was never built.
    """
    out_root = Path(out_root)
    if not out_root.is_dir():
        return dict(root=str(out_root), n_frames=0, n_ok=0, n_bad=1, total_points=0,
                    bad=[{"frame": "-", "problems": [f"{out_root} does not exist"]}], missing=[])

    expected = None
    exp_file = out_root / "_EXPECTED.json"
    if exp_file.is_file():
        try:
            expected = json.loads(exp_file.read_text()).get("frames")
        except Exception:
            expected = None

    present = sorted(p for p in out_root.iterdir() if p.is_dir() and p.name.isdigit())
    missing = []
    if expected is not None:
        have = {int(p.name) for p in present}
        missing = sorted(set(expected) - have)
        reports = [verify_frame(out_root / f"{f:05d}", deep=deep) for f in sorted(expected)
                   if f in have]
    else:
        reports = [verify_frame(f, deep=deep) for f in present]

    ok = [r for r in reports if r["ok"]]
    n_bad = len(reports) - len(ok) + len(missing)
    if expected is None and not present:
        n_bad += 1
        missing = ["<no _EXPECTED.json and no frames on disk: nothing was verified>"]
    return dict(root=str(out_root), n_frames=len(reports) + len(missing), n_ok=len(ok),
                n_bad=n_bad, expected=len(expected) if expected is not None else None,
                missing=missing, total_points=sum(r.get("n_points", 0) for r in ok),
                bad=[r for r in reports if not r["ok"]])
