"""Prove the durability properties, don't assert them in a README.

Each test here corresponds to a failure that actually happened in the source project:
truncated surface.npz files with intact zip magic bytes, and a re-run that overwrote a good
output and then died.

Run:  python tests/test_atomicio.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orhsurf.atomicio import FrameStage, atomic_savez, is_done, verify_frame  # noqa: E402

N = 20_000


def _arrays(n=N):
    return dict(xyz=np.random.rand(n, 3).astype(np.float32),
                normal=np.random.rand(n, 3).astype(np.float32),
                rgb=(np.random.rand(n, 3) * 255).astype(np.uint8),
                confidence=np.random.rand(n).astype(np.float32),
                observed=np.ones(n, bool),
                support=np.full(n, 5, np.int16))


def _write_good(d: Path):
    with FrameStage(d) as st:
        atomic_savez(st.path / "surface.npz", **_arrays())
        st.record("surface.npz", n_points=N)
    return d


def test_healthy_frame_verifies():
    with tempfile.TemporaryDirectory() as t:
        d = _write_good(Path(t) / "00000")
        r = verify_frame(d)
        assert r["ok"], r["problems"]
        assert r["n_points"] == N
        assert is_done(d)


def test_truncation_is_caught_despite_intact_magic_bytes():
    """The exact failure mode: a cheap header check passes, the real read explodes later."""
    with tempfile.TemporaryDirectory() as t:
        d = _write_good(Path(t) / "00000")
        p = d / "surface.npz"
        with open(p, "r+b") as f:
            f.truncate(int(p.stat().st_size * 0.65))
        assert open(p, "rb").read(4) == b"PK\x03\x04", "magic bytes should still look fine"
        deep = verify_frame(d, deep=True)
        assert not deep["ok"]
        assert any("unreadable" in x or "bytes on disk" in x for x in deep["problems"])
        # The size recorded in _DONE.json catches it even without decompressing.
        shallow = verify_frame(d, deep=False)
        assert not shallow["ok"], "shallow verify must still catch a size mismatch"


def test_crash_midframe_leaves_nothing():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "00000"
        try:
            with FrameStage(d) as st:
                atomic_savez(st.path / "surface.npz", **_arrays())
                raise RuntimeError("simulated preemption")
        except RuntimeError:
            pass
        assert not d.exists(), "a crashed frame must not leave a partial directory behind"
        assert not list(Path(t).glob("*")), f"staging leftovers: {list(Path(t).glob('*'))}"


def test_crashed_rerun_does_not_destroy_a_good_frame():
    """The worst incident: a re-run overwrote a complete output in place and then died."""
    with tempfile.TemporaryDirectory() as t:
        d = _write_good(Path(t) / "00000")
        before = (d / "surface.npz").stat().st_size
        try:
            with FrameStage(d) as st:
                atomic_savez(st.path / "surface.npz", **_arrays(N // 2))
                raise RuntimeError("simulated preemption")
        except RuntimeError:
            pass
        assert (d / "surface.npz").stat().st_size == before, "the good frame was damaged"
        assert verify_frame(d)["ok"]


def test_successful_rerun_replaces_atomically():
    with tempfile.TemporaryDirectory() as t:
        d = _write_good(Path(t) / "00000")
        with FrameStage(d) as st:
            atomic_savez(st.path / "surface.npz", **_arrays(N // 2))
            st.record("surface.npz", n_points=N // 2)
        assert verify_frame(d)["n_points"] == N // 2
        assert not list(Path(t).glob(".*old*")), "swap leftovers were not cleaned up"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all durability tests passed")
