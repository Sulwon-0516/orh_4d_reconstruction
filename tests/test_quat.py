"""Prove the PyTorch3D replacement, don't assume it.

Run:  python -m pytest tests/ -q      (or just: python tests/test_quat.py)
"""
import sys
from pathlib import Path

import torch


class _Skipped(Exception):
    """Raised when a test could not make its comparison. Reported as SKIP, never as PASS."""


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orhsurf.quat import quaternion_to_matrix  # noqa: E402


def test_is_rotation_matrix():
    """Output must be orthonormal with det +1 for arbitrary, un-normalised input."""
    torch.manual_seed(0)
    q = torch.randn(512, 4, dtype=torch.float64) * torch.rand(512, 1, dtype=torch.float64) * 10
    q = q[q.norm(dim=-1) > 1e-3]
    R = quaternion_to_matrix(q)
    eye = torch.eye(3, dtype=torch.float64).expand_as(R)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-10), "not orthonormal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(len(R), dtype=torch.float64), atol=1e-10)


def test_identity_and_known_rotations():
    q = torch.tensor([[1.0, 0, 0, 0],                      # identity
                      [0.0, 1, 0, 0],                      # 180 deg about x
                      [2 ** -0.5, 2 ** -0.5, 0, 0]],       # 90 deg about x
                     dtype=torch.float64)
    R = quaternion_to_matrix(q)
    assert torch.allclose(R[0], torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(R[1], torch.diag(torch.tensor([1.0, -1, -1], dtype=torch.float64)),
                          atol=1e-12)
    expect = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)
    assert torch.allclose(R[2], expect, atol=1e-12)


def test_batch_shapes_preserved():
    for shape in [(4,), (7, 4), (2, 3, 4)]:
        R = quaternion_to_matrix(torch.randn(*shape, dtype=torch.float64))
        assert R.shape == shape[:-1] + (3, 3), (shape, R.shape)


def test_matches_pytorch3d_if_available():
    """The real comparison. Skipped (not failed) where PyTorch3D is absent -- which is the point."""
    try:
        from pytorch3d.transforms import quaternion_to_matrix as p3d
    except Exception as e:                                  # pragma: no cover
        # This test exists to make ONE comparison. install.sh deliberately does not install
        # pytorch3d, so on every clean install this skips -- and it used to still print PASS,
        # which let the README cite "equal to 1.1e-15" on machines that never checked it.
        # Report SKIP distinctly so the result cannot be read as a proof that did not happen.
        raise _Skipped(f"pytorch3d not importable ({type(e).__name__}); the equivalence "
                       f"comparison did NOT run. The orthonormality/known-rotation tests did.")
    torch.manual_seed(1)
    q = torch.randn(4096, 4, dtype=torch.float64)
    q = q[q.norm(dim=-1) > 1e-3]
    ours, theirs = quaternion_to_matrix(q), p3d(q)
    err = (ours - theirs).abs().max().item()
    assert err < 1e-6, f"disagrees with pytorch3d by {err:.3e}"
    print(f"  [ok] matches pytorch3d on {len(q)} quaternions, max abs diff {err:.3e}")


if __name__ == "__main__":
    ran = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except _Skipped as e:
                skipped += 1
                print(f"SKIP {name}: {e}")
            else:
                ran += 1
                print(f"PASS {name}")
    print(f"{ran} passed, {skipped} SKIPPED")
    if skipped:
        print("NOTE: a skipped test proved nothing. The README's pytorch3d equivalence figure "
              "comes from a machine where pytorch3d WAS installed; this run did not verify it.")
