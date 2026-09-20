"""quaternion_to_matrix without PyTorch3D.

WHY: `third_party/AmbiSuR/scene/gaussian_model.py` imports exactly one symbol from PyTorch3D --
`from pytorch3d.transforms import quaternion_to_matrix` -- and that is the ONLY PyTorch3D
reference in the whole AmbiSuR tree (verified by recursive grep).  PyTorch3D has no universal
wheel for recent torch builds, so it is normally compiled from source; on the reference machine it
was built from commit 978cd99.  A from-source PyTorch3D build is the single most likely thing to
fail on a stranger's cluster: it is long, needs nvcc and a matching host compiler, and its errors
are opaque.  Trading that whole dependency for the twelve lines below is the best install-risk
reduction available in this package.

CONVENTION (PyTorch3D's, which is what AmbiSuR's callers assume): real part FIRST, i.e. (w, x, y, z),
and the quaternion need not be normalised -- pytorch3d normalises internally, so we must too, or
AmbiSuR's un-normalised optimiser state would produce scaled rotations.

`tests/test_quat.py` asserts this agrees with the real PyTorch3D to 1e-6 on random quaternions
whenever PyTorch3D happens to be importable, so the replacement is proven rather than assumed.
"""
from __future__ import annotations

import torch


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """(..., 4) real-part-first quaternions -> (..., 3, 3) rotation matrices."""
    if quaternions.shape[-1] != 4:
        raise ValueError(f"expected (..., 4) quaternions, got {tuple(quaternions.shape)}")
    # Normalise exactly as pytorch3d does; an un-normalised quaternion otherwise yields a matrix
    # scaled by |q|^2, which would silently rescale every Gaussian in the model.
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(
        quaternions.dtype).tiny)
    w, x, y, z = torch.unbind(q, dim=-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    out = torch.stack([
        1.0 - (tyy + tzz), txy - twz,         txz + twy,
        txy + twz,         1.0 - (txx + tzz), tyz - twx,
        txz - twy,         tyz + twx,         1.0 - (txx + tyy),
    ], dim=-1)
    return out.reshape(q.shape[:-1] + (3, 3))


def get_quaternion_to_matrix(prefer_pytorch3d: bool = False):
    """Return the implementation to use.

    Default is ours.  `prefer_pytorch3d=True` (config `use_pytorch3d: true`) restores the original
    import, which is the documented escape hatch if this replacement is ever suspected.
    """
    if prefer_pytorch3d:
        from pytorch3d.transforms import quaternion_to_matrix as p3d  # noqa: F401
        return p3d
    return quaternion_to_matrix
