"""Small real CUDA checks; no model weights or input dataset required."""
import argparse
import json
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--da3', action='store_true')
a = parser.parse_args()
t0 = time.perf_counter()
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 1, 'Run with exactly one visible allocated GPU'
x = torch.randn(128, 128, device='cuda', requires_grad=True)
y = (x @ x.T).square().mean()
y.backward()
assert torch.isfinite(y) and torch.isfinite(x.grad).all()
checks = ['torch CUDA matmul + backward']
if a.da3:
    import numpy
    from depth_anything_3.api import DepthAnything3
    from xformers.ops import memory_efficient_attention
    assert torch.__version__ == '2.6.0+cu124'
    assert numpy.__version__ == '2.2.6'
    q = torch.randn(1, 64, 4, 32, device='cuda', dtype=torch.float16, requires_grad=True)
    result = memory_efficient_attention(q, q, q)
    result.float().square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(q.grad).all()
    checks += ['DA3 API import', 'xformers CUDA attention + backward']
else:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party/AmbiSuR'))
    import scene.cameras
    import gaussian_renderer
    import train
    from simple_knn._C import distCUDA2
    from diff_plane_rasterization_ambisur import _C
    assert torch.__version__ == '2.7.1+cu128'
    points = torch.tensor([[0.,0.,1.], [1.,0.,1.], [0.,1.,1.], [0.,0.,2.]], device='cuda')
    distances = distCUDA2(points)
    expected = torch.tensor([1., 5./3., 5./3., 5./3.], device='cuda')
    assert torch.allclose(distances, expected, atol=1e-5), distances
    visible = _C.mark_visible(points, torch.eye(4, device='cuda'), torch.eye(4, device='cuda'))
    assert visible.shape == (4,) and visible.dtype == torch.bool and visible.all()
    checks += ['training imports', 'simple_knn CUDA known distances', 'rasterizer CUDA visibility']
torch.cuda.synchronize()
print(json.dumps(dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0), checks=checks,
                     seconds=round(time.perf_counter()-t0, 3)), indent=2))
