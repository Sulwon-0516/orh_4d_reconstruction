"""CPU-only point subsampling; uniform random sampling by default.

python -m orhsurf.simplify --source out/C001/00000 --out out/C001_simplified
This is a derived product: it never modifies the source reconstruction.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
from . import atomicio, cpubudget


def normal_codes(normal, max_angle_deg=30.0):
    import numpy as np
    if not 0 < max_angle_deg <= 90:
        raise ValueError('normal angle must be in (0,90] degrees')
    length = np.linalg.norm(normal, axis=1)
    if not np.isfinite(normal).all():
        raise ValueError('normals must be finite')
    unknown = length < 1e-8
    length = np.where(unknown, 1.0, length)
    # Same bin => chord distance <= sqrt(3)*step => angle <= max_angle_deg.
    step = 2 * math.sin(math.radians(max_angle_deg) / 2) / math.sqrt(3)
    base = int(math.floor(2 / step)) + 1
    code = np.zeros(len(normal), np.uint64)
    for axis in range(3):
        q = np.floor((np.clip(normal[:, axis] / length, -1, 1) + 1) / step).astype(np.uint64)
        code = code * np.uint64(base) + q
    code[unknown] = base ** 3  # unknown orientation is separate from every valid direction
    return code, base ** 3 + 1


def grid_keys(xyz, codes, normal_span, origin, voxel_m):
    import numpy as np
    if not math.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('voxel size must be positive and finite')
    span = np.floor((xyz.max(axis=0).astype(np.float64) - origin) / voxel_m).astype(object) + 1
    span = [int(v) for v in span]
    if math.prod(span) * normal_span >= 2**64:
        raise ValueError('voxel key exceeds uint64; increase voxel size')
    key = np.zeros(len(xyz), np.uint64)
    for axis in range(3):
        q = np.floor((xyz[:, axis].astype(np.float64) - origin[axis]) / voxel_m).astype(np.uint64)
        key = key * np.uint64(span[axis]) + q
    return key * np.uint64(normal_span) + codes


def occupied_count(keys):
    import numpy as np
    keys.sort()  # scratch keys only; never mutates input attributes
    return 1 + int(np.count_nonzero(keys[1:] != keys[:-1])) if len(keys) else 0


def representatives(keys, support):
    import numpy as np
    # Stable passes: key ascending, support descending, original index as final tie-breaker.
    order = np.argsort(-support.astype(np.int32), kind='stable')
    order = order[np.argsort(keys[order], kind='stable')]
    ordered_keys = keys[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    return order[first], ordered_keys[first]


def select_points(xyz, normal, support, target, angle=30.0, attempts=12, log=print):
    import numpy as np
    n = len(xyz)
    if target < 1 or target > n:
        raise ValueError(f'target must be between 1 and {n}')
    if not np.isfinite(xyz).all():
        raise ValueError('positions must be finite')
    if target == n:
        return np.arange(n), dict(voxel_m=0.0, clusters=n, budget_trimmed=0)
    codes, normal_span = normal_codes(normal, angle)
    origin = xyz.min(axis=0).astype(np.float64)
    extent = float(np.linalg.norm(xyz.max(axis=0).astype(np.float64) - origin))
    if extent == 0:
        raise ValueError('zero spatial extent: cannot select this budget with voxel clustering')
    voxel = extent / math.sqrt(target)
    lower, upper, best = 0.0, None, None
    # Occupancy is only approximately monotonic for a non-nested grid. Keep the best observed
    # feasible grid, not a claim that bisection has found a global optimum.
    for _ in range(attempts):
        count = occupied_count(grid_keys(xyz, codes, normal_span, origin, voxel))
        log(f'[simplify] voxel={voxel*1000:.4f} mm clusters={count:,} target={target:,}', flush=True)
        if count >= target:
            if best is None or count < best[0]:
                best = count, voxel
            if count <= target * 1.03:
                break
            lower = voxel
            voxel = math.sqrt(lower * upper) if upper is not None else voxel * 2
        else:
            upper = voxel
            voxel = math.sqrt(lower * upper) if lower else voxel / 2
    if best is None:
        raise ValueError('no grid reached the requested budget; increase --search-steps')
    count, voxel = best
    indices, keys = representatives(grid_keys(xyz, codes, normal_span, origin, voxel), support)
    if len(indices) > target:
        # Uniform seeded trimming of whole representatives, never global support ranking.
        # This final exact-budget step can remove coverage; report its size explicitly.
        keep = np.random.default_rng(0).choice(len(indices), target, replace=False)
        indices = indices[keep]
    indices.sort()  # original order improves compression and preserves all per-point attributes
    return indices, dict(voxel_m=voxel, voxel_diagonal_m=math.sqrt(3)*voxel,
                         clusters=count, budget_trimmed=count-target,
                         budget_trim_fraction=(count-target)/count,
                         normal_angle_bound_deg=angle, grid_origin=origin.tolist(),
                         source_zero_normals=int(np.count_nonzero(np.linalg.norm(normal, axis=1) < 1e-8)),
                         zero_normal_policy='separate unknown-direction bin; never merged with valid normals')


def stratified_points(xyz, target, voxel_m=0.05, seed=0):
    """Proportional per-cell quotas, rounded by largest remainder; random within each cell.

    No minimum-one policy: preserves density ratios to within one point per cell while
    hitting the exact total. Cells whose ideal quota is below one may receive zero.
    """
    import numpy as np
    n = len(xyz)
    if not 1 <= target <= n or not np.isfinite(xyz).all():
        raise ValueError('invalid target or non-finite positions')
    if not math.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('stratum voxel size must be positive')
    origin = np.floor(xyz.min(axis=0).astype(np.float64)/voxel_m)*voxel_m
    keys = grid_keys(xyz, np.zeros(n,np.uint64), 1, origin, voxel_m)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    order = order[np.argsort(keys[order],kind='stable')]
    sorted_keys = keys[order]
    starts = np.r_[0,np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])+1]
    counts = np.diff(np.r_[starts,n])
    # Integer arithmetic avoids floating-point quota rounding and ensures the exact total.
    if n*target > np.iinfo(np.int64).max:
        raise ValueError('point count exceeds int64 quota arithmetic')
    numerator = counts.astype(np.int64)*target
    quotas = numerator//n
    remainder = numerator%n
    extra = target-int(quotas.sum())
    if extra:
        ties = rng.permutation(len(counts))
        winners = ties[np.argsort(-remainder[ties],kind='stable')[:extra]]
        quotas[winners] += 1
    local_index = np.arange(n)-np.repeat(starts,counts)
    selected = order[local_index < np.repeat(quotas,counts)]
    selected.sort()
    assert len(selected)==target
    info = dict(stratum_voxel_m=voxel_m, grid_origin=origin.tolist(), seed=seed,
                source_strata=len(counts), retained_strata=int(np.count_nonzero(quotas)),
                dropped_strata=int(np.count_nonzero(quotas==0)), retention_ratio=target/n,
                max_quota_error_points=float(np.max(np.abs(quotas-counts*(target/n)))),
                quota_policy='largest remainder; no minimum-one guarantee',
                normal_policy='not used for sampling; original normals retained')
    return selected,info


def parse_targets(value):
    targets = []
    for token in value.split(','):
        token = token.strip().upper()
        targets.append(int(token[:-1]) * 1_000_000 if token.endswith('M') else int(token))
    if not targets or min(targets) < 1 or len(set(targets)) != len(targets):
        raise ValueError('targets must be unique positive counts, e.g. 10M,5M,1M')
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='completed source frame directory')
    parser.add_argument('--out', type=Path, required=True, help='new derived-output root')
    parser.add_argument('--targets', default='1M,5M')
    parser.add_argument('--method', choices=('normal-voxel', 'random', 'stratified'), default='random')
    parser.add_argument('--voxel-mm', type=float, default=50, help='stratified spatial cell size; default 50 mm')
    parser.add_argument('--seed', type=int, default=0, help='random subsampling seed')
    parser.add_argument('--normal-angle', type=float, default=30)
    parser.add_argument('--search-steps', type=int, default=12)
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='reuse verified outputs with matching source and sampling settings')
    args = parser.parse_args(argv)
    if not 1 <= args.cpus <= cpubudget.allocation_cpus():
        parser.error('--cpus must be within the current allocation')
    cpubudget.apply(args.cpus)
    import numpy as np
    targets = parse_targets(args.targets)
    if len(set(targets)) != len(targets) or args.search_steps < 1:
        parser.error('targets must be unique; search steps must be positive')
    source = args.source.resolve()
    if not source.name.isdigit() or not atomicio.is_done(source):
        parser.error('--source must be a completed numeric frame directory')
    output = args.out.resolve()
    source_npz = source/'surface.npz'
    with np.load(source_npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in atomicio.SURFACE_ARRAYS}
    n = len(arrays['xyz'])
    for k, (dtype, rank, width) in atomicio.SURFACE_ARRAYS.items():
        a = arrays[k]
        if a.dtype != np.dtype(dtype) or a.ndim != rank or len(a) != n or (width and a.shape[1] != width):
            raise ValueError(f'unexpected source schema: {k} {a.shape} {a.dtype}')
    if any(t < 1 or t > n for t in targets):
        parser.error(f'targets must be in 1..{n}')
    summaries = []
    for target in targets:
        label = f'{target//1_000_000}M' if target % 1_000_000 == 0 else str(target)
        root = output/label
        dest = root/source.name
        import hashlib
        fingerprint = dict(derived_from=str(source), target=target, method=args.method,
                           seed=args.seed, normal_angle=args.normal_angle,
                           search_steps=args.search_steps, voxel_mm=args.voxel_mm,
                           source_done_sha256=hashlib.sha256((source/'_DONE.json').read_bytes()).hexdigest(),
                           source_mtime_ns=source_npz.stat().st_mtime_ns)
        if dest.exists() or dest == source:
            marker = dest/'_DONE.json'
            if (args.resume and marker.is_file()
                    and json.loads(marker.read_text()).get('fingerprint') == fingerprint):
                report = atomicio.verify_frame(dest, deep=True)
                if not report['ok']:
                    raise RuntimeError(report)
                summaries.append(json.loads((dest/'provenance.json').read_text()))
                print(f'[simplify] reuse verified {dest}', flush=True)
                continue
            raise FileExistsError(f'refusing to overwrite incompatible or incomplete {dest}')
        start = time.monotonic()
        if args.method == 'stratified':
            indices, info = stratified_points(arrays['xyz'],target,args.voxel_mm/1000,args.seed)
            method = 'spatially stratified proportional random sample'
        elif args.method == 'random':
            indices = np.random.default_rng(args.seed).choice(n, target, replace=False)
            indices.sort()
            info = dict(seed=args.seed)
            method = 'uniform random sample without replacement'
        else:
            indices, info = select_points(arrays['xyz'], arrays['normal'], arrays['support'], target,
                                          args.normal_angle, args.search_steps)
            method = 'normal-aware voxel, max-support representative, seeded budget trim'
        info.update(method=method,
                    source=str(source_npz), source_bytes=source_npz.stat().st_size,
                    source_n_points=n, n_points=target, search_seconds=time.monotonic()-start,
                    attributes='unmodified source representatives; no averaging or support union',
                    independent_lod=True)
        with atomicio.FrameStage(dest, fingerprint=fingerprint) as stage:
            atomicio.atomic_savez(stage.path/'surface.npz', **{k:a[indices] for k,a in arrays.items()})
            stage.record('surface.npz', n_points=target)
            atomicio.atomic_write_json(stage.path/'surface_export.json', info)
            atomicio.atomic_write_json(stage.path/'provenance.json', dict(operation='simplify', **info))
            metadata = json.loads((source/'metadata.json').read_text())
            metadata.update(method=method, source_surface=str(source_npz))
            atomicio.atomic_write_json(stage.path/'metadata.json', metadata)
        report = atomicio.verify_frame(dest, deep=True)
        if not report['ok']:
            raise RuntimeError(report)
        # Read and compare every output attribute against the selected original points.
        with np.load(dest/'surface.npz', allow_pickle=False) as z:
            for k, a in arrays.items():
                if not np.array_equal(z[k], a[indices]):
                    raise AssertionError(f'output attribute changed: {k}')
        info.update(output=str(dest), output_bytes=(dest/'surface.npz').stat().st_size,
                    total_seconds=time.monotonic()-start, verified=True)
        atomicio.atomic_write_json(dest/'simplify_report.json', info)
        summaries.append(info)
        print(json.dumps(info), flush=True)
    atomicio.atomic_write_json(output/f'{source.name}_summary.json', summaries)


if __name__ == '__main__':
    main()
