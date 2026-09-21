"""Sequential whole-clip workflow: prepare, reconstruct, verify; stop on failure."""
from __future__ import annotations
import json
import re
from pathlib import Path
from . import alloc, cpubudget, fetch, paths


def full_frames(manifest: Path) -> str:
    """Require the full declared encoded range, not the CLI run's historical 0-149 default."""
    man = paths.read_manifest(manifest)
    n = int(man['window']['n_timestamps'])
    if n < 1 or man.get('decoded_frames') != list(range(n)):
        raise ValueError(f'{manifest} does not contain the whole clip')
    if not man.get('valid_serials'):
        raise ValueError(f'{manifest} has no valid cameras')
    for serial in man['valid_serials']:
        entries = man['cameras'][serial]['frames']
        indices = {int(k) for k in entries} if isinstance(entries, dict) else {f['index'] for f in entries}
        if indices != set(range(n)):
            raise ValueError(f'{serial} has an incomplete frame range in {manifest}')
    return f'0-{n-1}'


def prepare(clip: str, data_root: Path) -> Path:
    # Dedicated full-clip inputs leave existing *_prepared subsets and their fingerprints intact.
    prepared = data_root / f'{clip}_full_prepared'
    manifest = prepared / 'manifest.json'
    if manifest.exists():
        full_frames(manifest)  # refuse stale/partial input; do not silently rewrite a used manifest
        print(f'[process] reuse {manifest}', flush=True)
        return manifest
    raw = data_root / '_hevc' / clip
    if (raw / 'video_manifest.json').is_file():
        rc = fetch.convert_clip(raw, prepared)
    else:
        rc = fetch.fetch_clip(clip, data_root, convert=True, prepared_dir=prepared)
    if rc:
        raise RuntimeError(f'preparation failed for {clip} (exit {rc})')
    full_frames(manifest)
    return manifest


def run(a) -> int:
    from . import cli
    # Validate the entire request before any downloads or work; no paths/globs as clip IDs.
    if len(set(a.clips)) != len(a.clips) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', c) for c in a.clips):
        raise SystemExit('--clips requires unique clip IDs, e.g. C001 C002 C003')
    if a.gpus < 1:
        raise SystemExit('--gpus must be >= 1')
    available = alloc.resolve_gpus(None)
    if len(available) < a.gpus:
        raise SystemExit(f'--gpus {a.gpus} requires {a.gpus} visible GPUs on this node; '
                         f'only {len(available)} are allocated. Obtain the allocation first.')
    gpus = available[:a.gpus]
    budget = cpubudget.resolve(a.cpus_per_job, concurrent_jobs=len(gpus))
    total = cpubudget.allocation_cpus()
    if budget * len(gpus) > total:
        raise SystemExit(f'{budget} CPU threads/job x {len(gpus)} GPUs exceeds {total} allocated CPUs. '
                         'Lower --cpus-per-job (or ORHSURF_CPUS_PER_JOB).')
    print(f'[process] {len(gpus)} GPUs, {budget} CPU threads per worker; clips remain sequential', flush=True)
    cpubudget.apply(budget)
    snapshot = paths.cache_dir() / 'hf/hub' / ('models--'+fetch.DA3_REPO.replace('/', '--')) / 'snapshots' / fetch.DA3_REVISION
    if not all((snapshot/name).is_file() for name in ('config.json','model.safetensors')):
        rc = fetch.fetch_weights(paths.cache_dir())
        if rc:
            return rc
    else:
        print('[process] reuse pinned DA3 weights', flush=True)
    parser = cli.build_parser()
    rc = cli.cmd_doctor(parser.parse_args(['doctor','--cpus-per-job',str(budget)]))
    if rc:
        return rc
    root = Path(a.out_root).expanduser().resolve() if a.out_root else paths.out_root()
    for i, clip in enumerate(a.clips, 1):
        print(f'[process] {i}/{len(a.clips)}: {clip}', flush=True)
        manifest = prepare(clip, paths.data_root())
        frames = full_frames(manifest)
        out = root/clip
        args = parser.parse_args(['run','--clip',str(manifest),'--frames',frames,
                                  '--gpus',str(len(gpus)),'--cpus-per-job',str(budget),'--out',str(out),
                                  '--shard','0','--shards','1'])
        rc = cli.cmd_run(args)
        if rc:
            print(f'[process] {clip} reconstruction failed; stopping (exit {rc})', flush=True)
            return rc
        rc = cli.cmd_verify(parser.parse_args(['verify','--out',str(out)]))
        if rc:
            print(f'[process] {clip} verification failed; stopping (exit {rc})', flush=True)
            return rc
        print(f'[process] {clip} complete and verified: {out}', flush=True)
    return 0
