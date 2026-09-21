"""Sequential clip workflow: prepare, reconstruct, verify; stop on failure."""
from __future__ import annotations
import json
import re
from pathlib import Path
from . import alloc, cpubudget, fetch, paths


def full_frames(manifest: Path, smoke: bool = False, all_frames: bool = False, frame_limit: int = 150) -> str:
    """Validate every requested frame, including short clips and explicit full clips."""
    man = paths.read_manifest(manifest)
    n = int(man['window']['n_timestamps'])
    wanted = [0] if smoke else list(range(n if all_frames else min(n, frame_limit)))
    if n < 1 or man.get('decoded_frames') != wanted:
        raise ValueError(f'{manifest} does not contain the requested {"smoke" if smoke else "selected clip"} range')
    if not man.get('valid_serials'):
        raise ValueError(f'{manifest} has no valid cameras')
    for serial in man['valid_serials']:
        entries = man['cameras'][serial]['frames']
        indices = {int(k) for k in entries} if isinstance(entries, dict) else {f['index'] for f in entries}
        if indices != set(wanted):
            raise ValueError(f'{serial} has an incomplete frame range in {manifest}')
    return f'0-{wanted[-1]}'


def prepare(clip: str, data_root: Path, smoke: bool = False, all_frames: bool = False, frame_limit: int = 150, require_fps: int | None = None) -> Path:
    # Isolate selections so existing manifests and resume fingerprints stay unchanged.
    selection = (('smoke' if frame_limit == 150 else f'smoke{frame_limit}')
                 if smoke else ('full' if all_frames else f'first{frame_limit}'))
    prepared = data_root / f'{clip}_{selection}_prepared'
    manifest = prepared / 'manifest.json'
    if (prepared/'_DECODED_REMOVED.json').exists():
        raise RuntimeError(f'{prepared}: decoded inputs were deliberately cleaned; reuse the matching process output or explicitly re-decode into a new prepared directory')
    if manifest.exists():
        if require_fps and json.loads(manifest.read_text()).get('conventions', {}).get('output_fps') != require_fps:
            raise ValueError(f'duration mode requires output_fps={require_fps}')
        full_frames(manifest, smoke, all_frames, frame_limit)  # refuse stale/partial input; do not silently rewrite a used manifest
        print(f'[process] reuse {manifest}', flush=True)
        return manifest
    raw = data_root / '_hevc' / clip
    if not (raw / 'video_manifest.json').is_file():
        rc = fetch.fetch_clip(clip, data_root, convert=True, download_only=True)
        if rc:
            raise RuntimeError(f'download failed for {clip} (exit {rc})')
    if require_fps:
        vm = json.loads((raw/'video_manifest.json').read_text())
        if vm.get('conventions', {}).get('output_fps') != require_fps:
            raise ValueError(f'duration mode requires dataset output_fps={require_fps}')
    if smoke:
        frames = '0'
    elif all_frames:
        frames = None
    else:
        n = int(json.loads((raw / 'video_manifest.json').read_text())['window']['n_timestamps'])
        if n < 1:
            raise ValueError(f'{raw} has no frames')
        frames = f'0-{min(n, frame_limit)-1}'
    rc = fetch.convert_clip(raw, prepared, frames=frames)
    if rc:
        raise RuntimeError(f'preparation failed for {clip} (exit {rc})')
    full_frames(manifest, smoke, all_frames, frame_limit)
    return manifest


def resolve_recipe(a):
    """The recipe `process` will ACTUALLY execute for these arguments.

    `cli.recipe_from_args` is not enough on a `process` namespace, because `process` makes two
    substitutions of its own: `--preset` defaults to None there (so `--smoke` can tell an explicit
    choice from silence) and `--smoke` forces quality.  Anything that needs to know what a batch
    run will do -- retention identity, the log line, a test -- must ask HERE, or it will describe
    a different reconstruction than the one that runs.
    """
    import argparse
    from . import cli
    b = argparse.Namespace(**vars(a))
    if getattr(b, 'smoke', False):
        b.preset = 'quality'
    elif getattr(b, 'preset', None) is None:
        b.preset = cli.DEFAULT_PRESET
    return cli.recipe_from_args(b)


def _forward_recipe(recipe, args, cli) -> None:
    """Carry the resolved recipe onto the `run` namespace, and PROVE it arrived.

    `run` and `process` have separate parsers, and `process` used to build the inner namespace
    from a fixed argv listing only paths, frames and allocation.  Every recipe option therefore
    stopped at the `process` boundary: `process --preset quality --resolution 1` logged quality
    and then reconstructed at whatever `run`'s own default happened to be.  That was invisible
    while both defaults were `quality`; making `fast` the default turned it into every batch and
    Slurm run silently reconstructing at -r 4.

    Worse, the retention receipt hashed the REQUESTED recipe while the frames carried the
    executed one, so a `--cleanup-decoded` run could delete its inputs and leave a receipt
    describing a reconstruction that never happened.  Retention now hashes this same object.

    Only cli.RECIPE_FLAGS -- the fields a user can actually set -- are forwarded; every other
    Recipe field is fixed and keeps the same dataclass default on both sides.  The hash check is
    the real guarantee, and it fails loudly rather than reconstructing the wrong thing.
    """
    args.preset = None  # the resolved values below ARE the preset; applying it twice hides typos
    for field in cli.RECIPE_FLAGS:
        if not hasattr(args, field):
            raise AssertionError(
                f"orhsurf run has no --{field.replace('_','-')}, so `process` cannot forward it. "
                f"Add the option to the run parser; otherwise batch reconstructions silently "
                f"ignore it.")
        setattr(args, field, getattr(recipe, field))
    got = cli.recipe_from_args(args)
    if got.hash() != recipe.hash():
        raise AssertionError(f'recipe did not survive dispatch: asked {recipe}, run would use {got}')


def run(a) -> int:
    from . import cli
    from . import retention
    if len(set(a.clips)) != len(a.clips) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', c) for c in a.clips):
        raise SystemExit('--clips requires unique clip IDs')
    durations = getattr(a, 'durations', None)
    if durations:
        if getattr(a, 'all_frames', False):
            raise SystemExit('--durations and --all-frames cannot be combined')
        import argparse
        try:
            selected_durations = [int(v) for v in durations.split(',')]
        except ValueError:
            raise SystemExit('--durations accepts 10, 15, or 10,15')
        if not selected_durations or len(set(selected_durations)) != len(selected_durations) or any(v not in (10,15) for v in selected_durations):
            raise SystemExit('--durations accepts unique values 10 and/or 15')
        base = Path(a.out_root).expanduser().resolve() if a.out_root else paths.out_root()
        # Each clip finishes all requested versions before moving to the next clip.
        for clip in a.clips:
            for duration in selected_durations:
                child = argparse.Namespace(**vars(a))
                child.clips = [clip]
                child.durations = None
                child._frame_limit = duration * 15
                child._require_fps = 15
                child.out_root = str(base/f'{duration}s')
                rc = run(child)
                if rc:
                    return rc
        return 0
    frame_limit = getattr(a, '_frame_limit', 150)
    simplify_targets = getattr(a, 'simplify', None)
    only = getattr(a, 'simplify_only', False)
    if only and simplify_targets is None:
        simplify_targets = '1M,5M'
    cleanup = getattr(a, 'cleanup_decoded', False)
    if simplify_targets is not None:
        from .simplify import parse_targets
        parse_targets(simplify_targets)  # reject invalid selections before downloads
    # Validate the entire request before any downloads or work; no paths/globs as clip IDs.
    if len(set(a.clips)) != len(a.clips) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', c) for c in a.clips):
        raise SystemExit('--clips requires unique clip IDs, e.g. C001 C002 C003')
    if getattr(a, 'smoke', False):
        # A smoke run exists to prove the install and the data, so it uses the FULL quality recipe
        # whatever the batch default is -- otherwise "the smoke test passed" would say nothing
        # about a quality run.  That promise has to be unconditional: the old code promoted only
        # an absent/economy/fast preset, so `--smoke --preset draft` quietly smoke-tested draft,
        # contradicting both its own comment and the README.  Refuse the conflict instead, and
        # refuse it HERE, before any download -- the file's rule is to validate the whole request
        # before doing work.
        conflicting = [n for n in ('preset', 'iterations', 'resolution', 'nn_max_mm',
                                   'densify_from_iter', 'densification_interval')
                       if getattr(a, n, None) is not None]
        if conflicting:
            raise SystemExit('--smoke always runs the full quality recipe, so it cannot be combined '
                             'with ' + ', '.join('--'+n.replace('_', '-') for n in conflicting)
                             + '. Drop those, or drop --smoke and run the recipe you want.')
    if getattr(a, 'preset', None) is None:
        a.preset = cli.DEFAULT_PRESET
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
    smoke = getattr(a, 'smoke', False)
    all_frames = getattr(a, 'all_frames', False)
    if smoke:
        root = root / '_smoke'
        a.preset = 'quality'  # refused above if that conflicts with an explicit option
    # ONE resolved recipe for this invocation.  Everything downstream -- the log line, the actual
    # reconstruction, and the retention receipt -- must come from this object and nothing else.
    recipe = resolve_recipe(a)
    label = 'smoke: frame 0 only, quality' if smoke else f'preset {getattr(a, "preset", None) or "quality"}'
    print(f'[process] {label}: {recipe.iterations} it, -r {recipe.resolution}, '
          f'densify {recipe.densify_from_iter}/{recipe.densification_interval}, '
          f'isolation {recipe.nn_max_mm} mm', flush=True)
    for i, clip in enumerate(a.clips, 1):
        print(f'[process] {i}/{len(a.clips)}: {clip}', flush=True)
        out = root/clip
        receipt_path = out/retention.RECEIPT
        targets = parse_targets(simplify_targets) if simplify_targets else []
        method = getattr(a, 'simplify_method', 'random')
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            selection = (('smoke' if frame_limit == 150 else f'smoke{frame_limit}')
                 if smoke else ('full' if all_frames else f'first{frame_limit}'))
            manifest = paths.data_root()/f'{clip}_{selection}_prepared/manifest.json'
            selected = cli.parse_frames(full_frames(manifest, smoke, all_frames, frame_limit))
            spec = retention.identity(manifest, recipe.hash(), selected, targets, method, only, cleanup)
            if receipt['identity'] != spec:
                raise RuntimeError(f'{out}: retained run settings differ; use a distinct output root')
            retention.finish(out, spec)
            continue
        manifest = prepare(clip, paths.data_root(), smoke=smoke, all_frames=all_frames, frame_limit=frame_limit, require_fps=getattr(a, '_require_fps', None))
        frames = full_frames(manifest, smoke, all_frames, frame_limit)
        out = root/clip
        args = parser.parse_args(['run','--clip',str(manifest),'--frames',frames,
                                  '--gpus',str(len(gpus)),'--cpus-per-job',str(budget),'--out',str(out),
                                  '--shard','0','--shards','1'])
        _forward_recipe(recipe, args, cli)
        rc = cli.cmd_run(args)
        if rc:
            print(f'[process] {clip} reconstruction failed; stopping (exit {rc})', flush=True)
            return rc
        rc = cli.cmd_verify(parser.parse_args(['verify','--out',str(out)]))
        if rc:
            print(f'[process] {clip} verification failed; stopping (exit {rc})', flush=True)
            return rc
        if simplify_targets:
            from .simplify import main as simplify_frame
            method = getattr(a, 'simplify_method', 'random')
            for frame in cli.parse_frames(frames):
                simplify_frame(['--source', str(out/f'{frame:05d}'),
                                '--out', str(root/'_simplified'/clip/method),
                                '--targets', simplify_targets, '--method', method,
                                '--cpus', str(budget), '--resume'])
        if only or cleanup:
            spec = retention.identity(manifest, recipe.hash(), cli.parse_frames(frames),
                                      targets, method, only, cleanup)
            retention.finish(out, spec)
        print(f'[process] {clip} complete and verified: {out}', flush=True)
    return 0
