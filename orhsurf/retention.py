"""Verified, restartable clip retention; never scan shared data trees."""
from pathlib import Path
import hashlib
import json
from . import atomicio

RECEIPT = '_RETENTION.json'


def identity(manifest, recipe_hash, frames, targets, method, only, cleanup):
    return dict(manifest=str(manifest.resolve()),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                recipe_hash=recipe_hash, frames=frames, targets=targets, method=method,
                simplify_only=only, cleanup_decoded=cleanup)


def derived_root(out, method, target):
    label = f'{target//1_000_000}M' if target % 1_000_000 == 0 else str(target)
    return out.parent/'_simplified'/out.name/method/label


def verify(out, receipt, deep=True):
    spec = receipt['identity']
    reports = []
    for frame in spec['frames']:
        roots = ([] if spec['simplify_only'] else [out]) + [
            derived_root(out, spec['method'], t) for t in spec['targets']]
        checks = [atomicio.verify_frame(root/f'{frame:05d}', deep=deep) for root in roots]
        reports.append(dict(frame=str(frame), ok=all(c['ok'] for c in checks),
                            problems=[f"{c['path']}: {p}" for c in checks for p in c['problems']],
                            n_points=sum(c.get('n_points',0) for c in checks)))
    bad = [r for r in reports if not r['ok']]
    return dict(n_frames=len(reports), expected=len(reports), n_ok=len(reports)-len(bad),
                n_bad=len(bad), bad=bad, missing=[], total_points=sum(r['n_points'] for r in reports if r['ok']))


def cleanup_paths(manifest):
    """Allow only exact generated PNGs named by this manifest; preflight before deletion."""
    man = json.loads(manifest.read_text())
    prepared = manifest.parent.resolve()
    if man.get('generator') != 'orhsurf.convert (HuggingFace HEVC archive)':
        raise ValueError('decoded cleanup requires an orhsurf.convert manifest')
    remove = set()
    for serial in man['valid_serials']:
        entries = man['cameras'][serial]['frames']
        for entry in (entries.values() if isinstance(entries,dict) else entries):
            for key, folder in [('frame_path','rgb'), ('mask_path','masks_all_foreground')]:
                if key == 'mask_path' and man.get('mask_policy',{}).get('mode') != 'all_foreground':
                    continue  # user-supplied masks are never removed
                p = Path(entry[key])
                expected = prepared/folder/serial/f"{int(entry['index']):05d}.png"
                if p != expected or p.resolve() != expected or not expected.is_relative_to(prepared):
                    raise ValueError(f'refusing cleanup outside generated input layout: {p}')
                remove.add(p)
    return sorted(remove)


def finish(out, spec):
    """Write recovery receipt before deleting anything; verification is required on every retry."""
    receipt = dict(identity=spec, state='verified_pending_cleanup')
    report = verify(out, receipt)
    if report['n_bad'] or not report['n_frames']:
        raise RuntimeError(f'retention verification failed: {report}')
    manifest = Path(spec['manifest'])
    inputs = cleanup_paths(manifest) if spec['cleanup_decoded'] else []
    # Reject symlinked output paths before any deletion.
    for frame in spec['frames']:
        p = out/f'{frame:05d}'
        if p.is_symlink() or p.resolve() != p:
            raise ValueError(f'unsafe original frame path: {p}')
    atomicio.atomic_write_json(out/RECEIPT, receipt)
    if spec['simplify_only']:
        for frame in spec['frames']:
            folder = out/f'{frame:05d}'
            # Preserve source provenance, export metadata and the old completion record.
            done = folder/'_DONE.json'
            if done.exists():
                done.replace(folder/'_ORIGINAL_DONE.json')
            for name in ('surface.npz', 'surface.ply'):
                (folder/name).unlink(missing_ok=True)
    if spec['cleanup_decoded']:
        atomicio.atomic_write_json(manifest.parent/'_DECODED_REMOVED.json',
                                   dict(manifest_sha256=spec['manifest_sha256'], retention=str(out/RECEIPT)))
        for p in inputs:
            p.unlink(missing_ok=True)
    receipt['state'] = 'complete'
    atomicio.atomic_write_json(out/RECEIPT, receipt)
    print(f'[retention] complete: {out}; simplify_only={spec["simplify_only"]}, cleanup_decoded={spec["cleanup_decoded"]}', flush=True)
