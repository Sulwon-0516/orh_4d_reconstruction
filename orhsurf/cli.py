"""orhsurf command line.

The headline path is ONE command:

    orhsurf run --clip <clip-id> --gpus 8 --frames 0-149

which does prep -> DA3 1008 -> rewarp -> train 7k -> export WITH the filter, per frame, across the
GPUs, and leaves filtered surface.npz on disk.  The staged subcommands exist for debugging; they
are not a workflow anyone has to remember.

Scheduling: the frame list is partitioned into CONTIGUOUS in-order slices, one per GPU, and each
slice is worked by its own worker PROCESS (not thread).  One job per GPU -- packing two AmbiSuR
trainings onto one GPU was measured twice on the reference node and lost both times (451 s solo vs
1705 s each packed at 7k), and the DA3 stage OOMs at two per GPU regardless of -r.  Separate
processes rather than threads because each needs its own CUDA context pinned to one device, and
because a segfault in one must not take the run down.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import alloc, atomicio, paths


# --------------------------------------------------------------------------- helpers --
def parse_frames(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise SystemExit(f"--frames: '{part}' counts down; give it as {hi}-{lo}")
            out += list(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise SystemExit("--frames selected nothing")
    return sorted(set(out))


def _int_env(name: str):
    v = os.environ.get(name)
    return int(v) if v and v.lstrip("-").isdigit() else None


def resolve_clip(clip: str) -> Path:
    """clip id -> manifest.json. Accepts an id under the data root, or a direct path."""
    p = Path(clip)
    if p.is_file() and p.name.endswith(".json"):
        return p.resolve()
    if p.is_dir() and (p / "manifest.json").is_file():
        return (p / "manifest.json").resolve()
    cand = paths.data_root() / clip / "manifest.json"
    if cand.is_file():
        return cand.resolve()
    raise SystemExit(
        f"cannot find a manifest for clip '{clip}'.\n"
        f"  looked at: {cand}\n"
        f"  set ORHSURF_DATA_ROOT (currently {paths.data_root()}) or pass a path to manifest.json.\n"
        f"  to fetch the clip:  orhsurf fetch --clip {clip}")


def recipe_from_args(a) -> "object":
    from .pipeline import Recipe
    r = Recipe()
    for k in ("resolution", "iterations", "min_views", "nn_k", "nn_max_mm",
              "consistency_mm", "group_size", "write_ply"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(r, k, v)
    return r


# ------------------------------------------------------------------------ subcommands --
def cmd_run(a) -> int:
    from .pipeline import Recipe, run_frame     # noqa: F401  (Recipe used via recipe_from_args)

    manifest = resolve_clip(a.clip)
    clip_id = a.clip if not Path(a.clip).exists() else manifest.parent.name
    frames = parse_frames(a.frames)

    # Job-array sharding: take this task's contiguous slice BEFORE anything else, so each task
    # only ever considers, resumes and renders its own frames.
    shard = a.shard if a.shard is not None else _int_env("SLURM_ARRAY_TASK_ID")
    shards = a.shards if a.shards is not None else _int_env("SLURM_ARRAY_TASK_COUNT")
    if shards and shards > 1:
        if shard is None or not (0 <= shard < shards):
            raise SystemExit(f"--shard must be in [0,{shards}), got {shard}")
        all_frames = frames
        frames = alloc.contiguous_slices(frames, shards)[shard]
        print(f"[run] shard {shard}/{shards}: {len(frames)} of {len(all_frames)} frames"
              + (f" ({frames[0]:05d}..{frames[-1]:05d})" if frames else " (none)"))
        if not frames:
            print("[run] this shard has no work; exiting 0")
            return 0

    gpus = alloc.resolve_gpus(a.gpus)
    out_root = Path(a.out or (paths.out_root() / clip_id)).resolve()
    work_root = Path(a.work or (out_root / "_work")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    recipe = recipe_from_args(a)

    todo = frames if a.force else [f for f in frames if not atomicio.is_done(out_root / f"{f:05d}")]
    print(f"[run] clip {clip_id}")
    print(f"[run] {len(frames)} frames, {len(frames)-len(todo)} already done, {len(todo)} to run")
    print(f"[run] {len(gpus)} GPU(s) {gpus}, one job per GPU, -r {recipe.resolution} "
          f"x {recipe.iterations} it")
    print(f"[run] filter: min-views {recipe.min_views}, nn-k {recipe.nn_k}, "
          f"nn-max-mm {recipe.nn_max_mm}")
    print(f"[run] out  {out_root}")
    alloc_info = alloc.describe()
    print(f"[run] cpus {alloc_info['cpu_count']} (from {alloc_info['cpu_source']})"
          + (" [slurm]" if alloc_info["under_slurm"] else ""))

    if a.dry_run:
        for i, part in enumerate(alloc.contiguous_slices(todo, len(gpus))):
            print(f"  gpu {gpus[i]}: {len(part)} frames"
                  + (f" {part[0]:05d}..{part[-1]:05d}" if part else ""))
        return 0

    t0 = time.time()
    rc = 0
    if todo:
        rc = _dispatch(todo, gpus, manifest, out_root, work_root, recipe, a)
    el = time.time() - t0
    print(f"[run] reconstruction finished in {el/60:.1f} min (rc={rc})")

    # --- always-on cheap sanity artifact: one static render of the first requested frame ---
    done = [f for f in frames if atomicio.is_done(out_root / f"{f:05d}")]
    if done:
        from .render import static_check_render
        try:
            p = static_check_render(out_root, done[0], out_root / "_check", gpu=None)
            print(f"[run] check render -> {p}")
        except Exception as e:
            print(f"[run] WARNING: check render failed: {type(e).__name__}: {e}")
    else:
        print("[run] no completed frames; skipping the check render")

    if a.render_orbit or a.render_time or a.render_both:
        from .render import debug_renders
        debug_renders(out_root, done, out_root / "_video",
                      orbit=a.render_orbit, time_=a.render_time, both=a.render_both)
    return rc


def _dispatch(todo, gpus, manifest, out_root, work_root, recipe, a) -> int:
    """One worker subprocess per GPU, each given a contiguous in-order slice of the frame list."""
    parts = alloc.contiguous_slices(todo, len(gpus))
    # Translate our logical GPU indices to absolute device ids before handing them to children.
    phys = alloc.physical_gpu_ids()
    cpus_total = alloc.cpu_count()
    # Never let the workers collectively claim more CPU than we were allocated.  8 is what one
    # AmbiSuR training actually uses (~280% CPU measured), so more does not help anyway.
    per = max(1, min(8, cpus_total // max(1, len(gpus))))
    logdir = out_root / "_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    procs = []
    for gpu, part in zip(gpus, parts):
        if not part:
            continue
        dev = phys[gpu] if gpu < len(phys) else gpu
        spec = ",".join(str(f) for f in part)
        cmd = [sys.executable, "-m", "orhsurf.cli", "_worker",
               "--manifest", str(manifest), "--frames", spec, "--gpu", str(dev),
               "--out", str(out_root), "--work", str(work_root / f"gpu{gpu}"),
               "--cpus", str(per), "--recipe-json", json.dumps(_recipe_json(recipe))]
        if a.keep_work:
            cmd.append("--keep-work")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            filter(None, [str(paths.REPO_ROOT), os.environ.get("PYTHONPATH", "")])))
        # absolute id, so the child does not re-resolve it against our restricted list
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
        log = logdir / f"gpu{gpu}.log"
        print(f"[run] gpu {dev}: {len(part)} frames {part[0]:05d}..{part[-1]:05d} -> {log}")
        procs.append((dev, subprocess.Popen(cmd, stdout=open(log, "w"),
                                            stderr=subprocess.STDOUT, env=env), log))

    rc = 0
    for gpu, p, log in procs:
        r = p.wait()
        if r != 0:
            rc = r
            print(f"[run] gpu {gpu} worker exited rc={r}; see {log}")
    return rc


def _recipe_json(recipe) -> dict:
    from dataclasses import asdict
    return asdict(recipe)


def cmd_worker(a) -> int:
    """Internal: reconstruct a list of frames on one GPU, in order. Not a user-facing command."""
    from dataclasses import fields
    from .pipeline import Recipe, run_frame

    r = Recipe()
    data = json.loads(a.recipe_json)
    names = {f.name for f in fields(Recipe)}
    for k, v in data.items():
        if k in names:
            setattr(r, k, v)

    out_root, work_root = Path(a.out), Path(a.work)
    frames = parse_frames(a.frames)
    failed = []
    for f in frames:
        out_dir = out_root / f"{f:05d}"
        if atomicio.is_done(out_dir):
            print(f"[gpu{a.gpu}] frame {f:05d} already done, skipping", flush=True)
            continue
        wd = work_root / f"{f:05d}"
        try:
            run_frame(Path(a.manifest), f, out_dir, wd, r, gpu=a.gpu, cpus=a.cpus,
                      log=lambda m: print(m, flush=True))
        except Exception as e:
            print(f"[gpu{a.gpu}] frame {f:05d} FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            failed.append(f)
        finally:
            if not a.keep_work:
                import shutil
                shutil.rmtree(wd, ignore_errors=True)
    if failed:
        print(f"[gpu{a.gpu}] {len(failed)} frame(s) failed: {failed}", flush=True)
        return 1
    return 0


def cmd_verify(a) -> int:
    root = Path(a.out or paths.out_root()).resolve()
    if a.clip and not a.out:
        root = paths.out_root() / a.clip
    rep = atomicio.verify_tree(root, deep=not a.shallow)
    print(f"[verify] {root}")
    print(f"[verify] {rep['n_ok']}/{rep['n_frames']} frames ok, "
          f"{rep['total_points']:,} points total")
    for bad in rep["bad"]:
        print(f"  BAD {bad['frame']}:")
        for p in bad["problems"]:
            print(f"      {p}")
    if a.json:
        print(json.dumps(rep, indent=1))
    return 0 if rep["n_bad"] == 0 else 1


def cmd_doctor(a) -> int:
    """Report everything missing at once, rather than failing one stage at a time."""
    ok = True

    def check(label, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok ' if good else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

    print("orhsurf doctor")
    print(f"  repo      {paths.REPO_ROOT}")
    print(f"  cache     {paths.cache_dir()}")
    print(f"  data root {paths.data_root()}")
    print(f"  out root  {paths.out_root()}")
    info = alloc.describe()
    print(f"  cpus      {info['cpu_count']} (from {info['cpu_source']})")
    print(f"  gpus      {info['visible_gpus']} (from {info['gpu_source']})"
          + ("  [slurm]" if info["under_slurm"] else ""))

    cb = paths.colmap_bin()
    check(f"colmap at {cb}", cb.is_file(),
          "" if cb.is_file() else "set ORHSURF_COLMAP_BIN or run install.sh")
    repo = paths.ambisur_repo()
    check(f"AmbiSuR at {repo}", (repo / "train.py").is_file())

    try:
        import torch
        check(f"torch {torch.__version__}", True)
        check("CUDA available", torch.cuda.is_available(),
              "" if torch.cuda.is_available() else "no usable GPU in this allocation")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gb = p.total_memory / 2 ** 30
                check(f"gpu{i} {p.name} {gb:.1f} GiB", gb >= 23.0,
                      "" if gb >= 23.0 else "DA3 at 1008 peaks ~24 GB; see docs/INSTALL_SLURM.md")
    except Exception as e:
        check("torch import", False, f"{type(e).__name__}: {e}")

    for mod in ("cv2", "scipy", "numpy", "depth_anything_3"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as e:
            check(f"import {mod}", False, f"{type(e).__name__}: {e}")
    for ext in ("diff_plane_rasterization_ambisur", "simple_knn._C"):
        try:
            __import__(ext)
            check(f"CUDA extension {ext}", True)
        except Exception as e:
            check(f"CUDA extension {ext}", False, f"{type(e).__name__}: {e}")
    try:
        from .quat import quaternion_to_matrix
        import torch as _t
        q = _t.randn(8, 4, dtype=_t.float64)
        R = quaternion_to_matrix(q)
        check("quaternion_to_matrix (pytorch3d replacement)",
              bool(_t.allclose(R @ R.transpose(-1, -2),
                               _t.eye(3, dtype=_t.float64).expand_as(R), atol=1e-10)))
    except Exception as e:
        check("quaternion_to_matrix", False, f"{type(e).__name__}: {e}")

    ck = paths.cache_dir() / "hf/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1"
    check(f"DA3 checkpoint cached ({ck})", ck.is_dir(),
          "" if ck.is_dir() else "run: orhsurf fetch --weights   (6.76 GB)")
    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED (see above)"))
    return 0 if ok else 1


def cmd_fetch(a) -> int:
    from .fetch import fetch_weights, fetch_clip
    rc = 0
    if a.weights or not a.clip:
        rc |= fetch_weights(paths.cache_dir())
    if a.clip:
        rc |= fetch_clip(a.clip, paths.data_root())
    return rc


def cmd_render(a) -> int:
    from .render import debug_renders, static_check_render
    root = Path(a.out or (paths.out_root() / a.clip)).resolve()
    done = sorted(int(p.name) for p in root.iterdir()
                  if p.is_dir() and p.name.isdigit() and atomicio.is_done(p))
    if not done:
        raise SystemExit(f"no completed frames under {root}")
    if a.check or not (a.orbit or a.time or a.both):
        print(static_check_render(root, done[0], root / "_check", gpu=None))
    if a.orbit or a.time or a.both:
        debug_renders(root, done, root / "_video", orbit=a.orbit, time_=a.time, both=a.both)
    return 0


# ------------------------------------------------------------------------------ main --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("orhsurf", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="reconstruct a clip end to end (prep -> DA3 -> train -> export)")
    r.add_argument("--clip", required=True, help="clip id under the data root, or a manifest path")
    r.add_argument("--frames", default="0-149", help="e.g. 0-149 or 30-74 or 0,5,9")
    r.add_argument("--gpus", type=int, default=None,
                   help="how many GPUs to use; default = the whole allocation")
    r.add_argument("--out", default=None)
    r.add_argument("--work", default=None, help="scratch dir (large); default <out>/_work")
    r.add_argument("--keep-work", action="store_true", help="keep per-frame scratch for debugging")
    r.add_argument("--force", action="store_true", help="redo frames that are already complete")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--shard", type=int, default=None,
                   help="Slurm job arrays: this task's index. Takes the SHARD-th contiguous slice "
                        "of the frame list. Defaults to SLURM_ARRAY_TASK_ID.")
    r.add_argument("--shards", type=int, default=None,
                   help="total number of array tasks. Defaults to SLURM_ARRAY_TASK_COUNT.")
    r.add_argument("--iterations", type=int, default=None, help="default 7000 (settled)")
    r.add_argument("--resolution", type=int, default=None, help="AmbiSuR -r, default 2")
    r.add_argument("--min-views", type=int, default=None, dest="min_views")
    r.add_argument("--nn-k", type=int, default=None, dest="nn_k")
    r.add_argument("--nn-max-mm", type=float, default=None, dest="nn_max_mm")
    r.add_argument("--consistency-mm", type=float, default=None, dest="consistency_mm")
    r.add_argument("--group-size", type=int, default=None, dest="group_size",
                   help="DA3 views per batch (default 18, ~24 GB). Lower it for a smaller GPU, "
                        "but note it CHANGES the prior.")
    r.add_argument("--write-ply", action="store_true", dest="write_ply",
                   help="also write surface.ply (~688 MB/frame; off by default)")
    r.add_argument("--render-orbit", action="store_true", help="debug video: static frame, orbit cam")
    r.add_argument("--render-time", action="store_true", help="debug video: all frames, static cam")
    r.add_argument("--render-both", action="store_true", help="debug video: all frames, orbit cam")
    r.set_defaults(fn=cmd_run)

    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    w.add_argument("--manifest", required=True)
    w.add_argument("--frames", required=True)
    w.add_argument("--gpu", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--work", required=True)
    w.add_argument("--cpus", type=int, default=8)
    w.add_argument("--recipe-json", required=True)
    w.add_argument("--keep-work", action="store_true")
    w.set_defaults(fn=cmd_worker)

    v = sub.add_parser("verify", help="open every output and check it (use after any interruption)")
    v.add_argument("--clip", default=None)
    v.add_argument("--out", default=None)
    v.add_argument("--shallow", action="store_true", help="markers only, do not decompress arrays")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_verify)

    d = sub.add_parser("doctor", help="check the install and the allocation")
    d.set_defaults(fn=cmd_doctor)

    f = sub.add_parser("fetch", help="download model weights and/or a clip")
    f.add_argument("--clip", default=None)
    f.add_argument("--weights", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    rr = sub.add_parser("render", help="debug renders from finished frames")
    rr.add_argument("--clip", default=None)
    rr.add_argument("--out", default=None)
    rr.add_argument("--check", action="store_true", help="just the one static check image")
    rr.add_argument("--orbit", action="store_true")
    rr.add_argument("--time", action="store_true")
    rr.add_argument("--both", action="store_true")
    rr.set_defaults(fn=cmd_render)
    return p


def main(argv=None) -> int:
    # Thread limits must be set before numpy/torch are imported anywhere downstream.
    alloc.apply_thread_limits()
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
