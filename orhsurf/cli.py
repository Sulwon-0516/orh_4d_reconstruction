"""Reconstruct complete clips with one command:

    orhsurf process --clips C001 C002 C003 --gpus 1

Downloads/prepares one clip, reconstructs every frame, verifies outputs, then starts the next.
Re-run the same command to resume completed matching frames. Requires installation and a
compute allocation; it does not submit or renew Slurm jobs.

Advanced commands: fetch (inputs), run (selected frames), verify (outputs), render and view.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import alloc, atomicio, cpubudget, paths


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


def _run_fingerprint(manifest: Path, recipe) -> dict:
    """What a completed frame must match for resume to accept it.

    `is_done()` used to check only `complete: true`, so changing the filter parameters or swapping
    the manifest silently produced a clip containing mutually incompatible frames.  The recipe hash
    was already recorded in provenance and simply never compared.
    """
    m = Path(manifest)
    st = m.stat()
    return dict(recipe_hash=recipe.hash(), manifest=str(m.resolve()),
                manifest_size=st.st_size, manifest_mtime_ns=st.st_mtime_ns)


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

    # A converted clip carries only the frames that were decoded (orhsurf.convert writes the set to
    # `decoded_frames`).  --frames defaults to 0-149, so a clip converted with `--frames 0-4` used to
    # fail INSIDE a worker -- after resolving GPUs, building the scene and paying for prep.  Check it
    # here, where the argument was typed.
    try:
        _man = json.load(open(manifest))
    except Exception:
        _man = None
    if _man is not None and _man.get("decoded_frames") is not None:
        have = set(_man["decoded_frames"])
        missing = [f for f in frames if f not in have]
        if missing:
            avail = sorted(have)
            span = (f"{avail[0]}-{avail[-1]}" if avail and avail == list(range(avail[0], avail[-1] + 1))
                    else (str(avail[:12])[:-1] + ", ...]" if len(avail) > 12 else str(avail)))
            raise SystemExit(
                f"--frames asks for {len(missing)} frame(s) this clip does not carry "
                f"(first missing: {missing[0]}).\n"
                f"  {manifest} was converted with only these frames decoded: {span}\n"
                f"  Either pass --frames within that set, or re-run "
                f"`orhsurf fetch --clip <id> --convert --frames <range>` to decode more.")

    # Job-array sharding: take this task's contiguous slice BEFORE anything else, so each task
    # only ever considers, resumes and renders its own frames.
    shard = a.shard if a.shard is not None else _int_env("SLURM_ARRAY_TASK_ID")
    shards = a.shards if a.shards is not None else _int_env("SLURM_ARRAY_TASK_COUNT")
    if shards and shards > 1:
        # Slurm arrays are NOT necessarily dense or zero-based.  `--array=1-8` gives TASK_ID 1..8
        # with COUNT 8, so slice 0 would never run and task 8 would be out of range; resubmitting
        # a subset changes COUNT and so repartitions the clip entirely.  Refuse both rather than
        # silently producing a different partition than the first submission.
        lo = _int_env("SLURM_ARRAY_TASK_MIN")
        hi = _int_env("SLURM_ARRAY_TASK_MAX")
        if a.shard is None and lo is not None and hi is not None:
            if lo != 0 or hi != shards - 1:
                raise SystemExit(
                    f"this job array is {lo}-{hi} with {shards} tasks, which is not dense and "
                    f"zero-based.\n"
                    f"  orhsurf partitions the frame list by task index, so a 1-based or sparse "
                    f"array would skip slice 0 and repartition on resubmission.\n"
                    f"  Submit as --array=0-{shards - 1}, or pass --shard/--shards explicitly "
                    f"(keeping --shards equal to the ORIGINAL submission's task count when you "
                    f"retry a subset).")
        if shard is None or not (0 <= shard < shards):
            raise SystemExit(
                f"--shard must be in [0,{shards}), got {shard}. When retrying failed array tasks, "
                f"keep --shards at the original count so the partition does not move.")
        all_frames = frames
        frames = alloc.contiguous_slices(frames, shards)[shard]
        print(f"[run] shard {shard}/{shards}: {len(frames)} of {len(all_frames)} frames"
              + (f" ({frames[0]:05d}..{frames[-1]:05d})" if frames else " (none)"))
        if not frames:
            print("[run] this shard has no work; exiting 0")
            return 0

    gpus = alloc.resolve_gpus(a.gpus)
    if not a.no_gpu_check:
        usable = alloc.usable_gpus(gpus)
        if not usable:
            raise SystemExit(
                f"none of the {len(gpus)} visible GPU(s) has ~{alloc.REQUIRED_FREE_MIB} MiB free; "
                f"the DA3 stage would OOM after ~90 s of prep.\n"
                f"  Wait for them to free, request exclusive GPUs, or pass --no-gpu-check to "
                f"proceed anyway (e.g. with a smaller --group-size).")
        gpus = usable
    # One budget: the allocation divided by the number of CONCURRENT JOBS (one per GPU).
    cpus_per_job = cpubudget.resolve(a.cpus_per_job, concurrent_jobs=len(gpus))
    cpubudget.apply(cpus_per_job)
    out_root = Path(a.out or (paths.out_root() / clip_id)).resolve()
    # Scratch is namespaced by clip + a run id.  It used to be keyed only by logical GPU index and
    # frame, so two overlapping submissions built the same scene in the same directory and either
    # one's cleanup could delete the other's live workspace.
    run_id = os.environ.get("SLURM_JOB_ID") or f"{int(time.time())}.{os.getpid()}"
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        run_id += f".{os.environ['SLURM_ARRAY_TASK_ID']}"
    work_root = Path(a.work).resolve() if a.work else (out_root / "_work" / run_id)
    out_root.mkdir(parents=True, exist_ok=True)
    recipe = recipe_from_args(a)

    atomicio.recover_interrupted(out_root)
    fp = _run_fingerprint(manifest, recipe)
    if a.force:
        todo = frames
    else:
        todo, stale = [], []
        for f in frames:
            state = atomicio.frame_state(out_root / f"{f:05d}", fp)
            if state == "done":
                continue
            if state == "mismatch":
                stale.append(f)
            todo.append(f)
        if stale:
            print(f"[run] {len(stale)} frame(s) exist but were built with a DIFFERENT recipe or "
                  f"manifest and will be rebuilt: {stale[:8]}{'...' if len(stale) > 8 else ''}")
    print(f"[run] clip {clip_id}")
    print(f"[run] {len(frames)} frames, {len(frames)-len(todo)} already done, {len(todo)} to run")
    print(f"[run] {len(gpus)} GPU(s) {gpus}, one job per GPU, -r {recipe.resolution} "
          f"x {recipe.iterations} it")
    print(f"[run] filter: min-views {recipe.min_views}, nn-k {recipe.nn_k}, "
          f"nn-max-mm {recipe.nn_max_mm}")
    print(f"[run] out  {out_root}")
    alloc_info = alloc.describe()
    print(f"[run] cpus {cpus_per_job}/job x {len(gpus)} job(s) "
          f"of {cpubudget.allocation_cpus()} allocated"
          + (" [slurm]" if alloc_info["under_slurm"] else ""))

    if not a.dry_run:
        atomicio.atomic_write_json(out_root / "_EXPECTED.json", dict(
            clip=clip_id, frames=frames, recipe_hash=recipe.hash(),
            manifest=str(manifest), cpus_per_job=cpus_per_job,
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    if a.dry_run:
        for i, part in enumerate(alloc.contiguous_slices(todo, len(gpus))):
            print(f"  gpu {gpus[i]}: {len(part)} frames"
                  + (f" {part[0]:05d}..{part[-1]:05d}" if part else ""))
        return 0

    t0 = time.time()
    rc = 0
    if todo:
        rc = _dispatch(todo, gpus, manifest, out_root, work_root, recipe, a, cpus_per_job, fp)
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


def _dispatch(todo, gpus, manifest, out_root, work_root, recipe, a, cpus_per_job,
              _fingerprint=None) -> int:
    """One worker subprocess per GPU, each given a contiguous in-order slice of the frame list."""
    parts = alloc.contiguous_slices(todo, len(gpus))
    # Translate our logical GPU indices to absolute device ids before handing them to children.
    phys = alloc.physical_gpu_ids()
    per = cpus_per_job
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
               "--cpus", str(per), "--recipe-json", json.dumps(_recipe_json(recipe)),
               "--fingerprint", json.dumps(_fingerprint)]
        if a.keep_work:
            cmd.append("--keep-work")
        if a.force:
            cmd.append("--force")          # previously never reached the worker, so --force was a no-op
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            filter(None, [str(paths.REPO_ROOT), os.environ.get("PYTHONPATH", "")])))
        # absolute id, so the child does not re-resolve it against our restricted list
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
        # Include job/task/pid: every single-GPU array task used to truncate the same gpu0.log.
        tag = "_".join(str(x) for x in filter(None, (
            os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID"))))
        log = logdir / (f"gpu{dev}{('_' + tag) if tag else ''}_{os.getpid()}.log")
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
    fp = json.loads(a.fingerprint) if a.fingerprint else None
    failed = []
    for f in frames:
        out_dir = out_root / f"{f:05d}"
        # --force used to stop at the parent: the worker skipped completed frames regardless, so
        # `--force` was a no-op end to end.
        if not a.force and atomicio.frame_state(out_dir, fp) == "done":
            print(f"[gpu{a.gpu}] frame {f:05d} already done, skipping", flush=True)
            continue
        wd = work_root / f"{f:05d}"
        try:
            run_frame(Path(a.manifest), f, out_dir, wd, r, gpu=a.gpu, cpus=a.cpus,
                      fingerprint=fp, log=lambda m: print(m, flush=True))
        except Exception as e:
            print(f"[gpu{a.gpu}] frame {f:05d} FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            failed.append(f)
        finally:
            # Keep scratch when the frame FAILED: it holds the trained model and the DA3 depths,
            # which are ~13 min of GPU time, and the logs needed to diagnose the failure.
            if not a.keep_work and f not in failed:
                import shutil
                shutil.rmtree(wd, ignore_errors=True)
            elif f in failed:
                print(f"[gpu{a.gpu}] kept scratch for diagnosis: {wd}", flush=True)
    if failed:
        print(f"[gpu{a.gpu}] {len(failed)} frame(s) failed: {failed}", flush=True)
        return 1
    return 0


def cmd_verify(a) -> int:
    # `run` resolves a clip through resolve_clip(), which accepts an id, a directory, or a path to
    # manifest.json. `verify` used to do `out_root() / a.clip`, and pathlib DISCARDS the left
    # operand when the right is absolute -- so `verify --clip <path-to-manifest.json>` tried to
    # list a file as a directory and raised NotADirectoryError. Accept the same three forms.
    if a.out:
        root = Path(a.out).resolve()
    elif a.clip:
        p = Path(a.clip)
        clip_id = p.parent.name if p.suffix == ".json" else (p.name if p.exists() else a.clip)
        root = (paths.out_root() / clip_id).resolve()
    else:
        root = paths.out_root().resolve()
    rep = atomicio.verify_tree(root, deep=not a.shallow)
    print(f"[verify] {root}")
    exp = f" (expected {rep['expected']})" if rep.get("expected") is not None else \
          "  [no _EXPECTED.json: only frames already on disk were checked]"
    print(f"[verify] {rep['n_ok']}/{rep['n_frames']} frames ok{exp}, "
          f"{rep['total_points']:,} points total")
    if rep.get("missing"):
        print(f"  MISSING (never published): {rep['missing']}")
    for bad in rep["bad"]:
        print(f"  BAD {bad['frame']}:")
        for p in bad["problems"]:
            print(f"      {p}")
    if a.json:
        print(json.dumps(rep, indent=1))
    return 0 if rep["n_bad"] == 0 else 1


def cmd_doctor(a) -> int:
    """Report everything missing at once, rather than failing one stage at a time.

    Phase-aware: `envs` (login node, nothing compiled), `noweights`, or `full`.  Each environment
    is probed with ITS OWN interpreter -- DA3 lives only in env-da3, so asking the AmbiSuR
    interpreter for it reported a failure on every correct install.
    """
    phase = getattr(a, "phase", "full") or "full"
    want_ext = phase in ("full", "noweights")
    want_gpu = phase in ("full", "noweights")
    want_weights = phase == "full"
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
    cb = cpubudget.describe()
    print(f"  cpus      budget {cb['cpus_per_job']}/job of {cb['allocation_cpus']} allocated "
          f"(source: {cb['source']}); OMP_NUM_THREADS={cb['thread_vars']['OMP_NUM_THREADS']}")
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
        if want_gpu:
            check("CUDA available", torch.cuda.is_available(),
                  "" if torch.cuda.is_available() else "no usable GPU in this allocation")
        elif not torch.cuda.is_available():
            print(f"  [skip] GPU checks (phase={phase}; a login node normally has no GPU)")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gb = p.total_memory / 2 ** 30
                check(f"gpu{i} {p.name} {gb:.1f} GiB", gb >= 23.0,
                      "" if gb >= 23.0 else "DA3 at 1008 peaks ~24 GB; see docs/INSTALL_SLURM.md")
    except Exception as e:
        check("torch import", False, f"{type(e).__name__}: {e}")

    for mod in ("cv2", "scipy", "numpy", "imageio", "plyfile"):
        try:
            __import__(mod)
            check(f"import {mod} (ambisur env)", True)
        except Exception as e:
            check(f"import {mod} (ambisur env)", False, f"{type(e).__name__}: {e}")
    # DA3 is installed ONLY in env-da3; probe it with that interpreter.
    da3_py = paths.da3_python()
    if da3_py.is_file():
        # Import what the STAGE imports. `import depth_anything_3` alone always succeeds: it is a
        # namespace package with __file__ is None, so it reported [ok] on an env where every run
        # then died on `from addict import Dict`. The check must fail where the run fails.
        probe = ("import numpy, torch, addict; "
                 "from depth_anything_3.api import DepthAnything3; "
                 "print(f'numpy {numpy.__version__} torch {torch.__version__}')")
        r = subprocess.run([str(da3_py), "-c", probe], capture_output=True, text=True)
        detail = (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else ""
        check(f"DA3 stage imports (da3 env)", r.returncode == 0, detail)
        if r.returncode == 0 and "numpy 2.2.6" not in r.stdout:
            check("env-da3 numpy == 2.2.6 (the pinned reference stack)", False, detail)
    else:
        check(f"DA3 interpreter at {da3_py}", False, "run install.sh")
    if want_ext:
        for ext in ("diff_plane_rasterization_ambisur", "simple_knn._C"):
            try:
                __import__(ext)
                check(f"CUDA extension {ext}", True)
            except Exception as e:
                check(f"CUDA extension {ext}", False, f"{type(e).__name__}: {e}")
    else:
        print(f"  [skip] CUDA extensions (phase={phase})")
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
    if want_weights:
        check(f"DA3 checkpoint cached ({ck})", ck.is_dir(),
              "" if ck.is_dir() else "run: ./fetch_ckpts.sh   (6.76 GB)")
    else:
        print(f"  [skip] weights (phase={phase})")
    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED (see above)"))
    return 0 if ok else 1


def cmd_fetch(a) -> int:
    from .fetch import fetch_weights, fetch_clip
    rc = 0
    if a.weights or not a.clip:
        rc |= fetch_weights(paths.cache_dir())
    if a.clip:
        rc |= fetch_clip(a.clip, paths.data_root(),
                         convert=getattr(a, "convert", False), masks=getattr(a, "masks", None),
                         frames=getattr(a, "frames", None))
    return rc


def cmd_view(a) -> int:
    """Open one finished frame in viser. Optional dependency; see orhsurf/viewer.py."""
    from . import viewer
    if a.npz:
        npz = Path(a.npz)
        if not npz.is_file():
            raise SystemExit(f"no such file: {npz}")
        manifest = Path(a.manifest) if a.manifest else None
    else:
        if not a.clip:
            raise SystemExit("give --clip (a clip id / dir / manifest.json) or --npz <path>")
        manifest = Path(a.manifest) if a.manifest else resolve_clip(a.clip)
        p = Path(a.clip)
        clip_id = p.parent.name if p.suffix == ".json" else (p.name if p.exists() else a.clip)
        npz = viewer.find_frame(Path(a.out).resolve() if a.out
                                else (paths.out_root() / clip_id).resolve(), a.frame)
    return viewer.serve(npz, manifest, port=a.port, host=a.host,
                        point_size=a.point_size, budget=a.budget)


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

    from .process import run as process_clips
    whole = sub.add_parser("process", help="download, prepare, reconstruct and verify whole clips sequentially")
    whole.add_argument("--clips", nargs="+", required=True, help="clip IDs in execution order, e.g. C001 C002")
    whole.add_argument("--gpus", type=int, default=1, help="GPUs on this node; parallel frames within each clip, sequential clips")
    whole.add_argument("--out-root", default=None, help="results under <root>/<clip>/<frame>; default out/")
    whole.add_argument("--cpus-per-job", type=int, default=None, help="thread limit within the existing allocation")
    whole.add_argument("--smoke", action="store_true", help="frame 0 only at full quality; isolated inputs and out/_smoke/<clip>")
    whole.set_defaults(fn=process_clips)

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
    r.add_argument("--no-gpu-check", action="store_true", dest="no_gpu_check",
                   help="dispatch even onto GPUs that look too full for the DA3 stage")
    r.add_argument("--cpus-per-job", type=int, default=None, dest="cpus_per_job",
                   help="CPU budget for EACH concurrent job. Default: the allocation divided by "
                        "the number of GPUs. Applied to OMP/MKL/BLAS before any numeric import, "
                        "to torch, to the k-NN gate and to COLMAP.")
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
    w.add_argument("--gpu", required=True)   # str: may be a GPU-<uuid>, not only an index
    w.add_argument("--out", required=True)
    w.add_argument("--work", required=True)
    w.add_argument("--cpus", type=int, default=8)
    w.add_argument("--recipe-json", required=True)
    w.add_argument("--keep-work", action="store_true")
    w.add_argument("--force", action="store_true")
    w.add_argument("--fingerprint", default=None)
    w.set_defaults(fn=cmd_worker)

    v = sub.add_parser("verify", help="open every output and check it (use after any interruption)")
    v.add_argument("--clip", default=None)
    v.add_argument("--out", default=None)
    v.add_argument("--shallow", action="store_true", help="markers only, do not decompress arrays")
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=cmd_verify)

    d = sub.add_parser("doctor", help="check the install and the allocation")
    d.add_argument("--phase", choices=("envs", "noweights", "full"), default="full",
                   help="envs: login node, nothing compiled yet; full: everything (default)")
    d.add_argument("--cpus-per-job", type=int, default=None, dest="cpus_per_job")
    d.set_defaults(fn=cmd_doctor)

    f = sub.add_parser("fetch", help="download model weights and/or a clip")
    f.add_argument("--clip", default=None)
    f.add_argument("--weights", action="store_true")
    f.add_argument("--convert", action="store_true",
                   help="after extracting, decode the videos and write a loadable manifest")
    f.add_argument("--masks", default=None,
                   help="<dir>/<serial>/<frame:05d>.png RGBA masks to reference from the manifest")
    f.add_argument("--frames", default=None,
                   help="which frames to decode, e.g. 0-149 for the first 10 s of a 15 s clip. "
                        "Default: all of them. Indices are encoded_frame_index.")
    f.set_defaults(fn=cmd_fetch)

    rr = sub.add_parser("render", help="debug renders from finished frames")
    rr.add_argument("--clip", default=None)
    rr.add_argument("--out", default=None)
    rr.add_argument("--check", action="store_true", help="just the one static check image")
    rr.add_argument("--orbit", action="store_true")
    rr.add_argument("--time", action="store_true")
    rr.add_argument("--both", action="store_true")
    rr.set_defaults(fn=cmd_render)

    v = sub.add_parser("view", help="open a finished frame in an interactive viser viewer")
    v.add_argument("--clip", default=None,
                   help="clip id, clip dir, or path to manifest.json (same forms as `run`)")
    v.add_argument("--frame", type=int, default=None,
                   help="frame index; default = the first COMPLETED frame")
    v.add_argument("--npz", default=None, help="a surface.npz directly, instead of --clip")
    v.add_argument("--manifest", default=None,
                   help="manifest for camera centres (enables the `facing` mode); "
                        "inferred from --clip when possible")
    v.add_argument("--out", default=None, help="output root, if not the default")
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 so a cluster node is reachable through an SSH tunnel")
    v.add_argument("--point-size", type=float, default=0.004, dest="point_size")
    v.add_argument("--budget", default="3 M", choices=list(__import__(
        "orhsurf.viewer", fromlist=["BUDGETS"]).BUDGETS),
                   help="how many points to stream to the browser")
    v.set_defaults(fn=cmd_view)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    # ONE CPU budget, resolved and applied BEFORE anything numeric is imported downstream.
    # The old code applied the whole allocation here, so every per-GPU worker re-claimed all 64
    # cores before it had even parsed its own, smaller --cpus.  See orhsurf/cpubudget.py.
    if getattr(a, "cmd", None) == "_worker":
        cpubudget.apply(a.cpus)                      # the parent already divided the allocation
    elif getattr(a, "cpus_per_job", None) is not None:
        cpubudget.apply(cpubudget.resolve(a.cpus_per_job))
    # For `run`, the budget depends on how many GPUs we end up using, so cmd_run resolves it once
    # it knows; for the read-only commands the default is fine.
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
