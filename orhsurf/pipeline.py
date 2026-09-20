"""One frame, end to end: prep -> DA3 1008 -> rewarp -> train 7k -> export (filtered).

This is the headline path.  There is no separate filtering step and nothing to remember to run
afterwards: when `run_frame` returns, a FILTERED `surface.npz` is on disk, complete, with a
`_DONE.json` marker beside it.

EVERY tunable is passed explicitly at every call site.  That is deliberate.  In the source project
the exporter's argparse defaults were changed and it silently altered the output of an unrelated
analysis script that had been relying on those defaults (`bin/reexport_minviews.sh` had to be given
`--nn-max-mm 0` after the fact to stay raw).  Nothing here relies on a default it did not name.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from . import atomicio, paths
from .stages import da3_prior, prep as prep_stage


@dataclass
class Recipe:
    """The settled ORH recipe. Changing any of these changes the output; none is cosmetic."""
    resolution: int = 2                 # AmbiSuR -r
    iterations: int = 7000              # settled by the user; NOT a knob to sweep here
    # --- export filter (the 2026-09-20 default) ---
    consistency_mm: float = 5.0
    min_views: int = 2
    nn_k: int = 5
    nn_max_mm: float = 5.0
    alpha_min: float = 0.5
    voxel_mm: float = 1.0
    # --- AmbiSuR losses (DTU defaults, validated in envs/ambisur.report.md) ---
    ncc_scale: float = 0.5
    depth_weight: float = 0.1
    sh_unc_lower_max: float = 0.2
    multi_view_num: int = 8
    multi_view_max_angle: int = 30      # INT: train.py rejects "30.0"
    multi_view_max_dis: float = 1.5     # SCENE UNITS = metres here
    multi_view_min_dis: float = 0.01
    # --- DA3 prior ---
    da3_model: str = da3_prior.DA3_MODEL
    process_res: int = da3_prior.PROCESS_RES
    group_size: int = da3_prior.GROUP_SIZE
    group_overlap: int = da3_prior.GROUP_OVERLAP
    max_points_for_colmap: int = da3_prior.MAX_POINTS
    write_ply: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def warmup_from_iter(self) -> int:
        """Warmups at 0.4x the budget -- the ratio used for every measured run (7000 -> 2800)."""
        return int(self.iterations * 0.4)

    def hash(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class StageTimer:
    def __init__(self):
        self.stages: list[dict] = []

    def run(self, name: str, fn, **meta):
        t0 = time.time()
        out = fn()
        self.stages.append(dict(name=name, wall_s=round(time.time() - t0, 2), **meta))
        return out


def _check(cmd: list, *, cwd: Path, env: dict, log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"{label} failed (rc={rc}); last lines of {log_path}:\n{tail}")


def _run_da3_subprocess(manifest_path: Path, serials, prep_info, work_dir: Path,
                        recipe: "Recipe", env: dict, logs: Path, log) -> dict:
    """Run the DA3 stage in its OWN interpreter (torch 2.6.0+cu124).

    Not an in-process import: paths.da3_python() documents the measurement showing that torch
    2.7.1+cu128 produces materially different depths from the pinned 2.6.0+cu124, ~900x the
    run-to-run nondeterminism.  The AmbiSuR env cannot be used for this stage.
    """
    da3_py = paths.da3_python()
    if not da3_py.is_file():
        raise RuntimeError(
            f"DA3 interpreter not found at {da3_py}.\n"
            f"  Run install.sh (it creates env-da3), or set ORHSURF_DA3_PYTHON.\n"
            f"  It must be a torch 2.6.0+cu124 env -- see orhsurf/paths.py::da3_python.")
    spec_p, res_p = work_dir / "da3_spec.json", work_dir / "da3_result.json"
    spec_p.parent.mkdir(parents=True, exist_ok=True)
    spec_p.write_text(json.dumps(dict(
        manifest=str(manifest_path), serials=list(serials),
        undist={s: str(prep_info[s]["undist"]) for s in serials},
        mask={s: str(prep_info[s]["mask"]) for s in serials},
        out_dir=str(work_dir / "native/da3"),
        model_id=recipe.da3_model, process_res=recipe.process_res,
        group_size=recipe.group_size, group_overlap=recipe.group_overlap)))
    da3_env = dict(env)
    # The DA3 env has its own site-packages; do not leak the AmbiSuR repo onto its PYTHONPATH.
    da3_env["PYTHONPATH"] = str(paths.REPO_ROOT)
    # A group of 18 views at 1008 peaks ~24,090 MiB, and a 24 GB card exposes only 23.53 GiB. With
    # the default caching allocator, fragmentation reserved a further ~1.9 GiB and the run OOMed on
    # an OTHERWISE EMPTY card. expandable_segments lets the allocator grow one segment instead of
    # stranding blocks, which is what makes this fit at all. It changes allocation strategy only,
    # not arithmetic, so the depths are unaffected.
    da3_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _check([da3_py, "-m", "orhsurf.stages._da3_worker", spec_p, res_p],
           cwd=paths.REPO_ROOT, env=da3_env, log_path=logs / "da3.log",
           label="DA3 1008 prior")
    meta = json.loads(res_p.read_text())
    log(f"[da3] {len(meta['per_view'])} views written "
        f"(peak {max(g['torch_peak_mib'] for g in meta['groups'])} MiB)")
    return meta


def run_frame(manifest_path: Path, frame_index: int, out_dir: Path, work_dir: Path,
              recipe: Recipe, *, gpu: int = 0, cpus: int = 8, log=print) -> dict:
    """Reconstruct ONE frame. Returns the provenance dict that was written into the output.

    `out_dir`   final per-frame directory (created atomically at the very end)
    `work_dir`  scratch for this frame: undistorted images, COLMAP workspace, scene, model.
                Large (~GBs) and safe to delete once the frame is done.
    """
    import numpy as np

    manifest_path, out_dir, work_dir = Path(manifest_path), Path(out_dir), Path(work_dir)
    env = paths.subprocess_env(gpu=gpu, cpus=cpus)
    env["ORHSURF_WRITE_PLY"] = "1" if recipe.write_ply else "0"
    repo = paths.ambisur_repo()
    py = paths.python_bin()
    logs = work_dir / "logs"
    tl = StageTimer()

    import sys
    sys.path.insert(0, str(paths.vendor_dir()))
    from colmap_dataset import build_dataset, load_manifest, select_serials   # noqa: E402
    sys.path.insert(0, str(repo))
    from s0_run import build_scene                                            # noqa: E402

    man = load_manifest(manifest_path)
    serials, heldout = select_serials(man, ())
    log(f"[frame {frame_index:05d}] {len(serials)} cameras, gpu {gpu}, {cpus} cpus")

    # ---- 1. undistorted views + masks (DA3's input pinhole) --------------------------------
    prep_info = tl.run("prep", lambda: prep_stage.prep_views(
        man, serials, frame_index, work_dir / "native", workers=cpus, log=log))

    # ---- 2. COLMAP workspace + AmbiSuR scene ------------------------------------------------
    ds = tl.run("dataset_build", lambda: build_dataset(
        manifest_path, frame_index, work_dir / "native/dataset", (), force=False))
    scene = work_dir / "native/scene"
    tl.run("scene_build", lambda: build_scene(ds, man, serials, scene, None))

    # ---- 3. DA3 @ 1008 with our poses, then rewarp onto the scene pinhole -------------------
    da3_meta = tl.run("da3_1008", lambda: _run_da3_subprocess(
        manifest_path, serials, prep_info, work_dir, recipe, env, logs, log))
    rewarp_meta = tl.run("rewarp", lambda: da3_prior.rewarp(
        scene, work_dir / "native/da3",
        target_w=recipe.process_res, conf_percent=da3_prior.CONF_PERCENT,
        max_points=recipe.max_points_for_colmap, seed=da3_prior.SEED, log=log))

    # ---- 4. AmbiSuR training ----------------------------------------------------------------
    model = work_dir / "native/model"
    model.mkdir(parents=True, exist_ok=True)
    w = recipe.warmup_from_iter
    tl.run("train", lambda: _check(
        [py, repo / "train.py", "-s", scene, "-m", model,
         "-r", recipe.resolution, "--ncc_scale", recipe.ncc_scale,
         "--depth_weight", recipe.depth_weight, "--sh_unc_lower_max", recipe.sh_unc_lower_max,
         "--iterations", recipe.iterations,
         "--single_view_weight_from_iter", w, "--multi_view_weight_from_iter", w,
         "--unc_from_iter", w,
         "--multi_view_num", recipe.multi_view_num,
         "--multi_view_max_angle", recipe.multi_view_max_angle,
         "--multi_view_max_dis", recipe.multi_view_max_dis,
         "--multi_view_min_dis", recipe.multi_view_min_dis,
         "--test_iterations", recipe.iterations, "--save_iterations", recipe.iterations],
        cwd=repo, env=env, log_path=logs / "train.log", label="AmbiSuR train.py"),
        iterations=recipe.iterations, resolution=recipe.resolution)

    # ---- 5. export + filter, into a staging dir that is swapped in only when complete -------
    with atomicio.FrameStage(out_dir) as st:
        tl.run("export", lambda: _check(
            [py, paths.vendor_dir() / "export_surface.py", "-s", scene, "-m", model,
             "--iteration", recipe.iterations, "--out", st.path,
             # every filter parameter named explicitly -- never inherited from argparse defaults
             "--consistency-mm", recipe.consistency_mm,
             "--min-views", recipe.min_views,
             "--nn-k", recipe.nn_k,
             "--nn-max-mm", recipe.nn_max_mm,
             "--alpha-min", recipe.alpha_min,
             "--voxel-mm", recipe.voxel_mm,
             "--manifest", manifest_path,
             "--dataset-json", work_dir / "native/dataset/dataset.json"],
            cwd=repo, env=env, log_path=logs / "export.log", label="export_surface.py"),
            min_views=recipe.min_views, nn_k=recipe.nn_k, nn_max_mm=recipe.nn_max_mm)

        surf = json.loads((st.path / "surface_export.json").read_text())
        with np.load(st.path / "surface.npz") as z:      # prove it before we call it done
            n = int(z["xyz"].shape[0])
        assert n == surf["n_points"], (n, surf["n_points"])

        prov = dict(
            orhsurf_version=1, frame_index=frame_index, recipe=asdict(recipe),
            recipe_hash=recipe.hash(), manifest=str(manifest_path),
            n_cameras=len(serials), heldout=list(heldout),
            n_points=n, stages=tl.stages,
            da3=da3_meta | {"rewarp": rewarp_meta},
            export=surf,
        )
        atomicio.atomic_write_json(st.path / "provenance.json", prov)
        st.record("surface.npz", n_points=n)
    log(f"[frame {frame_index:05d}] done: {n:,} points -> {out_dir}")
    return prov
