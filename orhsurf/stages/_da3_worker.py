"""Internal: run the DA3 1008 stage inside the DA3 environment.

Invoked as a subprocess by orhsurf.pipeline because the DA3 stage is pinned to torch 2.6.0+cu124
while AmbiSuR runs on torch 2.7.1+cu128 -- see orhsurf/paths.py::da3_python for the measurement
that forced that split.  Communicates by JSON file so nothing depends on the two envs sharing an
object model.

    <da3-python> -m orhsurf.stages._da3_worker <spec.json> <result.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orhsurf.stages.da3_prior import run_da3_1008   # noqa: E402


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    manifest = json.loads(Path(spec["manifest"]).read_text())
    meta = run_da3_1008(
        manifest, spec["serials"],
        undist_png={s: Path(p) for s, p in spec["undist"].items()},
        mask_png={s: Path(p) for s, p in spec["mask"].items()},
        out_dir=Path(spec["out_dir"]),
        model_id=spec["model_id"], process_res=spec["process_res"],
        group_size=spec["group_size"], group_overlap=spec["group_overlap"],
        revision=spec.get("revision"),
        log=lambda m: print(m, flush=True))
    Path(sys.argv[2]).write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
