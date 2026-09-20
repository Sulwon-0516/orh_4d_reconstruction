"""Does replacing lane R's mask visual-hull centre with the rig centroid change the DA3 groups?

orhsurf/stages/da3_prior.py drops the mask-based visual hull that lanes_src/R/run.py:108-165 used
to locate the subject, and orders cameras by azimuth about the camera-rig centroid instead.  Only
the cyclic ORDER is consumed, so this is a no-op iff both centres produce the same order and hence
the same groups.  This script checks that against the real visual-hull box recorded by lane R.

  python tools/check_azimuth_order.py <manifest.json> <lane_R_metadata.json>
"""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from orhsurf.stages.da3_prior import azimuth_groups, scene_center

man = json.load(open(sys.argv[1]))
serials = sorted(man["valid_serials"])
cams = man["cameras"]
centers = {s: np.asarray(cams[s]["camera_center_world"], float) for s in serials}

rig = scene_center(centers, serials)
hull = np.asarray(json.load(open(sys.argv[2]))["visual_hull"]["center_world"], float)
print(f"rig centroid      {rig}")
print(f"visual-hull centre {hull}   (distance {np.linalg.norm(rig-hull):.4f} m)")

g_rig, o_rig = azimuth_groups(centers, serials, rig)
g_hull, o_hull = azimuth_groups(centers, serials, hull)
same_groups = g_rig == g_hull
same_owner = o_rig == o_hull
print(f"groups identical : {same_groups}  ({len(g_rig)} groups of {len(g_rig[0])})")
print(f"owners identical : {same_owner}")
if not same_groups:
    for i, (a, b) in enumerate(zip(g_rig, g_hull)):
        if a != b:
            print(f"  group {i} differs:\n    rig  {a}\n    hull {b}")
raise SystemExit(0 if (same_groups and same_owner) else 1)

# RESULT ON THE ORH CLIP 20260807_cand1_quant06__20260807_161358 (2026-09-20):
#   rig centroid       [ 0.0153  0.9823  1.9919]
#   visual-hull centre [-1.0524  2.0262  1.2500]   distance 1.667 m
#   groups identical : False   (all 4 groups differ)
# => the rig centroid is NOT a valid substitute for the visual hull. da3_prior.py keeps the hull.
