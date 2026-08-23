#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "hot3d/visualize-code/eval_metric.py",
    "hot3d/interaction_common.py",
    "hot3d/mano.py",
    "hot3d/rot.py",
    "hot3d/data_loaders/pytorch3d_rotation/rotation_conversions.py",
    "mano.yaml",
    "models/MANO_LEFT.pkl",
    "models/MANO_RIGHT.pkl",
    "data/obj.pkl",
    "inputs/results_test_hot3d_seen_fair_unique5_seeds0-9_gaze2hoi.pkl",
    "inputs/results_test_hot3d_unseen_fair_unique5_seeds0-9_gaze2hoi.pkl",
]

missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
mesh_count = len(list((ROOT / "data/object_mesh").glob("*.ply")))

if missing or mesh_count != 32:
    for relative in missing:
        print(f"MISSING: {relative}", file=sys.stderr)
    if mesh_count != 32:
        print(f"INVALID: expected 32 object meshes, found {mesh_count}", file=sys.stderr)
    raise SystemExit(1)

print(f"Package OK: {len(REQUIRED)} required files, {mesh_count} object meshes")

