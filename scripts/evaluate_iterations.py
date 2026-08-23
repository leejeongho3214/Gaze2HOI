#!/usr/bin/env python3
"""Evaluate Gaze2HOI iteration checkpoints with the official grounding metrics."""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))


def _optional_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def summarize_official_metrics(csv_path, gsr_contact_frames, gsr_lift_cm):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_sample = defaultdict(list)
    for row in rows:
        key = (
            row.get("file_name", ""),
            row.get("sample_idx", row.get("batch_sample_idx", "")),
            row.get("seed", ""),
        )
        by_sample[key].append(row)

    final_rows = []
    gsr_values = []
    for sample_rows in by_sample.values():
        ordered = sorted(sample_rows, key=lambda row: int(row["frame_idx"]))
        if not ordered:
            continue
        final_row = ordered[-1]
        final_rows.append(final_row)
        if len(ordered) < gsr_contact_frames:
            gsr_values.append(0.0)
            continue
        sustained_contact = all(
            row.get("Con_pass") == "1"
            for row in ordered[-gsr_contact_frames:]
        )
        final_id_mm = _optional_float(final_row.get("ID_max_mm"))
        final_lift_m = _optional_float(final_row.get("Object_lift_y_m"))
        gsr_values.append(
            float(
                sustained_contact
                and final_id_mm is not None
                and final_id_mm <= 10.0
                and final_lift_m is not None
                and final_lift_m >= gsr_lift_cm / 100.0
            )
        )

    def mean_column(name, scale=1.0):
        values = [
            value
            for value in (
                _optional_float(row.get(name)) for row in final_rows
            )
            if value is not None
        ]
        return float(np.mean(values) * scale) if values else None

    return {
        "GSR_percent": float(np.mean(gsr_values) * 100.0)
        if gsr_values
        else None,
        "PartAcc_percent": mean_column("Part_Acc", 100.0),
        "PCP_percent": mean_column("PCP", 100.0),
        "G2C_cm": mean_column("G2C_m", 100.0),
        "evaluated_samples": len(final_rows),
        "gsr_contact_frames": int(gsr_contact_frames),
        "gsr_lift_cm": float(gsr_lift_cm),
    }


def metric_rank(metrics):
    """Lexicographic paper-metric checkpoint rank."""
    return (
        metrics["GSR_percent"]
        if metrics["GSR_percent"] is not None
        else float("-inf"),
        metrics["PartAcc_percent"]
        if metrics["PartAcc_percent"] is not None
        else float("-inf"),
        metrics["PCP_percent"]
        if metrics["PCP_percent"] is not None
        else float("-inf"),
        -metrics["G2C_cm"]
        if metrics["G2C_cm"] is not None
        else float("-inf"),
    )


def run_command(command, env):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--official-test",
        action="store_true",
        help=(
            "Score best_metric_model.pth once on gaze_test. This mode never "
            "changes checkpoint selection."
        ),
    )
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--gsr-contact-frames", type=int, default=5)
    parser.add_argument("--gsr-lift-cm", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if args.official_test:
        best_checkpoint = experiment_dir / "best_metric_model.pth"
        checkpoints = [best_checkpoint] if best_checkpoint.exists() else []
    else:
        checkpoints = sorted(experiment_dir.glob("iteration_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            (
                f"No best_metric_model.pth found under {experiment_dir}"
                if args.official_test
                else f"No iteration_*.pth checkpoints found under {experiment_dir}"
            )
        )

    eval_root = experiment_dir / (
        "official_test_evaluation"
        if args.official_test
        else "metric_evaluation"
    )
    eval_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHON_BIN"] = str(args.python)

    all_results = []
    for checkpoint in checkpoints:
        checkpoint_state = torch.load(checkpoint, map_location="cpu")
        iteration = int(
            checkpoint_state.get(
                "global_step",
                checkpoint.stem.split("_")[-1],
            )
        )
        stem = (
            f"best_metric_iteration_{iteration:07d}"
            if args.official_test
            else f"iteration_{iteration:07d}"
        )
        prediction_base = eval_root / stem
        prediction_path = prediction_base.with_suffix(".pkl")
        csv_path = eval_root / f"{stem}_metrics.csv"
        markdown_path = eval_root / f"{stem}_metrics.md"
        json_path = eval_root / f"{stem}_metrics.json"
        dataset_overrides = []
        evaluation_split = "gaze_test" if args.official_test else "gaze_train_validation"
        if args.official_test:
            dataset_overrides.append("dataset.data_name=ori_dataset/gaze_test")
        else:
            validation_indices = checkpoint_state.get("validation_indices")
            if not validation_indices:
                raise RuntimeError(
                    f"{checkpoint} has no validation_indices. Use --official-test "
                    "only for final reporting, or retrain with the new split logic."
                )
            indices_path = eval_root / f"{stem}_validation_indices.json"
            with open(indices_path, "w", encoding="utf-8") as handle:
                json.dump([int(index) for index in validation_indices], handle)
            dataset_overrides.extend(
                [
                    "dataset.data_name=ori_dataset/gaze_train",
                    f"gaze2hoi.exp.eval_dataset_indices_path={indices_path}",
                ]
            )
        del checkpoint_state

        if args.force or not prediction_path.exists():
            run_command(
                [
                    args.python,
                    "gaze2hoi/test.py",
                    f"gaze2hoi.exp.weight_path={checkpoint}",
                    f"gaze2hoi.exp.save_name={prediction_base}",
                    f"gaze2hoi.exp.num_test_seeds={args.num_seeds}",
                    f"gaze2hoi.exp.max_test_batches={args.max_test_batches}",
                    *dataset_overrides,
                ],
                env,
            )
        if args.force or not csv_path.exists():
            run_command(
                [
                    "eval/run_eval_gpu.sh",
                    "--device",
                    "cuda:0",
                    "--quick",
                    "--input",
                    prediction_path,
                    "--part-labels",
                    PROJECT_ROOT / "assets" / "label_merged_3parts.json",
                    "--new-metric-csv-output",
                    csv_path,
                    "--new-metric-md-output",
                    markdown_path,
                    "--gsr-contact-frames",
                    args.gsr_contact_frames,
                    "--gsr-lift-cm",
                    args.gsr_lift_cm,
                ],
                env,
            )
        metrics = summarize_official_metrics(
            csv_path,
            gsr_contact_frames=args.gsr_contact_frames,
            gsr_lift_cm=args.gsr_lift_cm,
        )
        metrics.update(
            {
                "iteration": iteration,
                "checkpoint": str(checkpoint),
                "prediction": str(prediction_path),
                "csv": str(csv_path),
                "evaluation_split": evaluation_split,
            }
        )
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        all_results.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    if args.official_test:
        summary = {
            "evaluation_split": "gaze_test",
            "checkpoint_selection": "disabled to prevent test leakage",
            "results": all_results,
        }
        summary_path = experiment_dir / "official_test_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(
            f"Saved final test metrics to {summary_path}; "
            "best_metric_model.pth was not changed."
        )
        return

    best = max(all_results, key=metric_rank)
    best_path = experiment_dir / "best_metric_model.pth"
    shutil.copy2(best["checkpoint"], best_path)
    summary = {
        "selection_policy": (
            "lexicographic: max GSR, max PartAcc, max PCP, min G2C"
        ),
        "evaluation_split": "held-out 10% of gaze_train",
        "best": best,
        "all_results": all_results,
    }
    with open(
        experiment_dir / "best_metric_model.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"Selected {best_path} from iteration {best['iteration']}.")


if __name__ == "__main__":
    main()
