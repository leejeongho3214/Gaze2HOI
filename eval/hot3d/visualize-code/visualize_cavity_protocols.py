#!/usr/bin/env python3
"""Compare Current, DiffH2O, and LatentHOI penetration metrics in Rerun.

The tool scans cavity-prone objects, selects the frame with the largest number
of hand vertices removed by the Current evaluator's +Y open-cavity heuristic,
and logs the same geometry under three protocol branches.  DiffH2O and
LatentHOI deliberately produce identical geometry here because their released
ID/IV implementations are effectively the same.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import rerun as rr
import torch
import trimesh

import eval_metric as base
import eval_metric_gpu as gpu


DEFAULT_CANDIDATES = (
    "bowl",
    "mug_white",
    "mug_patterned",
    "coffee_pot",
    "flask",
    "vase",
    "holder_black",
    "holder_gray",
    "bottle_bbq",
    "bottle_ranch",
    "bottle_mustard",
)


@dataclass
class FrameCase:
    object_name: str
    text: str
    sample_idx: int
    frame_idx: int
    frames_total: int
    object_mesh: trimesh.Trimesh
    hands: list[tuple[str, np.ndarray, np.ndarray]]
    raw_inside: list[np.ndarray]
    filtered_inside: list[np.ndarray]

    @property
    def removed_count(self) -> int:
        return int(
            sum(
                np.count_nonzero(raw & ~filtered)
                for raw, filtered in zip(self.raw_inside, self.filtered_inside)
            )
        )

    @property
    def raw_inside_count(self) -> int:
        return int(sum(np.count_nonzero(mask) for mask in self.raw_inside))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Result pickle to inspect.")
    parser.add_argument(
        "--obj-pkl",
        default=os.path.join(base.PACKAGE_ROOT, "data", "obj.pkl"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Candidate object name; repeat to override the default hollow objects.",
    )
    parser.add_argument(
        "--object",
        action="append",
        dest="objects",
        help="Visualize these exact objects instead of selecting top-k automatically.",
    )
    parser.add_argument(
        "--rrd",
        default="reports/metric_comparison/cavity_protocols.rerun.rrd",
        help="Recording output. Pass an empty string to disable saving.",
    )
    parser.add_argument("--spawn", action="store_true", help="Open the Rerun viewer.")
    parser.add_argument("--gpu-query-chunk", type=int, default=512)
    parser.add_argument("--gpu-face-chunk", type=int, default=4096)
    return parser.parse_args()


def _setup_gpu(args: argparse.Namespace) -> None:
    gpu.DEVICE = torch.device(args.device)
    gpu.DTYPE = torch.float32
    gpu.QUERY_CHUNK = max(1, int(args.gpu_query_chunk))
    gpu.FACE_CHUNK = max(1, int(args.gpu_face_chunk))
    gpu.ALLOW_CPU_FALLBACK = True


def _load_metric_meshes(
    obj_pkl: str,
    wanted: set[str],
) -> tuple[base.ObjectModel, dict[str, trimesh.Trimesh]]:
    model = base.ObjectModel(obj_pkl)
    meshes = {}
    for source_key, point_cloud in model.obj_pcs.items():
        key = str(source_key).lower()
        if key not in wanted:
            continue
        mesh = base._load_object_mesh(
            obj_pkl,
            model.obj_path.get(source_key),
            np.asarray(point_cloud, dtype=np.float32),
            object_key=key,
        )
        if mesh is not None:
            meshes[key] = mesh
    return model, meshes


def _reconstruct_hands(sample: dict, left_layer, right_layer, nframes: int):
    text = str(sample.get("text", ""))
    use_left, use_right = base._selected_hands(text)
    output = []
    for name, enabled, layer, key in (
        ("left", use_left, left_layer, "lhand_params"),
        ("right", use_right, right_layer, "rhand_params"),
    ):
        if not enabled:
            continue
        vertices, _joints, faces = base.process_hand_result(
            layer,
            base._slice_frame_indices(sample[key], np.arange(nframes)),
        )
        output.append((name, base._to_numpy(vertices), base._to_numpy(faces)))
    return output


def _frame_geometry(sample: dict, source_mesh, hand_sequences, frame_idx: int):
    obj_params = base._slice_frame_indices(
        sample["obj_params"], np.asarray([frame_idx], dtype=np.int64)
    )
    object_world = base._transform_object_mesh_to_world(source_mesh, obj_params)
    object_metric = base._transform_object_mesh_world_to_object_frame(
        object_world, obj_params
    )
    hands = [
        (
            name,
            base._transform_points_world_to_object_frame(sequence[frame_idx], obj_params),
            faces,
        )
        for name, sequence, faces in hand_sequences
    ]
    return object_metric, hands


def _inside_masks(object_mesh, hands):
    vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
    faces = np.asarray(object_mesh.faces, dtype=np.int64)
    raw_masks = []
    filtered_masks = []
    for _name, hand_vertices, _hand_faces in hands:
        raw = gpu._mesh_contains(vertices, faces, hand_vertices)
        filtered = gpu._remove_open_cavity_gpu(
            vertices, faces, hand_vertices, raw
        )
        raw_masks.append(raw)
        filtered_masks.append(filtered)
    return raw_masks, filtered_masks


def _find_cases(input_path: str, meshes: dict[str, trimesh.Trimesh]):
    left_layer = base.build_mano_aa(is_rhand=False, flat_hand=False)
    right_layer = base.build_mano_aa(is_rhand=True, flat_hand=False)
    best_by_object: dict[str, FrameCase] = {}
    sample_idx = -1

    for record in base._load_items_from_path(input_path):
        for sample in base._iter_samples_from_record(record):
            sample_idx += 1
            text = str(sample.get("text", ""))
            object_name = str(
                sample.get("object") or base._extract_object_key(text)
            ).lower()
            source_mesh = meshes.get(object_name)
            if source_mesh is None:
                continue
            nframes = base._sequence_length(sample["obj_params"])
            use_left, use_right = base._selected_hands(text)
            if use_left:
                nframes = min(nframes, base._sequence_length(sample["lhand_params"]))
            if use_right:
                nframes = min(nframes, base._sequence_length(sample["rhand_params"]))
            if nframes <= 0:
                continue
            hand_sequences = _reconstruct_hands(
                sample, left_layer, right_layer, nframes
            )
            for frame_idx in range(nframes):
                object_mesh, hands = _frame_geometry(
                    sample, source_mesh, hand_sequences, frame_idx
                )
                raw, filtered = _inside_masks(object_mesh, hands)
                candidate = FrameCase(
                    object_name=object_name,
                    text=text,
                    sample_idx=sample_idx,
                    frame_idx=frame_idx,
                    frames_total=nframes,
                    object_mesh=object_mesh.copy(),
                    hands=[
                        (name, vertices.copy(), np.asarray(faces).copy())
                        for name, vertices, faces in hands
                    ],
                    raw_inside=[mask.copy() for mask in raw],
                    filtered_inside=[mask.copy() for mask in filtered],
                )
                previous = best_by_object.get(object_name)
                score = (candidate.removed_count, candidate.raw_inside_count)
                previous_score = (
                    (previous.removed_count, previous.raw_inside_count)
                    if previous is not None
                    else (-1, -1)
                )
                if score > previous_score:
                    best_by_object[object_name] = candidate
    return best_by_object


def _released_metric(object_mesh, hand_vertices, hand_faces, hand_name: str):
    """Geometry used by the released DiffH2O and LatentHOI ID/IV code."""
    object_vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
    object_faces = np.asarray(object_mesh.faces, dtype=np.int64)
    hand_vertices = np.asarray(hand_vertices, dtype=np.float64)
    raw_inside = gpu._mesh_contains(object_vertices, object_faces, hand_vertices)
    closest, distances, _face_ids = gpu._mesh_closest_point(
        object_vertices, object_faces, hand_vertices
    )
    inside_indices = np.flatnonzero(raw_inside)
    inside_points = hand_vertices[inside_indices]
    closest_inside = closest[inside_indices]
    inside_distances = distances[inside_indices]
    iv_points, iv_m3 = gpu._gpu_voxel_intersection_volume(
        object_mesh,
        hand_vertices,
        hand_faces,
        pitch_m=base.IV_VOXEL_PITCH_M,
        enabled=inside_indices.size > 0,
    )
    id_mm = float(np.max(inside_distances) * 1000.0) if inside_distances.size else 0.0
    return {
        "hand": hand_name,
        "inside_mask": raw_inside,
        "inside_points": inside_points,
        "closest_points": closest_inside,
        "id_line_hand_points": inside_points,
        "id_line_object_points": closest_inside,
        "id_distances_m": inside_distances,
        "id_max_mm": id_mm,
        "inside_count": int(inside_indices.size),
        "iv_points": iv_points,
        "iv_volume_cm3": float(iv_m3 * 1e6),
    }


def _current_metric(object_mesh, hand_vertices, hand_faces, hand_name: str):
    base.QUICK_EVAL_ACTIVE = False
    return gpu._gpu_inside_vertex_metric(
        object_mesh, hand_vertices, hand_faces, hand_name
    )


def _offset_geometry(points, offset):
    points = np.asarray(points, dtype=np.float32)
    return points + np.asarray(offset, dtype=np.float32).reshape(1, 3)


def _log_protocol(
    root: str,
    case: FrameCase,
    protocol: str,
    offset: np.ndarray,
):
    object_vertices = _offset_geometry(case.object_mesh.vertices, offset)
    object_faces = np.asarray(case.object_mesh.faces, dtype=np.int32)
    rr.log(
        f"{root}/object",
        rr.Mesh3D(
            vertex_positions=object_vertices,
            triangle_indices=object_faces,
            vertex_colors=np.tile(
                np.asarray([150, 150, 150], dtype=np.uint8),
                (object_vertices.shape[0], 1),
            ),
        ),
        static=True,
    )

    id_values = []
    iv_values = []
    inside_counts = []
    for hand_idx, (hand_name, hand_vertices, hand_faces) in enumerate(case.hands):
        metric = (
            _current_metric(case.object_mesh, hand_vertices, hand_faces, hand_name)
            if protocol == "current"
            else _released_metric(
                case.object_mesh, hand_vertices, hand_faces, hand_name
            )
        )
        shifted_hand = _offset_geometry(hand_vertices, offset)
        inside_mask = np.asarray(metric["inside_mask"], dtype=bool)
        colors = np.tile(
            np.asarray([70, 190, 230], dtype=np.uint8),
            (shifted_hand.shape[0], 1),
        )
        colors[inside_mask] = np.asarray([255, 45, 45], dtype=np.uint8)
        rr.log(
            f"{root}/{hand_name}/mesh",
            rr.Mesh3D(
                vertex_positions=shifted_hand,
                triangle_indices=np.asarray(hand_faces, dtype=np.int32),
                vertex_colors=colors,
            ),
            static=True,
        )
        if np.any(inside_mask):
            rr.log(
                f"{root}/{hand_name}/inside_vertices",
                rr.Points3D(
                    positions=shifted_hand[inside_mask],
                    radii=0.002,
                    colors=[255, 0, 0],
                    labels=["penetrating hand vertex"] * int(np.count_nonzero(inside_mask)),
                    show_labels=False,
                ),
                static=True,
            )

        hand_line = np.asarray(metric.get("id_line_hand_points"), dtype=np.float32)
        object_line = np.asarray(metric.get("id_line_object_points"), dtype=np.float32)
        distances = np.asarray(metric.get("id_distances_m"), dtype=np.float32)
        if hand_line.shape == object_line.shape and hand_line.shape[0] > 0:
            segments = np.stack(
                [_offset_geometry(hand_line, offset), _offset_geometry(object_line, offset)],
                axis=1,
            )
            labels = [f"ID {value * 1000.0:.3f} mm" for value in distances]
            rr.log(
                f"{root}/{hand_name}/id_lines",
                rr.LineStrips3D(
                    segments,
                    radii=0.0007,
                    colors=[0, 255, 0],
                    labels=labels,
                    show_labels=False,
                ),
                static=True,
            )
            deepest = int(np.argmax(distances))
            rr.log(
                f"{root}/{hand_name}/deepest_id",
                rr.LineStrips3D(
                    segments[deepest : deepest + 1],
                    radii=0.0018,
                    colors=[255, 255, 0],
                    labels=[labels[deepest]],
                    show_labels=True,
                ),
                static=True,
            )

        iv_points = np.asarray(metric.get("iv_points"), dtype=np.float32)
        if iv_points.ndim == 2 and iv_points.shape[0] > 0:
            rr.log(
                f"{root}/{hand_name}/iv_voxel_centers",
                rr.Points3D(
                    positions=_offset_geometry(iv_points, offset),
                    radii=0.0015,
                    colors=[190, 70, 255],
                    labels=["5 mm object voxel center inside hand"] * iv_points.shape[0],
                    show_labels=False,
                ),
                static=True,
            )

        if protocol == "current":
            raw = case.raw_inside[hand_idx]
            filtered = case.filtered_inside[hand_idx]
            removed = raw & ~filtered
            if np.any(removed):
                removed_points = hand_vertices[removed]
                rr.log(
                    f"{root}/{hand_name}/cavity_removed_magenta",
                    rr.Points3D(
                        positions=_offset_geometry(removed_points, offset),
                        radii=0.003,
                        colors=[255, 0, 255],
                        labels=["removed by +Y cavity filter"] * removed_points.shape[0],
                        show_labels=True,
                    ),
                    static=True,
                )
                extent = max(float(np.ptp(case.object_mesh.vertices[:, 1])), 0.05)
                ends = removed_points.copy()
                ends[:, 1] += extent * 0.4
                rr.log(
                    f"{root}/{hand_name}/cavity_escape_rays",
                    rr.LineStrips3D(
                        np.stack(
                            [
                                _offset_geometry(removed_points, offset),
                                _offset_geometry(ends, offset),
                            ],
                            axis=1,
                        ),
                        radii=0.0008,
                        colors=[255, 0, 255],
                    ),
                    static=True,
                )

        id_values.append(float(metric.get("id_max_mm", 0.0)))
        iv_values.append(float(metric.get("iv_volume_cm3", 0.0)))
        inside_counts.append(int(metric.get("inside_count", 0)))

    center = np.asarray(case.object_mesh.vertices).mean(axis=0) + offset
    span = max(float(np.ptp(case.object_mesh.vertices, axis=0).max()), 0.05)
    label_pos = center + np.asarray([0.0, span * 0.8, 0.0])
    protocol_note = (
        "2 mm filter + open-cavity filter + seed-face distance"
        if protocol == "current"
        else "released code: raw ray-parity + full-surface distance; no 2 mm/cavity filter"
    )
    label = "\n".join(
        [
            protocol.upper(),
            protocol_note,
            f"frame ID max = {max(id_values, default=0.0):.4f} mm",
            f"IV sum = {sum(iv_values):.4f} cm^3",
            f"inside vertices = {sum(inside_counts)}",
        ]
    )
    rr.log(
        f"{root}/metric_label",
        rr.Points3D(
            positions=[label_pos],
            radii=0.006,
            colors=[255, 255, 255],
            labels=[label],
            show_labels=True,
        ),
        static=True,
    )


def _log_cases(cases: list[FrameCase]) -> None:
    rr.log(
        "cavity_protocol_comparison/README",
        rr.TextDocument(
            "# Cavity protocol comparison\n\n"
            "Each row is one actual output frame. Columns are Current, DiffH2O, "
            "and LatentHOI. Red = inside hand vertices; green = all ID segments; "
            "yellow = deepest ID; purple = IV voxel centers; magenta = vertices "
            "removed only by Current's +Y open-cavity filter.\n\n"
            "DiffH2O and LatentHOI columns are expected to be identical because "
            "their released ID/IV evaluator code is effectively identical."
        ),
        static=True,
    )
    for row_idx, case in enumerate(cases):
        object_span = max(
            float(np.ptp(case.object_mesh.vertices, axis=0).max()), 0.12
        )
        x_spacing = object_span * 2.2
        z_spacing = object_span * 2.2
        row_root = (
            f"cavity_protocol_comparison/{row_idx + 1:02d}_{case.object_name}"
        )
        for column_idx, protocol in enumerate(
            ("current", "diffh2o", "latenthoi")
        ):
            offset = np.asarray(
                [column_idx * x_spacing, 0.0, -row_idx * z_spacing],
                dtype=np.float32,
            )
            _log_protocol(
                f"{row_root}/{protocol}", case, protocol, offset
            )
        rr.log(
            f"{row_root}/sample_label",
            rr.TextDocument(
                f"## {case.object_name}\n"
                f"sample `{case.sample_idx}`, frame `{case.frame_idx}/{case.frames_total - 1}`\n\n"
                f"`{case.text}`\n\n"
                f"raw inside `{case.raw_inside_count}`, cavity removed `{case.removed_count}`"
            ),
            static=True,
        )


def main() -> None:
    args = _args()
    _setup_gpu(args)
    candidates = {
        str(item).lower()
        for item in (args.candidates if args.candidates else DEFAULT_CANDIDATES)
    }
    if args.objects:
        candidates.update(str(item).lower() for item in args.objects)
    obj_pkl = os.path.abspath(os.path.expanduser(args.obj_pkl))
    input_path = os.path.abspath(os.path.expanduser(args.input))
    _model, meshes = _load_metric_meshes(obj_pkl, candidates)
    missing = sorted(candidates - set(meshes))
    if missing:
        print(f"[WARN] unavailable candidate meshes: {', '.join(missing)}")
    cases_by_object = _find_cases(input_path, meshes)
    if args.objects:
        requested = [str(item).lower() for item in args.objects]
        cases = [cases_by_object[name] for name in requested if name in cases_by_object]
    else:
        cases = sorted(
            cases_by_object.values(),
            key=lambda case: (case.removed_count, case.raw_inside_count),
            reverse=True,
        )[: max(1, int(args.top_k))]
    if not cases:
        raise RuntimeError("No matching object samples were found in the input.")
    print("[CAVITY] selected cases:")
    for case in cases:
        current_metrics = [
            _current_metric(case.object_mesh, vertices, faces, name)
            for name, vertices, faces in case.hands
        ]
        released_metrics = [
            _released_metric(case.object_mesh, vertices, faces, name)
            for name, vertices, faces in case.hands
        ]
        current_id = max(
            (float(metric.get("id_max_mm", 0.0)) for metric in current_metrics),
            default=0.0,
        )
        released_id = max(
            (float(metric.get("id_max_mm", 0.0)) for metric in released_metrics),
            default=0.0,
        )
        current_iv = sum(
            float(metric.get("iv_volume_cm3", 0.0)) for metric in current_metrics
        )
        released_iv = sum(
            float(metric.get("iv_volume_cm3", 0.0)) for metric in released_metrics
        )
        print(
            f"  {case.object_name}: sample={case.sample_idx} "
            f"frame={case.frame_idx}/{case.frames_total - 1} "
            f"raw_inside={case.raw_inside_count} removed={case.removed_count}\n"
            f"    Current  ID={current_id:.4f} mm IV={current_iv:.4f} cm^3\n"
            f"    DiffH2O  ID={released_id:.4f} mm IV={released_iv:.4f} cm^3\n"
            f"    LatentHOI ID={released_id:.4f} mm IV={released_iv:.4f} cm^3 "
            "(released evaluator is identical)"
        )

    rr.init("Text2HOI cavity protocol comparison", spawn=False)
    if args.rrd:
        output_path = os.path.abspath(os.path.expanduser(args.rrd))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        rr.save(output_path)
    _log_cases(cases)
    if args.spawn:
        rr.spawn()
    if args.rrd:
        print(f"[CAVITY] saved Rerun recording: {output_path}")
    if args.spawn:
        input("Press Enter to close the Rerun comparison... ")


if __name__ == "__main__":
    main()
