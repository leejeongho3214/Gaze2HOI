#!/usr/bin/env python3
"""GPU implementation of the expensive geometry in ``eval_metric.py``.

The original evaluator remains the source of truth for input parsing, object
transforms, aggregation, CSV writing, Markdown summaries (including the
sample-count-ordered per-target-part tables), and visualization.
This launcher replaces its CPU geometry kernels with chunked PyTorch CUDA
kernels:

* closest point from hand vertices to object triangles (ID);
* ray-parity point-in-mesh tests for hand/object penetration;
* point-in-hand tests for object voxels (IV);
* nearest object-vertex queries for CR and fingertip contact.

MANO is reconstructed on CUDA by default. Pass ``--no-gpu-mano`` to keep the
reference CPU implementation.

Usage is the same as eval_metric.py, with optional GPU launcher arguments:

    python eval_metric_gpu.py --device cuda:0 --input result_a.pkl result_b.pkl
    python eval_metric_gpu.py --input result_a.pkl --input result_b.pkl

The output columns and metric thresholds are unchanged.  Surface voxelization
itself is kept in trimesh because it defines the original 5 mm sampling grid;
the expensive containment of those voxels is performed on the GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import trimesh

import eval_metric as base


# Edit this list to choose the input pickle files evaluated by default.
# Paths are resolved relative to --input-dir (the same behavior as eval_metric.py).
DEFAULT_INPUT_FILES = []  # pass prediction pickles with --input

# Keep generated tables out of the repository root by default. Explicit
# --new-metric-*-output arguments still take precedence.
DEFAULT_RESULT_DIR = "result_md"


DEVICE = torch.device("cpu")
DTYPE = torch.float64
QUERY_CHUNK = 256
FACE_CHUNK = 2048
ALLOW_CPU_FALLBACK = True
_CPU_INSIDE_VERTEX_METRIC = base._new_inside_vertex_metric_for_hand
_CPU_VOXEL_INTERSECTION_VOLUME = base._voxel_intersection_volume_for_hand


def _launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--device",
        default="auto",
        help="GPU device: auto, cuda, cuda:0, ... (default: auto).",
    )
    parser.add_argument(
        "--input",
        dest="input_files",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Input pickle file(s). Pass one or more paths per option, or repeat "
            "--input. Paths are forwarded to eval_metric.py."
        ),
    )
    parser.add_argument(
        "--gpu-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Geometry precision (default: float32; float64 best matches CPU).",
    )
    parser.add_argument(
        "--gpu-query-chunk",
        type=int,
        default=256,
        help="Number of query points processed together (default: 256).",
    )
    parser.add_argument(
        "--gpu-face-chunk",
        type=int,
        default=2048,
        help="Number of triangles processed together (default: 2048).",
    )
    parser.add_argument(
        "--no-gpu-fallback",
        action="store_true",
        help="Raise a CUDA error instead of retrying a failed kernel on CPU.",
    )
    mano_group = parser.add_mutually_exclusive_group()
    mano_group.add_argument(
        "--gpu-mano",
        dest="gpu_mano",
        action="store_true",
        help="Reconstruct MANO on CUDA (default).",
    )
    mano_group.add_argument(
        "--no-gpu-mano",
        dest="gpu_mano",
        action="store_false",
        help="Keep MANO reconstruction on CPU for reference-compatible output.",
    )
    parser.set_defaults(gpu_mano=True)
    return parser.parse_known_args()


def _resolve_device(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run eval_metric.py for CPU evaluation."
            )
        return torch.device("cuda:0")
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("eval_metric_gpu.py requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    return device


def _add_default_output_paths(evaluator_argv: list[str]) -> list[str]:
    """Place regular evaluation outputs in ``DEFAULT_RESULT_DIR``."""
    modes_without_regular_output = (
        "--summary-from-csv",
        "--new-metric-visualize",
        "--new-metric-top10-visualize",
        "--new-metric-pairdiff-visualize",
        "--diversity-visualize",
    )
    if any(flag in evaluator_argv for flag in modes_without_regular_output):
        return evaluator_argv

    os.makedirs(DEFAULT_RESULT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = "quick_metrics" if "--quick" in evaluator_argv else "new_metrics"
    defaults = []
    if "--new-metric-csv-output" not in evaluator_argv:
        defaults.extend(
            [
                "--new-metric-csv-output",
                os.path.join(
                    DEFAULT_RESULT_DIR, f"{prefix}_per_frame_{timestamp}.csv"
                ),
            ]
        )
    if "--new-metric-md-output" not in evaluator_argv:
        defaults.extend(
            [
                "--new-metric-md-output",
                os.path.join(
                    DEFAULT_RESULT_DIR, f"{prefix}_summary_{timestamp}.md"
                ),
            ]
        )
    return [*defaults, *evaluator_argv]


def _tensor(array, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(array),
        dtype=DTYPE if dtype is None else dtype,
        device=DEVICE,
    )


def _closest_on_segment(
    points: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    edge = end - start
    denom = (edge * edge).sum(dim=-1).clamp_min(torch.finfo(DTYPE).eps)
    t = ((points - start) * edge).sum(dim=-1) / denom
    return start + t.clamp(0.0, 1.0).unsqueeze(-1) * edge


def _closest_points_to_triangles(
    points: torch.Tensor,
    triangles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest point for every point/triangle pair.

    Returns ``(closest, squared_distance)`` with shapes ``[P,F,3]`` and
    ``[P,F]``.  The computation considers the triangle plane interior and all
    three edges, which also handles degenerate triangles safely.
    """
    p = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]

    ab = b - a
    ac = c - a
    normal = torch.cross(ab, ac, dim=-1)
    normal_sq = (normal * normal).sum(dim=-1)
    signed_numerator = ((p - a) * normal).sum(dim=-1)
    projected = p - (
        signed_numerator / normal_sq.clamp_min(torch.finfo(DTYPE).eps)
    ).unsqueeze(-1) * normal

    v0 = ab
    v1 = ac
    v2 = projected - a
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0).sum(dim=-1)
    d21 = (v2 * v1).sum(dim=-1)
    bary_denom = d00 * d11 - d01 * d01
    bary_v = (d11 * d20 - d01 * d21) / bary_denom.clamp_min(
        torch.finfo(DTYPE).eps
    )
    bary_w = (d00 * d21 - d01 * d20) / bary_denom.clamp_min(
        torch.finfo(DTYPE).eps
    )
    bary_u = 1.0 - bary_v - bary_w
    plane_valid = (
        (normal_sq > torch.finfo(DTYPE).eps)
        & (bary_denom.abs() > torch.finfo(DTYPE).eps)
        & (bary_u >= 0.0)
        & (bary_v >= 0.0)
        & (bary_w >= 0.0)
    )

    candidates = torch.stack(
        (
            projected,
            _closest_on_segment(p, a, b),
            _closest_on_segment(p, b, c),
            _closest_on_segment(p, c, a),
        ),
        dim=2,
    )
    distances_sq = ((candidates - p.unsqueeze(2)) ** 2).sum(dim=-1)
    distances_sq[:, :, 0] = torch.where(
        plane_valid,
        distances_sq[:, :, 0],
        torch.full_like(distances_sq[:, :, 0], torch.inf),
    )
    choice = distances_sq.argmin(dim=2)
    closest = candidates.gather(
        2,
        choice[:, :, None, None].expand(-1, -1, 1, 3),
    ).squeeze(2)
    return closest, distances_sq.gather(2, choice.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def _mesh_closest_point(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_points: np.ndarray,
    face_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    points_np = np.asarray(query_points, dtype=np.float64)
    if face_indices is None:
        selected_ids = np.arange(faces_np.shape[0], dtype=np.int64)
    else:
        selected_ids = np.unique(np.asarray(face_indices, dtype=np.int64))
        selected_ids = selected_ids[
            (selected_ids >= 0) & (selected_ids < faces_np.shape[0])
        ]
    if points_np.shape[0] == 0 or selected_ids.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    vertices_t = _tensor(vertices_np)
    faces_t = _tensor(faces_np, dtype=torch.long)
    selected_t = _tensor(selected_ids, dtype=torch.long)
    triangles = vertices_t[faces_t[selected_t]]
    all_closest = []
    all_distances = []
    all_face_ids = []

    for query_start in range(0, points_np.shape[0], QUERY_CHUNK):
        points = _tensor(points_np[query_start : query_start + QUERY_CHUNK])
        best_sq = torch.full(
            (points.shape[0],), torch.inf, dtype=DTYPE, device=DEVICE
        )
        best_points = torch.zeros(
            (points.shape[0], 3), dtype=DTYPE, device=DEVICE
        )
        best_faces = torch.full(
            (points.shape[0],), -1, dtype=torch.long, device=DEVICE
        )
        for face_start in range(0, triangles.shape[0], FACE_CHUNK):
            triangle_chunk = triangles[face_start : face_start + FACE_CHUNK]
            closest, distances_sq = _closest_points_to_triangles(
                points, triangle_chunk
            )
            local_sq, local_idx = distances_sq.min(dim=1)
            update = local_sq < best_sq
            if torch.any(update):
                rows = torch.arange(points.shape[0], device=DEVICE)
                local_points = closest[rows, local_idx]
                best_sq = torch.where(update, local_sq, best_sq)
                best_points[update] = local_points[update]
                best_faces[update] = selected_t[face_start + local_idx[update]]
        all_closest.append(best_points.cpu())
        all_distances.append(best_sq.clamp_min(0.0).sqrt().cpu())
        all_face_ids.append(best_faces.cpu())

    return (
        torch.cat(all_closest).numpy().astype(np.float32),
        torch.cat(all_distances).numpy().astype(np.float32),
        torch.cat(all_face_ids).numpy().astype(np.int64),
    )


@torch.inference_mode()
def _ray_intersection_counts(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_points: np.ndarray,
    direction: tuple[float, float, float],
) -> np.ndarray:
    """Möller–Trumbore ray/triangle intersection counts on CUDA."""
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    points_np = np.asarray(query_points, dtype=np.float64)
    if points_np.shape[0] == 0 or faces_np.shape[0] == 0:
        return np.zeros((points_np.shape[0],), dtype=np.int64)

    vertices_t = _tensor(vertices_np)
    faces_t = _tensor(faces_np, dtype=torch.long)
    triangles = vertices_t[faces_t]
    ray_dir = torch.tensor(direction, dtype=DTYPE, device=DEVICE)
    ray_dir = ray_dir / torch.linalg.vector_norm(ray_dir)
    eps = 1e-7 if DTYPE == torch.float32 else 1e-12
    output = []

    for query_start in range(0, points_np.shape[0], QUERY_CHUNK):
        origins = _tensor(points_np[query_start : query_start + QUERY_CHUNK])
        counts = torch.zeros(
            (origins.shape[0],), dtype=torch.long, device=DEVICE
        )
        for face_start in range(0, triangles.shape[0], FACE_CHUNK):
            tri = triangles[face_start : face_start + FACE_CHUNK]
            v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = torch.cross(ray_dir.expand_as(edge2), edge2, dim=-1)
            determinant = (edge1 * h).sum(dim=-1)
            valid_det = determinant.abs() > eps
            inv_det = torch.where(
                valid_det, determinant.reciprocal(), torch.zeros_like(determinant)
            )
            s = origins[:, None, :] - v0[None, :, :]
            u = (s * h[None, :, :]).sum(dim=-1) * inv_det[None, :]
            q = torch.cross(s, edge1[None, :, :], dim=-1)
            v = (q * ray_dir[None, None, :]).sum(dim=-1) * inv_det[None, :]
            distance = (
                q * edge2[None, :, :]
            ).sum(dim=-1) * inv_det[None, :]
            hits = (
                valid_det[None, :]
                & (u >= -eps)
                & (v >= -eps)
                & ((u + v) <= 1.0 + eps)
                & (distance > eps)
            )
            counts += hits.sum(dim=1)
        output.append(counts.cpu())
    return torch.cat(output).numpy().astype(np.int64)


def _mesh_contains(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """Match trimesh's bidirectional ray-parity containment policy."""
    vertices = np.asarray(vertices, dtype=np.float64)
    points = np.asarray(query_points, dtype=np.float64)
    result = np.zeros((points.shape[0],), dtype=bool)
    if vertices.shape[0] == 0 or points.shape[0] == 0:
        return result

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    in_bounds = np.all((points >= bounds_min) & (points <= bounds_max), axis=1)
    if not np.any(in_bounds):
        return result

    # This is trimesh.ray.ray_util.contains_points' fixed default vector.
    direction = np.array(
        [0.4395064455, 0.617598629942, 0.652231566745],
        dtype=np.float64,
    )
    bounded_points = points[in_bounds]
    forward = _ray_intersection_counts(
        vertices, faces, bounded_points, tuple(direction)
    )
    backward = _ray_intersection_counts(
        vertices, faces, bounded_points, tuple(-direction)
    )
    forward_inside = (forward % 2) == 1
    backward_inside = (backward % 2) == 1
    agree = forward_inside == backward_inside
    bounded_result = np.zeros((bounded_points.shape[0],), dtype=bool)
    bounded_result[agree] = forward_inside[agree]

    # The original implementation retries only inconsistent rays for which
    # neither direction reaches free space. Use a fixed second vector here so
    # repeated GPU evaluations remain deterministic.
    broken = (~agree) & (forward > 0) & (backward > 0)
    if np.any(broken):
        retry_direction = np.array(
            [-0.25717225, 0.93276191, 0.25299132],
            dtype=np.float64,
        )
        retry_points = bounded_points[broken]
        retry_forward = _ray_intersection_counts(
            vertices, faces, retry_points, tuple(retry_direction)
        )
        retry_backward = _ray_intersection_counts(
            vertices, faces, retry_points, tuple(-retry_direction)
        )
        retry_inside = (retry_forward % 2) == 1
        retry_agree = retry_inside == ((retry_backward % 2) == 1)
        broken_values = np.zeros((retry_points.shape[0],), dtype=bool)
        broken_values[retry_agree] = retry_inside[retry_agree]
        bounded_result[broken] = broken_values

    result[in_bounds] = bounded_result
    return result


def _remove_open_cavity_gpu(
    vertices: np.ndarray,
    faces: np.ndarray,
    query_points: np.ndarray,
    inside: np.ndarray,
) -> np.ndarray:
    inside = np.asarray(inside, dtype=bool).copy()
    candidates = np.flatnonzero(inside)
    if candidates.size == 0:
        return inside
    origins = np.asarray(query_points, dtype=np.float64)[candidates].copy()
    origins[:, 1] += 1e-5
    hit_up = (
        _ray_intersection_counts(vertices, faces, origins, (0.0, 1.0, 0.0)) > 0
    )
    inside[candidates[~hit_up]] = False
    return inside


@torch.inference_mode()
def _nearest_vertex_distances(
    query_points: np.ndarray,
    reference_vertices: np.ndarray,
) -> np.ndarray:
    queries_np = np.asarray(query_points, dtype=np.float64)
    reference_np = np.asarray(reference_vertices, dtype=np.float64)
    if queries_np.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    reference = _tensor(reference_np)
    chunks = []
    for start in range(0, queries_np.shape[0], QUERY_CHUNK):
        queries = _tensor(queries_np[start : start + QUERY_CHUNK])
        chunks.append(torch.cdist(queries, reference).amin(dim=1).cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def _empty_metric(
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    hand_name: str,
    expected_vertex_count: int,
) -> dict:
    return _CPU_INSIDE_VERTEX_METRIC(
        None,
        hand_vertices,
        hand_faces,
        hand_name,
        expected_vertex_count=expected_vertex_count,
    )


def _gpu_voxel_intersection_volume(
    obj_mesh: Optional[trimesh.Trimesh],
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    *,
    pitch_m: float = base.IV_VOXEL_PITCH_M,
    enabled: bool = True,
) -> tuple[np.ndarray, float]:
    empty = np.zeros((0, 3), dtype=np.float32)
    if not enabled or obj_mesh is None or pitch_m <= 0.0:
        return empty, 0.0
    try:
        voxel_points = np.asarray(
            obj_mesh.voxelized(pitch=float(pitch_m)).points,
            dtype=np.float64,
        )
        if voxel_points.ndim != 2 or voxel_points.shape[0] == 0:
            return empty, 0.0
        overlap = _mesh_contains(hand_vertices, hand_faces, voxel_points)
        overlap_points = voxel_points[overlap].astype(np.float32)
        return overlap_points, float(overlap_points.shape[0]) * float(pitch_m) ** 3
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if not ALLOW_CPU_FALLBACK:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[GPU WARN] IV CUDA kernel failed; using CPU for this hand: {error}")
        return _CPU_VOXEL_INTERSECTION_VOLUME(
            obj_mesh,
            hand_vertices,
            hand_faces,
            pitch_m=pitch_m,
            enabled=enabled,
        )


def _gpu_inside_vertex_metric(
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_vertices_world: np.ndarray,
    hand_faces: np.ndarray,
    hand_name: str,
    expected_vertex_count: int = 778,
) -> dict:
    hand_vertices = np.asarray(hand_vertices_world, dtype=np.float64)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    metric = _empty_metric(
        hand_vertices, hand_faces, hand_name, expected_vertex_count
    )
    if (
        obj_mesh_world is None
        or hand_vertices.ndim != 2
        or hand_vertices.shape[1] != 3
        or hand_vertices.shape[0] == 0
    ):
        return metric

    obj_vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    obj_faces = np.asarray(obj_mesh_world.faces, dtype=np.int64)
    finite = np.all(np.isfinite(hand_vertices), axis=1)
    if (
        obj_vertices.ndim != 2
        or obj_faces.ndim != 2
        or not np.any(finite)
    ):
        return metric
    query_points = hand_vertices[finite]

    try:
        closest, _distances, triangle_ids = _mesh_closest_point(
            obj_vertices, obj_faces, query_points
        )
        inside_query = _mesh_contains(obj_vertices, obj_faces, query_points)
        inside_query = _remove_open_cavity_gpu(
            obj_vertices, obj_faces, query_points, inside_query
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if not ALLOW_CPU_FALLBACK:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[GPU WARN] penetration CUDA kernel failed; "
            f"using CPU for {hand_name}: {error}"
        )
        return _CPU_INSIDE_VERTEX_METRIC(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            hand_name,
            expected_vertex_count=expected_vertex_count,
        )

    valid_indices = np.flatnonzero(finite)
    inside_mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
    inside_mask[valid_indices] = inside_query
    inside_indices = np.flatnonzero(inside_mask).astype(np.int64)
    metric["inside_mask"] = inside_mask
    metric["inside_indices"] = inside_indices
    metric["inside_count"] = int(inside_indices.shape[0])
    metric["inside_ratio_per_778"] = float(inside_indices.shape[0]) / float(
        expected_vertex_count
    )
    metric["inside_method"] = "cuda_ray_parity"

    inside_query_indices = np.flatnonzero(inside_query).astype(np.int64)
    if inside_query_indices.size > 0:
        inside_points = hand_vertices[inside_indices].astype(np.float32)
        seed_faces = triangle_ids[inside_query_indices]
        patch_faces = base._object_surface_patch_faces_for_penetration(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            seed_faces,
        )
        # Keep this final, usually very small query on trimesh so the emitted
        # ID decimals match the reference evaluator exactly. CUDA already did
        # the expensive all-vertex containment and seed-face search above.
        closest_inside, id_distances = base._closest_points_on_object_patch(
            obj_mesh_world,
            inside_points,
            patch_faces,
        )
        line_count = min(inside_points.shape[0], closest_inside.shape[0])
        inside_points = inside_points[:line_count]
        closest_inside = closest_inside[:line_count]
        id_distances = np.asarray(id_distances[:line_count], dtype=np.float32)
        inside_indices = inside_indices[:line_count]
        inside_query_indices = inside_query_indices[:line_count]
        valid_penetration = np.isfinite(id_distances) & (
            id_distances > base.MIN_ID_PENETRATION_DEPTH_M
        )
        inside_points = inside_points[valid_penetration]
        closest_inside = closest_inside[valid_penetration]
        id_distances = id_distances[valid_penetration]
        inside_indices = inside_indices[valid_penetration]
        inside_query_indices = inside_query_indices[valid_penetration]

        filtered_mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
        filtered_mask[inside_indices] = True
        metric["inside_mask"] = filtered_mask
        metric["inside_indices"] = inside_indices
        metric["inside_count"] = int(inside_indices.shape[0])
        metric["inside_ratio_per_778"] = float(inside_indices.shape[0]) / float(
            expected_vertex_count
        )
        if id_distances.size > 0:
            deepest_mm = float(np.max(id_distances) * 1000.0)
            metric["id_mean_mm"] = deepest_mm
            metric["id_max_mm"] = deepest_mm
            patch_faces = base._object_surface_patch_faces_for_penetration(
                obj_mesh_world,
                hand_vertices,
                hand_faces,
                triangle_ids[inside_query_indices],
            )
        else:
            patch_faces = np.zeros((0,), dtype=np.int64)
        metric["inside_points"] = inside_points
        metric["closest_points"] = closest_inside
        metric["id_line_hand_points"] = inside_points
        metric["id_line_object_points"] = closest_inside
        metric["id_distances_m"] = id_distances
        metric["id_face_indices"] = patch_faces

    if getattr(base, "QUICK_EVAL_ACTIVE", False):
        iv_points = np.zeros((0, 3), dtype=np.float32)
        iv_volume_m3 = 0.0
    else:
        iv_points, iv_volume_m3 = _gpu_voxel_intersection_volume(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            pitch_m=base.IV_VOXEL_PITCH_M,
            enabled=metric["inside_count"] > 0,
        )
    metric["hand_volume_m3"] = 0.0
    metric["iv_points"] = iv_points
    metric["iv_method"] = "object_surface_voxels_inside_hand_cuda"
    metric["iv_voxel_pitch_m"] = float(base.IV_VOXEL_PITCH_M)
    metric["iv_volume_m3"] = float(iv_volume_m3)
    metric["iv_volume_cm3"] = float(iv_volume_m3 * 1e6)
    return metric


def _gpu_cr_for_frame(
    obj_mesh_world,
    hand_vertices_parts: list[np.ndarray],
    threshold_m: float = 0.005,
) -> float:
    if obj_mesh_world is None:
        return 0.0
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    hands = [
        np.asarray(part, dtype=np.float64)
        for part in hand_vertices_parts
        if np.asarray(part).ndim == 2 and np.asarray(part).shape[0] > 0
    ]
    if vertices.shape[0] == 0 or not hands:
        return 0.0
    points = np.concatenate(hands, axis=0)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        return 0.0
    distances = _nearest_vertex_distances(points, vertices)
    return float(np.count_nonzero(distances < threshold_m)) / float(points.shape[0])


def _gpu_contact_mask(
    obj_mesh_world,
    hand_vertices: np.ndarray,
    threshold_m: float = 0.005,
) -> np.ndarray:
    hand_vertices = np.asarray(hand_vertices, dtype=np.float64)
    mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
    if obj_mesh_world is None or hand_vertices.ndim != 2:
        return mask
    finite = np.all(np.isfinite(hand_vertices), axis=1)
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    if not np.any(finite) or vertices.shape[0] == 0:
        return mask
    distances = _nearest_vertex_distances(hand_vertices[finite], vertices)
    finite_indices = np.flatnonzero(finite)
    mask[finite_indices[distances < threshold_m]] = True
    return mask


@torch.inference_mode()
def _gpu_process_hand_result(hand_layer, hand_params):
    params = torch.as_tensor(hand_params, dtype=torch.float32, device=DEVICE)
    hand_pose = base.rot6d_to_axis_angle(params[:, 3:]).reshape(-1, 48)
    translation = params[:, :3]
    output = hand_layer(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:48],
        betas=torch.zeros(
            (translation.shape[0], 10), dtype=torch.float32, device=DEVICE
        ),
    )
    vertices = output.vertices + translation.unsqueeze(1)
    joints = (
        output.joints_w_tip
        if getattr(output, "joints_w_tip", None) is not None
        else output.joints
    )
    joints = joints + translation.unsqueeze(1)
    faces = torch.as_tensor(
        hand_layer.faces.copy().astype(np.int64), dtype=torch.long
    )
    return vertices, joints, faces


def _install_gpu_kernels(*, gpu_mano: bool) -> None:
    if gpu_mano:
        original_build_mano = base.build_mano_aa

        def build_mano_gpu(*args, **kwargs):
            return original_build_mano(*args, **kwargs).to(DEVICE).eval()

        base.build_mano_aa = build_mano_gpu
        base.process_hand_result = _gpu_process_hand_result
    base._new_inside_vertex_metric_for_hand = _gpu_inside_vertex_metric
    # Even if an old command contains this flag, keep the GPU metric rather
    # than selecting the old CPU KD-tree approximation.
    base._new_inside_vertex_metric_for_hand_fast = _gpu_inside_vertex_metric
    base._voxel_intersection_volume_for_hand = _gpu_voxel_intersection_volume
    base._fast_cr_for_frame = _gpu_cr_for_frame
    base._fast_contact_mask_for_hand = _gpu_contact_mask


def main() -> None:
    global DEVICE, DTYPE, QUERY_CHUNK, FACE_CHUNK, ALLOW_CPU_FALLBACK
    launcher, evaluator_argv = _launcher_args()
    selected_input_files = (
        [file_name for input_group in launcher.input_files for file_name in input_group]
        if launcher.input_files
        else list(DEFAULT_INPUT_FILES)
    )
    if selected_input_files:
        input_args = [
            argument
            for file_name in selected_input_files
            for argument in ("--input", file_name)
        ]
        evaluator_argv = [*input_args, *evaluator_argv]
    evaluator_argv = _add_default_output_paths(evaluator_argv)
    DEVICE = _resolve_device(launcher.device)
    DTYPE = torch.float32 if launcher.gpu_dtype == "float32" else torch.float64
    QUERY_CHUNK = max(1, int(launcher.gpu_query_chunk))
    FACE_CHUNK = max(1, int(launcher.gpu_face_chunk))
    ALLOW_CPU_FALLBACK = not bool(launcher.no_gpu_fallback)

    torch.backends.cudnn.benchmark = True
    _install_gpu_kernels(gpu_mano=bool(launcher.gpu_mano))
    properties = torch.cuda.get_device_properties(DEVICE)
    dtype_name = str(DTYPE)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name[len("torch.") :]
    print(
        "[GPU] "
        f"device={DEVICE} name={properties.name!r} "
        f"dtype={dtype_name} "
        f"query_chunk={QUERY_CHUNK} face_chunk={FACE_CHUNK}"
    )
    print(
        "[GPU] CUDA kernels enabled for ID closest-point, inside-mesh, "
        "IV containment, CR, and fingertip contact; "
        f"MANO={'cuda' if launcher.gpu_mano else 'cpu-reference'}."
    )

    sys.argv = [sys.argv[0], *evaluator_argv]
    base.main()


if __name__ == "__main__":
    main()
