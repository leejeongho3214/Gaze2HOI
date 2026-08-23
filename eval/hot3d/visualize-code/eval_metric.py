from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import torch
import trimesh

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None
try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None
try:
    import tqdm
except ModuleNotFoundError:

    class _TqdmFallback:
        @staticmethod
        def tqdm(iterable=None, total=None, desc=None, leave=True):
            if iterable is not None:
                return iterable

            class _DummyPbar:
                def update(self, _n=1):
                    return None

                def close(self):
                    return None

            return _DummyPbar()

    tqdm = _TqdmFallback()

try:
    import rerun as rr
except ModuleNotFoundError:
    rr = None

for _name, _type in [
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("object", object),
    ("str", str),
    ("complex", complex),
    ("unicode", str),
]:
    if _name not in np.__dict__:
        setattr(np, _name, _type)

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT3D_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PACKAGE_ROOT = os.path.abspath(os.path.join(HOT3D_ROOT, ".."))
if HOT3D_ROOT not in sys.path:
    sys.path.insert(0, HOT3D_ROOT)

from interaction_common import (
    ObjectModel,
    _extract_object_key,
    _pose9_sequence,
    _safe_mesh_volume,
    _sequence_length,
    _standard_rot6d_to_matrix,
    _to_numpy,
    _to_torch,
    process_hand_result_standard as process_hand_result,
    process_obj_result_standard as process_obj_result,
)
from mano import MANO_C, MODEL_DIR, build_mano_aa
from rot import (
    axis_angle_to_rotmat,
    rotation_matrix_to_angle_axis,
    rot6d_to_axis_angle,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
)


PART_ALIASES = {("whiteboard_eraser", "long edge"): "side"}
IV_SUCCESS_THRESHOLD_CM3 = 100.0
_MISSING_OBJ_MESH_WARNED = False
LATETHOI_CONTACT_THRESHOLD_M = 0.005
OFF_GROUND_THRESHOLD_M = 0.005
MIN_ID_PENETRATION_DEPTH_M = 0.002
IV_VOXEL_PITCH_M = 0.005
_MESH_CACHE_VERSION = 4
_DEFAULT_EXPECTED_SAMPLE_COUNT = 106
_EVAL_WORKER_CONTEXT = None
_EVAL_WORKER_L_HAND_LAYER = None
_EVAL_WORKER_R_HAND_LAYER = None
_GT_ONLY_SOURCE_FILE_LABELS = {}
_RERUN_VIEWER_PROCESS = None
_DIFFH2O_PCA_HAND_LAYERS = {}
QUICK_EVAL_ACTIVE = False


def _diag_extent(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return 0.0
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return 0.0
    ext = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(ext))


def _align_mesh_scale_to_object_pc(
    mesh: trimesh.Trimesh,
    obj_pc,
    object_key: str,
    ratio_threshold: float = 3.0,
) -> trimesh.Trimesh:
    pc_diag = _diag_extent(np.asarray(obj_pc, dtype=np.float64))
    mesh_diag = _diag_extent(np.asarray(mesh.vertices, dtype=np.float64))
    if pc_diag <= 0.0 or mesh_diag <= 0.0:
        return mesh
    ratio = mesh_diag / pc_diag
    if (1.0 / ratio_threshold) <= ratio <= ratio_threshold:
        return mesh
    scale = pc_diag / mesh_diag
    mesh_aligned = mesh.copy()
    mesh_aligned.apply_scale(float(scale))
    print(
        f"[WARN] mesh scale mismatch for '{object_key}': "
        f"mesh/pc diag ratio={ratio:.4f}. applying scale {scale:.6f}."
    )
    return mesh_aligned


def _prepare_mesh_for_proximity(
    mesh: trimesh.Trimesh,
    object_key: str,
    max_faces: int = 20000,
) -> trimesh.Trimesh:
    if mesh is None or not hasattr(mesh, "faces"):
        return mesh
    try:
        face_count = int(mesh.faces.shape[0])
    except Exception:
        return mesh
    if face_count <= max_faces:
        return mesh
    try:
        face_idx = np.linspace(0, face_count - 1, max_faces, dtype=np.int64)
        faces_sampled = np.asarray(mesh.faces[face_idx], dtype=np.int64)
        unique_vertices, inverse = np.unique(
            faces_sampled.reshape(-1), return_inverse=True
        )
        vertices_sampled = np.asarray(mesh.vertices[unique_vertices], dtype=np.float64)
        faces_remapped = inverse.reshape(-1, 3).astype(np.int64)
        sampled_mesh = trimesh.Trimesh(
            vertices=vertices_sampled,
            faces=faces_remapped,
            process=False,
        )
        print(
            f"[WARN] heavy mesh '{object_key}' face-sampled for metrics: "
            f"{face_count} -> {int(sampled_mesh.faces.shape[0])} faces, "
            f"{int(mesh.vertices.shape[0])} -> {int(sampled_mesh.vertices.shape[0])} vertices."
        )
        return sampled_mesh
    except Exception as ex:
        print(
            f"[WARN] failed to face-sample heavy mesh '{object_key}' "
            f"({face_count} faces): {ex}"
        )
    return mesh


def _remap_mesh_faces_to_object_pc(
    mesh: trimesh.Trimesh,
    obj_pc,
    object_key: str,
) -> Optional[trimesh.Trimesh]:
    if mesh is None:
        return None
    obj_vertices = np.asarray(obj_pc, dtype=np.float64)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        obj_vertices.ndim != 2
        or obj_vertices.shape[1] != 3
        or obj_vertices.shape[0] == 0
        or mesh_vertices.ndim != 2
        or mesh_vertices.shape[1] != 3
        or mesh_vertices.shape[0] == 0
        or mesh_faces.ndim != 2
        or mesh_faces.shape[1] != 3
        or mesh_faces.shape[0] == 0
    ):
        return None
    try:
        if cKDTree is not None:
            tree = cKDTree(obj_vertices)
            _dists, nearest_idx = tree.query(mesh_vertices, k=1)
            nearest_idx = np.asarray(nearest_idx, dtype=np.int64)
        else:
            nearest_idx = (
                np.linalg.norm(
                    mesh_vertices[:, None, :] - obj_vertices[None, :, :], axis=2
                )
                .argmin(axis=1)
                .astype(np.int64)
            )
        remapped_faces = nearest_idx[mesh_faces]
        valid = (
            (remapped_faces[:, 0] != remapped_faces[:, 1])
            & (remapped_faces[:, 1] != remapped_faces[:, 2])
            & (remapped_faces[:, 0] != remapped_faces[:, 2])
        )
        remapped_faces = remapped_faces[valid]
        if remapped_faces.shape[0] == 0:
            print(
                f"[WARN] remapped mesh for '{object_key}' has no non-degenerate faces."
            )
            return None
        face_keys = np.sort(remapped_faces, axis=1)
        _unique_keys, unique_indices = np.unique(face_keys, axis=0, return_index=True)
        remapped_faces = remapped_faces[np.sort(unique_indices)]
        out = trimesh.Trimesh(
            vertices=obj_vertices,
            faces=remapped_faces.astype(np.int64),
            process=False,
        )
        print(
            f"[INFO] remapped mesh for '{object_key}': "
            f"obj.pkl vertices={int(obj_vertices.shape[0])}, "
            f"original faces={int(mesh_faces.shape[0])} -> "
            f"usable faces={int(remapped_faces.shape[0])}."
        )
        return out
    except Exception as ex:
        print(f"[WARN] failed to remap mesh faces for '{object_key}': {ex}")
        return None


def _object_mesh_cache_dir(obj_pkl_path: str) -> str:
    del obj_pkl_path
    return os.path.join(HOT3D_ROOT, ".mesh_cache")


def _object_mesh_cache_path(
    obj_pkl_path: str,
    mesh_path: str,
    object_key: str,
    max_faces: int,
    obj_vertex_count: int,
) -> str:
    stat = os.stat(mesh_path)
    token = "|".join(
        [
            str(_MESH_CACHE_VERSION),
            os.path.abspath(mesh_path),
            str(int(stat.st_mtime_ns)),
            str(int(stat.st_size)),
            str(int(max_faces)),
            str(int(obj_vertex_count)),
        ]
    )
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    safe_key = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in object_key
    )
    return os.path.join(
        _object_mesh_cache_dir(obj_pkl_path),
        f"{safe_key}_{digest}.pkl",
    )


def _load_cached_object_mesh(
    cache_path: str,
    object_key: str,
    expected_pc_diag: float,
    expected_vertex_count: int,
) -> Optional[trimesh.Trimesh]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if float(payload.get("pc_diag", -1.0)) != float(expected_pc_diag):
            return None
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        faces = np.asarray(payload["faces"], dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return None
        if int(vertices.shape[0]) != int(expected_vertex_count):
            return None
        if faces.ndim != 2 or faces.shape[1] != 3:
            return None
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as ex:
        print(f"[WARN] failed to read mesh cache for '{object_key}': {ex}")
        return None


def _write_cached_object_mesh(
    cache_path: str,
    mesh: trimesh.Trimesh,
    pc_diag: float,
    object_key: str,
) -> None:
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "pc_diag": float(pc_diag),
            "vertices": np.asarray(mesh.vertices, dtype=np.float64),
            "faces": np.asarray(mesh.faces, dtype=np.int64),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(f"[WARN] failed to write mesh cache for '{object_key}': {ex}")


def _resolve_object_mesh_path(
    obj_pkl_path: str,
    obj_path_value,
) -> tuple[Optional[str], list[str]]:
    raw_path = os.path.expanduser(str(obj_path_value))
    tried = []

    if os.path.isabs(raw_path):
        tried.append(raw_path)
        return (raw_path if os.path.exists(raw_path) else None), tried

    obj_pkl_dir = os.path.dirname(os.path.abspath(obj_pkl_path))
    candidates = [
        os.path.join(obj_pkl_dir, raw_path),
        os.path.join(obj_pkl_dir, os.path.basename(raw_path)),
        os.path.abspath(raw_path),
    ]

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        norm = os.path.abspath(os.path.expanduser(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        ordered_candidates.append(norm)

    tried.extend(ordered_candidates)
    for candidate in ordered_candidates:
        if os.path.exists(candidate):
            return candidate, tried
    return None, tried


def _load_object_mesh(
    obj_pkl_path: str,
    obj_path_value,
    obj_pc,
    object_key: str,
    max_faces: int = 20000,
) -> Optional[trimesh.Trimesh]:
    mesh_path, tried_paths = _resolve_object_mesh_path(obj_pkl_path, obj_path_value)
    if mesh_path is None:
        print(
            f"[WARN] failed to resolve object mesh for '{object_key}' from obj_path "
            f"'{obj_path_value}'. tried: {tried_paths}"
        )
        return None
    pc_diag = _diag_extent(np.asarray(obj_pc, dtype=np.float64))
    obj_vertex_count = int(np.asarray(obj_pc).shape[0])
    cache_path = _object_mesh_cache_path(
        obj_pkl_path,
        mesh_path,
        object_key=object_key,
        max_faces=max_faces,
        obj_vertex_count=obj_vertex_count,
    )
    cached_mesh = _load_cached_object_mesh(
        cache_path,
        object_key=object_key,
        expected_pc_diag=pc_diag,
        expected_vertex_count=obj_vertex_count,
    )
    if cached_mesh is not None:
        return cached_mesh
    try:
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"expected Trimesh, got {type(loaded)!r}")
        mesh = trimesh.Trimesh(
            vertices=np.asarray(loaded.vertices, dtype=np.float64),
            faces=np.asarray(loaded.faces, dtype=np.int64),
            process=False,
        )
        mesh = _align_mesh_scale_to_object_pc(mesh, obj_pc, object_key)
        remapped_mesh = _remap_mesh_faces_to_object_pc(mesh, obj_pc, object_key)
        if remapped_mesh is None:
            print(
                f"[WARN] failed to remap original mesh faces onto obj.pkl vertices "
                f"for '{object_key}'."
            )
            return None
        mesh = remapped_mesh
        mesh = _prepare_mesh_for_proximity(
            mesh, object_key=object_key, max_faces=max_faces
        )
        _write_cached_object_mesh(
            cache_path, mesh, pc_diag=pc_diag, object_key=object_key
        )
        return mesh
    except Exception as ex:
        print(
            f"[WARN] failed to load object mesh for '{object_key}' from "
            f"'{mesh_path}': {ex}"
        )
        return None


def _build_proxy_mesh_from_object_pc(
    obj_pc,
    object_key: str,
) -> Optional[trimesh.Trimesh]:
    pts = np.asarray(obj_pc, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(
            f"[WARN] invalid object point cloud for proxy mesh '{object_key}': {pts.shape}"
        )
        return None
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] < 4:
        print(
            f"[WARN] insufficient object points for proxy mesh '{object_key}': {pts.shape[0]}"
        )
        return None

    def _finalize_proxy_mesh(mesh, method: str) -> Optional[trimesh.Trimesh]:
        if mesh is None or not hasattr(mesh, "faces"):
            return None
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or faces.ndim != 2
                or faces.shape[1] != 3
                or faces.shape[0] == 0
            ):
                return None
            proxy_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )
            components = proxy_mesh.split(only_watertight=False)
            if components:
                proxy_mesh = max(components, key=lambda m: int(m.faces.shape[0]))
            face_count = int(proxy_mesh.faces.shape[0])
            print(
                f"[INFO] proxy mesh for '{object_key}' built from object point cloud "
                f"using {method}: {pts.shape[0]} points -> {face_count} faces."
            )
            return proxy_mesh
        except Exception:
            return None

    if o3d is not None and cKDTree is not None:
        try:
            tree = cKDTree(pts)
            k = min(9, pts.shape[0])
            dists, _ = tree.query(pts, k=k)
            if dists.ndim == 1:
                dists = dists[:, None]
            neighbor_dists = dists[:, 1:] if dists.shape[1] > 1 else dists[:, :0]
            finite_neighbor_dists = neighbor_dists[np.isfinite(neighbor_dists)]
            if finite_neighbor_dists.size > 0:
                base_spacing = float(np.median(finite_neighbor_dists))
                if base_spacing > 0.0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    for alpha_scale in (2.0, 2.5, 3.0, 3.5):
                        alpha = base_spacing * alpha_scale
                        if not np.isfinite(alpha) or alpha <= 0.0:
                            continue
                        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                            pcd, alpha
                        )
                        vertices = np.asarray(mesh_o3d.vertices)
                        faces = np.asarray(mesh_o3d.triangles)
                        if vertices.size == 0 or faces.size == 0:
                            continue
                        proxy_mesh = _finalize_proxy_mesh(
                            trimesh.Trimesh(
                                vertices=vertices,
                                faces=faces,
                                process=False,
                            ),
                            method=f"alpha-shape(alpha={alpha:.6f})",
                        )
                        if proxy_mesh is not None:
                            return proxy_mesh
        except Exception as ex:
            print(
                f"[WARN] alpha-shape proxy mesh reconstruction failed for "
                f"'{object_key}': {ex}"
            )
    try:
        point_cloud = trimesh.points.PointCloud(pts)
        proxy_mesh = point_cloud.convex_hull
        if proxy_mesh is None or not hasattr(proxy_mesh, "faces"):
            print(
                f"[WARN] proxy mesh reconstruction returned no faces for '{object_key}'"
            )
            return None
        proxy_mesh = _finalize_proxy_mesh(proxy_mesh, method="convex-hull")
        if proxy_mesh is not None:
            return proxy_mesh
        print(
            f"[WARN] convex-hull proxy mesh reconstruction returned invalid mesh "
            f"for '{object_key}'"
        )
        return None
    except Exception as ex:
        print(f"[WARN] failed to build proxy mesh for '{object_key}': {ex}")
        return None


def _object_pc_proxy_cache_key(obj_pc, object_key: str) -> Optional[tuple]:
    pts = np.asarray(obj_pc, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    pts = np.ascontiguousarray(pts)
    digest = hashlib.sha1(pts.view(np.uint8)).hexdigest()
    return (str(object_key).lower(), tuple(pts.shape), pts.dtype.str, digest)


def _get_or_build_proxy_mesh_from_object_pc(
    obj_pc,
    object_key: str,
    proxy_cache: dict,
) -> Optional[trimesh.Trimesh]:
    cache_key = _object_pc_proxy_cache_key(obj_pc, object_key)
    if cache_key is None:
        return _build_proxy_mesh_from_object_pc(obj_pc, object_key=object_key)
    if cache_key not in proxy_cache:
        proxy_cache[cache_key] = _build_proxy_mesh_from_object_pc(
            obj_pc, object_key=object_key
        )
    return proxy_cache[cache_key]


def _transform_object_mesh_to_world(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_pose_params,
) -> Optional[trimesh.Trimesh]:
    if obj_mesh is None:
        return None
    try:
        vertices_local = np.asarray(obj_mesh.vertices, dtype=np.float64)
        faces = np.asarray(obj_mesh.faces, dtype=np.int64)
        if (
            vertices_local.ndim != 2
            or vertices_local.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or vertices_local.shape[0] == 0
        ):
            return None
        obj_pose = _pose9_sequence(obj_pose_params)
        last_pose = _to_numpy(obj_pose[-1]).astype(np.float64)
        trans = last_pose[:3]
        rot = (
            _to_numpy(rot6d_to_rotmat(_to_torch(last_pose[3:9]).reshape(1, 6)))
            .reshape(3, 3)
            .astype(np.float64)
        )
        vertices_world = np.einsum("ni,ji->nj", vertices_local, rot) + trans[None, :]
        return trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    except Exception:
        return None


def _object_pose_trans_rot(
    obj_pose_params,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    obj_pose = _pose9_sequence(obj_pose_params)
    last_pose = _to_numpy(obj_pose[-1]).astype(np.float64)
    if last_pose.ndim != 1 or last_pose.shape[0] < 9:
        raise ValueError(f"invalid object pose shape: {last_pose.shape}")
    trans = last_pose[:3]
    rot6d = last_pose[3:9]
    if not np.isfinite(trans).all():
        raise ValueError(f"non-finite object translation: {trans}")
    if not np.isfinite(rot6d).all():
        raise ValueError(f"non-finite object rot6d: {rot6d}")
    rot = (
        _to_numpy(rot6d_to_rotmat(_to_torch(rot6d).reshape(1, 6)))
        .reshape(3, 3)
        .astype(np.float64)
    )
    if not np.isfinite(rot).all():
        raise ValueError(f"non-finite object rotation matrix from rot6d: {rot6d}")
    return trans, rot


def _transform_points_world_to_object_frame(
    points_world: np.ndarray,
    obj_pose_params,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"invalid hand vertex shape: {points.shape}")
    if not np.isfinite(points).all():
        bad_count = int(np.count_nonzero(~np.isfinite(points).all(axis=1)))
        raise ValueError(
            f"non-finite hand vertices before object inverse transform: {bad_count}"
        )
    trans, rot = _object_pose_trans_rot(obj_pose_params)
    try:
        with np.errstate(divide="raise", over="raise", invalid="raise"):
            transformed = np.einsum("ni,ij->nj", points - trans.reshape(1, 3), rot)
    except FloatingPointError as ex:
        raise ValueError(
            "floating-point failure during object inverse transform: "
            f"{ex}; points_abs_max={float(np.max(np.abs(points))):.6g}, "
            f"trans_abs_max={float(np.max(np.abs(trans))):.6g}, "
            f"rot_abs_max={float(np.max(np.abs(rot))):.6g}"
        ) from ex
    if not np.isfinite(transformed).all():
        bad_count = int(np.count_nonzero(~np.isfinite(transformed).all(axis=1)))
        raise ValueError(
            f"non-finite hand vertices after object inverse transform: {bad_count}"
        )
    return transformed.astype(np.float32)


def _transform_object_mesh_world_to_object_frame(
    obj_mesh_world: Optional[trimesh.Trimesh],
    obj_pose_params,
) -> Optional[trimesh.Trimesh]:
    if obj_mesh_world is None:
        return None
    vertices = _transform_points_world_to_object_frame(
        np.asarray(obj_mesh_world.vertices, dtype=np.float64),
        obj_pose_params,
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(obj_mesh_world.faces, dtype=np.int64),
        process=False,
    )


def _object_overlap_region(
    obj_mesh: Optional[trimesh.Trimesh],
    obj_pose_params,
    obj_points_world: np.ndarray,
    hand_mesh_parts: list[tuple[np.ndarray, np.ndarray]],
    voxel_pitch: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, float]:
    object_point_mask = np.zeros((int(obj_points_world.shape[0]),), dtype=bool)
    overlap_voxel_points = np.zeros((0, 3), dtype=np.float32)
    if obj_mesh is None or not hand_mesh_parts:
        return object_point_mask, overlap_voxel_points, 0.0

    obj_mesh_world = _transform_object_mesh_to_world(obj_mesh, obj_pose_params)
    if obj_mesh_world is None:
        return object_point_mask, overlap_voxel_points, 0.0

    hand_meshes = []
    for verts, faces in hand_mesh_parts:
        verts_np = np.asarray(verts, dtype=np.float64)
        faces_np = np.asarray(faces, dtype=np.int64)
        if (
            verts_np.ndim != 2
            or verts_np.shape[1] != 3
            or verts_np.shape[0] == 0
            or faces_np.ndim != 2
            or faces_np.shape[1] != 3
            or faces_np.shape[0] == 0
        ):
            continue
        hand_meshes.append(
            trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        )
    if not hand_meshes:
        return object_point_mask, overlap_voxel_points, 0.0

    try:
        obj_points_world = np.asarray(obj_points_world, dtype=np.float64)
        valid_points = (
            obj_points_world.ndim == 2
            and obj_points_world.shape[1] == 3
            and obj_points_world.shape[0] > 0
        )
        if valid_points:
            finite_mask = np.all(np.isfinite(obj_points_world), axis=1)
            valid_idx = np.flatnonzero(finite_mask)
            if valid_idx.size > 0:
                points_valid = obj_points_world[valid_idx]
                inside_any = np.zeros((points_valid.shape[0],), dtype=bool)
                for hand_mesh in hand_meshes:
                    inside_any |= np.asarray(
                        hand_mesh.contains(points_valid), dtype=bool
                    )
                object_point_mask[valid_idx] = inside_any
    except Exception:
        pass

    try:
        obj_vox = obj_mesh_world.voxelized(pitch=float(voxel_pitch))
        obj_voxel_points = np.asarray(obj_vox.points, dtype=np.float64)
        if obj_voxel_points.ndim == 2 and obj_voxel_points.shape[0] > 0:
            inside_any = np.zeros((obj_voxel_points.shape[0],), dtype=bool)
            for hand_mesh in hand_meshes:
                inside_any |= np.asarray(
                    hand_mesh.contains(obj_voxel_points), dtype=bool
                )
            overlap_voxel_points = np.asarray(
                obj_voxel_points[inside_any], dtype=np.float32
            )
            overlap_volume_m3 = float(overlap_voxel_points.shape[0]) * float(
                voxel_pitch**3
            )
            return object_point_mask, overlap_voxel_points, overlap_volume_m3
    except Exception:
        pass

    return object_point_mask, overlap_voxel_points, 0.0


def _selected_hands(text: str) -> tuple[bool, bool]:
    t = str(text).lower()
    use_right = "right" in t
    use_left = "left" in t
    if "both" in t:
        use_left = True
        use_right = True
    return use_left, use_right


def _target_part_from_text(text: str) -> str:
    value = str(text).lower()
    prefix, marker, hand_marker = "grab ", " of ", " with "
    if not value.startswith(prefix) or marker not in value or hand_marker not in value:
        return ""
    return value[len(prefix) : value.index(marker)].strip()


def _contact_object_point_indices(
    hand_vertices: list[np.ndarray],
    object_points: np.ndarray,
    threshold_m: float = LATETHOI_CONTACT_THRESHOLD_M,
) -> np.ndarray:
    """Canonical object-point indices within the contact threshold."""
    if not hand_vertices:
        return np.zeros((0,), dtype=np.int64)
    points = np.asarray(object_points, dtype=np.float64)
    hands = np.concatenate(hand_vertices, axis=0)
    if (
        points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0
        or hands.ndim != 2 or hands.shape[1] != 3 or hands.shape[0] == 0
    ):
        return np.zeros((0,), dtype=np.int64)
    valid = np.all(np.isfinite(hands), axis=1)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)
    if cKDTree is not None:
        distance, index = cKDTree(points).query(hands[valid], k=1)
    else:
        dist = np.linalg.norm(hands[valid, None] - points[None], axis=2)
        index = np.argmin(dist, axis=1)
        distance = dist[np.arange(dist.shape[0]), index]
    return np.unique(np.asarray(index)[np.asarray(distance) < threshold_m]).astype(np.int64)


def _part_contact_metrics(
    contact_indices: np.ndarray,
    object_key: str,
    target_part: str,
    part_labels: Optional[dict],
) -> tuple[Optional[float], Optional[float]]:
    """Return PCP and dominant-part accuracy; None means no annotation exists."""
    if not part_labels or not target_part:
        return None, None
    parts = part_labels.get(str(object_key).lower(), {})
    annotation_part = PART_ALIASES.get((str(object_key).lower(), target_part), target_part)
    target_indices = set(int(index) for index in parts.get(annotation_part, []))
    if not target_indices:
        return None, None
    if contact_indices.size == 0:
        return 0.0, 0.0
    pcp = float(sum(int(index) in target_indices for index in contact_indices)) / float(contact_indices.size)
    counts = {
        name: int(np.isin(contact_indices, np.asarray(indices, dtype=np.int64)).sum())
        for name, indices in parts.items()
    }
    dominant_part = max(counts, key=counts.get) if counts else None
    return pcp, float(dominant_part == annotation_part)


def _gaze_target_point_index(
    gaze,
    gt_obj_params,
    object_points: np.ndarray,
    nframes: int,
    sigma_m: float = 0.05,
    frame_end_exclusive: Optional[int] = None,
) -> Optional[int]:
    if gaze is None or gt_obj_params is None:
        return None
    rays = _to_numpy(gaze).squeeze(-1).astype(np.float64)
    gaze_frames = (
        int(nframes)
        if frame_end_exclusive is None
        else max(0, int(frame_end_exclusive))
    )
    count = min(gaze_frames, int(rays.shape[0]), _sequence_length(gt_obj_params))
    if count <= 0 or rays.ndim != 3 or rays.shape[1:] != (2, 3):
        return None
    try:
        object_world = _to_numpy(process_obj_result(object_points, _slice_frame_indices(gt_obj_params, np.arange(count))))
    except Exception:
        return None
    origin, direction = rays[:count, 0], rays[:count, 1]
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    valid = np.isfinite(rays[:count]).all(axis=(1, 2)) & (norm[:, 0] > 1e-8)
    if not np.any(valid):
        return None
    direction = direction / np.maximum(norm, 1e-8)
    offset = object_world - origin[:, None, :]
    forward = np.einsum("tni,ti->tn", offset, direction)
    closest = origin[:, None, :] + np.maximum(forward, 0.0)[..., None] * direction[:, None, :]
    score = np.exp(-0.5 * (np.linalg.norm(object_world - closest, axis=2) / sigma_m) ** 2)
    score[~valid] = np.nan
    mean_score = np.nanmean(score, axis=0)
    return int(np.nanargmax(mean_score)) if np.isfinite(mean_score).any() else None


def _normalize_motion_npy_array(motion, nsamples_hint=None) -> np.ndarray:
    """
    Normalize motion array to shape [N, T, D].
    Supports common layouts such as [N,T,D], [N,D,T], [N,D,1,T].
    """
    motion = np.asarray(motion)
    if motion.ndim < 2:
        raise ValueError(f"unsupported motion shape: {tuple(motion.shape)}")
    motion = np.squeeze(motion)
    if motion.ndim == 2:
        motion = motion[None, ...]
    if motion.ndim < 3:
        raise ValueError(f"unsupported motion shape: {tuple(motion.shape)}")

    sample_axis = 0
    if nsamples_hint is not None:
        try:
            hint = int(np.asarray(nsamples_hint).item())
            matches = [ax for ax, sz in enumerate(motion.shape) if int(sz) == hint]
            if matches:
                sample_axis = matches[0]
        except Exception:
            pass
    motion = np.moveaxis(motion, sample_axis, 0)  # [N, ...]

    tail_shape = motion.shape[1:]
    candidate_axes = [ax for ax, sz in enumerate(tail_shape, start=1) if sz >= 207]
    if not candidate_axes:
        raise ValueError(
            f"motion has no feature axis (>=207) after normalization: {tuple(motion.shape)}"
        )
    if any(motion.shape[ax] == 207 for ax in candidate_axes):
        feat_axis = next(ax for ax in candidate_axes if motion.shape[ax] == 207)
    else:
        feat_axis = min(candidate_axes, key=lambda ax: abs(int(motion.shape[ax]) - 207))

    motion = np.moveaxis(motion, feat_axis, -1)  # [N, ..., D]
    d = motion.shape[-1]
    motion = motion.reshape(motion.shape[0], -1, d)  # [N, T, D]
    return motion


def _load_items_from_path(path: str):
    def _diffh2o_dataset_dir(payload: dict) -> str:
        split_set = str(payload.get("split_set", "hot3d_seen"))
        dataset_name = (
            "GRAB_HANDS_hot3d_unseen"
            if "unseen" in split_set
            else "GRAB_HANDS_hot3d_seen"
        )
        candidates = []
        model_path = payload.get("model_path")
        if model_path:
            model_path = os.path.abspath(os.path.expanduser(str(model_path)))
            current = os.path.dirname(model_path)
            while current and current != os.path.dirname(current):
                candidates.append(os.path.join(current, "dataset", dataset_name))
                current = os.path.dirname(current)
        candidates.append(
            os.path.abspath(
                os.path.join(PACKAGE_ROOT, "..", "..", "diffh2o", "dataset", dataset_name)
            )
        )
        for candidate in candidates:
            if os.path.isfile(os.path.join(candidate, "hot3d_pca24_local_pose.npz")):
                return candidate
        raise FileNotFoundError(
            "DiffH2O PCA file not found; expected "
            f"dataset/{dataset_name}/hot3d_pca24_local_pose.npz near the model repository."
        )

    def _diffh2o_motion_array(value) -> np.ndarray:
        motion = np.asarray(value)
        while motion.ndim > 3:
            singleton_axis = next(
                (axis for axis in range(1, motion.ndim - 1) if motion.shape[axis] == 1),
                None,
            )
            if singleton_axis is None:
                raise ValueError(f"unsupported DiffH2O motion shape: {motion.shape}")
            motion = np.squeeze(motion, axis=singleton_axis)
        if motion.ndim != 3 or motion.shape[-1] != 117:
            raise ValueError(f"expected DiffH2O [N,T,117] motion, got {motion.shape}")
        return motion.astype(np.float32, copy=False)

    def _diffh2o_rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        a1, a2 = value[..., :3], value[..., 3:6]
        b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
        b3 = np.cross(b1, b2)
        return np.stack((b1, b2, b3), axis=-2)

    def _diffh2o_feature117_to_99d(
        feature: np.ndarray,
        object_pc: np.ndarray,
        pca: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feature = np.asarray(feature, dtype=np.float32)
        object_pc = np.asarray(object_pc, dtype=np.float32).reshape(-1, 3)
        object_center = (object_pc.min(axis=0) + object_pc.max(axis=0)) * 0.5

        def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
            tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
            return rotmat_to_rot6d(tensor).numpy().reshape(matrix.shape[0], 6)

        pos_l, pos_r = feature[:, 0:3], feature[:, 3:6]
        coeff_l, coeff_r = feature[:, 6:30], feature[:, 36:60]
        glob_l = _matrix_to_rot6d(_diffh2o_rot6d_to_matrix(feature[:, 30:36]))
        glob_r = _matrix_to_rot6d(_diffh2o_rot6d_to_matrix(feature[:, 60:66]))

        aa_l = coeff_l @ pca["left_components"] + pca["left_mean"][None, :]
        aa_r = coeff_r @ pca["right_components"] + pca["right_mean"][None, :]
        local_l = rotmat_to_rot6d(
            axis_angle_to_rotmat(torch.from_numpy(aa_l.reshape(-1, 3)))
        ).numpy().reshape(feature.shape[0], 90)
        local_r = rotmat_to_rot6d(
            axis_angle_to_rotmat(torch.from_numpy(aa_r.reshape(-1, 3)))
        ).numpy().reshape(feature.shape[0], 90)
        pred_l = np.concatenate((pos_l, glob_l, local_l), axis=-1)
        pred_r = np.concatenate((pos_r, glob_r, local_r), axis=-1)

        pred_obj = feature[:, -9:].copy()
        obj_rot = np.asarray(
            rot6d_to_rotmat(torch.from_numpy(pred_obj[:, 3:9])).numpy(),
            dtype=np.float32,
        )
        pred_obj[:, :3] -= np.einsum("tij,j->ti", obj_rot, object_center)
        return pred_obj, pred_l, pred_r

    def _normalize_diffh2o_multiseed_dict(payload: dict):
        results_by_seed = payload.get("results_by_seed")
        if not isinstance(results_by_seed, dict):
            return None
        dataset_dir = _diffh2o_dataset_dir(payload)
        pca_raw = np.load(os.path.join(dataset_dir, "hot3d_pca24_local_pose.npz"))
        pca = {
            key: np.asarray(pca_raw[key], dtype=np.float32)
            for key in ("left_components", "left_mean", "right_components", "right_mean")
        }
        with open(os.path.join(PACKAGE_ROOT, "data", "obj.pkl"), "rb") as handle:
            object_pcs = pickle.load(handle)["obj_pcs"]

        records = []
        for seed, result in sorted(results_by_seed.items(), key=lambda item: int(item[0])):
            if not isinstance(result, dict) or "motion" not in result:
                continue
            motions = _diffh2o_motion_array(result["motion"])
            count = min(int(result.get("num_samples", len(motions))), len(motions))
            lengths = np.asarray(result.get("lengths", []), dtype=np.int64).reshape(-1)
            texts = result.get("text", [])
            object_names = result.get("object_name", [])
            prompt_indices = np.asarray(result.get("unique_prompt_index", [])).reshape(-1)
            repeat_indices = np.asarray(result.get("prompt_repeat_index", [])).reshape(-1)
            sample_ids = result.get("sample_id", [])
            data_ids = result.get("data_id", [])
            prompt_metadata = result.get("prompt_metadata", [])
            pred_objs, pred_lhands, pred_rhands, text_list, metadata = [], [], [], [], []
            for index in range(count):
                object_name = str(object_names[index]).strip().lower()
                if object_name not in object_pcs:
                    raise KeyError(f"object '{object_name}' is missing from obj.pkl")
                length = int(lengths[index]) if index < len(lengths) else motions.shape[1]
                length = max(1, min(length, motions.shape[1]))
                pred_obj, pred_lhand, pred_rhand = _diffh2o_feature117_to_99d(
                    motions[index, :length], object_pcs[object_name], pca
                )
                pred_objs.append(pred_obj)
                pred_lhands.append(pred_lhand)
                pred_rhands.append(pred_rhand)
                text_list.append(str(texts[index]) if index < len(texts) else "")
                saved_meta = (
                    prompt_metadata[index]
                    if index < len(prompt_metadata) and isinstance(prompt_metadata[index], dict)
                    else {}
                )
                meta = dict(saved_meta)
                meta.update(
                    {
                        "object_name": object_name,
                        "source_object_name": object_name,
                        "seed": int(seed),
                        "source_seed": int(seed),
                        "seed_sample_idx": index,
                        "source_format": "diffh2o_hot3d_multiseed_v2",
                    }
                )
                if index < len(prompt_indices):
                    meta["prompt_idx"] = int(prompt_indices[index])
                if index < len(repeat_indices):
                    meta["repeat_idx"] = int(repeat_indices[index])
                if index < len(sample_ids):
                    meta["sample_id"] = str(sample_ids[index])
                    sample_id = str(sample_ids[index])
                    if len(sample_id) > 1 and sample_id[1:].isdigit():
                        meta["dataset_idx"] = int(sample_id[1:])
                if index < len(data_ids):
                    meta["data_id"] = str(data_ids[index])
                metadata.append(meta)
            records.append(
                [
                    pred_objs,
                    pred_lhands,
                    pred_rhands,
                    text_list,
                    metadata,
                    None,
                    None,
                    None,
                    None,
                    None,
                    metadata,
                ]
            )
        return records

    def _normalize_seed_keyed_record_dict(payload: dict):
        seed_items = []
        for key, records in payload.items():
            key_text = str(key)
            if not key_text.startswith("seed_"):
                return None
            try:
                seed = int(key_text.removeprefix("seed_"))
            except ValueError:
                return None
            if not isinstance(records, (list, tuple)):
                return None
            seed_items.append((seed, records))

        if not seed_items:
            return None
        flattened = []
        for seed, records in sorted(seed_items, key=lambda item: item[0]):
            for record in records:
                if not isinstance(record, (list, tuple)):
                    continue
                normalized_record = list(record)
                if len(normalized_record) > 10 and isinstance(
                    normalized_record[10], (list, tuple)
                ):
                    normalized_meta = []
                    for value in normalized_record[10]:
                        if not isinstance(value, dict):
                            normalized_meta.append(value)
                            continue
                        meta = dict(value)
                        meta["seed"] = seed
                        meta.setdefault("repeat_idx", meta.get("repeat_index"))
                        meta.setdefault("prompt_idx", meta.get("dataset_sample_idx"))
                        meta.setdefault("dataset_idx", meta.get("dataset_sample_idx"))
                        normalized_meta.append(meta)
                    normalized_record[10] = normalized_meta
                flattened.append(normalized_record)
        return flattened

    def _normalize_seeded_hot3d_motion_dict(payload: dict):
        if "conditioning_batches" not in payload or "seeds" not in payload:
            return None
        conditioning_batches = payload.get("conditioning_batches")
        seed_payloads = payload.get("seeds")
        if not isinstance(conditioning_batches, (list, tuple)):
            return None
        if not isinstance(seed_payloads, dict):
            return None

        metadata = (
            payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        )
        inference_order = metadata.get("inference_order")
        if not isinstance(inference_order, (list, tuple)):
            inference_order = []
        effective_prompts = metadata.get("effective_prompts")
        if not isinstance(effective_prompts, (list, tuple)):
            effective_prompts = []

        def _conditioning_item(key: str, global_idx: int):
            """Return a conditioning value using a global sample index.

            ``conditioning_batches`` stores each dataloader batch separately.
            Using ``global_idx`` directly inside every batch works only for the
            first batch (typically 64 samples), then silently returns ``None``
            for all later batches.  Convert it to the batch-local index while
            traversing the batches instead.
            """
            remaining_idx = int(global_idx)
            for batch in conditioning_batches:
                if not isinstance(batch, dict):
                    continue
                values = batch.get(key)
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                if not isinstance(values, (list, tuple)):
                    continue
                if 0 <= remaining_idx < len(values):
                    return values[remaining_idx]
                remaining_idx -= len(values)
            return None

        def _text_for_sample(global_idx: int) -> str:
            order = (
                inference_order[global_idx] if global_idx < len(inference_order) else {}
            )
            if isinstance(order, dict):
                prompt_idx = _meta_int(order.get("prompt_index"))
                if prompt_idx is not None and 0 <= prompt_idx < len(effective_prompts):
                    return str(effective_prompts[prompt_idx])
            text_value = _conditioning_item("text", global_idx)
            return "" if text_value is None else str(text_value)

        def _object_name_for_sample(global_idx: int, text: str) -> str:
            object_names = _conditioning_item("object_names", global_idx)
            if isinstance(object_names, (list, tuple)) and object_names:
                return str(object_names[0]).strip().lower()
            if object_names is not None:
                return str(object_names).strip().lower()
            return _extract_object_key(text)

        records = []
        for seed, seed_records in sorted(
            seed_payloads.items(),
            key=lambda item: int(item[0]),
        ):
            if not isinstance(seed_records, (list, tuple)):
                continue
            global_idx = 0
            for seed_record in seed_records:
                if not isinstance(seed_record, dict):
                    continue
                x_obj = seed_record.get("pred_obj")
                x_lhand = seed_record.get("pred_lhand")
                x_rhand = seed_record.get("pred_rhand")
                if not all(
                    isinstance(v, (list, tuple)) for v in (x_obj, x_lhand, x_rhand)
                ):
                    continue
                batch_size = min(len(x_obj), len(x_lhand), len(x_rhand))
                if batch_size <= 0:
                    continue
                text_list = []
                object_meta_list = []
                for local_idx in range(batch_size):
                    sample_global_idx = global_idx + local_idx
                    order = (
                        inference_order[sample_global_idx]
                        if sample_global_idx < len(inference_order)
                        else {}
                    )
                    text = _text_for_sample(sample_global_idx)
                    object_name = _object_name_for_sample(sample_global_idx, text)
                    meta = {
                        "object_name": object_name,
                        "obj_name": object_name,
                        "seed": seed,
                        "repeat_idx": (
                            _meta_int(order.get("repeat_index"))
                            if isinstance(order, dict)
                            else None
                        ),
                        "prompt_idx": (
                            _meta_int(order.get("prompt_index"))
                            if isinstance(order, dict)
                            else None
                        ),
                        "dataset_idx": (
                            _meta_int(order.get("dataset_index"))
                            if isinstance(order, dict)
                            else None
                        ),
                        "seed_sample_idx": sample_global_idx,
                        "source_format": "seeded_hot3d_motion_dict",
                    }
                    text_list.append(text)
                    object_meta_list.append(meta)
                records.append(
                    [
                        list(x_obj[:batch_size]),
                        list(x_lhand[:batch_size]),
                        list(x_rhand[:batch_size]),
                        text_list,
                        object_meta_list,
                        None,
                        None,
                        None,
                        None,
                        None,
                        object_meta_list,
                    ]
                )
                global_idx += batch_size
        return records

    def _normalize_hot3d_motion_dict(payload: dict):
        required = ("x_obj", "x_lhand", "x_rhand")
        if not all(key in payload for key in required):
            return None

        x_obj = payload.get("x_obj")
        x_lhand = payload.get("x_lhand")
        x_rhand = payload.get("x_rhand")
        if (
            not isinstance(x_obj, (list, tuple))
            or not isinstance(x_lhand, (list, tuple))
            or not isinstance(x_rhand, (list, tuple))
        ):
            return None

        batch_size = min(len(x_obj), len(x_lhand), len(x_rhand))
        if batch_size <= 0:
            return []

        action_names = payload.get("action_name")
        instance_names = payload.get("instance_name")
        gaze_map = payload.get("gaze_map")
        gaze = payload.get("gaze")

        def _payload_item(key: str, idx: int):
            values = payload.get(key)
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if isinstance(values, (list, tuple)) and idx < len(values):
                value = values[idx]
                return value.item() if hasattr(value, "item") else value
            return None

        text_list = []
        object_meta_list = []
        for idx in range(batch_size):
            action_text = (
                str(action_names[idx])
                if isinstance(action_names, (list, tuple, np.ndarray))
                and len(action_names) > idx
                else ""
            )
            instance_name = (
                str(instance_names[idx]).strip().lower()
                if isinstance(instance_names, (list, tuple, np.ndarray))
                and len(instance_names) > idx
                else ""
            )
            text_list.append(action_text)
            meta = {
                "object_name": instance_name,
                "instance_name": instance_name,
                "source_format": "hot3d_motion_dict",
            }
            seed = _payload_item("seed", idx)
            repeat_idx = _payload_item("repeat_index", idx)
            dataset_idx = _payload_item("dataset_sample_idx", idx)
            if seed is not None:
                meta["seed"] = seed
            if repeat_idx is not None:
                meta["repeat_idx"] = repeat_idx
            if dataset_idx is not None:
                meta["prompt_idx"] = dataset_idx
                meta["dataset_idx"] = dataset_idx
            object_meta_list.append(meta)

        return [
            [
                list(x_obj[:batch_size]),
                list(x_lhand[:batch_size]),
                list(x_rhand[:batch_size]),
                text_list,
                object_meta_list,
                (
                    list(gaze_map[:batch_size])
                    if isinstance(gaze_map, (list, tuple))
                    else gaze_map
                ),
                list(gaze[:batch_size]) if isinstance(gaze, (list, tuple)) else gaze,
                None,
                None,
                None,
                object_meta_list,
            ]
        ]

    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            normalized = _normalize_diffh2o_multiseed_dict(payload)
            if normalized is not None:
                return normalized
        if (
            "_seed" in os.path.basename(path)
            and isinstance(payload, (list, tuple))
            and payload
            and all(
                isinstance(record, (list, tuple)) and len(record) == 7
                for record in payload
            )
        ):
            seeded_records = []
            for seed, record in enumerate(payload):
                batch_size = _batch_len_hint(record[0]) or 0
                object_meta_list = [
                    {
                        "object_name": _extract_object_key(
                            record[3][sample_idx]
                            if isinstance(record[3], (list, tuple))
                            and sample_idx < len(record[3])
                            else ""
                        ),
                        "seed": seed,
                        "repeat_idx": seed,
                        "source_format": "seed_grouped_record",
                    }
                    for sample_idx in range(batch_size)
                ]
                seeded_records.append(
                    [
                        record[0],
                        record[1],
                        record[2],
                        record[3],
                        object_meta_list,
                        record[5],
                        record[6],
                        None,
                        None,
                        None,
                        object_meta_list,
                    ]
                )
            return seeded_records
        if isinstance(payload, dict) and "save_list" in payload:
            save_list = payload["save_list"]
            if isinstance(save_list, np.ndarray):
                save_list = save_list.tolist()
            return save_list
        if isinstance(payload, dict):
            normalized = _normalize_seed_keyed_record_dict(payload)
            if normalized is not None:
                return normalized
        if isinstance(payload, dict):
            normalized = _normalize_seeded_hot3d_motion_dict(payload)
            if normalized is not None:
                return normalized
        if isinstance(payload, dict):
            normalized = _normalize_hot3d_motion_dict(payload)
            if normalized is not None:
                return normalized
        return payload
    if ext == ".npy":
        raw = np.load(path, allow_pickle=True)
        payload = raw.item() if isinstance(raw, np.ndarray) and raw.shape == () else raw
        if isinstance(payload, dict) and "save_list" in payload:
            save_list = payload["save_list"]
            if isinstance(save_list, np.ndarray):
                save_list = save_list.tolist()
            return save_list
        if not isinstance(payload, dict) or "motion" not in payload:
            raise ValueError(
                f"unsupported npy payload; expected dict with 'motion' or 'save_list': {path}"
            )

        motion = _normalize_motion_npy_array(
            payload["motion"], nsamples_hint=payload.get("num_samples", None)
        )
        if motion.shape[-1] < 207:
            raise ValueError(
                f"motion last dim must be >=207 (lhand99+rhand99+obj9), got {tuple(motion.shape)}"
            )
        text_arr = payload.get("text", None)
        lengths_arr = payload.get("lengths", None)
        nsamples = min(
            int(payload.get("num_samples", motion.shape[0])), motion.shape[0]
        )

        x_obj_list, x_lhand_list, x_rhand_list, text_list = [], [], [], []
        text_np = np.asarray(text_arr, dtype=object) if text_arr is not None else None
        lengths_np = (
            np.asarray(lengths_arr).reshape(-1) if lengths_arr is not None else None
        )
        for i in range(nsamples):
            seq = motion[i]
            t = int(seq.shape[0])
            if lengths_np is not None and i < lengths_np.shape[0]:
                try:
                    t = int(lengths_np[i])
                except Exception:
                    t = int(seq.shape[0])
            t = max(1, min(t, int(seq.shape[0])))
            seq = seq[:t]

            # motion layout: [:99]=lhand, [99:198]=rhand, [198:207]=obj
            x_lhand_list.append(seq[:, :99].astype(np.float32, copy=False))
            x_rhand_list.append(seq[:, 99:198].astype(np.float32, copy=False))
            x_obj_list.append(seq[:, 198:207].astype(np.float32, copy=False))
            if text_np is not None and i < text_np.shape[0]:
                text_list.append(str(text_np[i]))
            else:
                text_list.append("")

        return [[x_obj_list, x_lhand_list, x_rhand_list, text_list, None, None, None]]

    raise ValueError(f"unsupported input extension: {path}")


def _looks_like_eval_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "contact" in first
        or "penetration_ok" in first
        or "success" in first
        or "pen_max_mm" in first
        or "pen_vertex_info" in first
        or "contact_vertex_info" in first
    )


def _looks_like_text_field(value) -> bool:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if not isinstance(value, str):
        return False
    return len(value.strip()) > 0


def _looks_like_object_meta_list(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return (
            "object_name" in value
            or "object_names" in value
            or "obj_name" in value
            or "target_obj_name" in value
            or "pred_target_obj_name" in value
            or "obj_pc_org" in value
            or "obj_pc_org_all" in value
            or "data_id" in value
        )
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and (
        "object_name" in first
        or "object_names" in first
        or "obj_name" in first
        or "target_obj_name" in first
        or "pred_target_obj_name" in first
        or "obj_pc_org" in first
        or "obj_pc_org_all" in first
        or "data_id" in first
    )


def _looks_like_object_name_field(value) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        return len(value.strip()) > 0
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    if isinstance(first, str):
        return len(first.strip()) > 0
    if isinstance(first, (list, tuple)) and first:
        return isinstance(first[0], str) and len(first[0].strip()) > 0
    return False


def _batch_len_hint(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return len(value)
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        if value.ndim >= 3:
            return int(value.shape[0])
        if value.ndim >= 2:
            return 1
    return None


def _object_meta_list_from_names(value, batch_size_hint: Optional[int] = None):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if _looks_like_object_meta_list(value):
        return value

    def _meta_from_name_item(item):
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if isinstance(item, str):
            name = str(item).strip().lower()
            names = [name] if name else []
        elif isinstance(item, (list, tuple)):
            names = [str(name).strip().lower() for name in item if str(name).strip()]
            name = names[0] if names else ""
        else:
            name = str(item).strip().lower()
            names = [name] if name else []
        return {
            "object_name": name,
            "obj_name": name,
            "target_obj_name": name,
            "pred_target_obj_name": name,
            "object_names": names,
            "target_obj_idx": 0 if names else None,
            "pred_target_obj_idx": 0 if names else None,
            "source_format": "object_names_field",
        }

    if isinstance(value, str):
        meta = _meta_from_name_item(value)
        count = max(1, int(batch_size_hint or 1))
        return [dict(meta) for _ in range(count)]
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if batch_size_hint is not None and len(value) == int(batch_size_hint):
        return [_meta_from_name_item(item) for item in value]
    meta = _meta_from_name_item(value)
    count = max(1, int(batch_size_hint or 1))
    return [dict(meta) for _ in range(count)]


def _meta_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(np.asarray(value).item())
    except Exception:
        return None


def _meta_sequence_item(value, idx: int):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and 0 <= idx < len(value):
        return value[idx]
    return None


def _target_object_name_from_meta(
    object_meta: Optional[dict],
    target_idx: Optional[int],
    *,
    name_key: Optional[str] = None,
    fallback_text: str = "",
) -> str:
    if isinstance(object_meta, dict):
        if name_key and object_meta.get(name_key) is not None:
            return str(object_meta[name_key]).strip().lower()
        object_names = object_meta.get("object_names")
        if target_idx is not None:
            name = _meta_sequence_item(object_names, int(target_idx))
            if name is not None:
                return str(name).strip().lower()
        for key in (
            "object_name",
            "obj_name",
            "target_obj_name",
            "target_object_name",
        ):
            if object_meta.get(key) is not None:
                return str(object_meta[key]).strip().lower()
    return _extract_object_key(fallback_text)


def _select_object_pose_slot(obj_params, target_idx: Optional[int]):
    if target_idx is None:
        return obj_params
    arr = _to_numpy(obj_params)
    if arr.ndim >= 3 and arr.shape[0] <= 8 and arr.shape[-1] >= 9:
        idx = int(target_idx)
        if 0 <= idx < arr.shape[0]:
            return arr[idx]
    return obj_params


def _looks_like_batched_multi_object_pose(value) -> bool:
    if isinstance(value, np.ndarray):
        arr = value
        return arr.ndim >= 4 and arr.shape[1] <= 8 and arr.shape[-1] >= 9
    if torch.is_tensor(value):
        return value.ndim >= 4 and value.shape[1] <= 8 and value.shape[-1] >= 9
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        try:
            arr = _to_numpy(first)
        except Exception:
            return False
        return arr.ndim >= 3 and arr.shape[0] <= 8 and arr.shape[-1] >= 9
    return False


def _target_object_meta(
    object_meta: Optional[dict], target_idx: Optional[int], name: str
):
    if not isinstance(object_meta, dict):
        return object_meta
    out = dict(object_meta)
    if name:
        out["object_name"] = name
        out["obj_name"] = name
    if target_idx is not None:
        out["selected_target_obj_idx"] = int(target_idx)
    return out


def _gt_only_output_file_name(file_name: str) -> str:
    return _GT_ONLY_SOURCE_FILE_LABELS.get(os.path.basename(str(file_name)), file_name)


def _sample_as_gt_only_if_requested(sample: dict, file_name: str) -> Optional[dict]:
    if os.path.basename(str(file_name)) not in _GT_ONLY_SOURCE_FILE_LABELS:
        return sample
    gt_obj_params = sample.get("gt_obj_params")
    if gt_obj_params is None:
        return None
    out = dict(sample)
    out["obj_params"] = gt_obj_params
    out["lhand_params"] = (
        sample.get("gt_lhand_params")
        if sample.get("gt_lhand_params") is not None
        else sample.get("lhand_params")
    )
    out["rhand_params"] = (
        sample.get("gt_rhand_params")
        if sample.get("gt_rhand_params") is not None
        else sample.get("rhand_params")
    )
    if sample.get("gt_object"):
        out["object"] = sample.get("gt_object")
    return out


def _allow_gt_source_file(file_name: str) -> bool:
    base = os.path.basename(str(file_name))
    return base in {"s_bps_bim_cano_gt.pkl", *set(_GT_ONLY_SOURCE_FILE_LABELS)}


def _slice_frame_indices(params, indices: np.ndarray):
    arr = _to_numpy(params)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.ndim == 1:
        if indices.size == 0:
            return arr[:0]
        return arr.reshape(1, -1)[np.zeros(indices.shape[0], dtype=np.int64)]
    if arr.ndim >= 2:
        valid = indices[(indices >= 0) & (indices < arr.shape[0])]
        return arr[valid]
    return arr


def _iter_samples_from_record(record):
    if not isinstance(record, (list, tuple)):
        return

    gt_lhand_params = None
    gt_rhand_params = None
    gt_obj_params = None
    gt_cov_map = None
    object_meta_list = None
    gaze = None
    if (
        len(record) >= 11
        and _looks_like_text_field(record[3])
        and _looks_like_object_meta_list(record[10])
        and (
            _looks_like_batched_multi_object_pose(record[8])
            or _looks_like_batched_multi_object_pose(record[9])
        )
    ):
        # Text2HOI attention-target two-object save format:
        # [pred_target_obj, pred_lhand, pred_rhand, text, gaze_map, gaze,
        #  cov_map, gt_target_obj, pred_all_obj, gt_all_obj, object_meta,
        #  gt_lhand?, gt_rhand?]
        #
        # pred_all_obj / gt_all_obj contain the target object and the context
        # object together. Evaluate only the explicit target-object trajectory.
        x_obj = record[0]
        course_lhand = record[1]
        course_rhand = record[2]
        text = record[3]
        gt_cov_map = record[6]
        gaze = record[5]
        gt_obj_params = record[7]
        gt_lhand_params = record[11] if len(record) > 11 else None
        gt_rhand_params = record[12] if len(record) > 12 else None
        object_meta_list = record[10]
    elif len(record) >= 10 and _looks_like_eval_meta_list(record[4]):
        # New save format:
        # [x_obj, x_lhand, x_rhand, text, eval_meta_list, gaze|contact_list,
        #  pen_max_list, gt_x_obj, gt_x_lhand, gt_x_rhand, object_meta_list?]
        # The fifth slot is gaze in current HOT3D saves and contact data in
        # older saves; _gaze_target_point_index validates the shape at use time.
        x_obj, course_lhand, course_rhand, text = record[:4]
        gaze = record[5] if len(record) > 5 else None
        gt_obj_params = record[7] if len(record) > 7 else None
        gt_lhand_params = record[8] if len(record) > 8 else None
        gt_rhand_params = record[9] if len(record) > 9 else None
        if len(record) > 10 and _looks_like_object_meta_list(record[10]):
            object_meta_list = record[10]
    elif len(record) == 11:
        # HOT3D motion save format:
        # [x_obj, x_lhand, x_rhand, text, object_meta_list, ..., ..., ..., ..., ..., object_meta_list]
        if _looks_like_object_meta_list(record[4]) or _looks_like_object_meta_list(
            record[10]
        ):
            x_obj = record[0]
            course_lhand = record[1]
            course_rhand = record[2]
            text = record[3]
            object_meta_list = (
                record[10] if _looks_like_object_meta_list(record[10]) else record[4]
            )
        # New format also handled in texthoi_vis:
        # [coarse_lhand, coarse_rhand, coarse_obj, refined_obj,
        #  refined_lhand, refined_rhand, text, gaze_map, gaze, cov_map, gt_x_obj]
        elif _looks_like_text_field(record[6]):
            x_obj = record[3]
            course_lhand = record[4]
            course_rhand = record[5]
            text = record[6]
            gt_cov_map = record[9]
            gt_obj_params = record[10]
            gaze = record[8]
        else:
            (
                _fine_lhand,
                _fine_rhand,
                x_obj,
                text,
                course_lhand,
                course_rhand,
                gt_obj_params,
                _cond_enc,
                gt_cov_map,
                _est_cov_map,
                _extra,
            ) = record
    elif len(record) >= 10 and _looks_like_object_name_field(record[8]):
        # Current coarse save format:
        # [pred_obj, pred_lhand, pred_rhand, text, gaze_map, gaze,
        #  cov_map, est_contact_map, object_names|obj_name, gt_x_obj]
        #
        # Prompt-ablation saves append extra metadata fields after gt_x_obj:
        # [ ..., object_names|obj_name, gt_x_obj, original_text, prompt_mode]
        x_obj = record[0]
        course_lhand = record[1]
        course_rhand = record[2]
        text = record[3]
        gt_cov_map = record[6]
        gt_obj_params = record[9]
        gaze = record[5]
        object_meta_list = _object_meta_list_from_names(
            record[8],
            batch_size_hint=_batch_len_hint(x_obj),
        )
    elif len(record) == 10:
        (
            x_obj,
            course_lhand,
            course_rhand,
            text,
            _gaze_map,
            gaze,
            gt_cov_map,
            gt_obj_params,
            gt_lhand_params,
            gt_rhand_params,
        ) = record
    elif len(record) == 8:
        (
            x_obj,
            course_lhand,
            course_rhand,
            text,
            _gaze_map,
            gaze,
            gt_cov_map,
            gt_obj_params,
        ) = record
    elif len(record) == 7:
        (
            x_obj,
            course_lhand,
            course_rhand,
            text,
            _gaze_map,
            _gaze,
            gt_cov_map,
        ) = record
        gt_obj_params = None
    else:
        return

    def _batch_size(data) -> int:
        if data is None:
            return 0
        if isinstance(data, (list, tuple)):
            return len(data)
        if torch.is_tensor(data) or isinstance(data, np.ndarray):
            if data.ndim >= 3:
                return int(data.shape[0])
            if data.ndim >= 2:
                return 1
            return 0
        return 1

    def _batch_item(data, idx: int):
        if data is None:
            return None
        if isinstance(data, (list, tuple)):
            return data[idx] if len(data) > idx else None
        if torch.is_tensor(data) or isinstance(data, np.ndarray):
            if data.ndim >= 3:
                return data[idx] if data.shape[0] > idx else None
            if data.ndim >= 2:
                return data if idx == 0 else None
            return data if idx == 0 else None
        return data

    candidate_sizes = [
        _batch_size(x_obj),
        _batch_size(course_lhand),
        _batch_size(course_rhand),
        _batch_size(text),
        _batch_size(object_meta_list),
    ]
    batch_size = max(candidate_sizes) if any(s > 0 for s in candidate_sizes) else 0
    for i in range(batch_size):
        x_obj_i = _batch_item(x_obj, i)
        l_i = _batch_item(course_lhand, i)
        r_i = _batch_item(course_rhand, i)
        if x_obj_i is None or l_i is None or r_i is None:
            continue
        text_entry = _batch_item(text, i)
        if text_entry is None:
            text_entry = text
        object_meta = _batch_item(object_meta_list, i)
        pred_target_idx = (
            _meta_int(object_meta.get("pred_target_obj_idx"))
            if isinstance(object_meta, dict)
            else None
        )
        if pred_target_idx is None and isinstance(object_meta, dict):
            pred_target_idx = _meta_int(object_meta.get("target_object_index"))
        gt_target_idx = (
            _meta_int(object_meta.get("target_obj_idx"))
            if isinstance(object_meta, dict)
            else None
        )
        if gt_target_idx is None and isinstance(object_meta, dict):
            gt_target_idx = _meta_int(object_meta.get("target_object_index"))
        object_name = _target_object_name_from_meta(
            object_meta,
            pred_target_idx,
            name_key="pred_target_obj_name",
            fallback_text=str(text_entry),
        )
        gt_object_name = _target_object_name_from_meta(
            object_meta,
            gt_target_idx,
            name_key="target_obj_name",
            fallback_text=str(text_entry),
        )
        x_obj_i = _select_object_pose_slot(x_obj_i, pred_target_idx)
        gt_obj_i = _batch_item(gt_obj_params, i)
        gt_obj_i = _select_object_pose_slot(gt_obj_i, gt_target_idx)
        yield {
            "text": str(text_entry),
            "obj_params": x_obj_i,
            "lhand_params": l_i,
            "rhand_params": r_i,
            "gt_cov_map": _batch_item(gt_cov_map, i),
            "gaze": _batch_item(gaze, i),
            "gt_obj_params": gt_obj_i,
            "gt_lhand_params": _batch_item(gt_lhand_params, i),
            "gt_rhand_params": _batch_item(gt_rhand_params, i),
            "object_meta": _target_object_meta(
                object_meta, pred_target_idx, object_name
            ),
            "object": object_name,
            "gt_object": gt_object_name,
            "target_obj_idx": pred_target_idx,
            "gt_target_obj_idx": gt_target_idx,
            "sample_idx": i,
        }


def _voxel_intersection_volume_for_hand(
    obj_mesh: Optional[trimesh.Trimesh],
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    *,
    pitch_m: float = IV_VOXEL_PITCH_M,
    enabled: bool = True,
) -> tuple[np.ndarray, float]:
    """LatentHOI/DiffH2O-style voxel intersection volume.

    The object surface is voxelized at ``pitch_m`` and voxel centers contained
    by the closed hand mesh are counted.  This intentionally matches the
    reference implementation instead of estimating volume from the fraction
    of penetrating MANO vertices.
    """
    empty = np.zeros((0, 3), dtype=np.float32)
    if not enabled or obj_mesh is None or pitch_m <= 0.0:
        return empty, 0.0
    vertices = np.asarray(hand_vertices, dtype=np.float64)
    faces = np.asarray(hand_faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or vertices.shape[0] == 0
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.shape[0] == 0
    ):
        return empty, 0.0
    try:
        hand_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        object_voxel_points = np.asarray(
            obj_mesh.voxelized(pitch=float(pitch_m)).points,
            dtype=np.float64,
        )
        if object_voxel_points.ndim != 2 or object_voxel_points.shape[0] == 0:
            return empty, 0.0
        overlap_mask = np.asarray(
            hand_mesh.contains(object_voxel_points), dtype=bool
        ).reshape(-1)
        overlap_points = np.asarray(object_voxel_points[overlap_mask], dtype=np.float32)
        volume_m3 = float(overlap_points.shape[0]) * float(pitch_m) ** 3
        return overlap_points, volume_m3
    except Exception as ex:
        print(f"[WARN] voxel IV failed: {ex}")
        return empty, 0.0


def _new_inside_vertex_metric_for_hand(
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_vertices_world: np.ndarray,
    hand_faces: np.ndarray,
    hand_name: str,
    expected_vertex_count: int = 778,
) -> dict:
    hand_vertices = np.asarray(hand_vertices_world, dtype=np.float64)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    empty = {
        "hand": hand_name,
        "inside_mask": np.zeros((int(hand_vertices.shape[0]),), dtype=bool),
        "inside_indices": np.zeros((0,), dtype=np.int64),
        "inside_points": np.zeros((0, 3), dtype=np.float32),
        "closest_points": np.zeros((0, 3), dtype=np.float32),
        "id_distances_m": np.zeros((0,), dtype=np.float32),
        "id_mean_mm": 0.0,
        "id_max_mm": 0.0,
        "inside_count": 0,
        "inside_ratio_per_778": 0.0,
        "hand_volume_m3": 0.0,
        "iv_volume_m3": 0.0,
        "iv_volume_cm3": 0.0,
        "iv_points": np.zeros((0, 3), dtype=np.float32),
        "iv_method": "object_surface_voxels_inside_hand",
        "iv_voxel_pitch_m": float(IV_VOXEL_PITCH_M),
        "id_face_indices": np.zeros((0,), dtype=np.int64),
        "inside_method": "unavailable",
    }
    if (
        obj_mesh_world is None
        or hand_vertices.ndim != 2
        or hand_vertices.shape[1] != 3
        or hand_vertices.shape[0] == 0
    ):
        return empty

    finite = np.all(np.isfinite(hand_vertices), axis=1)
    if not np.any(finite):
        return empty
    query_points = hand_vertices[finite]

    closest_points = np.full(query_points.shape, np.nan, dtype=np.float64)
    distances_m = np.full((query_points.shape[0],), np.nan, dtype=np.float64)
    tri_ids = np.full((query_points.shape[0],), -1, dtype=np.int64)
    try:
        closest_points, distances_m, tri_ids = trimesh.proximity.closest_point(
            obj_mesh_world,
            query_points,
        )
        closest_points = np.asarray(closest_points, dtype=np.float64)
        distances_m = np.asarray(distances_m, dtype=np.float64).reshape(-1)
        tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)
    except Exception:
        vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
        if vertices.ndim == 2 and vertices.shape[0] > 0:
            nearest_idx = np.argmin(
                np.linalg.norm(query_points[:, None, :] - vertices[None, :, :], axis=2),
                axis=1,
            )
            closest_points = vertices[nearest_idx]
            distances_m = np.linalg.norm(query_points - closest_points, axis=1)

    contains_inside = None
    try:
        contains_inside = np.asarray(obj_mesh_world.contains(query_points), dtype=bool)
    except Exception:
        contains_inside = None

    signed_inside = None
    try:
        signed = np.asarray(
            trimesh.proximity.signed_distance(obj_mesh_world, query_points),
            dtype=np.float64,
        ).reshape(-1)
        signed_inside = np.isfinite(signed) & (signed > 1e-7)
    except Exception:
        signed_inside = None

    normal_inside = None
    try:
        face_normals = np.asarray(obj_mesh_world.face_normals, dtype=np.float64)
        valid_tri = (
            (tri_ids >= 0)
            & (tri_ids < face_normals.shape[0])
            & np.all(np.isfinite(closest_points), axis=1)
        )
        normal_inside = np.zeros((query_points.shape[0],), dtype=bool)
        if np.any(valid_tri):
            direction = query_points[valid_tri] - closest_points[valid_tri]
            dot = np.einsum("ij,ij->i", direction, face_normals[tri_ids[valid_tri]])
            normal_inside[valid_tri] = dot < -1e-7
    except Exception:
        normal_inside = None

    if contains_inside is not None:
        inside_query = contains_inside
        method = "mesh.contains"
    elif signed_inside is not None:
        inside_query = signed_inside
        method = "signed_distance"
    elif normal_inside is not None:
        inside_query = normal_inside
        method = "face_normal_direction"
    else:
        inside_query = np.zeros((query_points.shape[0],), dtype=bool)
        method = "unavailable"
    inside_query = _remove_open_cavity_points(
        obj_mesh_world,
        query_points,
        inside_query,
        height_axis=1,
    )

    inside_mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
    valid_idx = np.flatnonzero(finite)
    inside_mask[valid_idx] = inside_query
    inside_indices = np.flatnonzero(inside_mask).astype(np.int64)
    empty["inside_mask"] = inside_mask
    empty["inside_indices"] = inside_indices
    empty["inside_count"] = int(inside_indices.shape[0])
    empty["inside_ratio_per_778"] = float(inside_indices.shape[0]) / float(
        expected_vertex_count
    )
    empty["inside_method"] = method

    if inside_indices.shape[0] > 0:
        inside_query_idx = np.flatnonzero(inside_query).astype(np.int64)
        inside_points = np.asarray(hand_vertices[inside_indices], dtype=np.float32)
        seed_faces = tri_ids[inside_query_idx]
        valid_faces = _object_surface_patch_faces_for_penetration(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            seed_faces,
        )
        closest_inside, id_distances_m = _closest_points_on_object_patch(
            obj_mesh_world,
            inside_points,
            valid_faces,
        )
        line_count = min(int(inside_points.shape[0]), int(closest_inside.shape[0]))
        inside_points = inside_points[:line_count]
        closest_inside = closest_inside[:line_count]
        id_distances_m = np.asarray(id_distances_m[:line_count], dtype=np.float32)
        inside_indices = inside_indices[:line_count]
        inside_query_idx = inside_query_idx[:line_count]
        valid_penetration = np.isfinite(id_distances_m) & (
            id_distances_m > MIN_ID_PENETRATION_DEPTH_M
        )
        inside_points = inside_points[valid_penetration]
        closest_inside = closest_inside[valid_penetration]
        id_distances_m = id_distances_m[valid_penetration]
        inside_indices = inside_indices[valid_penetration]
        inside_query_idx = inside_query_idx[valid_penetration]
        filtered_inside_mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
        filtered_inside_mask[inside_indices] = True
        empty["inside_mask"] = filtered_inside_mask
        empty["inside_indices"] = inside_indices
        empty["inside_count"] = int(inside_indices.shape[0])
        empty["inside_ratio_per_778"] = float(inside_indices.shape[0]) / float(
            expected_vertex_count
        )
        if id_distances_m.shape[0] > 0:
            # Match the LatentHOI/DiffH2O penetration-depth definition: a
            # frame is represented by its deepest penetrating hand vertex,
            # rather than by the mean over all penetrating vertices.
            frame_id_max_mm = float(np.max(id_distances_m) * 1000.0)
            empty["id_mean_mm"] = frame_id_max_mm
            empty["id_max_mm"] = frame_id_max_mm
            filtered_seed_faces = tri_ids[inside_query_idx]
            valid_faces = _object_surface_patch_faces_for_penetration(
                obj_mesh_world,
                hand_vertices,
                hand_faces,
                filtered_seed_faces,
            )
        else:
            valid_faces = np.zeros((0,), dtype=np.int64)
        empty["inside_points"] = inside_points
        empty["closest_points"] = closest_inside
        empty["id_line_hand_points"] = inside_points
        empty["id_line_object_points"] = closest_inside
        empty["id_distances_m"] = id_distances_m
        empty["id_face_indices"] = valid_faces

    if QUICK_EVAL_ACTIVE:
        iv_points = np.zeros((0, 3), dtype=np.float32)
        iv_volume_m3 = 0.0
    else:
        iv_points, iv_volume_m3 = _voxel_intersection_volume_for_hand(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            pitch_m=IV_VOXEL_PITCH_M,
            enabled=empty["inside_count"] > 0,
        )
    empty["hand_volume_m3"] = 0.0
    empty["iv_points"] = iv_points
    empty["iv_method"] = "object_surface_voxels_inside_hand"
    empty["iv_voxel_pitch_m"] = float(IV_VOXEL_PITCH_M)
    empty["iv_volume_m3"] = float(iv_volume_m3)
    empty["iv_volume_cm3"] = float(iv_volume_m3 * 1e6)
    return empty


def _new_inside_vertex_metric_for_hand_fast(
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_vertices_world: np.ndarray,
    hand_faces: np.ndarray,
    hand_name: str,
    expected_vertex_count: int = 778,
) -> dict:
    hand_vertices = np.asarray(hand_vertices_world, dtype=np.float64)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    metric = _new_inside_vertex_metric_for_hand(
        None,
        hand_vertices,
        hand_faces,
        hand_name,
        expected_vertex_count=expected_vertex_count,
    )
    if obj_mesh_world is None or hand_vertices.ndim != 2 or hand_vertices.shape[1] != 3:
        return metric
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        return metric
    finite = np.all(np.isfinite(hand_vertices), axis=1)
    if not np.any(finite):
        return metric
    query_points = hand_vertices[finite]
    if cKDTree is not None:
        tree = cKDTree(vertices)
        distances_m, nearest_idx = tree.query(query_points, k=1)
    else:
        nearest_idx = np.argmin(
            np.linalg.norm(query_points[:, None, :] - vertices[None, :, :], axis=2),
            axis=1,
        )
        distances_m = np.linalg.norm(query_points - vertices[nearest_idx], axis=1)
    nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
    distances_m = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    closest_points = vertices[nearest_idx]

    normals = np.asarray(obj_mesh_world.vertex_normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape != vertices.shape:
        normals = np.zeros_like(vertices, dtype=np.float64)
    center = np.mean(vertices, axis=0, keepdims=True)
    radial_vertices = vertices - center
    normal_radial_dot = np.einsum("ij,ij->i", normals, radial_vertices)
    finite_orientation = np.isfinite(normal_radial_dot)
    if np.any(finite_orientation) and float(np.nanmean(normal_radial_dot)) < 0.0:
        normals = -normals
        normal_radial_dot = -normal_radial_dot
    flip_local = finite_orientation & (normal_radial_dot < 0.0)
    if np.any(flip_local):
        normals = normals.copy()
        normals[flip_local] *= -1.0
    direction = query_points - closest_points
    dot = np.einsum("ij,ij->i", direction, normals[nearest_idx])
    query_radius = np.linalg.norm(query_points - center, axis=1)
    surface_radius = np.linalg.norm(closest_points - center, axis=1)
    radial_inside = query_radius <= (surface_radius + 1e-4)
    inside_query = np.isfinite(dot) & (dot < -1e-7) & radial_inside
    inside_query = _remove_open_cavity_points(
        obj_mesh_world,
        query_points,
        inside_query,
        height_axis=1,
    )

    inside_mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
    valid_idx = np.flatnonzero(finite)
    inside_mask[valid_idx] = inside_query
    inside_indices = np.flatnonzero(inside_mask).astype(np.int64)

    metric["inside_mask"] = inside_mask
    metric["inside_indices"] = inside_indices
    metric["inside_count"] = int(inside_indices.shape[0])
    metric["inside_ratio_per_778"] = float(inside_indices.shape[0]) / float(
        expected_vertex_count
    )
    metric["inside_method"] = "fast_kdtree_vertex_normal_radial"

    inside_query_idx = np.flatnonzero(inside_query).astype(np.int64)
    if inside_query_idx.shape[0] > 0:
        inside_points = hand_vertices[inside_indices].astype(np.float32)
        seed_faces = np.zeros((0,), dtype=np.int64)
        seed_tri_ids_for_points = None
        try:
            _seed_closest, _seed_distances, seed_tri_ids = (
                trimesh.proximity.closest_point(
                    obj_mesh_world,
                    inside_points,
                )
            )
            faces = np.asarray(obj_mesh_world.faces, dtype=np.int64)
            seed_tri_ids = np.asarray(seed_tri_ids, dtype=np.int64).reshape(-1)
            seed_tri_ids_for_points = seed_tri_ids.copy()
            seed_faces = seed_tri_ids[
                (seed_tri_ids >= 0) & (seed_tri_ids < faces.shape[0])
            ]
        except Exception:
            faces = np.asarray(obj_mesh_world.faces, dtype=np.int64)
            if faces.ndim == 2 and faces.shape[1] == 3 and faces.shape[0] > 0:
                vertex_to_faces = getattr(obj_mesh_world, "vertex_faces", None)
                face_parts = []
                if vertex_to_faces is not None:
                    for vertex_idx in nearest_idx[inside_query_idx]:
                        vf = np.asarray(
                            vertex_to_faces[int(vertex_idx)], dtype=np.int64
                        )
                        vf = vf[(vf >= 0) & (vf < faces.shape[0])]
                        if vf.size:
                            face_parts.append(vf)
                seed_faces = (
                    np.unique(np.concatenate(face_parts)).astype(np.int64)
                    if face_parts
                    else np.zeros((0,), dtype=np.int64)
                )
        patch_faces = _object_surface_patch_faces_for_penetration(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            seed_faces,
        )
        closest_inside, id_distances_m = _closest_points_on_object_patch(
            obj_mesh_world,
            inside_points,
            patch_faces,
        )
        line_count = min(int(inside_points.shape[0]), int(closest_inside.shape[0]))
        inside_points = inside_points[:line_count]
        closest_inside = closest_inside[:line_count]
        id_distances_m = np.asarray(id_distances_m[:line_count], dtype=np.float32)
        inside_indices = inside_indices[:line_count]
        inside_query_idx = inside_query_idx[:line_count]
        valid_penetration = np.isfinite(id_distances_m) & (
            id_distances_m > MIN_ID_PENETRATION_DEPTH_M
        )
        inside_points = inside_points[valid_penetration]
        closest_inside = closest_inside[valid_penetration]
        id_distances_m = id_distances_m[valid_penetration]
        inside_indices = inside_indices[valid_penetration]
        inside_query_idx = inside_query_idx[valid_penetration]
        filtered_inside_mask = np.zeros((hand_vertices.shape[0],), dtype=bool)
        filtered_inside_mask[inside_indices] = True
        metric["inside_mask"] = filtered_inside_mask
        metric["inside_indices"] = inside_indices
        metric["inside_count"] = int(inside_indices.shape[0])
        metric["inside_ratio_per_778"] = float(inside_indices.shape[0]) / float(
            expected_vertex_count
        )
        if id_distances_m.shape[0] > 0:
            frame_id_max_mm = float(np.max(id_distances_m) * 1000.0)
            metric["id_mean_mm"] = frame_id_max_mm
            metric["id_max_mm"] = frame_id_max_mm
            if (
                seed_tri_ids_for_points is not None
                and seed_tri_ids_for_points.shape[0] >= line_count
            ):
                filtered_seed_faces = seed_tri_ids_for_points[:line_count][
                    valid_penetration
                ]
                patch_faces = _object_surface_patch_faces_for_penetration(
                    obj_mesh_world,
                    hand_vertices,
                    hand_faces,
                    filtered_seed_faces,
                )
        else:
            patch_faces = np.zeros((0,), dtype=np.int64)
        metric["inside_points"] = inside_points
        metric["closest_points"] = closest_inside
        metric["id_line_hand_points"] = inside_points
        metric["id_line_object_points"] = closest_inside
        metric["id_distances_m"] = id_distances_m
        metric["id_face_indices"] = patch_faces

    if QUICK_EVAL_ACTIVE:
        iv_points, iv_volume_m3 = np.zeros((0, 3), dtype=np.float32), 0.0
    else:
        iv_points, iv_volume_m3 = _voxel_intersection_volume_for_hand(
            obj_mesh_world,
            hand_vertices,
            hand_faces,
            pitch_m=IV_VOXEL_PITCH_M,
            enabled=metric["inside_count"] > 0,
        )
    metric["hand_volume_m3"] = 0.0
    metric["iv_points"] = iv_points
    metric["iv_method"] = "object_surface_voxels_inside_hand"
    metric["iv_voxel_pitch_m"] = float(IV_VOXEL_PITCH_M)
    metric["iv_volume_m3"] = float(iv_volume_m3)
    metric["iv_volume_cm3"] = float(iv_volume_m3 * 1e6)
    return metric


def _fast_cr_for_frame(
    obj_mesh_world: Optional[trimesh.Trimesh],
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
    if vertices.ndim != 2 or vertices.shape[0] == 0 or not hands:
        return 0.0
    hand_vertices = np.concatenate(hands, axis=0)
    finite = np.all(np.isfinite(hand_vertices), axis=1)
    if not np.any(finite):
        return 0.0
    points = hand_vertices[finite]
    if cKDTree is not None:
        tree = cKDTree(vertices)
        distances_m, _idx = tree.query(points, k=1)
    else:
        distances_m = np.min(
            np.linalg.norm(points[:, None, :] - vertices[None, :, :], axis=2),
            axis=1,
        )
    distances_m = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    return float(np.count_nonzero(distances_m < float(threshold_m))) / float(
        points.shape[0]
    )


def _fast_contact_mask_for_hand(
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_vertices: np.ndarray,
    threshold_m: float = 0.005,
) -> np.ndarray:
    hand_vertices = np.asarray(hand_vertices, dtype=np.float64)
    mask = np.zeros((int(hand_vertices.shape[0]),), dtype=bool)
    if (
        obj_mesh_world is None
        or hand_vertices.ndim != 2
        or hand_vertices.shape[1] != 3
        or hand_vertices.shape[0] == 0
    ):
        return mask
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    finite = np.all(np.isfinite(hand_vertices), axis=1)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or not np.any(finite):
        return mask
    points = hand_vertices[finite]
    if cKDTree is not None:
        tree = cKDTree(vertices)
        distances_m, _idx = tree.query(points, k=1)
    else:
        distances_m = np.min(
            np.linalg.norm(points[:, None, :] - vertices[None, :, :], axis=2),
            axis=1,
        )
    local = np.flatnonzero(finite)
    mask[local[np.asarray(distances_m, dtype=np.float64) < float(threshold_m)]] = True
    return mask


def _remove_open_cavity_points(
    obj_mesh_world: Optional[trimesh.Trimesh],
    query_points: np.ndarray,
    inside_query: np.ndarray,
    height_axis: int = 1,
) -> np.ndarray:
    if obj_mesh_world is None:
        return inside_query
    query_points = np.asarray(query_points, dtype=np.float64)
    inside_query = np.asarray(inside_query, dtype=bool).reshape(-1).copy()
    candidate_idx = np.flatnonzero(inside_query)
    if candidate_idx.size == 0:
        return inside_query
    directions = np.zeros((candidate_idx.size, 3), dtype=np.float64)
    directions[:, int(height_axis)] = 1.0
    origins = query_points[candidate_idx].copy()
    origins[:, int(height_axis)] += 1e-5
    try:
        hit_up = np.asarray(
            obj_mesh_world.ray.intersects_any(origins, directions),
            dtype=bool,
        ).reshape(-1)
    except Exception:
        return inside_query
    # If a point can go upward through an opening without crossing the mesh, it
    # is in an empty cavity, not in object material. This fixes bowls/cups whose
    # hollow interior should not count as penetration.
    inside_query[candidate_idx[~hit_up]] = False
    return inside_query


def _object_surface_patch_faces_for_penetration(
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_vertices: np.ndarray,
    hand_faces: np.ndarray,
    seed_face_indices: np.ndarray,
) -> np.ndarray:
    if obj_mesh_world is None:
        return np.zeros((0,), dtype=np.int64)
    obj_vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    obj_faces = np.asarray(obj_mesh_world.faces, dtype=np.int64)
    hand_vertices = np.asarray(hand_vertices, dtype=np.float64)
    hand_faces = np.asarray(hand_faces, dtype=np.int64)
    if (
        obj_vertices.ndim != 2
        or obj_vertices.shape[1] != 3
        or obj_faces.ndim != 2
        or obj_faces.shape[1] != 3
        or obj_faces.shape[0] == 0
    ):
        return np.zeros((0,), dtype=np.int64)
    seed_face_indices = np.asarray(seed_face_indices, dtype=np.int64).reshape(-1)
    valid_seed = seed_face_indices[
        (seed_face_indices >= 0) & (seed_face_indices < obj_faces.shape[0])
    ]
    if valid_seed.size > 0:
        return np.unique(valid_seed).astype(np.int64)
    return np.zeros((0,), dtype=np.int64)


def _closest_points_on_object_patch(
    obj_mesh_world: Optional[trimesh.Trimesh],
    query_points: np.ndarray,
    patch_face_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_points = np.asarray(query_points, dtype=np.float64)
    empty_points = np.zeros((0, 3), dtype=np.float32)
    empty_distances = np.zeros((0,), dtype=np.float32)
    if obj_mesh_world is None or query_points.ndim != 2 or query_points.shape[0] == 0:
        return empty_points, empty_distances
    obj_vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    obj_faces = np.asarray(obj_mesh_world.faces, dtype=np.int64)
    patch_face_indices = np.asarray(patch_face_indices, dtype=np.int64).reshape(-1)
    valid_faces = np.unique(
        patch_face_indices[
            (patch_face_indices >= 0) & (patch_face_indices < obj_faces.shape[0])
        ]
    )
    if valid_faces.size == 0:
        return empty_points, empty_distances
    try:
        patch_mesh = trimesh.Trimesh(
            vertices=obj_vertices,
            faces=obj_faces[valid_faces],
            process=False,
        )
        closest_points, distances_m, _tri_ids = trimesh.proximity.closest_point(
            patch_mesh,
            query_points,
        )
        return (
            np.asarray(closest_points, dtype=np.float32),
            np.asarray(distances_m, dtype=np.float32).reshape(-1),
        )
    except Exception:
        patch_vertex_idx = np.unique(obj_faces[valid_faces].reshape(-1))
        patch_vertices = obj_vertices[patch_vertex_idx]
        if patch_vertices.shape[0] == 0:
            return empty_points, empty_distances
        nearest_idx = np.argmin(
            np.linalg.norm(
                query_points[:, None, :] - patch_vertices[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        closest_points = patch_vertices[nearest_idx]
        distances_m = np.linalg.norm(query_points - closest_points, axis=1)
        return closest_points.astype(np.float32), distances_m.astype(np.float32)


def _object_face_area_sum_m2(
    obj_mesh_world: Optional[trimesh.Trimesh],
    face_indices: np.ndarray,
) -> float:
    if obj_mesh_world is None:
        return 0.0
    face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    if face_indices.size == 0:
        return 0.0
    faces = np.asarray(obj_mesh_world.faces, dtype=np.int64)
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float64)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or vertices.ndim != 2
        or vertices.shape[1] != 3
    ):
        return 0.0
    valid = np.unique(
        face_indices[(face_indices >= 0) & (face_indices < faces.shape[0])]
    )
    if valid.size == 0:
        return 0.0
    tri = vertices[faces[valid]]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    areas = areas[np.isfinite(areas)]
    return float(np.sum(areas)) if areas.size else 0.0


def _combine_new_inside_vertex_metrics(hand_metrics: list[dict]) -> dict:
    valid = [m for m in hand_metrics if isinstance(m, dict)]
    distances = [
        np.asarray(m.get("id_distances_m"), dtype=np.float64)
        for m in valid
        if np.asarray(m.get("id_distances_m"), dtype=np.float64).size > 0
    ]
    distance_values = (
        np.concatenate(distances, axis=0)
        if distances
        else np.zeros((0,), dtype=np.float64)
    )
    finite_dist = distance_values[np.isfinite(distance_values)]
    inside_count = int(sum(int(m.get("inside_count", 0)) for m in valid))
    active_hands = int(len(valid))
    ratio_denominator = 778 * max(1, active_hands)
    iv_volume_m3 = float(sum(float(m.get("iv_volume_m3", 0.0)) for m in valid))
    iv_volume_cm3 = float(sum(float(m.get("iv_volume_cm3", 0.0)) for m in valid))
    frame_id_max_mm = float(np.max(finite_dist) * 1000.0) if finite_dist.size else 0.0
    return {
        # Both legacy per-frame CSV fields carry the frame penetration depth.
        # Their distinction is made at sample aggregation time: mean versus
        # max over the sequence of frame maxima.
        "id_mean_mm": frame_id_max_mm,
        "id_max_mm": frame_id_max_mm,
        "iv_volume_m3": iv_volume_m3,
        "iv_volume_cm3": iv_volume_cm3,
        "inside_count": inside_count,
        "inside_ratio_per_778_per_hand": float(inside_count) / float(ratio_denominator),
        "active_hands": active_hands,
    }


def _combine_new_frame_metrics(
    hand_metrics: list[dict],
    obj_mesh_world: Optional[trimesh.Trimesh],
) -> dict:
    combined = _combine_new_inside_vertex_metrics(hand_metrics)
    return combined


def _safe_path_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    token = "_".join(part for part in token.split("_") if part)
    return token[:80] if token else "unknown"


def _rerun_set_time_sequence(name: str, value: int) -> None:
    if rr is None:
        return
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(name, value)
    else:
        rr.set_time(name, sequence=value)


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tcp_port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def _spawn_rerun_viewer_process(
    port: int,
    *,
    viewer: str,
    web_port: int,
) -> Optional[subprocess.Popen]:
    viewer_exe = shutil.which("rerun")
    print(f"[RERUN] viewer executable: {viewer_exe}")
    if viewer_exe is None:
        return None
    cmd = [
        viewer_exe,
        "--port",
        str(int(port)),
        "--expect-data-soon",
        "--hide-welcome-screen",
    ]
    if viewer == "web":
        cmd.extend(["--web-viewer", "--web-viewer-port", str(int(web_port))])
        print(f"[RERUN] web viewer URL: http://127.0.0.1:{int(web_port)}")
    else:
        print("[RERUN] launching native viewer")
    proc = subprocess.Popen(cmd, start_new_session=True)
    time.sleep(1.0)
    if proc.poll() is not None:
        if proc.returncode == 0:
            print(
                "[RERUN] viewer command exited cleanly; using the existing "
                f"viewer on port {int(port)}."
            )
            return None
        raise RuntimeError(
            "Rerun viewer process exited immediately. Try running "
            f"`{viewer_exe}` directly from the terminal."
        )
    return proc


def _init_rerun_viewer(
    application_id: str,
    *,
    rerun_port: int = 9876,
    rerun_web_port: int = 9090,
    viewer: str = "native",
) -> None:
    global _RERUN_VIEWER_PROCESS
    if rr is None:
        raise RuntimeError("rerun is not installed")

    python_bin_dir = os.path.dirname(sys.executable)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin_dir and python_bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join([python_bin_dir, *path_entries])

    rerun_port = int(rerun_port)
    if rerun_port <= 0:
        rerun_port = 9876
    rerun_web_port = int(rerun_web_port)
    if viewer == "web" and rerun_web_port == rerun_port:
        rerun_web_port = _pick_free_tcp_port()

    print(f"[RERUN] using viewer for '{application_id}' on port {rerun_port}...")
    rr.init(application_id, spawn=False)
    try:
        if _tcp_port_is_open(rerun_port):
            print(f"[RERUN] reusing existing viewer on port {rerun_port}.")
            _RERUN_VIEWER_PROCESS = None
        else:
            _RERUN_VIEWER_PROCESS = _spawn_rerun_viewer_process(
                rerun_port,
                viewer=viewer,
                web_port=rerun_web_port,
            )
        rr.connect_grpc(f"rerun+http://127.0.0.1:{rerun_port}/proxy")
        print(f"[RERUN] connected to viewer on port {rerun_port}.")
    except RuntimeError as exc:
        if "Failed to find Rerun Viewer executable in PATH" not in str(exc):
            raise
        raise RuntimeError(
            "Rerun Viewer executable was not found. Activate the hot3d "
            "environment or pass --new-metric-output-dir for GLB export."
        ) from exc


def _finish_rerun_viewer(wait_seconds: float) -> None:
    global _RERUN_VIEWER_PROCESS
    if rr is None:
        return
    wait_seconds = float(wait_seconds)
    try:
        if wait_seconds < 0.0:
            print("[RERUN] live viewer is open. Press Ctrl-C to stop.")
            while True:
                time.sleep(1.0)
        elif wait_seconds > 0.0:
            print(
                f"[RERUN] keeping stream alive for {wait_seconds:g}s "
                "so the viewer can finish receiving data..."
            )
            time.sleep(wait_seconds)
    except KeyboardInterrupt:
        print("\n[RERUN] stopping live stream.")
    finally:
        try:
            rr.disconnect()
        except Exception as ex:
            print(f"[WARN] failed to disconnect Rerun cleanly: {ex}")
        if _RERUN_VIEWER_PROCESS is not None and _RERUN_VIEWER_PROCESS.poll() is None:
            print("[RERUN] leaving viewer open for reuse.")
        _RERUN_VIEWER_PROCESS = None


def _colors_for_inside_vertices(
    vertex_count: int,
    inside_mask: np.ndarray,
    base_color=(180, 180, 180),
    inside_color=(255, 0, 0),
) -> np.ndarray:
    colors = np.tile(np.asarray(base_color, dtype=np.uint8), (int(vertex_count), 1))
    mask = np.asarray(inside_mask, dtype=bool).reshape(-1)
    if mask.shape[0] == colors.shape[0] and np.any(mask):
        colors[mask] = np.asarray(inside_color, dtype=np.uint8)
    return colors


def _new_metric_distance_labels_mm(distances_m: np.ndarray, count: int) -> list[str]:
    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    labels = []
    for idx in range(int(count)):
        if idx < distances.shape[0] and np.isfinite(distances[idx]):
            labels.append(f"ID {distances[idx] * 1000.0:.2f} mm")
        else:
            labels.append("ID")
    return labels


def _inside_face_indices_from_mask(
    faces: np.ndarray,
    inside_mask: np.ndarray,
) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    mask = np.asarray(inside_mask, dtype=bool).reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3 or mask.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    valid_faces = np.all((faces >= 0) & (faces < mask.shape[0]), axis=1)
    if not np.any(valid_faces):
        return np.zeros((0, 3), dtype=np.int32)
    faces_valid = faces[valid_faces]
    keep = np.any(mask[faces_valid], axis=1)
    return faces_valid[keep].astype(np.int32)


def _format_success_fail(value) -> str:
    if value is None:
        return "NA"
    return "Success" if bool(value) else "Fail"


def _frame_has_positive_cr(cr_value) -> Optional[bool]:
    cr_float = _finite_float_or_none(cr_value)
    if cr_float is None:
        return None
    return bool(cr_float > 0.0)


def _pen_1cm_event_from_new_id(frame_metric: dict, cr_value=None):
    # Pen_1cm is defined only on contact frames.  Approach/non-contact frames
    # must remain unavailable rather than becoming artificial successes from
    # their zero penetration depth.
    has_eval_cr = _frame_has_positive_cr(cr_value)
    if has_eval_cr is not True:
        return None
    try:
        id_max_mm = float(frame_metric.get("id_max_mm", 0.0))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(id_max_mm):
        return None
    return bool(id_max_mm <= 10.0)


def _finite_float_or_none(value) -> Optional[float]:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_float):
        return None
    return value_float


def _frame_con_pass_from_cr(cr_value) -> Optional[bool]:
    cr_float = _finite_float_or_none(cr_value)
    if cr_float is None:
        return None
    return bool(cr_float > 0.0)


def _success_from_contact_and_pen_1cm(
    con_pass: Optional[bool],
    pen_1cm_event: Optional[bool],
) -> Optional[bool]:
    if con_pass is None:
        return None
    return bool(con_pass and pen_1cm_event is True)


def _mesh_bottom_axis_value(
    mesh: Optional[trimesh.Trimesh],
    axis: int,
) -> Optional[float]:
    if mesh is None:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        return None
    finite = np.all(np.isfinite(vertices), axis=1)
    if not np.any(finite):
        return None
    return float(np.min(vertices[finite, int(axis)]))


def _active_hand_vertices_world_for_frame(
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
    frame_idx: int,
) -> Optional[np.ndarray]:
    parts = []
    if use_left and l_seq.ndim == 3 and frame_idx < l_seq.shape[0]:
        parts.append(np.asarray(l_seq[frame_idx], dtype=np.float64))
    if use_right and r_seq.ndim == 3 and frame_idx < r_seq.shape[0]:
        parts.append(np.asarray(r_seq[frame_idx], dtype=np.float64))
    parts = [
        item
        for item in parts
        if item.ndim == 2 and item.shape[1] == 3 and item.shape[0] > 0
    ]
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _frame_hand_accel_m(
    l_seq: np.ndarray,
    r_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
    frame_idx: int,
) -> Optional[float]:
    if frame_idx < 2:
        return None
    verts_t = _active_hand_vertices_world_for_frame(
        l_seq, r_seq, use_left, use_right, frame_idx
    )
    verts_t1 = _active_hand_vertices_world_for_frame(
        l_seq, r_seq, use_left, use_right, frame_idx - 1
    )
    verts_t2 = _active_hand_vertices_world_for_frame(
        l_seq, r_seq, use_left, use_right, frame_idx - 2
    )
    if verts_t is None or verts_t1 is None or verts_t2 is None:
        return None
    if verts_t.shape != verts_t1.shape or verts_t.shape != verts_t2.shape:
        return None
    accel = verts_t - 2.0 * verts_t1 + verts_t2
    finite = np.all(np.isfinite(accel), axis=1)
    if not np.any(finite):
        return None
    return float(np.linalg.norm(accel[finite], axis=1).mean())


def _canonical_wrist_frame_from_joints(
    l_joints_seq: np.ndarray,
    r_joints_seq: np.ndarray,
    use_left: bool,
    use_right: bool,
    frame_idx: int,
    first_obj_params,
) -> np.ndarray:
    wrists_world = np.full((2, 3), np.nan, dtype=np.float32)
    if use_left and l_joints_seq.ndim == 3 and frame_idx < l_joints_seq.shape[0]:
        joints = np.asarray(l_joints_seq[frame_idx], dtype=np.float64)
        if joints.ndim == 2 and joints.shape[0] > 0 and joints.shape[1] == 3:
            wrists_world[0] = joints[0]
    if use_right and r_joints_seq.ndim == 3 and frame_idx < r_joints_seq.shape[0]:
        joints = np.asarray(r_joints_seq[frame_idx], dtype=np.float64)
        if joints.ndim == 2 and joints.shape[0] > 0 and joints.shape[1] == 3:
            wrists_world[1] = joints[0]

    valid = np.all(np.isfinite(wrists_world), axis=1)
    wrists_canonical = np.full((2, 3), np.nan, dtype=np.float32)
    if np.any(valid):
        wrists_canonical[valid] = _transform_points_world_to_object_frame(
            wrists_world[valid],
            first_obj_params,
        )
    return wrists_canonical


def _active_wrist_centroid(traj: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(traj), axis=2)
    out = np.full((traj.shape[0], 3), np.nan, dtype=np.float32)
    for frame_idx in range(traj.shape[0]):
        if np.any(valid[frame_idx]):
            out[frame_idx] = np.mean(traj[frame_idx, valid[frame_idx]], axis=0)
    return out


def _wrist_trajectory_distance_m(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3 or a.shape[-1] != 3 or b.shape[-1] != 3:
        return None
    t = min(int(a.shape[0]), int(b.shape[0]))
    if t <= 0:
        return None

    vals = []
    shared_hands = min(a.shape[1], b.shape[1])
    for hand_idx in range(shared_hands):
        a_hand = a[:t, hand_idx, :]
        b_hand = b[:t, hand_idx, :]
        valid = np.all(np.isfinite(a_hand), axis=1) & np.all(
            np.isfinite(b_hand), axis=1
        )
        if np.any(valid):
            vals.extend(np.linalg.norm(a_hand[valid] - b_hand[valid], axis=1).tolist())
    if vals:
        return float(np.mean(vals))

    a_center = _active_wrist_centroid(a[:t])
    b_center = _active_wrist_centroid(b[:t])
    valid = np.all(np.isfinite(a_center), axis=1) & np.all(
        np.isfinite(b_center), axis=1
    )
    if not np.any(valid):
        return None
    return float(np.linalg.norm(a_center[valid] - b_center[valid], axis=1).mean())


def _mean_pairwise_wrist_distance_m(trajs: list[np.ndarray]) -> float:
    if len(trajs) < 2:
        return 0.0

    normalized = []
    for traj in trajs:
        arr = np.asarray(traj, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] > 0:
            normalized.append(arr)
    if len(normalized) < 2:
        return 0.0

    # Pad once, then compare every trajectory against all following
    # trajectories in NumPy.  This preserves the original equal weighting of
    # pair distances while avoiding a Python call for every O(N^2) pair.
    ntraj = len(normalized)
    max_frames = max(int(arr.shape[0]) for arr in normalized)
    max_hands = max(int(arr.shape[1]) for arr in normalized)
    coords = np.zeros((ntraj, max_frames, max_hands, 3), dtype=np.float32)
    valid = np.zeros((ntraj, max_frames, max_hands), dtype=bool)
    centers = np.zeros((ntraj, max_frames, 3), dtype=np.float32)
    center_valid = np.zeros((ntraj, max_frames), dtype=bool)
    for idx, arr in enumerate(normalized):
        frames, hands = int(arr.shape[0]), int(arr.shape[1])
        arr_valid = np.all(np.isfinite(arr), axis=2)
        coords[idx, :frames, :hands] = np.where(
            arr_valid[..., None], arr, 0.0
        )
        valid[idx, :frames, :hands] = arr_valid
        counts = arr_valid.sum(axis=1)
        center_valid[idx, :frames] = counts > 0
        centers[idx, :frames] = (
            np.where(arr_valid[..., None], arr, 0.0).sum(axis=1)
            / np.maximum(counts[:, None], 1)
        )

    pair_sum = 0.0
    pair_count = 0
    for i in range(ntraj - 1):
        shared = valid[i][None, ...] & valid[i + 1 :]
        frame_hand_dist = np.linalg.norm(
            coords[i][None, ...] - coords[i + 1 :], axis=3
        )
        shared_count = shared.sum(axis=(1, 2))
        shared_sum = np.where(shared, frame_hand_dist, 0.0).sum(axis=(1, 2))
        pair_values = np.zeros(ntraj - i - 1, dtype=np.float64)
        has_shared = shared_count > 0
        pair_values[has_shared] = (
            shared_sum[has_shared] / shared_count[has_shared]
        )

        # Match the original fallback for pairs with no common active hand:
        # compare their per-frame active-wrist centroids.
        fallback_indices = np.flatnonzero(~has_shared)
        if fallback_indices.size:
            fallback_valid = (
                center_valid[i][None, ...]
                & center_valid[i + 1 :][fallback_indices]
            )
            fallback_dist = np.linalg.norm(
                centers[i][None, ...]
                - centers[i + 1 :][fallback_indices],
                axis=2,
            )
            fallback_count = fallback_valid.sum(axis=1)
            fallback_sum = np.where(
                fallback_valid, fallback_dist, 0.0
            ).sum(axis=1)
            usable = fallback_count > 0
            pair_values[fallback_indices[usable]] = (
                fallback_sum[usable] / fallback_count[usable]
            )
            valid_pairs = has_shared.copy()
            valid_pairs[fallback_indices[usable]] = True
        else:
            valid_pairs = has_shared

        pair_sum += float(pair_values[valid_pairs].sum())
        pair_count += int(valid_pairs.sum())
    return pair_sum / pair_count if pair_count else 0.0


def _read_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _csv_float_or_none(row: dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _csv_bool_or_none(row: dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value in {"0", "1"}:
        return float(value)
    return None


def _csv_row_has_contact_cr(row: dict) -> bool:
    cr_value = _csv_float_or_none(row, "CR")
    return bool(cr_value is not None and cr_value > 0.0)


def _top10_metric_specs() -> list[dict]:
    return [
        {
            "name": "ID_mean_mm_contact_mean",
            "column": "ID_mean_mm",
            "scale": 1.0,
            "unit": "mm",
            "filter": "positive_over_contact",
            "kind": "mean",
        },
        {
            "name": "ID_max_mm_contact_mean",
            "column": "ID_max_mm",
            "scale": 1.0,
            "unit": "mm",
            "filter": "positive_over_contact",
            "kind": "mean",
        },
        {
            "name": "IV_volume_cm3_contact_mean",
            "column": "IV_volume_cm3",
            "scale": 1.0,
            "unit": "cm3",
            "filter": "contact",
            "kind": "mean",
        },
        {
            "name": "CR_percent_nonzero_mean",
            "column": "CR",
            "scale": 100.0,
            "unit": "%",
            "filter": "nonzero",
            "kind": "mean",
        },
        {
            "name": "Con_pass_percent",
            "column": "Con_pass",
            "scale": 100.0,
            "unit": "%",
            "filter": "bool",
            "kind": "mean",
        },
        {
            "name": "Pen_1cm_percent_contact",
            "column": "Pen_1cm",
            "scale": 100.0,
            "unit": "%",
            "filter": "bool_contact",
            "kind": "mean",
        },
        {
            "name": "Success_percent_last",
            "column": "Success",
            "scale": 100.0,
            "unit": "%",
            "filter": "bool",
            "kind": "mean",
        },
        {
            "name": "Accel_m_nonzero_mean",
            "column": "Accel_m",
            "scale": 1.0,
            "unit": "m",
            "filter": "nonzero",
            "kind": "mean",
        },
        {
            "name": "Off_ground_percent",
            "column": "Off_ground",
            "scale": 100.0,
            "unit": "%",
            "filter": "bool",
            "kind": "mean",
        },
        {
            "name": "Phy_percent_contact",
            "column": "Phy",
            "scale": 100.0,
            "unit": "%",
            "filter": "bool_contact",
            "kind": "mean",
        },
    ]


def _sample_metric_value_from_rows(
    sample_rows: list[dict], spec: dict
) -> Optional[dict]:
    values = []
    denominator = None
    for row in sample_rows:
        filter_name = str(spec.get("filter", "all"))
        if filter_name in {"contact", "bool_contact"} and not _csv_row_has_contact_cr(
            row
        ):
            continue
        if filter_name == "positive_over_contact":
            if not _csv_row_has_contact_cr(row):
                continue
            denominator = 0 if denominator is None else denominator
            denominator += 1
            value = _csv_float_or_none(row, str(spec["column"]))
            if value is None:
                continue
        elif filter_name == "nonzero":
            value = _csv_float_or_none(row, str(spec["column"]))
            if value is None or np.isclose(value, 0.0):
                continue
        elif filter_name.startswith("bool"):
            value = _csv_bool_or_none(row, str(spec["column"]))
            if value is None:
                continue
        else:
            value = _csv_float_or_none(row, str(spec["column"]))
            if value is None:
                continue
        values.append(float(value))
    if denominator is not None:
        if denominator <= 0:
            return None
        raw_value = float(sum(values)) / float(denominator)
        if raw_value <= 0.0:
            return None
        raw_numerator = float(sum(values))
        scale = float(spec.get("scale", 1.0))
        return {
            "value": raw_value * scale,
            "raw_value": raw_value,
            "numerator": raw_numerator * scale,
            "raw_numerator": raw_numerator,
            "denominator": int(denominator),
        }
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    raw_value = float(np.max(arr)) if spec.get("kind") == "max" else float(np.mean(arr))
    raw_numerator = (
        float(np.max(arr)) if spec.get("kind") == "max" else float(np.sum(arr))
    )
    scale = float(spec.get("scale", 1.0))
    return {
        "value": raw_value * scale,
        "raw_value": raw_value,
        "numerator": raw_numerator * scale,
        "raw_numerator": raw_numerator,
        "denominator": int(arr.shape[0]),
    }


def _top10_current_frame_value(
    frame_summary: dict,
    extra: dict,
    top10_context: dict,
) -> Optional[float]:
    column = str(top10_context.get("column", ""))
    filter_name = str(top10_context.get("filter", "all"))
    cr_value = _finite_float_or_none(extra.get("cr"))
    if filter_name in {"contact", "bool_contact"} and not (
        cr_value is not None and cr_value > 0.0
    ):
        return None

    if column == "ID_mean_mm":
        value = _finite_float_or_none(frame_summary.get("id_mean_mm"))
    elif column == "ID_max_mm":
        value = _finite_float_or_none(frame_summary.get("id_max_mm"))
    elif column == "IV_volume_cm3":
        value = _finite_float_or_none(frame_summary.get("iv_volume_cm3"))
    elif column == "CR":
        value = cr_value
    elif column == "Con_pass":
        value = extra.get("con_pass")
        value = None if value is None else float(bool(value))
    elif column == "Pen_1cm":
        value = extra.get("pen_1cm_event")
        value = None if value is None else float(bool(value))
    elif column == "Success":
        value = extra.get("success")
        value = None if value is None else float(bool(value))
    elif column == "Accel_m":
        value = _finite_float_or_none(extra.get("accel_m"))
    elif column == "Off_ground":
        value = extra.get("off_ground")
        value = None if value is None else float(bool(value))
    elif column == "Phy":
        value = extra.get("phy_success")
        value = None if value is None else float(bool(value))
    else:
        value = None

    if value is None:
        return None
    value = float(value)
    if filter_name == "nonzero" and np.isclose(value, 0.0):
        return None
    return value * float(top10_context.get("scale", 1.0))


def _top10_current_frame_contribution(
    frame_summary: dict,
    extra: dict,
    top10_context: dict,
) -> dict:
    value = _top10_current_frame_value(frame_summary, extra, top10_context)
    filter_name = str(top10_context.get("filter", "all"))
    scale = float(top10_context.get("scale", 1.0))
    cr_value = _finite_float_or_none(extra.get("cr"))
    included = value is not None
    denominator_add = 1
    numerator_add = 0.0
    reason = "included"

    if filter_name == "positive_over_contact":
        if not (cr_value is not None and cr_value > 0.0):
            included = False
            denominator_add = 0
            reason = "skip: CR<=0"
        else:
            denominator_add = 1
            if value is not None and value > 0.0:
                numerator_add = float(value)
                reason = "denom += CR>0, numerator += value"
            else:
                included = True
                numerator_add = 0.0
                reason = "denom += CR>0, numerator += 0 because value<=0"
    elif value is None:
        denominator_add = 0
        reason = "skip: filter/value unavailable"
    else:
        numerator_add = float(value)
        denominator_add = 1
        if filter_name in {"contact", "bool_contact"}:
            reason = "denom += CR>0, numerator += value"
        elif filter_name == "nonzero":
            reason = "denom += nonzero value, numerator += value"
        elif filter_name.startswith("bool"):
            reason = "denom += bool value, numerator += 0/1"

    raw_value = None if value is None else float(value) / scale
    raw_numerator_add = float(numerator_add) / scale if scale != 0.0 else numerator_add
    return {
        "value": value,
        "raw_value": raw_value,
        "numerator_add": float(numerator_add),
        "raw_numerator_add": float(raw_numerator_add),
        "denominator_add": int(denominator_add),
        "included": bool(included),
        "reason": reason,
    }


def build_new_metric_top10_contexts_from_csv(
    csv_path: str,
    *,
    top_k: int = 10,
    model_filters: Optional[list[str]] = None,
    metric_filters: Optional[list[str]] = None,
    directions: tuple[str, ...] = ("high", "low"),
    per_model: bool = False,
) -> dict[int, list[dict]]:
    rows = _read_csv_rows(csv_path)
    if model_filters:
        allowed = {os.path.basename(str(item)) for item in model_filters}
        rows = [
            row
            for row in rows
            if os.path.basename(str(row.get("file_name", ""))) in allowed
        ]
    if not rows:
        return {}

    specs = _top10_metric_specs()
    if metric_filters:
        wanted = {str(item) for item in metric_filters}
        specs = [
            spec
            for spec in specs
            if str(spec["name"]) in wanted or str(spec["column"]) in wanted
        ]

    contexts_by_sample: dict[int, list[dict]] = {}
    metric_slot = 0
    if per_model:
        row_partitions = [
            (
                file_name,
                [row for row in rows if str(row.get("file_name", "")) == file_name],
            )
            for file_name in sorted({str(row.get("file_name", "")) for row in rows})
        ]
    else:
        row_partitions = [("all_models", rows)]

    for partition_idx, (partition_name, partition_rows) in enumerate(row_partitions):
        grouped: dict[tuple[str, int], list[dict]] = {}
        for row in partition_rows:
            try:
                sample_idx = int(row.get("sample_idx", row.get("batch_sample_idx", "")))
            except (TypeError, ValueError):
                continue
            grouped.setdefault((str(row.get("file_name", "")), sample_idx), []).append(
                row
            )
        for spec in specs:
            scored = []
            for (file_name, sample_idx), sample_rows in grouped.items():
                aggregate = _sample_metric_value_from_rows(sample_rows, spec)
                if aggregate is None:
                    continue
                first = sample_rows[0]
                scored.append(
                    {
                        "file_name": file_name,
                        "sample_idx": sample_idx,
                        "object": first.get("object", ""),
                        "text": first.get("text", ""),
                        "metric": str(spec["name"]),
                        "column": str(spec["column"]),
                        "scale": float(spec.get("scale", 1.0)),
                        "unit": str(spec.get("unit", "")),
                        "filter": str(spec.get("filter", "all")),
                        "kind": str(spec.get("kind", "mean")),
                        **aggregate,
                    }
                )
            if not scored:
                continue
            for direction, reverse in (("high", True), ("low", False)):
                if direction not in directions:
                    continue
                selected = sorted(
                    scored,
                    key=lambda item: (
                        float(item["value"]),
                        str(item["file_name"]),
                        int(item["sample_idx"]),
                    ),
                    reverse=reverse,
                )[: max(1, int(top_k))]
                for rank_idx, item in enumerate(selected, start=1):
                    context = dict(item)
                    context["direction"] = direction
                    context["rank"] = int(rank_idx)
                    context["path"] = (
                        f"{_safe_path_token(context['file_name'])}/"
                        "new_metric_top10/"
                        f"{_safe_path_token(context['metric'])}/"
                        f"{direction}/rank_{rank_idx:02d}/"
                        f"sample_{int(context['sample_idx']):04d}_"
                        f"{_safe_path_token(context.get('object', 'object'))}"
                    )
                    context["offset"] = (
                        float(metric_slot) * 1.6 + float(partition_idx) * 1.8,
                        float(rank_idx - 1) * 0.42,
                        0.0 if direction == "high" else 0.75,
                    )
                    contexts_by_sample.setdefault(
                        int(context["sample_idx"]), []
                    ).append(context)
                print(
                    "[TOP10] "
                    f"model={partition_name} metric={spec['name']} "
                    f"direction={direction} "
                    f"samples={[int(item['sample_idx']) for item in selected]}"
                )
                for rank_idx, item in enumerate(selected, start=1):
                    print(
                        "[TOP10-DETAIL] "
                        f"model={item['file_name']} "
                        f"metric={spec['name']} "
                        f"direction={direction} "
                        f"rank={rank_idx} "
                        f"sample={int(item['sample_idx'])} "
                        f"value={float(item['value']):.6f}{item.get('unit', '')} "
                        f"numerator={float(item['numerator']):.6f} "
                        f"denominator={int(item['denominator'])} "
                        f"object={item.get('object', '')} "
                        f"text={item.get('text', '')}"
                    )
            metric_slot += 1
    return contexts_by_sample


def build_new_metric_pairdiff_contexts_from_csv(
    csv_path: str,
    *,
    metric_name: str = "ID_mean_mm_contact_mean",
    top_k: int = 10,
    base_model: str = "s_bps.pkl",
    compare_model: str = "s_bps_9000.pkl",
) -> dict[int, list[dict]]:
    rows = _read_csv_rows(csv_path)
    if not rows:
        return {}
    spec = next(
        (
            item
            for item in _top10_metric_specs()
            if str(item["name"]) == str(metric_name)
            or str(item["column"]) == str(metric_name)
        ),
        None,
    )
    if spec is None:
        raise ValueError(f"unknown pairdiff metric: {metric_name}")

    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        file_name = os.path.basename(str(row.get("file_name", "")))
        if file_name not in {base_model, compare_model}:
            continue
        try:
            sample_idx = int(row.get("sample_idx", row.get("batch_sample_idx", "")))
        except (TypeError, ValueError):
            continue
        local_idx = sample_idx % int(_DEFAULT_EXPECTED_SAMPLE_COUNT)
        grouped.setdefault((file_name, local_idx), []).append(row)

    pairs = []
    for local_idx in range(int(_DEFAULT_EXPECTED_SAMPLE_COUNT)):
        base_rows = grouped.get((base_model, local_idx), [])
        compare_rows = grouped.get((compare_model, local_idx), [])
        if not base_rows or not compare_rows:
            continue
        base_value = _sample_metric_value_from_rows(base_rows, spec)
        compare_value = _sample_metric_value_from_rows(compare_rows, spec)
        if base_value is None or compare_value is None:
            continue
        base_first = base_rows[0]
        compare_first = compare_rows[0]
        pairs.append(
            {
                "local_idx": int(local_idx),
                "base_sample_idx": int(base_first.get("sample_idx")),
                "compare_sample_idx": int(compare_first.get("sample_idx")),
                "base_object": base_first.get("object", ""),
                "compare_object": compare_first.get("object", ""),
                "base_text": base_first.get("text", ""),
                "compare_text": compare_first.get("text", ""),
                "base": base_value,
                "compare": compare_value,
                "diff": float(base_value["value"]) - float(compare_value["value"]),
            }
        )

    selected = sorted(pairs, key=lambda item: item["diff"], reverse=True)[
        : max(1, int(top_k))
    ]
    contexts_by_sample: dict[int, list[dict]] = {}
    for rank_idx, pair in enumerate(selected, start=1):
        for side_idx, side in enumerate(("base", "compare")):
            model_name = base_model if side == "base" else compare_model
            sample_idx = (
                int(pair["base_sample_idx"])
                if side == "base"
                else int(pair["compare_sample_idx"])
            )
            object_name = (
                str(pair["base_object"])
                if side == "base"
                else str(pair["compare_object"])
            )
            aggregate = pair[side]
            context = {
                "file_name": model_name,
                "sample_idx": sample_idx,
                "object": object_name,
                "text": pair["base_text"] if side == "base" else pair["compare_text"],
                "metric": str(spec["name"]),
                "column": str(spec["column"]),
                "scale": float(spec.get("scale", 1.0)),
                "unit": str(spec.get("unit", "")),
                "filter": str(spec.get("filter", "all")),
                "kind": str(spec.get("kind", "mean")),
                "direction": "pairdiff",
                "rank": int(rank_idx),
                "value": float(aggregate["value"]),
                "raw_value": float(aggregate["raw_value"]),
                "numerator": float(aggregate["numerator"]),
                "raw_numerator": float(aggregate["raw_numerator"]),
                "denominator": int(aggregate["denominator"]),
                "pairdiff_local_idx": int(pair["local_idx"]),
                "pairdiff_side": side,
                "pairdiff_base_model": base_model,
                "pairdiff_compare_model": compare_model,
                "pairdiff_base_value": float(pair["base"]["value"]),
                "pairdiff_compare_value": float(pair["compare"]["value"]),
                "pairdiff_delta": float(pair["diff"]),
                "path": (
                    "new_metric_pairdiff/"
                    f"{_safe_path_token(str(spec['name']))}/"
                    f"rank_{rank_idx:02d}/"
                    f"{_safe_path_token(model_name)}/"
                    f"sample_{sample_idx:04d}_{_safe_path_token(object_name)}"
                ),
                "offset": (
                    float(rank_idx - 1) * 1.25,
                    0.0 if side == "base" else 0.48,
                    0.0,
                ),
            }
            contexts_by_sample.setdefault(sample_idx, []).append(context)
        print(
            "[PAIRDIFF] "
            f"rank={rank_idx} local={pair['local_idx']} "
            f"{base_model}[{pair['base_sample_idx']}]={pair['base']['value']:.6f} "
            f"{compare_model}[{pair['compare_sample_idx']}]={pair['compare']['value']:.6f} "
            f"delta={pair['diff']:.6f}"
        )
    return contexts_by_sample


def _csv_rows_to_wrist_trajectory_items(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        sample_key = (
            str(row.get("file_name", "")),
            str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
        )
        grouped.setdefault(sample_key, []).append(row)

    wrist_keys = [
        (
            "left_wrist_canonical_x",
            "left_wrist_canonical_y",
            "left_wrist_canonical_z",
        ),
        (
            "right_wrist_canonical_x",
            "right_wrist_canonical_y",
            "right_wrist_canonical_z",
        ),
    ]
    items = []
    for (file_name, sample_idx), sample_rows in grouped.items():
        try:
            ordered = sorted(
                sample_rows,
                key=lambda item: int(item.get("frame_idx", -1)),
            )
        except Exception:
            ordered = sample_rows
        traj = np.full((len(ordered), 2, 3), np.nan, dtype=np.float32)
        for frame_out_idx, row in enumerate(ordered):
            for hand_idx, keys in enumerate(wrist_keys):
                values = []
                for key in keys:
                    value = row.get(key)
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        value_f = float("nan")
                    values.append(value_f)
                if np.isfinite(values).all():
                    traj[frame_out_idx, hand_idx] = np.asarray(
                        values,
                        dtype=np.float32,
                    )
        if not np.any(np.isfinite(traj)):
            continue
        first = ordered[0]
        items.append(
            {
                "file_name": file_name,
                "sample_idx": sample_idx,
                "text": first.get("text", ""),
                "object": first.get("object", ""),
                "wrist_traj": traj,
            }
        )
    return items


def _pairwise_wrist_distances(items: list[dict]) -> list[tuple[float, int, int]]:
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            dist = _wrist_trajectory_distance_m(
                items[i]["wrist_traj"],
                items[j]["wrist_traj"],
            )
            if dist is not None:
                out.append((float(dist), i, j))
    return out


def _trajectory_item_label(item: dict) -> str:
    model = os.path.basename(str(item.get("file_name", "")))
    return f"{model} sample={item.get('sample_idx')} object={item.get('object')}"


def _log_diversity_trajectory_group(
    root_path: str,
    items: list[dict],
    title: str,
    metric_name: str,
    metric_value: float,
    max_samples: int,
    offset: tuple[float, float, float],
) -> None:
    if rr is None or not items:
        return
    selected = items[: max(1, int(max_samples))]
    offset_np = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    colors = np.asarray(
        [
            [230, 80, 70],
            [60, 160, 240],
            [70, 190, 120],
            [245, 180, 65],
            [180, 110, 230],
            [70, 210, 210],
            [235, 100, 170],
            [150, 150, 150],
            [255, 120, 0],
            [120, 220, 80],
        ],
        dtype=np.uint8,
    )
    pairwise = _pairwise_wrist_distances(selected)
    pairwise_sorted = sorted(pairwise, key=lambda item: item[0], reverse=True)
    label_lines = [
        title,
        f"{metric_name} {metric_value:.6f} m",
        f"shown_samples {len(selected)} / source_samples {len(items)}",
    ]
    for dist, i, j in pairwise_sorted[:10]:
        label_lines.append(
            f"pair {i}-{j}: {dist:.6f} m | "
            f"{selected[i].get('sample_idx')} vs {selected[j].get('sample_idx')}"
        )

    all_points = []
    for item in selected:
        traj = np.asarray(item["wrist_traj"], dtype=np.float32)
        finite = traj[np.all(np.isfinite(traj), axis=2)]
        if finite.size:
            all_points.append(finite.reshape(-1, 3))
    if all_points:
        label_pos = np.mean(np.concatenate(all_points, axis=0), axis=0, keepdims=True)
    else:
        label_pos = np.zeros((1, 3), dtype=np.float32)
    rr.log(
        f"{root_path}/summary_label",
        rr.Points3D(
            positions=label_pos + offset_np,
            radii=[0.015],
            colors=np.asarray([[255, 255, 255]], dtype=np.uint8),
            labels=["\n".join(label_lines)],
            show_labels=True,
        ),
        static=True,
    )

    for item_idx, item in enumerate(selected):
        traj = np.asarray(item["wrist_traj"], dtype=np.float32)
        color = colors[item_idx % colors.shape[0]]
        for hand_idx, hand_name in enumerate(("left", "right")):
            hand_traj = traj[:, hand_idx, :]
            valid = np.all(np.isfinite(hand_traj), axis=1)
            if np.count_nonzero(valid) < 2:
                continue
            rr.log(
                f"{root_path}/static_trajectories/sample_{item_idx:02d}/{hand_name}",
                rr.LineStrips3D(
                    [hand_traj[valid] + offset_np],
                    colors=np.tile(color.reshape(1, 3), (1, 1)),
                    radii=0.002,
                    labels=[f"{_trajectory_item_label(item)} {hand_name}"],
                    show_labels=True,
                ),
                static=True,
            )

    max_frame = max(int(item["wrist_traj"].shape[0]) for item in selected)
    for frame_idx in range(max_frame):
        _rerun_set_time_sequence("diversity_frame", frame_idx)
        centroid_points = []
        centroid_labels = []
        centroid_colors = []
        for item_idx, item in enumerate(selected):
            traj = np.asarray(item["wrist_traj"], dtype=np.float32)
            if frame_idx >= traj.shape[0]:
                continue
            center = _active_wrist_centroid(traj[frame_idx : frame_idx + 1])
            if center.shape[0] == 0 or not np.all(np.isfinite(center[0])):
                continue
            centroid_points.append(center[0] + offset_np.reshape(3))
            centroid_labels.append(f"{item_idx}: {_trajectory_item_label(item)}")
            centroid_colors.append(colors[item_idx % colors.shape[0]])
        if centroid_points:
            centroid_arr = np.asarray(centroid_points, dtype=np.float32)
            rr.log(
                f"{root_path}/frame_centroids",
                rr.Points3D(
                    positions=centroid_arr,
                    radii=[0.006] * int(centroid_arr.shape[0]),
                    colors=np.asarray(centroid_colors, dtype=np.uint8),
                    labels=centroid_labels,
                    show_labels=True,
                ),
            )
        line_segments = []
        line_labels = []
        for dist, i, j in pairwise_sorted:
            if i >= len(selected) or j >= len(selected):
                continue
            traj_i = np.asarray(selected[i]["wrist_traj"], dtype=np.float32)
            traj_j = np.asarray(selected[j]["wrist_traj"], dtype=np.float32)
            if frame_idx >= traj_i.shape[0] or frame_idx >= traj_j.shape[0]:
                continue
            ci = _active_wrist_centroid(traj_i[frame_idx : frame_idx + 1])
            cj = _active_wrist_centroid(traj_j[frame_idx : frame_idx + 1])
            if (
                ci.shape[0] == 0
                or cj.shape[0] == 0
                or not np.all(np.isfinite(ci[0]))
                or not np.all(np.isfinite(cj[0]))
            ):
                continue
            line_segments.append(
                np.stack(
                    [ci[0] + offset_np.reshape(3), cj[0] + offset_np.reshape(3)],
                    axis=0,
                )
            )
            current_dist = float(np.linalg.norm(ci[0] - cj[0]))
            line_labels.append(
                f"pair {i}-{j} frame {current_dist:.4f} m, mean {dist:.4f} m"
            )
        if line_segments:
            rr.log(
                f"{root_path}/frame_pairwise_links",
                rr.LineStrips3D(
                    np.asarray(line_segments, dtype=np.float32),
                    colors=np.tile(
                        np.asarray([[180, 180, 180]], dtype=np.uint8),
                        (len(line_segments), 1),
                    ),
                    radii=0.0007,
                    labels=line_labels,
                    show_labels=True,
                ),
            )


def _find_latest_new_metric_csv() -> Optional[str]:
    candidates = []
    for name in os.listdir("."):
        if name.startswith("new_metrics_per_frame_") and name.endswith(".csv"):
            try:
                mtime = os.path.getmtime(name)
            except OSError:
                continue
            candidates.append((mtime, name))
    if not candidates:
        return None
    return max(candidates)[1]


def visualize_diversity_from_csv(
    csv_path: str,
    model_filters: Optional[list[str]] = None,
    text_filter: Optional[str] = None,
    max_groups: int = 3,
    max_samples: int = 8,
) -> None:
    if rr is None:
        raise RuntimeError("rerun is not installed; cannot visualize SD/OD")
    rows = _read_csv_rows(csv_path)
    if model_filters:
        allowed = {os.path.basename(str(item)) for item in model_filters}
        rows = [
            row
            for row in rows
            if os.path.basename(str(row.get("file_name", ""))) in allowed
        ]
    if text_filter:
        needle = str(text_filter).lower()
        rows = [row for row in rows if needle in str(row.get("text", "")).lower()]
    items = _csv_rows_to_wrist_trajectory_items(rows)
    if len(items) < 2:
        print("[WARN] need at least two wrist trajectories to visualize diversity.")
        return

    _init_rerun_viewer("HOT3D SD/OD diversity debug")

    by_text: dict[str, list[dict]] = {}
    for item in items:
        by_text.setdefault(str(item.get("text", "")), []).append(item)
    od_value = _mean_pairwise_wrist_distance_m(
        [np.asarray(item["wrist_traj"], dtype=np.float32) for item in items]
    )
    _log_diversity_trajectory_group(
        "diversity/OD_all_texts",
        items,
        title="OD: all samples pairwise canonical wrist distance",
        metric_name="OD_m",
        metric_value=od_value,
        max_samples=max_samples,
        offset=(0.0, 0.0, 0.0),
    )

    sd_groups = []
    for text, text_items in by_text.items():
        if len(text_items) < 2:
            continue
        value = _mean_pairwise_wrist_distance_m(
            [np.asarray(item["wrist_traj"], dtype=np.float32) for item in text_items]
        )
        sd_groups.append((float(value), text, text_items))
    if not sd_groups:
        print("[WARN] no text group has at least two samples for SD visualization.")
        return
    sd_values = [value for value, _text, _items in sd_groups]
    print(f"[DIVERSITY] CSV={csv_path}")
    print(f"[DIVERSITY] OD_m={od_value:.6f} samples={len(items)}")
    print(f"[DIVERSITY] SD_m={float(np.mean(sd_values)):.6f} groups={len(sd_values)}")
    for group_idx, (value, text, text_items) in enumerate(
        sorted(sd_groups, key=lambda item: (-len(item[2]), -item[0], item[1]))[
            : max(1, int(max_groups))
        ]
    ):
        print(
            f"[DIVERSITY] group={group_idx} SD_m={value:.6f} "
            f"samples={len(text_items)} text={text}"
        )
        _log_diversity_trajectory_group(
            f"diversity/SD_text_group_{group_idx:02d}",
            text_items,
            title=f"SD text group {group_idx}: {text}",
            metric_name="SD_m",
            metric_value=value,
            max_samples=max_samples,
            offset=(float(group_idx + 1) * 1.0, 0.0, 0.0),
        )


def _log_new_metric_frame_label(
    root_path: str,
    obj_mesh_world: Optional[trimesh.Trimesh],
    frame_summary: dict,
    extra: dict,
    top10_context: Optional[dict] = None,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    if rr is None or obj_mesh_world is None:
        return
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return
    offset_np = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    position = vertices.mean(axis=0, keepdims=True) + offset_np
    extent = np.ptp(vertices, axis=0)
    position[0, 2] += float(np.max(extent) * 0.65 + 0.03)
    cr = extra.get("cr")
    is_last_frame = bool(extra.get("is_last_frame", False))
    accel = extra.get("accel_m")
    object_lift = extra.get("object_lift_m")
    success_rate = extra.get("success_rate_percent")
    label_lines = []
    if top10_context:
        numerator = float(top10_context.get("numerator", 0.0))
        denominator = int(top10_context.get("denominator", 0))
        value = float(top10_context.get("value", 0.0))
        unit = str(top10_context.get("unit", ""))
        metric = str(top10_context.get("metric", "metric"))
        label_lines.extend(
            [
                (
                    f"{metric} {top10_context.get('direction')} "
                    f"rank {int(top10_context.get('rank', 0))}"
                ),
                (
                    f"sample_avg {value:.6f} {unit} "
                    f"= {numerator:.6f} / {denominator}"
                ),
                (
                    f"pair {top10_context.get('pairdiff_base_model')} "
                    f"{float(top10_context.get('pairdiff_base_value')):.6f} {unit} vs "
                    f"{top10_context.get('pairdiff_compare_model')} "
                    f"{float(top10_context.get('pairdiff_compare_value')):.6f} {unit}"
                    if "pairdiff_delta" in top10_context
                    else ""
                ),
                (
                    f"pair_delta_base_minus_compare "
                    f"{float(top10_context.get('pairdiff_delta')):.6f} {unit} "
                    f"(local {int(top10_context.get('pairdiff_local_idx'))})"
                    if "pairdiff_delta" in top10_context
                    else ""
                ),
                (
                    f"avg_source column={top10_context.get('column')} "
                    f"filter={top10_context.get('filter')} "
                    f"kind={top10_context.get('kind')}"
                ),
                f"sample {top10_context.get('sample_idx')} object {top10_context.get('object')}",
            ]
        )
        label_lines = [line for line in label_lines if line != ""]
    label_lines.extend(
        [
            f"frame {int(frame_summary.get('frame_idx', -1))}",
            f"frame_ID_depth_max_mm {float(frame_summary.get('id_max_mm', 0.0)):.6f}",
            f"frame_IV_volume_cm3 {float(frame_summary.get('iv_volume_cm3', 0.0)):.6f}",
            f"frame_CR {float(cr):.6f}" if cr is not None else "frame_CR NA",
            f"frame_Pen_1cm {_format_success_fail(extra.get('pen_1cm_event'))}",
            f"frame_Con_pass {_format_success_fail(extra.get('con_pass'))}",
            (
                f"frame_Accel_m {float(accel):.6f}"
                if accel is not None
                else "frame_Accel_m NA"
            ),
            (
                f"frame_Object_lift_y_m {float(object_lift):.6f} "
                f"(threshold {OFF_GROUND_THRESHOLD_M:.3f} m)"
                if object_lift is not None
                else "frame_Object_lift_y_m NA"
            ),
            f"frame_Off_ground {_format_success_fail(extra.get('off_ground'))}",
            f"frame_Phy {_format_success_fail(extra.get('phy_success'))}",
            (
                f"last_Success {_format_success_fail(extra.get('success'))}"
                if is_last_frame
                else "last_Success NA"
            ),
            (
                f"last_Success_Rate_percent {float(success_rate):.6f}"
                if success_rate is not None
                else "last_Success_Rate_percent NA"
            ),
        ]
    )
    label = "\n".join(label_lines)
    rr.log(
        f"{root_path}/frame_metrics_label",
        rr.Points3D(
            positions=position,
            radii=[0.01],
            colors=np.asarray([[255, 255, 255]], dtype=np.uint8),
            labels=[label],
            show_labels=True,
        ),
    )


def _log_new_metric_running_mean_label(
    root_path: str,
    obj_mesh_world: Optional[trimesh.Trimesh],
    top10_context: dict,
    running_values: list[float],
    current_value: Optional[float],
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    if rr is None or obj_mesh_world is None or not top10_context:
        return
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return

    offset_np = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    position = vertices.mean(axis=0, keepdims=True) + offset_np
    extent = np.ptp(vertices, axis=0)
    position[0, 0] += float(np.max(extent) * 0.65 + 0.05)
    position[0, 2] += float(np.max(extent) * 0.25 + 0.02)

    unit = str(top10_context.get("unit", ""))
    values = np.asarray(running_values, dtype=np.float64)
    if values.size > 0:
        numerator = float(np.sum(values))
        denominator = int(values.size)
        mean_value = numerator / float(denominator)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
    else:
        numerator = 0.0
        denominator = 0
        mean_value = None
        min_value = None
        max_value = None

    label = "\n".join(
        [
            "running mean box",
            f"metric {top10_context.get('metric')}",
            (
                f"current_frame_value {float(current_value):.6f} {unit}"
                if current_value is not None
                else "current_frame_value skipped"
            ),
            (
                f"running_mean {float(mean_value):.6f} {unit} "
                f"= {numerator:.6f} / {denominator}"
                if mean_value is not None
                else "running_mean NA = 0 / 0"
            ),
            (
                f"running_min {float(min_value):.6f} {unit}"
                if min_value is not None
                else "running_min NA"
            ),
            (
                f"running_max {float(max_value):.6f} {unit}"
                if max_value is not None
                else "running_max NA"
            ),
            (
                f"include_rule column={top10_context.get('column')} "
                f"filter={top10_context.get('filter')}"
            ),
        ]
    )
    rr.log(
        f"{root_path}/running_mean_label",
        rr.Points3D(
            positions=position,
            radii=[0.009],
            colors=np.asarray([[80, 220, 255]], dtype=np.uint8),
            labels=[label],
            show_labels=True,
        ),
    )


def _log_new_metric_accumulator_debug_labels(
    root_path: str,
    obj_mesh_world: Optional[trimesh.Trimesh],
    frame_summary: dict,
    extra: dict,
    top10_context: dict,
    contribution: dict,
    running_state: dict,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    if rr is None or obj_mesh_world is None or not top10_context:
        return
    vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[0] == 0:
        return

    offset_np = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    center = vertices.mean(axis=0, keepdims=True) + offset_np
    extent = np.ptp(vertices, axis=0)
    scale = max(float(np.max(extent)), 0.05)
    frame_pos = center.copy()
    frame_pos[0, 2] += scale * 0.65 + 0.03
    accum_pos = center.copy()
    accum_pos[0, 0] += scale * 0.75 + 0.05
    accum_pos[0, 2] += scale * 0.35 + 0.02

    unit = str(top10_context.get("unit", ""))
    metric = str(top10_context.get("metric", "metric"))
    numerator = float(running_state.get("numerator", 0.0))
    denominator = int(running_state.get("denominator", 0))
    running_value = numerator / float(denominator) if denominator > 0 else None
    final_value = float(top10_context.get("value", 0.0))
    final_numerator = float(top10_context.get("numerator", 0.0))
    final_denominator = int(top10_context.get("denominator", 0))

    current_value = contribution.get("value")
    frame_label = "\n".join(
        [
            "FRAME METRIC",
            f"metric {metric}",
            f"sample {top10_context.get('sample_idx')} rank {top10_context.get('rank')}",
            f"frame {int(frame_summary.get('frame_idx', -1))}",
            f"column {top10_context.get('column')}",
            f"filter {top10_context.get('filter')}",
            (
                f"current_value {float(current_value):.6f} {unit}"
                if current_value is not None
                else "current_value skipped"
            ),
            (
                f"frame_numerator_add "
                f"{float(contribution.get('numerator_add', 0.0)):.6f} {unit}"
            ),
            f"frame_denominator_add {int(contribution.get('denominator_add', 0))}",
            f"rule {contribution.get('reason')}",
            (
                f"CR {float(extra.get('cr')):.6f}"
                if extra.get("cr") is not None
                else "CR NA"
            ),
        ]
    )
    accum_label = "\n".join(
        [
            "ACCUMULATED METRIC",
            f"metric {metric}",
            (
                f"running_value {float(running_value):.6f} {unit} "
                f"= {numerator:.6f} / {denominator}"
                if running_value is not None
                else "running_value NA = 0 / 0"
            ),
            (
                f"final_value {final_value:.6f} {unit} "
                f"= {final_numerator:.6f} / {final_denominator}"
            ),
            "final is computed over all selected frames in the sample",
        ]
    )
    rr.log(
        f"{root_path}/frame_contribution_label",
        rr.Points3D(
            positions=frame_pos,
            radii=[0.01],
            colors=np.asarray([[255, 255, 255]], dtype=np.uint8),
            labels=[frame_label],
            show_labels=True,
        ),
    )
    rr.log(
        f"{root_path}/accumulated_metric_label",
        rr.Points3D(
            positions=accum_pos,
            radii=[0.009],
            colors=np.asarray([[80, 220, 255]], dtype=np.uint8),
            labels=[accum_label],
            show_labels=True,
        ),
    )


def _log_xyz_axes(
    root_path: str,
    origin: np.ndarray,
    length: float,
    static: bool = False,
) -> None:
    if rr is None:
        return
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    length = float(max(length, 0.01))
    endpoints = np.asarray(
        [
            origin + np.asarray([length, 0.0, 0.0], dtype=np.float32),
            origin + np.asarray([0.0, length, 0.0], dtype=np.float32),
            origin + np.asarray([0.0, 0.0, length], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    segments = np.stack(
        [
            np.stack([origin, endpoints[0]], axis=0),
            np.stack([origin, endpoints[1]], axis=0),
            np.stack([origin, endpoints[2]], axis=0),
        ],
        axis=0,
    )
    colors = np.asarray(
        [
            [255, 0, 0],
            [0, 220, 0],
            [40, 120, 255],
        ],
        dtype=np.uint8,
    )
    rr.log(
        f"{root_path}/xyz_axes",
        rr.LineStrips3D(
            segments,
            colors=colors,
            radii=max(length * 0.015, 0.001),
            labels=["X", "Y", "Z"],
            show_labels=True,
        ),
        static=static,
    )
    rr.log(
        f"{root_path}/xyz_axis_labels",
        rr.Points3D(
            positions=endpoints,
            radii=[max(length * 0.04, 0.004)] * 3,
            colors=colors,
            labels=["X", "Y", "Z"],
            show_labels=True,
        ),
        static=static,
    )


def _log_new_metric_visualization(
    root_path: str,
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_entries: list[tuple[str, np.ndarray, np.ndarray, dict]],
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    show_id_values: bool = False,
    id_label_count: int = 20,
) -> None:
    if rr is None:
        return

    def clear_entity(path: str) -> None:
        rr.log(path, rr.Clear(recursive=True))

    offset_np = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    if obj_mesh_world is not None:
        obj_vertices = np.asarray(obj_mesh_world.vertices, dtype=np.float32) + offset_np
        rr.log(
            f"{root_path}/object_mesh",
            rr.Mesh3D(
                vertex_positions=obj_vertices,
                triangle_indices=np.asarray(obj_mesh_world.faces, dtype=np.int32),
                vertex_normals=np.asarray(
                    obj_mesh_world.vertex_normals, dtype=np.float32
                ),
                vertex_colors=np.tile(
                    np.asarray((120, 120, 120), dtype=np.uint8),
                    (int(len(obj_mesh_world.vertices)), 1),
                ),
            ),
        )
        rr.log(
            f"{root_path}/object_point_cloud",
            rr.Points3D(
                positions=obj_vertices,
                radii=[0.0025] * int(obj_vertices.shape[0]),
                colors=np.tile(
                    np.asarray((0, 210, 255), dtype=np.uint8),
                    (int(obj_vertices.shape[0]), 1),
                ),
            ),
        )
    for hand_name, vertices, faces, metric in hand_entries:
        vertices_np = np.asarray(vertices, dtype=np.float32) + offset_np
        faces_np = np.asarray(faces, dtype=np.int32)
        if vertices_np.ndim != 2 or vertices_np.shape[0] == 0:
            continue
        vertex_colors = _colors_for_inside_vertices(
            vertices_np.shape[0],
            metric.get("inside_mask", np.zeros((vertices_np.shape[0],), dtype=bool)),
        )
        inside_mask = np.asarray(
            metric.get("inside_mask", np.zeros((vertices_np.shape[0],), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if faces_np.ndim == 2 and faces_np.shape[0] > 0:
            hand_mesh = trimesh.Trimesh(
                vertices=vertices_np, faces=faces_np, process=False
            )
            rr.log(
                f"{root_path}/{hand_name}/mesh_iv_red",
                rr.Mesh3D(
                    vertex_positions=vertices_np,
                    triangle_indices=faces_np,
                    vertex_normals=np.asarray(
                        hand_mesh.vertex_normals, dtype=np.float32
                    ),
                    vertex_colors=vertex_colors,
                ),
            )
            iv_faces_np = _inside_face_indices_from_mask(faces_np, inside_mask)
            if iv_faces_np.shape[0] > 0:
                rr.log(
                    f"{root_path}/{hand_name}/iv_hand_patch_red",
                    rr.Mesh3D(
                        vertex_positions=vertices_np,
                        triangle_indices=iv_faces_np,
                        vertex_colors=np.tile(
                            np.asarray((255, 0, 0), dtype=np.uint8),
                            (int(vertices_np.shape[0]), 1),
                        ),
                    ),
                )
            else:
                clear_entity(f"{root_path}/{hand_name}/iv_hand_patch_red")
        inside_points = np.asarray(metric.get("inside_points"), dtype=np.float32)
        if inside_points.ndim == 2 and inside_points.shape[1] == 3:
            inside_points = inside_points + offset_np
        iv_points = np.asarray(metric.get("iv_points"), dtype=np.float32)
        if iv_points.ndim == 2 and iv_points.shape[1] == 3 and iv_points.shape[0] > 0:
            rr.log(
                f"{root_path}/{hand_name}/iv_points_red",
                rr.Points3D(
                    positions=iv_points + offset_np,
                    radii=[0.0012] * int(iv_points.shape[0]),
                    colors=np.tile(
                        np.asarray((255, 0, 0), dtype=np.uint8),
                        (iv_points.shape[0], 1),
                    ),
                ),
            )
        else:
            clear_entity(f"{root_path}/{hand_name}/iv_points_red")
        line_hand_points = np.asarray(
            metric.get("id_line_hand_points", metric.get("inside_points")),
            dtype=np.float32,
        )
        line_object_points = np.asarray(
            metric.get("id_line_object_points", metric.get("closest_points")),
            dtype=np.float32,
        )
        if line_hand_points.ndim == 2 and line_hand_points.shape[1] == 3:
            line_hand_points = line_hand_points + offset_np
        if line_object_points.ndim == 2 and line_object_points.shape[1] == 3:
            line_object_points = line_object_points + offset_np
        if inside_points.ndim == 2 and inside_points.shape[0] > 0:
            rr.log(
                f"{root_path}/{hand_name}/iv_inside_vertices_red",
                rr.Points3D(
                    positions=inside_points,
                    radii=[0.0012] * int(inside_points.shape[0]),
                    colors=np.tile(
                        np.asarray((255, 0, 0), dtype=np.uint8),
                        (inside_points.shape[0], 1),
                    ),
                    labels=[
                        f"IV vertex {idx}" for idx in range(int(inside_points.shape[0]))
                    ],
                    show_labels=False,
                ),
            )
            iv_label_position = np.mean(inside_points, axis=0, keepdims=True)
            rr.log(
                f"{root_path}/{hand_name}/iv_volume_label",
                rr.Points3D(
                    positions=iv_label_position,
                    radii=[0.003],
                    colors=np.asarray([[255, 0, 0]], dtype=np.uint8),
                    labels=[
                        "IV "
                        f"{float(metric.get('iv_volume_cm3', 0.0)):.4f} cm^3, "
                        f"vertices {int(metric.get('inside_count', 0))}"
                    ],
                    show_labels=True,
                ),
            )
        else:
            clear_entity(f"{root_path}/{hand_name}/iv_inside_vertices_red")
            clear_entity(f"{root_path}/{hand_name}/iv_volume_label")
        contact_mask = np.asarray(
            metric.get("contact_mask", np.zeros((vertices_np.shape[0],), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if contact_mask.shape[0] == vertices_np.shape[0] and np.any(contact_mask):
            contact_points = vertices_np[contact_mask]
            rr.log(
                f"{root_path}/{hand_name}/cr_contact_vertices_yellow",
                rr.Points3D(
                    positions=contact_points,
                    radii=[0.0015] * int(contact_points.shape[0]),
                    colors=np.tile(
                        np.asarray((255, 220, 0), dtype=np.uint8),
                        (contact_points.shape[0], 1),
                    ),
                    labels=[f"CR contact {hand_name}"] * int(contact_points.shape[0]),
                    show_labels=False,
                ),
            )
        else:
            clear_entity(f"{root_path}/{hand_name}/cr_contact_vertices_yellow")
        if (
            line_hand_points.ndim == 2
            and line_object_points.ndim == 2
            and line_hand_points.shape == line_object_points.shape
            and line_hand_points.shape[0] > 0
        ):
            line_segments = np.stack([line_hand_points, line_object_points], axis=1)
            id_distances_m = np.asarray(
                metric.get("id_distances_m", np.zeros((0,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            line_labels = _new_metric_distance_labels_mm(
                id_distances_m,
                int(line_segments.shape[0]),
            )
            rr.log(
                f"{root_path}/{hand_name}/id_lines_green",
                rr.LineStrips3D(
                    line_segments,
                    colors=np.tile(
                        np.asarray((0, 255, 0), dtype=np.uint8),
                        (line_segments.shape[0], 1),
                    ),
                    radii=0.0006,
                    labels=line_labels,
                    show_labels=bool(show_id_values),
                ),
            )
            label_count = min(int(line_segments.shape[0]), int(id_distances_m.shape[0]))
            finite_idx = np.flatnonzero(np.isfinite(id_distances_m[:label_count]))
            if show_id_values and finite_idx.size > 0:
                selected = finite_idx
                selected = selected[np.argsort(id_distances_m[selected])[::-1]]
                label_positions = (
                    line_hand_points[selected] + line_object_points[selected]
                ) * 0.5
                label_offsets = np.zeros_like(label_positions, dtype=np.float32)
                label_offsets[:, 1] = 0.004
                label_values = id_distances_m[selected] * 1000.0
                rr.log(
                    f"{root_path}/{hand_name}/id_value_labels",
                    rr.Points3D(
                        positions=label_positions + label_offsets,
                        radii=[0.0025] * int(selected.shape[0]),
                        colors=np.tile(
                            np.asarray((0, 255, 0), dtype=np.uint8),
                            (int(selected.shape[0]), 1),
                        ),
                        labels=[f"ID {value:.2f} mm" for value in label_values],
                        show_labels=True,
                    ),
                )
                rr.log(
                    f"{root_path}/{hand_name}/id_summary_label",
                    rr.Points3D(
                        positions=np.mean(label_positions, axis=0, keepdims=True)
                        + np.asarray([[0.0, 0.014, 0.0]], dtype=np.float32),
                        radii=[0.004],
                        colors=np.asarray([[0, 255, 0]], dtype=np.uint8),
                        labels=[
                            f"{hand_name} frame ID max "
                            f"{float(metric.get('id_max_mm', 0.0)):.2f} mm"
                        ],
                        show_labels=True,
                    ),
                )
            else:
                clear_entity(f"{root_path}/{hand_name}/id_value_labels")
                clear_entity(f"{root_path}/{hand_name}/id_summary_label")
        else:
            clear_entity(f"{root_path}/{hand_name}/id_lines_green")
            clear_entity(f"{root_path}/{hand_name}/id_value_labels")
            clear_entity(f"{root_path}/{hand_name}/id_summary_label")


def _make_cylinder_between_points(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[int, int, int, int],
) -> Optional[trimesh.Trimesh]:
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
        return None
    if np.linalg.norm(end - start) <= 1e-8:
        return None
    try:
        cylinder = trimesh.creation.cylinder(
            radius=float(radius),
            segment=np.stack([start, end], axis=0),
            sections=8,
        )
        cylinder.visual.vertex_colors = np.tile(
            np.asarray(color, dtype=np.uint8),
            (int(len(cylinder.vertices)), 1),
        )
        return cylinder
    except Exception:
        return None


def _export_new_metric_scene(
    output_dir: Optional[str],
    file_stem: str,
    obj_mesh_world: Optional[trimesh.Trimesh],
    hand_entries: list[tuple[str, np.ndarray, np.ndarray, dict]],
) -> Optional[str]:
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    scene = trimesh.Scene()
    if obj_mesh_world is not None:
        obj_vis = obj_mesh_world.copy()
        obj_vis.visual.vertex_colors = np.tile(
            np.asarray((130, 130, 130, 90), dtype=np.uint8),
            (int(len(obj_vis.vertices)), 1),
        )
        scene.add_geometry(obj_vis, node_name="object_mesh")
    for hand_name, vertices, faces, metric in hand_entries:
        vertices_np = np.asarray(vertices, dtype=np.float64)
        faces_np = np.asarray(faces, dtype=np.int64)
        if vertices_np.ndim != 2 or faces_np.ndim != 2 or faces_np.shape[0] == 0:
            continue
        hand_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
        rgb = _colors_for_inside_vertices(
            vertices_np.shape[0],
            metric.get("inside_mask", np.zeros((vertices_np.shape[0],), dtype=bool)),
            base_color=(185, 185, 185),
            inside_color=(255, 0, 0),
        )
        alpha = np.full((rgb.shape[0], 1), 220, dtype=np.uint8)
        hand_mesh.visual.vertex_colors = np.concatenate([rgb, alpha], axis=1)
        scene.add_geometry(hand_mesh, node_name=f"{hand_name}_hand_iv_red")

        inside_points = np.asarray(metric.get("inside_points"), dtype=np.float64)
        closest_points = np.asarray(metric.get("closest_points"), dtype=np.float64)
        if (
            inside_points.ndim == 2
            and closest_points.ndim == 2
            and inside_points.shape == closest_points.shape
        ):
            for idx, (inside_point, closest_point) in enumerate(
                zip(inside_points, closest_points)
            ):
                line_mesh = _make_cylinder_between_points(
                    inside_point,
                    closest_point,
                    radius=0.0007,
                    color=(0, 255, 0, 255),
                )
                if line_mesh is not None:
                    scene.add_geometry(
                        line_mesh,
                        node_name=f"{hand_name}_id_green_{idx:03d}",
                    )
    out_path = os.path.join(output_dir, f"{file_stem}.glb")
    scene.export(out_path)
    return out_path


def run_new_metric_visualization(
    input_paths: list[str],
    obj_pc_by_key: dict,
    obj_mesh_by_key: dict,
    l_hand_layer,
    r_hand_layer,
    sample_idx_filter: Optional[set[int]],
    max_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    show_id_values: bool = False,
    id_label_count: int = 20,
    rerun_port: int = 0,
    rerun_web_port: int = 9090,
    viewer: str = "native",
    fast_penetration_metric: bool = False,
    contact_only_penetration: bool = True,
    top10_contexts_by_sample: Optional[dict[int, list[dict]]] = None,
) -> list[dict]:
    if rr is None:
        raise RuntimeError("rerun is not installed; cannot visualize new metrics")
    _init_rerun_viewer(
        "New ID/IV metric debug",
        rerun_port=rerun_port,
        rerun_web_port=rerun_web_port,
        viewer=viewer,
    )

    results = []
    global_sample_idx = 0
    for path in input_paths:
        for record in _load_items_from_path(path):
            for sample in _iter_samples_from_record(record):
                sample_idx = int(global_sample_idx)
                global_sample_idx += 1
                if (
                    sample_idx_filter is not None
                    and sample_idx not in sample_idx_filter
                ):
                    continue
                text = str(sample.get("text", ""))
                obj_key = (
                    str(sample.get("object") or _extract_object_key(text))
                    .strip()
                    .lower()
                )
                object_meta = sample.get("object_meta")
                if obj_key not in obj_pc_by_key and isinstance(object_meta, dict):
                    meta_name = object_meta.get("object_name")
                    if meta_name is not None:
                        meta_key = str(meta_name).strip().lower()
                        if meta_key in obj_pc_by_key:
                            obj_key = meta_key
                obj_mesh = obj_mesh_by_key.get(obj_key)
                if obj_mesh is None:
                    print(
                        f"[WARN] skip sample {sample_idx}: no mesh for object '{obj_key}'"
                    )
                    continue
                use_left, use_right = _selected_hands(text)
                if not use_left and not use_right:
                    continue
                nframes = _sequence_length(sample["obj_params"])
                if use_left:
                    nframes = min(nframes, _sequence_length(sample["lhand_params"]))
                if use_right:
                    nframes = min(nframes, _sequence_length(sample["rhand_params"]))
                if nframes <= 0:
                    continue
                try:
                    l_seq = np.zeros((0, 0, 3), dtype=np.float32)
                    l_faces_np = np.zeros((0, 3), dtype=np.int64)
                    r_seq = np.zeros((0, 0, 3), dtype=np.float32)
                    r_faces_np = np.zeros((0, 3), dtype=np.int64)
                    if use_left:
                        l_seq_t, _l_joints, l_faces = process_hand_result(
                            l_hand_layer,
                            _slice_frame_indices(
                                sample["lhand_params"],
                                np.arange(int(nframes), dtype=np.int64),
                            ),
                        )
                        l_seq = _to_numpy(l_seq_t)
                        l_faces_np = _to_numpy(l_faces)
                    if use_right:
                        r_seq_t, _r_joints, r_faces = process_hand_result(
                            r_hand_layer,
                            _slice_frame_indices(
                                sample["rhand_params"],
                                np.arange(int(nframes), dtype=np.int64),
                            ),
                        )
                        r_seq = _to_numpy(r_seq_t)
                        r_faces_np = _to_numpy(r_faces)
                except Exception as ex:
                    print(
                        f"[WARN] skip sample {sample_idx}: hand reconstruction failed: {ex}"
                    )
                    continue

                sample_contexts = (
                    top10_contexts_by_sample.get(sample_idx, [])
                    if top10_contexts_by_sample
                    else []
                )
                if not sample_contexts:
                    sample_contexts = [
                        {
                            "path": (
                                f"new_metric/sample_{sample_idx:04d}_"
                                f"{_safe_path_token(obj_key)}"
                            ),
                            "offset": (0.0, 0.0, 0.0),
                        }
                    ]
                frame_summaries = []
                all_hand_metrics = []
                first_frame_glb = None
                frame_count = int(nframes)
                running_states_by_context: dict[str, dict[str, float]] = {}
                first_obj_params = _slice_frame_indices(
                    sample["obj_params"],
                    np.asarray([0], dtype=np.int64),
                )
                first_obj_mesh_world = _transform_object_mesh_to_world(
                    obj_mesh,
                    first_obj_params,
                )
                first_obj_mesh_metric = _transform_object_mesh_world_to_object_frame(
                    first_obj_mesh_world,
                    first_obj_params,
                )
                first_object_bottom_y = _mesh_bottom_axis_value(
                    first_obj_mesh_metric,
                    axis=1,
                )
                sample_last_success = None
                for frame_idx in range(frame_count):
                    obj_frame_params = _slice_frame_indices(
                        sample["obj_params"],
                        np.asarray([frame_idx], dtype=np.int64),
                    )
                    obj_mesh_world = _transform_object_mesh_to_world(
                        obj_mesh,
                        obj_frame_params,
                    )
                    obj_mesh_metric = _transform_object_mesh_world_to_object_frame(
                        obj_mesh_world,
                        first_obj_params,
                    )
                    if obj_mesh_metric is None:
                        continue

                    hand_entries = []
                    hand_metrics = []
                    metric_fn = (
                        _new_inside_vertex_metric_for_hand_fast
                        if fast_penetration_metric
                        else _new_inside_vertex_metric_for_hand
                    )
                    hand_vertex_parts = []
                    active_hand_parts = []
                    if use_left and l_seq.ndim == 3 and frame_idx < l_seq.shape[0]:
                        l_vertices = _transform_points_world_to_object_frame(
                            l_seq[frame_idx],
                            first_obj_params,
                        )
                        hand_vertex_parts.append(l_vertices)
                        active_hand_parts.append(("left", l_vertices, l_faces_np))
                    if use_right and r_seq.ndim == 3 and frame_idx < r_seq.shape[0]:
                        r_vertices = _transform_points_world_to_object_frame(
                            r_seq[frame_idx],
                            first_obj_params,
                        )
                        hand_vertex_parts.append(r_vertices)
                        active_hand_parts.append(("right", r_vertices, r_faces_np))
                    if not active_hand_parts:
                        continue

                    cr_value = _fast_cr_for_frame(obj_mesh_metric, hand_vertex_parts)
                    compute_penetration = (not contact_only_penetration) or (
                        _frame_con_pass_from_cr(cr_value) is True
                    )
                    for hand_name, hand_vertices, hand_faces in active_hand_parts:
                        if compute_penetration:
                            hand_metric = metric_fn(
                                obj_mesh_metric,
                                hand_vertices,
                                hand_faces,
                                hand_name,
                            )
                        else:
                            hand_metric = _new_inside_vertex_metric_for_hand(
                                None,
                                hand_vertices,
                                hand_faces,
                                hand_name,
                            )
                        hand_metric["contact_mask"] = _fast_contact_mask_for_hand(
                            obj_mesh_metric, hand_vertices
                        )
                        hand_entries.append(
                            (hand_name, hand_vertices, hand_faces, hand_metric)
                        )
                        hand_metrics.append(hand_metric)
                        all_hand_metrics.append(hand_metric)

                    frame_summary = _combine_new_frame_metrics(
                        hand_metrics,
                        obj_mesh_metric,
                    )
                    frame_summary["frame_idx"] = int(frame_idx)
                    frame_summaries.append(frame_summary)
                    pen_1cm_event = _pen_1cm_event_from_new_id(
                        frame_summary,
                        cr_value,
                    )
                    is_last_frame = frame_idx == frame_count - 1
                    con_pass = _frame_con_pass_from_cr(cr_value)
                    current_object_bottom_y = _mesh_bottom_axis_value(
                        obj_mesh_metric,
                        axis=1,
                    )
                    if (
                        first_object_bottom_y is not None
                        and current_object_bottom_y is not None
                    ):
                        object_lift_m = float(
                            current_object_bottom_y - first_object_bottom_y
                        )
                        off_ground = bool(object_lift_m > OFF_GROUND_THRESHOLD_M)
                    else:
                        object_lift_m = None
                        off_ground = None
                    phy_success = (
                        bool(off_ground and con_pass)
                        if off_ground is not None and con_pass is not None
                        else None
                    )
                    accel_m = _frame_hand_accel_m(
                        l_seq,
                        r_seq,
                        use_left,
                        use_right,
                        frame_idx,
                    )
                    success = (
                        _success_from_contact_and_pen_1cm(con_pass, pen_1cm_event)
                        if is_last_frame
                        else None
                    )
                    if is_last_frame:
                        sample_last_success = success
                    _rerun_set_time_sequence("frame", frame_idx)
                    frame_extra = {
                        "cr": cr_value,
                        "pen_1cm_event": pen_1cm_event,
                        "is_last_frame": is_last_frame,
                        "con_pass": con_pass,
                        "accel_m": accel_m,
                        "object_lift_m": object_lift_m,
                        "off_ground": off_ground,
                        "phy_success": phy_success,
                        "success": success,
                        "success_rate_percent": (
                            100.0 * float(bool(success))
                            if success is not None
                            else None
                        ),
                    }
                    for sample_context in sample_contexts:
                        context_root = str(sample_context.get("path", "new_metric"))
                        context_offset = tuple(
                            sample_context.get("offset", (0.0, 0.0, 0.0))
                        )
                        contribution = None
                        running_state = None
                        if "metric" in sample_context:
                            contribution = _top10_current_frame_contribution(
                                frame_summary,
                                frame_extra,
                                sample_context,
                            )
                            running_state = running_states_by_context.setdefault(
                                context_root,
                                {"numerator": 0.0, "denominator": 0.0},
                            )
                            running_state["numerator"] += float(
                                contribution.get("numerator_add", 0.0)
                            )
                            running_state["denominator"] += float(
                                contribution.get("denominator_add", 0)
                            )
                        ours_root = f"{context_root}/ours"
                        _log_new_metric_visualization(
                            ours_root,
                            obj_mesh_metric,
                            hand_entries,
                            offset=context_offset,
                            show_id_values=show_id_values,
                            id_label_count=id_label_count,
                        )
                        if "metric" in sample_context:
                            _log_new_metric_accumulator_debug_labels(
                                ours_root,
                                obj_mesh_metric,
                                frame_summary,
                                frame_extra,
                                sample_context,
                                contribution,
                                running_state,
                                offset=context_offset,
                            )
                        else:
                            _log_new_metric_frame_label(
                                ours_root,
                                obj_mesh_metric,
                                frame_summary,
                                frame_extra,
                                top10_context=None,
                                offset=context_offset,
                            )
                    if output_dir and frame_idx == 0:
                        first_frame_glb = _export_new_metric_scene(
                            output_dir,
                            f"sample_{sample_idx:04d}_{_safe_path_token(obj_key)}_frame_{frame_idx:04d}",
                            obj_mesh_metric,
                            hand_entries,
                        )

                if not frame_summaries:
                    print(
                        f"[WARN] skip sample {sample_idx}: no valid frames for new metric"
                    )
                    continue

                overall = _combine_new_inside_vertex_metrics(all_hand_metrics)
                frame_id_maxima_mm = np.asarray(
                    [item.get("id_max_mm", 0.0) for item in frame_summaries],
                    dtype=np.float64,
                )
                positive_frame_id_maxima_mm = frame_id_maxima_mm[
                    np.isfinite(frame_id_maxima_mm) & (frame_id_maxima_mm > 0.0)
                ]
                overall["id_mean_mm"] = (
                    float(np.mean(positive_frame_id_maxima_mm))
                    if positive_frame_id_maxima_mm.size
                    else 0.0
                )
                overall["id_max_mm"] = (
                    float(np.max(positive_frame_id_maxima_mm))
                    if positive_frame_id_maxima_mm.size
                    else 0.0
                )
                frame_iv = np.asarray(
                    [item.get("iv_volume_cm3", 0.0) for item in frame_summaries],
                    dtype=np.float64,
                )
                frame_inside = np.asarray(
                    [item.get("inside_count", 0) for item in frame_summaries],
                    dtype=np.float64,
                )
                overall["frame_count"] = int(len(frame_summaries))
                overall["iv_volume_cm3_sum"] = float(np.sum(frame_iv))
                overall["iv_volume_cm3_mean"] = float(np.mean(frame_iv))
                overall["iv_volume_cm3_max"] = float(np.max(frame_iv))
                overall["inside_count_mean_per_frame"] = float(np.mean(frame_inside))
                row = {
                    "file_name": os.path.basename(path),
                    "sample_idx": sample_idx,
                    "object": obj_key,
                    "text": text,
                    **overall,
                    "success": sample_last_success,
                    "frames": frame_summaries,
                }
                results.append(row)
                print(
                    "[NEW_METRIC] "
                    f"sample={sample_idx} object={obj_key} frames={overall['frame_count']} "
                    f"ID_frame_max_mean={overall['id_mean_mm']:.4f}mm "
                    f"ID_frame_max_max={overall['id_max_mm']:.4f}mm "
                    f"IV_mean={overall['iv_volume_cm3_mean']:.6f}cm^3 "
                    f"IV_max={overall['iv_volume_cm3_max']:.6f}cm^3 "
                    f"inside_mean={overall['inside_count_mean_per_frame']:.2f} "
                    f"ratio={overall['inside_ratio_per_778_per_hand']:.6f} "
                    f"Success={_format_success_fail(sample_last_success)}"
                    + (f" glb_first_frame={first_frame_glb}" if first_frame_glb else "")
                )
                if max_samples is not None and len(results) >= int(max_samples):
                    return results
    return results


def _csv_bool_or_empty(value) -> str:
    if value is None:
        return ""
    return str(int(bool(value)))


def _new_metric_csv_sample_key_from_values(file_name, sample_idx) -> tuple[str, str]:
    return (str(file_name), str(sample_idx))


def _new_metric_existing_completed_rows(
    path: str,
    header: list[str],
) -> tuple[list[dict], set[tuple[str, str]]]:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return [], set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != header:
            print(
                "[RESUME] existing CSV header does not match current format; "
                "starting a fresh CSV."
            )
            return [], set()
        rows = list(reader)

    by_sample: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        sample_idx = row.get("sample_idx")
        if sample_idx in (None, ""):
            continue
        key = _new_metric_csv_sample_key_from_values(
            row.get("file_name", ""),
            sample_idx,
        )
        by_sample.setdefault(key, []).append(row)

    completed_keys = set()
    for key, sample_rows in by_sample.items():
        for row in sample_rows:
            try:
                frame_idx = int(row.get("frame_idx", -1))
                frames_total = int(row.get("frames_total", -1))
            except (TypeError, ValueError):
                continue
            if frames_total > 0 and frame_idx == frames_total - 1:
                completed_keys.add(key)
                break

    completed_rows = [
        row
        for row in rows
        if _new_metric_csv_sample_key_from_values(
            row.get("file_name", ""),
            row.get("sample_idx", ""),
        )
        in completed_keys
    ]
    dropped = len(rows) - len(completed_rows)
    if completed_rows or dropped:
        print(
            "[RESUME] "
            f"kept_completed_samples={len(completed_keys)} "
            f"kept_rows={len(completed_rows)} dropped_partial_rows={dropped}"
        )
    return completed_rows, completed_keys


def write_new_metric_per_frame_csv(
    path: str,
    input_paths: list[str],
    obj_pc_by_key: dict,
    obj_mesh_by_key: dict,
    l_hand_layer,
    r_hand_layer,
    sample_idx_filter: Optional[set[int]],
    max_samples: Optional[int],
    fast_penetration_metric: bool = False,
    contact_only_penetration: bool = True,
    resume: bool = False,
    part_labels: Optional[dict] = None,
    target_parts: Optional[set[str]] = None,
    gaze_sigma_m: float = 0.05,
    quick: bool = False,
    grounding_only: bool = False,
    gsr_contact_frames: int = 3,
    id_pen_only: bool = False,
) -> int:
    global QUICK_EVAL_ACTIVE
    previous_quick_eval = QUICK_EVAL_ACTIVE
    # Both lightweight modes skip expensive IV voxelization.  ``grounding_only``
    # additionally skips every penetration query; it is intended for G2C/PCP/
    # Part Acc. analysis only.
    QUICK_EVAL_ACTIVE = bool(quick or grounding_only or id_pen_only)
    header = [
        "file_name",
        "split",
        "object",
        "text",
        "sample_idx",
        "batch_sample_idx",
        "seed",
        "repeat_idx",
        "prompt_idx",
        "dataset_idx",
        "frame_idx",
        "frames_total",
        "ID_mean_mm",
        "ID_max_mm",
        "IV_volume_cm3",
        "CR",
        "Con_pass",
        "Pen_1cm",
        "Success",
        "Accel_m",
        "Object_lift_y_m",
        "Off_ground",
        "Phy",
        "active_hands",
        "left_wrist_canonical_x",
        "left_wrist_canonical_y",
        "left_wrist_canonical_z",
        "right_wrist_canonical_x",
        "right_wrist_canonical_y",
        "right_wrist_canonical_z",
        "PCP",
        "Part_Acc",
        "G2C_m",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    completed_rows, completed_sample_keys = (
        _new_metric_existing_completed_rows(path, header) if resume else ([], set())
    )

    eligible_count = 0
    global_idx_for_count = 0
    for input_path in input_paths:
        file_name_for_count = _gt_only_output_file_name(os.path.basename(input_path))
        for record in _load_items_from_path(input_path):
            for _sample in _iter_samples_from_record(record):
                if (
                    target_parts is not None
                    and _target_part_from_text(_sample.get("text", ""))
                    not in target_parts
                ):
                    global_idx_for_count += 1
                    continue
                sample_key = _new_metric_csv_sample_key_from_values(
                    file_name_for_count,
                    global_idx_for_count,
                )
                if (
                    sample_idx_filter is None
                    or global_idx_for_count in sample_idx_filter
                ) and sample_key not in completed_sample_keys:
                    eligible_count += 1
                global_idx_for_count += 1
    progress_total = (
        min(int(max_samples), eligible_count)
        if max_samples is not None
        else eligible_count
    )

    rows_written = len(completed_rows)
    samples_written = 0
    global_sample_idx = 0
    pbar = tqdm.tqdm(total=progress_total, desc="new metric samples", unit="sample")
    try:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in completed_rows:
                writer.writerow([row.get(column, "") for column in header])
            for input_path in input_paths:
                source_file_name = os.path.basename(input_path)
                file_name = _gt_only_output_file_name(source_file_name)
                split = _split_tag_from_file_name(file_name)
                for record in _load_items_from_path(input_path):
                    for raw_sample in _iter_samples_from_record(record):
                        sample_idx = int(global_sample_idx)
                        global_sample_idx += 1
                        sample = _sample_as_gt_only_if_requested(
                            raw_sample, source_file_name
                        )
                        if sample is None:
                            continue
                        if (
                            sample_idx_filter is not None
                            and sample_idx not in sample_idx_filter
                        ):
                            continue
                        sample_key = _new_metric_csv_sample_key_from_values(
                            file_name,
                            sample_idx,
                        )
                        if sample_key in completed_sample_keys:
                            continue
                        if max_samples is not None and samples_written >= int(
                            max_samples
                        ):
                            return rows_written

                        text = str(sample.get("text", ""))
                        if (
                            target_parts is not None
                            and _target_part_from_text(text) not in target_parts
                        ):
                            continue
                        sample_meta = (
                            sample.get("object_meta")
                            if isinstance(sample.get("object_meta"), dict)
                            else {}
                        )
                        obj_key = (
                            str(sample.get("object") or _extract_object_key(text))
                            .strip()
                            .lower()
                        )
                        object_meta = sample.get("object_meta")
                        if obj_key not in obj_pc_by_key and isinstance(
                            object_meta, dict
                        ):
                            meta_name = object_meta.get("object_name")
                            if meta_name is not None:
                                meta_key = str(meta_name).strip().lower()
                                if meta_key in obj_pc_by_key:
                                    obj_key = meta_key
                        obj_mesh = obj_mesh_by_key.get(obj_key)
                        if obj_mesh is None:
                            print(
                                f"[WARN] skip sample {sample_idx}: no mesh for object '{obj_key}'"
                            )
                            pbar.update(1)
                            samples_written += 1
                            continue

                        use_left, use_right = _selected_hands(text)
                        if not use_left and not use_right:
                            pbar.update(1)
                            samples_written += 1
                            continue
                        nframes = _sequence_length(sample["obj_params"])
                        if use_left:
                            nframes = min(
                                nframes, _sequence_length(sample["lhand_params"])
                            )
                        if use_right:
                            nframes = min(
                                nframes, _sequence_length(sample["rhand_params"])
                            )
                        if nframes <= 0:
                            pbar.update(1)
                            samples_written += 1
                            continue

                        object_points = _to_numpy(obj_pc_by_key[obj_key]).astype(
                            np.float64
                        )
                        target_part = _target_part_from_text(text)
                        first_contact_frame = None

                        try:
                            l_seq = np.zeros((0, 0, 3), dtype=np.float32)
                            l_joints_seq = np.zeros((0, 0, 3), dtype=np.float32)
                            l_faces_np = np.zeros((0, 3), dtype=np.int64)
                            r_seq = np.zeros((0, 0, 3), dtype=np.float32)
                            r_joints_seq = np.zeros((0, 0, 3), dtype=np.float32)
                            r_faces_np = np.zeros((0, 3), dtype=np.int64)
                            if use_left:
                                l_seq_t, l_joints, l_faces = process_hand_result(
                                    l_hand_layer,
                                    _slice_frame_indices(
                                        sample["lhand_params"],
                                        np.arange(int(nframes), dtype=np.int64),
                                    ),
                                )
                                l_seq = _to_numpy(l_seq_t)
                                l_joints_seq = _to_numpy(l_joints)
                                l_faces_np = _to_numpy(l_faces)
                            if use_right:
                                r_seq_t, r_joints, r_faces = process_hand_result(
                                    r_hand_layer,
                                    _slice_frame_indices(
                                        sample["rhand_params"],
                                        np.arange(int(nframes), dtype=np.int64),
                                    ),
                                )
                                r_seq = _to_numpy(r_seq_t)
                                r_joints_seq = _to_numpy(r_joints)
                                r_faces_np = _to_numpy(r_faces)
                        except Exception as ex:
                            print(
                                f"[WARN] skip sample {sample_idx}: hand reconstruction failed: {ex}"
                            )
                            pbar.update(1)
                            samples_written += 1
                            continue

                        first_obj_params = _slice_frame_indices(
                            sample["obj_params"],
                            np.asarray([0], dtype=np.int64),
                        )
                        first_obj_mesh_world = _transform_object_mesh_to_world(
                            obj_mesh,
                            first_obj_params,
                        )
                        first_obj_mesh_metric = (
                            _transform_object_mesh_world_to_object_frame(
                                first_obj_mesh_world,
                                first_obj_params,
                            )
                        )
                        first_object_bottom_y = _mesh_bottom_axis_value(
                            first_obj_mesh_metric,
                            axis=1,
                        )
                        for frame_idx in range(int(nframes)):
                            is_last_frame = frame_idx == int(nframes) - 1
                            in_gsr_window = frame_idx >= max(
                                0, int(nframes) - max(1, int(gsr_contact_frames))
                            )
                            obj_frame_params = _slice_frame_indices(
                                sample["obj_params"],
                                np.asarray([frame_idx], dtype=np.int64),
                            )
                            obj_mesh_world = _transform_object_mesh_to_world(
                                obj_mesh,
                                obj_frame_params,
                            )
                            obj_mesh_metric = (
                                _transform_object_mesh_world_to_object_frame(
                                    obj_mesh_world,
                                    obj_frame_params,
                                )
                            )
                            # Grounding labels are indexed in the current
                            # canonical object frame, while the established
                            # GSR lift protocol measures displacement in the
                            # first-frame metric coordinate system.
                            obj_mesh_lift_metric = (
                                _transform_object_mesh_world_to_object_frame(
                                    obj_mesh_world,
                                    first_obj_params,
                                )
                            )
                            if obj_mesh_metric is None:
                                continue

                            hand_metrics = []
                            active_hand_parts = []
                            metric_fn = (
                                _new_inside_vertex_metric_for_hand_fast
                                if (fast_penetration_metric or id_pen_only)
                                else _new_inside_vertex_metric_for_hand
                            )
                            if (
                                use_left
                                and l_seq.ndim == 3
                                and frame_idx < l_seq.shape[0]
                            ):
                                l_vertices = _transform_points_world_to_object_frame(
                                    l_seq[frame_idx],
                                    obj_frame_params,
                                )
                                active_hand_parts.append(
                                    ("left", l_vertices, l_faces_np)
                                )
                            if (
                                use_right
                                and r_seq.ndim == 3
                                and frame_idx < r_seq.shape[0]
                            ):
                                r_vertices = _transform_points_world_to_object_frame(
                                    r_seq[frame_idx],
                                    obj_frame_params,
                                )
                                active_hand_parts.append(
                                    ("right", r_vertices, r_faces_np)
                                )
                            if not active_hand_parts:
                                continue

                            # Detect contact on the metric mesh (the same
                            # surface used for CR), then map those hand
                            # vertices to annotated 1,024-point indices.  A
                            # second 5mm test against the sparse point cloud
                            # would incorrectly discard valid mesh contact.
                            contact_hand_vertices = []
                            contact_vertex_count = 0
                            finite_hand_vertex_count = 0
                            for _hand_name, hand_vertices, _hand_faces in active_hand_parts:
                                contact_mask = _fast_contact_mask_for_hand(
                                    obj_mesh_metric,
                                    hand_vertices,
                                    threshold_m=LATETHOI_CONTACT_THRESHOLD_M,
                                )
                                finite_hand_vertex_count += int(
                                    np.count_nonzero(
                                        np.all(np.isfinite(hand_vertices), axis=1)
                                    )
                                )
                                contact_vertex_count += int(np.count_nonzero(contact_mask))
                                if np.any(contact_mask):
                                    contact_hand_vertices.append(hand_vertices[contact_mask])
                            if (
                                first_contact_frame is None
                                and bool(contact_hand_vertices)
                            ):
                                first_contact_frame = int(frame_idx)
                            contact_point_indices = (
                                _contact_object_point_indices(
                                    contact_hand_vertices,
                                    object_points,
                                    threshold_m=float("inf"),
                                )
                                if (not id_pen_only) and contact_hand_vertices and (
                                    not quick
                                    or is_last_frame
                                    or grounding_only
                                )
                                else np.zeros((0,), dtype=np.int64)
                            )
                            if id_pen_only:
                                pcp, part_acc = None, None
                            elif not quick or is_last_frame or grounding_only:
                                pcp, part_acc = _part_contact_metrics(
                                    contact_point_indices,
                                    obj_key,
                                    target_part,
                                    part_labels,
                                )
                            else:
                                pcp, part_acc = None, None

                            cr_value = (
                                float(contact_vertex_count)
                                / float(finite_hand_vertex_count)
                                if finite_hand_vertex_count > 0
                                else 0.0
                            )
                            compute_penetration = (not grounding_only) and (
                                (id_pen_only or not quick or in_gsr_window)
                                and (
                                    (not contact_only_penetration)
                                    or (_frame_con_pass_from_cr(cr_value) is True)
                                )
                            )
                            for (
                                hand_name,
                                hand_vertices,
                                hand_faces,
                            ) in active_hand_parts:
                                if compute_penetration:
                                    hand_metrics.append(
                                        metric_fn(
                                            obj_mesh_metric,
                                            hand_vertices,
                                            hand_faces,
                                            hand_name,
                                        )
                                    )
                                else:
                                    hand_metrics.append(
                                        _new_inside_vertex_metric_for_hand(
                                            None,
                                            hand_vertices,
                                            hand_faces,
                                            hand_name,
                                        )
                                    )

                            frame_metric = _combine_new_frame_metrics(
                                hand_metrics, obj_mesh_metric
                            )
                            pen_1cm_event = _pen_1cm_event_from_new_id(
                                frame_metric,
                                cr_value,
                            )
                            try:
                                cr_float = float(cr_value)
                            except (TypeError, ValueError):
                                cr_float = None
                            con_pass = bool(
                                cr_float is not None
                                and np.isfinite(cr_float)
                                and cr_float > 0.0
                            )
                            current_object_bottom_y = _mesh_bottom_axis_value(
                                obj_mesh_lift_metric,
                                axis=1,
                            )
                            if (
                                first_object_bottom_y is not None
                                and current_object_bottom_y is not None
                            ):
                                object_lift_m = float(
                                    current_object_bottom_y - first_object_bottom_y
                                )
                                off_ground = bool(
                                    object_lift_m > OFF_GROUND_THRESHOLD_M
                                )
                            else:
                                object_lift_m = None
                                off_ground = None
                            phy_success = (
                                bool(off_ground and con_pass)
                                if off_ground is not None and con_pass is not None
                                else None
                            )
                            accel_m = (
                                None
                                if id_pen_only or quick or grounding_only
                                else _frame_hand_accel_m(
                                    l_seq,
                                    r_seq,
                                    use_left,
                                    use_right,
                                    frame_idx,
                                )
                            )
                            success = (
                                _success_from_contact_and_pen_1cm(
                                    con_pass,
                                    pen_1cm_event,
                                )
                                if is_last_frame
                                else None
                            )
                            g2c_m = None
                            if (
                                is_last_frame
                                and not id_pen_only
                                and contact_point_indices.size > 0
                                and first_contact_frame is not None
                            ):
                                gaze_target_idx = _gaze_target_point_index(
                                    sample.get("gaze"),
                                    sample.get("gt_obj_params"),
                                    object_points,
                                    nframes,
                                    sigma_m=float(gaze_sigma_m),
                                    frame_end_exclusive=first_contact_frame,
                                )
                                if gaze_target_idx is not None:
                                    g2c_m = float(
                                        np.min(
                                            np.linalg.norm(
                                                object_points[int(gaze_target_idx)][None]
                                                - object_points[contact_point_indices],
                                                axis=1,
                                            )
                                        )
                                    )
                            wrist_canonical = (
                                np.full((2, 3), np.nan, dtype=np.float32)
                                if id_pen_only or quick or grounding_only
                                else _canonical_wrist_frame_from_joints(
                                    l_joints_seq,
                                    r_joints_seq,
                                    use_left,
                                    use_right,
                                    frame_idx,
                                    first_obj_params,
                                )
                            )
                            writer.writerow(
                                [
                                    file_name,
                                    split,
                                    obj_key,
                                    text,
                                    sample_idx,
                                    int(sample.get("sample_idx", -1)),
                                    sample_meta.get(
                                        "seed",
                                        sample_meta.get(
                                            "source_seed",
                                            sample_meta.get("test_seed", ""),
                                        ),
                                    ),
                                    sample_meta.get("repeat_idx", ""),
                                    sample_meta.get("prompt_idx", ""),
                                    sample_meta.get("dataset_idx", ""),
                                    frame_idx,
                                    int(nframes),
                                    _format_csv_float(frame_metric.get("id_mean_mm")),
                                    _format_csv_float(frame_metric.get("id_max_mm")),
                                    _format_csv_float(
                                        frame_metric.get("iv_volume_cm3")
                                    ),
                                    _format_csv_float(cr_value),
                                    _csv_bool_or_empty(con_pass),
                                    _csv_bool_or_empty(pen_1cm_event),
                                    (
                                        _csv_bool_or_empty(success)
                                        if is_last_frame
                                        else ""
                                    ),
                                    _format_csv_float(accel_m),
                                    _format_csv_float(object_lift_m),
                                    _csv_bool_or_empty(off_ground),
                                    _csv_bool_or_empty(phy_success),
                                    int(frame_metric.get("active_hands", 0)),
                                    _format_csv_float(wrist_canonical[0, 0]),
                                    _format_csv_float(wrist_canonical[0, 1]),
                                    _format_csv_float(wrist_canonical[0, 2]),
                                    _format_csv_float(wrist_canonical[1, 0]),
                                    _format_csv_float(wrist_canonical[1, 1]),
                                    _format_csv_float(wrist_canonical[1, 2]),
                                    _format_csv_float(pcp),
                                    _format_csv_float(part_acc),
                                    _format_csv_float(g2c_m),
                                ]
                            )
                            rows_written += 1

                        samples_written += 1
                        pbar.set_postfix(
                            sample=sample_idx, object=obj_key, rows=rows_written
                        )
                        pbar.update(1)
    finally:
        QUICK_EVAL_ACTIVE = previous_quick_eval
        pbar.close()
    return rows_written


def write_new_metric_summary_markdown(
    path: str,
    csv_path: str,
    object_scale_by_key: Optional[dict[str, float]] = None,
    seed_csv_path: Optional[str] = None,
    gsr_contact_frames: int = 3,
    gsr_lift_threshold_m: float = 0.005,
    quick: bool = False,
) -> None:
    def _read_rows(csv_path: str) -> list[dict]:
        if not os.path.exists(csv_path):
            return []
        with open(csv_path, newline="") as f:
            return list(csv.DictReader(f))

    def _recover_missing_seed_values(
        metric_rows: list[dict],
    ) -> dict[str, str]:
        """Recover blank seed columns only when the CSV ordering proves a cycle.

        Older DiffH2O CSVs repeat ``batch_sample_idx`` once per inference seed.
        Older LatentHOI CSVs instead use a globally increasing batch index, but
        their ``*_seed.pkl`` rows contain an identical ordered condition block
        per seed.  Require one of these complete repetition patterns; otherwise
        leave the seed blank rather than inventing an unsupported grouping.
        """

        def _numeric_sort_key(value) -> tuple[int, object]:
            try:
                return (0, int(str(value)))
            except (TypeError, ValueError):
                return (1, str(value))

        rows_by_file: dict[str, list[dict]] = {}
        for row in metric_rows:
            rows_by_file.setdefault(str(row.get("file_name", "")), []).append(row)

        recovered: dict[str, str] = {}
        for file_name, file_rows in rows_by_file.items():
            seeds = [row.get("seed") for row in file_rows]
            if any(seed not in (None, "") for seed in seeds):
                continue

            first_row_by_sample: dict[str, dict] = {}
            for row in file_rows:
                sample_idx = row.get("sample_idx")
                if sample_idx in (None, ""):
                    first_row_by_sample = {}
                    break
                first_row_by_sample.setdefault(str(sample_idx), row)
            if len(first_row_by_sample) < 2:
                continue

            ordered_samples = sorted(
                first_row_by_sample.items(),
                key=lambda item: _numeric_sort_key(item[0]),
            )
            sample_seed: dict[str, int] = {}
            recovery_method = ""

            batch_values = [
                sample_row.get("batch_sample_idx")
                for _sample_idx, sample_row in ordered_samples
            ]
            if all(value not in (None, "") for value in batch_values):
                batch_counts: dict[str, int] = {}
                for value in batch_values:
                    key = str(value)
                    batch_counts[key] = batch_counts.get(key, 0) + 1
                repeat_counts = set(batch_counts.values())
                if (
                    len(repeat_counts) == 1
                    and 2 <= next(iter(repeat_counts)) <= 100
                ):
                    occurrence: dict[str, int] = {}
                    for sample_idx, sample_row in ordered_samples:
                        batch_idx = str(sample_row.get("batch_sample_idx"))
                        seed = occurrence.get(batch_idx, 0)
                        occurrence[batch_idx] = seed + 1
                        sample_seed[sample_idx] = seed
                    expected_batch_ids = set(batch_counts)
                    inferred_seed_count = next(iter(repeat_counts))
                    inferred_sets = {
                        seed: {
                            str(sample_row.get("batch_sample_idx"))
                            for sample_idx, sample_row in ordered_samples
                            if sample_seed.get(sample_idx) == seed
                        }
                        for seed in range(inferred_seed_count)
                    }
                    if not all(
                        value == expected_batch_ids
                        for value in inferred_sets.values()
                    ):
                        sample_seed = {}
                    else:
                        recovery_method = "repeated batch_sample_idx"

            if not sample_seed and "_seed" in os.path.basename(file_name).lower():
                signatures = [
                    (
                        str(sample_row.get("object", "")),
                        str(sample_row.get("text", "")),
                        str(sample_row.get("frames_total", "")),
                    )
                    for _sample_idx, sample_row in ordered_samples
                ]
                sample_count = len(signatures)
                candidates = [10] + [
                    count
                    for count in range(2, min(100, sample_count // 2) + 1)
                    if count != 10
                ]
                for inferred_seed_count in candidates:
                    if sample_count % inferred_seed_count != 0:
                        continue
                    block_size = sample_count // inferred_seed_count
                    if block_size < 2:
                        continue
                    base = signatures[:block_size]
                    if not all(
                        signatures[
                            seed * block_size : (seed + 1) * block_size
                        ]
                        == base
                        for seed in range(inferred_seed_count)
                    ):
                        continue
                    sample_seed = {
                        sample_idx: position // block_size
                        for position, (sample_idx, _sample_row) in enumerate(
                            ordered_samples
                        )
                    }
                    recovery_method = "repeated ordered condition block"
                    break

            if not sample_seed:
                continue
            for row in file_rows:
                sample_idx = str(row.get("sample_idx", ""))
                if sample_idx in sample_seed:
                    row["seed"] = str(sample_seed[sample_idx])
            recovered[file_name] = recovery_method
        return recovered

    def _float_value(row: dict, key: str) -> Optional[float]:
        value = row.get(key)
        if value in (None, ""):
            return None
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return value_f if np.isfinite(value_f) else None

    def _mean_value(rows: list[dict], key: str) -> Optional[float]:
        vals = [_float_value(row, key) for row in rows]
        vals = [value for value in vals if value is not None]
        return float(np.mean(vals)) if vals else None

    def _mean_nonzero_value(rows: list[dict], key: str) -> Optional[float]:
        vals = [_float_value(row, key) for row in rows]
        vals = [
            value for value in vals if value is not None and not np.isclose(value, 0.0)
        ]
        return float(np.mean(vals)) if vals else None

    def _row_has_contact_cr(row: dict) -> bool:
        cr_value = _float_value(row, "CR")
        return bool(cr_value is not None and cr_value > 0.0)

    def _mean_value_for_contact_cr(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        vals = [_float_value(row, key) for row in rows if _row_has_contact_cr(row)]
        vals = [value for value in vals if value is not None]
        return float(np.mean(vals)) if vals else None

    def _max_value(rows: list[dict], key: str) -> Optional[float]:
        vals = [_float_value(row, key) for row in rows]
        vals = [value for value in vals if value is not None]
        return float(np.max(vals)) if vals else None

    def _sample_metric_groups(
        rows: list[dict],
        key: str,
        row_filter=None,
    ) -> list[list[float]]:
        by_sample: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if row_filter is not None and not row_filter(row):
                continue
            value = _float_value(row, key)
            if value is None:
                continue
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(value)
        return list(by_sample.values())

    def _sample_groups_with_min_filtered_frames(
        rows: list[dict],
        key: str,
        row_filter,
        min_count: int = 5,
    ) -> list[list[float]]:
        by_sample: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if not row_filter(row):
                continue
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            value = _float_value(row, key)
            if value is not None:
                by_sample.setdefault(sample_key, []).append(float(value))
        return [group for group in by_sample.values() if len(group) >= int(min_count)]

    def _sample_nonzero_metric_groups(rows: list[dict], key: str) -> list[list[float]]:
        return [
            [value for value in group if not np.isclose(value, 0.0)]
            for group in _sample_metric_groups(rows, key)
        ]

    def _mean_sample_mean(rows: list[dict], key: str) -> Optional[float]:
        vals = [
            float(np.mean(group)) for group in _sample_metric_groups(rows, key) if group
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_nonzero_mean(rows: list[dict], key: str) -> Optional[float]:
        vals = [
            float(np.mean(group))
            for group in _sample_nonzero_metric_groups(rows, key)
            if group
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_mean_for_contact_cr(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        vals = [
            float(np.mean(group))
            for group in _sample_metric_groups(rows, key, _row_has_contact_cr)
            if group
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_contact_mean_min5_samples(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        vals = [
            float(np.mean(group))
            for group in _sample_groups_with_min_filtered_frames(
                rows, key, _row_has_contact_cr, min_count=5
            )
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_filtered_mean_positive_samples(
        rows: list[dict],
        key: str,
        row_filter=None,
    ) -> Optional[float]:
        vals = [
            float(np.mean(group))
            for group in _sample_metric_groups(rows, key, row_filter)
            if group and float(np.mean(group)) > 0.0
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_nonzero_frame_mean(
        rows: list[dict],
        key: str,
        row_filter=None,
    ) -> Optional[float]:
        vals = []
        for group in _sample_metric_groups(rows, key, row_filter):
            filtered = [
                float(value)
                for value in group
                if value is not None and not np.isclose(float(value), 0.0)
            ]
            if filtered:
                vals.append(float(np.mean(filtered)))
        return float(np.mean(vals)) if vals else None

    def _mean_sample_nonzero_frame_max(
        rows: list[dict],
        key: str,
        row_filter=None,
    ) -> Optional[float]:
        vals = []
        for group in _sample_metric_groups(rows, key, row_filter):
            filtered = [
                float(value)
                for value in group
                if value is not None and not np.isclose(float(value), 0.0)
            ]
            if filtered:
                vals.append(float(np.max(filtered)))
        return float(np.mean(vals)) if vals else None

    def _mean_sample_max(rows: list[dict], key: str) -> Optional[float]:
        vals = [
            float(np.max(group)) for group in _sample_metric_groups(rows, key) if group
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_nonzero_max(rows: list[dict], key: str) -> Optional[float]:
        vals = [
            float(np.max(group))
            for group in _sample_nonzero_metric_groups(rows, key)
            if group
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_max_for_contact_cr(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        vals = [
            float(np.max(group))
            for group in _sample_metric_groups(rows, key, _row_has_contact_cr)
            if group
        ]
        return float(np.mean(vals)) if vals else None

    def _mean_sample_contact_max_min5_samples(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        vals = [
            float(np.max(group))
            for group in _sample_groups_with_min_filtered_frames(
                rows, key, _row_has_contact_cr, min_count=5
            )
        ]
        return float(np.mean(vals)) if vals else None

    def _row_has_positive_id(row: dict) -> bool:
        value = _float_value(row, "ID_mean_mm")
        return value is not None and value > 0.0

    def _mean_sample_positive_id_frame_mean(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        return _mean_sample_nonzero_frame_mean(rows, key, _row_has_positive_id)

    def _mean_sample_positive_id_frame_max(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        return _mean_sample_nonzero_frame_max(rows, key, _row_has_positive_id)

    def _scaled(value: Optional[float], scale: float) -> Optional[float]:
        return None if value is None else float(value) * float(scale)

    def _sum_value(rows: list[dict], key: str) -> float:
        vals = [_float_value(row, key) for row in rows]
        return float(sum(value for value in vals if value is not None))

    def _sample_count(rows: list[dict]) -> int:
        return len(
            {row.get("sample_idx", "") for row in rows if row.get("sample_idx") != ""}
        )

    def _contact_frame_count(rows: list[dict]) -> int:
        return sum(1 for row in rows if _row_has_contact_cr(row))

    def _bool_percent(rows: list[dict], key: str) -> Optional[float]:
        vals = []
        for row in rows:
            value = row.get(key)
            if value in {"0", "1"}:
                vals.append(float(value))
        return float(np.mean(vals) * 100.0) if vals else None

    def _bool_percent_for_contact_cr(
        rows: list[dict],
        key: str,
    ) -> Optional[float]:
        return _bool_percent(
            [row for row in rows if _row_has_contact_cr(row)],
            key,
        )

    def _pen_1cm_percent_for_contact_cr(rows: list[dict]) -> Optional[float]:
        vals = []
        for row in rows:
            if not _row_has_contact_cr(row):
                continue
            id_max_mm = _float_value(row, "ID_max_mm")
            if id_max_mm is None:
                continue
            vals.append(float(id_max_mm <= 10.0))
        return float(np.mean(vals) * 100.0) if vals else None

    def _sample_bool_percent(
        rows: list[dict],
        key: str,
        row_filter=None,
    ) -> Optional[float]:
        by_sample: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if row_filter is not None and not row_filter(row):
                continue
            value = row.get(key)
            if value not in {"0", "1"}:
                continue
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(float(value))
        vals = [float(np.mean(group)) for group in by_sample.values() if group]
        return float(np.mean(vals) * 100.0) if vals else None

    def _sample_pen_1cm_percent_for_contact_cr(rows: list[dict]) -> Optional[float]:
        by_sample: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if not _row_has_contact_cr(row):
                continue
            id_max_mm = _float_value(row, "ID_max_mm")
            if id_max_mm is None:
                continue
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(float(id_max_mm <= 10.0))
        vals = [float(np.mean(group)) for group in by_sample.values() if group]
        return float(np.mean(vals) * 100.0) if vals else None

    def _sample_pen_1cm_percent_for_contact_cr_min5(
        rows: list[dict],
    ) -> Optional[float]:
        by_sample: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if not _row_has_contact_cr(row):
                continue
            id_max_mm = _float_value(row, "ID_max_mm")
            if id_max_mm is None:
                continue
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(float(id_max_mm <= 10.0))
        vals = [
            float(np.mean(group)) for group in by_sample.values() if len(group) >= 5
        ]
        return float(np.mean(vals) * 100.0) if vals else None

    def _sample_pen_1cm_percent_for_positive_id_frames(
        rows: list[dict],
    ) -> Optional[float]:
        by_sample: dict[tuple[str, str], list[float]] = {}
        for row in rows:
            if not _row_has_positive_id(row):
                continue
            id_max_mm = _float_value(row, "ID_max_mm")
            if id_max_mm is None:
                continue
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(float(id_max_mm <= 10.0))
        vals = [float(np.mean(group)) for group in by_sample.values() if group]
        return float(np.mean(vals) * 100.0) if vals else None

    def _last_frame_rows(rows: list[dict]) -> list[dict]:
        by_sample: dict[tuple[str, str], dict] = {}
        for row in rows:
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            try:
                frame_idx = int(row.get("frame_idx", -1))
            except (TypeError, ValueError):
                frame_idx = -1
            previous = by_sample.get(sample_key)
            if previous is None:
                by_sample[sample_key] = row
                continue
            try:
                previous_frame_idx = int(previous.get("frame_idx", -1))
            except (TypeError, ValueError):
                previous_frame_idx = -1
            if frame_idx > previous_frame_idx:
                by_sample[sample_key] = row
        return list(by_sample.values())

    def _last_frame_bool_percent(rows: list[dict], key: str) -> Optional[float]:
        return _bool_percent(_last_frame_rows(rows), key)

    def _last_frame_mean(rows: list[dict], key: str) -> Optional[float]:
        values = [_float_value(row, key) for row in _last_frame_rows(rows)]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None

    def _gsr_percent(rows: list[dict]) -> Optional[float]:
        """Task success at the final frame with sustained contact and lift."""
        required_contact_frames = max(1, int(gsr_contact_frames))
        lift_threshold_m = max(0.0, float(gsr_lift_threshold_m))
        by_sample: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(row)

        successes = []
        for sample_rows in by_sample.values():
            try:
                ordered = sorted(
                    sample_rows, key=lambda item: int(item.get("frame_idx", -1))
                )
            except (TypeError, ValueError):
                continue
            if len(ordered) < required_contact_frames:
                successes.append(False)
                continue
            final_row = ordered[-1]
            final_window = ordered[-required_contact_frames:]
            # Paper protocol: contact must persist throughout the final window,
            # while penetration and lift are evaluated at the final frame.
            sustained_contact = all(row.get("Con_pass") == "1" for row in final_window)
            final_id_max_mm = _float_value(final_row, "ID_max_mm")
            final_penetration_ok = bool(
                final_id_max_mm is not None and final_id_max_mm <= 10.0
            )
            final_lift_m = _float_value(final_row, "Object_lift_y_m")
            final_lift_ok = bool(
                final_lift_m is not None and final_lift_m >= lift_threshold_m
            )
            successes.append(
                bool(sustained_contact and final_penetration_ok and final_lift_ok)
            )
        return float(np.mean(successes) * 100.0) if successes else None

    def _final_window_contact_percent(rows: list[dict]) -> Optional[float]:
        required_contact_frames = max(1, int(gsr_contact_frames))
        by_sample: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            by_sample.setdefault(sample_key, []).append(row)
        passes = []
        for sample_rows in by_sample.values():
            try:
                ordered = sorted(
                    sample_rows, key=lambda item: int(item.get("frame_idx", -1))
                )
            except (TypeError, ValueError):
                continue
            if len(ordered) < required_contact_frames:
                passes.append(False)
                continue
            passes.append(
                all(
                    row.get("Con_pass") == "1"
                    for row in ordered[-required_contact_frames:]
                )
            )
        return float(np.mean(passes) * 100.0) if passes else None

    def _wrist_trajectories_from_rows(rows: list[dict]) -> list[dict]:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            sample_key = (
                str(row.get("file_name", "")),
                str(row.get("sample_idx", row.get("batch_sample_idx", ""))),
            )
            grouped.setdefault(sample_key, []).append(row)

        traj_rows = []
        wrist_keys = [
            (
                "left_wrist_canonical_x",
                "left_wrist_canonical_y",
                "left_wrist_canonical_z",
            ),
            (
                "right_wrist_canonical_x",
                "right_wrist_canonical_y",
                "right_wrist_canonical_z",
            ),
        ]
        for sample_rows in grouped.values():
            try:
                ordered = sorted(
                    sample_rows,
                    key=lambda item: int(item.get("frame_idx", -1)),
                )
            except Exception:
                ordered = sample_rows
            traj = np.full((len(ordered), 2, 3), np.nan, dtype=np.float32)
            for frame_out_idx, row in enumerate(ordered):
                for hand_idx, keys in enumerate(wrist_keys):
                    values = [_float_value(row, key) for key in keys]
                    if all(value is not None for value in values):
                        traj[frame_out_idx, hand_idx] = np.asarray(
                            values,
                            dtype=np.float32,
                        )
            if np.any(np.isfinite(traj)):
                first = ordered[0]
                traj_rows.append(
                    {
                        "text": first.get("text"),
                        "object": first.get("object"),
                        "wrist_traj": traj,
                    }
                )
        return traj_rows

    def _sample_diversity_m(
        rows: list[dict],
        trajectory_items: Optional[list[dict]] = None,
    ) -> Optional[float]:
        by_text: dict[str, list[np.ndarray]] = {}
        items = (
            trajectory_items
            if trajectory_items is not None
            else _wrist_trajectories_from_rows(rows)
        )
        for item in items:
            text = item.get("text")
            traj = item.get("wrist_traj")
            if text is None or traj is None:
                continue
            by_text.setdefault(str(text), []).append(traj)
        values = [
            _mean_pairwise_wrist_distance_m(trajs)
            for trajs in by_text.values()
            if len(trajs) >= 2
        ]
        return float(np.mean(values)) if values else None

    def _overall_diversity_m(
        rows: list[dict],
        trajectory_items: Optional[list[dict]] = None,
    ) -> Optional[float]:
        items = (
            trajectory_items
            if trajectory_items is not None
            else _wrist_trajectories_from_rows(rows)
        )
        trajs = [
            item["wrist_traj"]
            for item in items
            if item.get("wrist_traj") is not None
        ]
        if len(trajs) < 2:
            return None
        return float(_mean_pairwise_wrist_distance_m(trajs))

    def _fmt(value, digits: int = 6) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.{digits}f}"

    def _fmt_mean_sd(values: list[Optional[float]], digits: int = 6) -> str:
        vals = [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ]
        if not vals:
            return "NA"
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return f"{mean:.{digits}f} +- {sd:.{digits}f}"

    def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
        out = ["| " + " | ".join(columns) + " |"]
        out.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            out.append("| " + " | ".join(str(item) for item in row) + " |")
        return "\n".join(out)

    excluded_model_files = {"s_bps_bim_cano_dist_obj.pkl"}

    def _exclude_from_best_values(name: str) -> bool:
        return os.path.splitext(os.path.basename(str(name)))[0].endswith("_gt")

    def _model_sort_key(row: list[str]) -> tuple[int, str]:
        name = str(row[0])
        return (0, name)

    def _numeric_cell(value: str) -> Optional[float]:
        value = str(value).replace("**", "")
        if "+-" in value:
            value = value.split("+-", 1)[0].strip()
        if "±" in value:
            value = value.split("±", 1)[0].strip()
        if value in ("", "NA"):
            return None
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return value_f if np.isfinite(value_f) else None

    def _bold_best_model_values(
        columns: list[str],
        rows: list[list[str]],
        excluded_names: set[str],
    ) -> list[list[str]]:
        out = [list(row) for row in rows]
        for col_idx, column in enumerate(columns):
            if "↑" not in column and "↓" not in column:
                continue
            candidates = []
            for row_idx, row in enumerate(out):
                if not row or str(row[0]) in excluded_names:
                    continue
                value = _numeric_cell(row[col_idx])
                if value is not None:
                    candidates.append((row_idx, value))
            if not candidates:
                continue
            best_value = (
                max(value for _row_idx, value in candidates)
                if "↑" in column
                else min(value for _row_idx, value in candidates)
            )
            for row_idx, value in candidates:
                if np.isclose(value, best_value, rtol=1e-12, atol=1e-12):
                    out[row_idx][col_idx] = f"**{out[row_idx][col_idx]}**"
        return out

    def _seed_groups(rows: list[dict]) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for row in rows:
            seed = row.get("seed")
            if seed in (None, ""):
                return {}
            groups.setdefault(str(seed), []).append(row)
        return groups

    seed_metric_records: list[dict] = []
    seed_labels: set[str] = set()

    def _record_seed_metrics(
        section: str,
        model: str,
        seed_values: dict[str, dict[str, Optional[float]]],
    ) -> None:
        if not seed_csv_path or not seed_values:
            return
        seed_labels.update(seed_values)
        metric_names = list(next(iter(seed_values.values())).keys())
        for metric in metric_names:
            values_by_seed = {
                seed: values.get(metric) for seed, values in seed_values.items()
            }
            finite_values = [
                float(value)
                for value in values_by_seed.values()
                if value is not None and np.isfinite(float(value))
            ]
            seed_metric_records.append(
                {
                    "section": section,
                    "model": model,
                    "metric": metric,
                    "values_by_seed": values_by_seed,
                    "mean": float(np.mean(finite_values)) if finite_values else None,
                    "sd": (
                        float(np.std(finite_values, ddof=1))
                        if len(finite_values) > 1
                        else (0.0 if finite_values else None)
                    ),
                }
            )

    def _summary_values(rows: list[dict]) -> dict[str, Optional[float]]:
        # Build wrist trajectories once for this exact aggregation group.  In
        # particular, when seed metadata is present this function is called
        # once per seed, so SD/OD never compare trajectories across seeds.
        trajectory_items = _wrist_trajectories_from_rows(rows)
        return {
            "contact_frames(CR>0) ↑": float(_contact_frame_count(rows)),
            "ID_frame_max_mean_mm ↓": _mean_sample_positive_id_frame_mean(
                rows, "ID_mean_mm"
            ),
            "ID_frame_max_max_mm ↓": _mean_sample_positive_id_frame_max(
                rows, "ID_max_mm"
            ),
            "IV_volume_cm3_mean ↓": _mean_sample_positive_id_frame_mean(
                rows, "IV_volume_cm3"
            ),
            "CR_percent ↑": _scaled(
                _mean_sample_filtered_mean_positive_samples(rows, "CR"), 100.0
            ),
            "Con_pass_percent ↑": _final_window_contact_percent(rows),
            "Pen_1cm_percent ↑": _sample_pen_1cm_percent_for_contact_cr(rows),
            "Success_Rate_percent_last ↑": _last_frame_bool_percent(rows, "Success"),
            "GSR_percent ↑": _gsr_percent(rows),
            "Part_Acc_percent ↑": _scaled(_last_frame_mean(rows, "Part_Acc"), 100.0),
            "PCP_percent ↑": _scaled(_last_frame_mean(rows, "PCP"), 100.0),
            "G2C_cm ↓": _scaled(_last_frame_mean(rows, "G2C_m"), 100.0),
            "Accel_m_mean ↓": _scaled(
                _mean_sample_filtered_mean_positive_samples(rows, "Accel_m"),
                30.0 ** 2,
            ),
            "Off_ground_percent ↑": _sample_bool_percent(rows, "Off_ground"),
            "Phy_percent ↑": _sample_bool_percent(
                rows, "Off_ground", _row_has_contact_cr
            ),
            "SD_m ↑": _sample_diversity_m(
                rows, trajectory_items=trajectory_items
            ),
            "OD_m ↑": _overall_diversity_m(
                rows, trajectory_items=trajectory_items
            ),
        }

    def _summary_row(name: str, rows: list[dict]) -> list[str]:
        seed_groups = _seed_groups(rows)
        seed_values = {
            seed: _summary_values(seed_rows)
            for seed, seed_rows in sorted(seed_groups.items())
        }
        _record_seed_metrics("Main Summary", name, seed_values)
        # Files without seed metadata still need one aggregate computation,
        # not one full recomputation per output column.
        aggregate_values = None if seed_values else _summary_values(rows)

        def _metric_cell(key: str) -> str:
            if seed_values:
                return _fmt_mean_sd(
                    [values.get(key) for values in seed_values.values()]
                )
            if key == "contact_frames(CR>0) ↑":
                return str(_contact_frame_count(rows))
            return _fmt(aggregate_values.get(key))

        def _metric_alias(*keys: str) -> str:
            """Return the first available metric, preserving seed-wise aggregation."""
            for key in keys:
                value = _metric_cell(key)
                if value not in ("", "NA", "--"):
                    return value
            return "--"

        return [
            name,
            str(_sample_count(rows)),
            str(len(rows)),
            str(len(seed_groups)) if seed_groups else "NA",
            _metric_cell("contact_frames(CR>0) ↑"),
            _metric_cell("ID_frame_max_mean_mm ↓"),
            _metric_cell("ID_frame_max_max_mm ↓"),
            _metric_cell("IV_volume_cm3_mean ↓"),
            _metric_cell("CR_percent ↑"),
            _metric_cell("Con_pass_percent ↑"),
            _metric_cell("Pen_1cm_percent ↑"),
            _metric_cell("Success_Rate_percent_last ↑"),
            _metric_cell("Accel_m_mean ↓"),
            _metric_cell("Off_ground_percent ↑"),
            _metric_cell("Phy_percent ↑"),
            _metric_cell("SD_m ↑"),
            _metric_cell("OD_m ↑"),
            _metric_alias("Part_Acc_percent ↑", "Part Acc (%)"),
            _metric_alias("PCP_percent ↑", "PCP (%)"),
            _metric_alias("G2C_cm ↓", "G2C (cm)"),
            _metric_cell("GSR_percent ↑"),
        ]

    def _target_grounding_values(
        metric_rows: list[dict],
    ) -> dict[str, Optional[float]]:
        """Final-frame grounding metrics for one requested target part."""
        return {
            "Part Acc. (%) ↑": _scaled(
                _last_frame_mean(metric_rows, "Part_Acc"), 100.0
            ),
            "PCP (%) ↑": _scaled(_last_frame_mean(metric_rows, "PCP"), 100.0),
            "G2C (cm) ↓": _scaled(
                _last_frame_mean(metric_rows, "G2C_m"), 100.0
            ),
        }

    def _target_grounding_part_row(
        model: str,
        target_part: str,
        metric_rows: list[dict],
    ) -> list[str]:
        """Aggregate one model/target-part group, preserving seed-level SD."""
        seed_groups = _seed_groups(metric_rows)
        if seed_groups:
            values_by_seed = {
                seed: _target_grounding_values(seed_rows)
                for seed, seed_rows in sorted(seed_groups.items())
            }
            _record_seed_metrics(
                f"Target Grounding by Part ({target_part})",
                model,
                values_by_seed,
            )
            cells = {
                key: _fmt_mean_sd([values.get(key) for values in values_by_seed.values()])
                for key in ("Part Acc. (%) ↑", "PCP (%) ↑", "G2C (cm) ↓")
            }
        else:
            values = _target_grounding_values(metric_rows)
            cells = {
                key: _fmt(values.get(key))
                for key in ("Part Acc. (%) ↑", "PCP (%) ↑", "G2C (cm) ↓")
            }
        return [
            model,
            target_part,
            str(_sample_count(metric_rows)),
            str(len(seed_groups)) if seed_groups else "NA",
            cells["Part Acc. (%) ↑"],
            cells["PCP (%) ↑"],
            cells["G2C (cm) ↓"],
        ]

    rows = [
        row
        for row in _read_rows(csv_path)
        if os.path.basename(str(row.get("file_name", ""))) not in excluded_model_files
    ]
    recovered_seed_models = _recover_missing_seed_values(rows)

    def _ordered_target_parts(
        groups: dict[tuple[str, str], list[dict]],
    ) -> list[tuple[str, int]]:
        """Order requested parts by their per-model sample count, descending.

        A target part normally has the same number of samples for every model.
        Use the largest observed count so an incomplete model does not change the
        dataset-driven ordering; ties remain alphabetical for reproducibility.
        """
        counts: dict[str, int] = {}
        for (_model, target_part), part_rows in groups.items():
            counts[target_part] = max(
                counts.get(target_part, 0), _sample_count(part_rows)
            )
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    if quick:
        def _quick_values(metric_rows: list[dict]) -> dict[str, Optional[float]]:
            return {
                "Part Acc. (%) ↑": _scaled(
                    _last_frame_mean(metric_rows, "Part_Acc"), 100.0
                ),
                "PCP (%) ↑": _scaled(_last_frame_mean(metric_rows, "PCP"), 100.0),
                "G2C (cm) ↓": _scaled(
                    _last_frame_mean(metric_rows, "G2C_m"), 100.0
                ),
                "GSR (%) ↑": _gsr_percent(metric_rows),
            }

        quick_columns = [
            "model",
            "samples",
            "seeds",
            "Part Acc. (%) ↑",
            "PCP (%) ↑",
            "G2C (cm) ↓",
            "GSR (%) ↑",
        ]
        quick_rows = []
        quick_by_file: dict[str, list[dict]] = {}
        for row in rows:
            quick_by_file.setdefault(str(row.get("file_name", "")), []).append(row)
        for file_name, file_rows in sorted(quick_by_file.items()):
            seed_groups = _seed_groups(file_rows)
            if seed_groups:
                values_by_seed = [
                    _quick_values(seed_rows)
                    for _seed, seed_rows in sorted(seed_groups.items())
                ]
                cells = {
                    key: _fmt_mean_sd([values.get(key) for values in values_by_seed])
                    for key in quick_columns[3:]
                }
            else:
                values = _quick_values(file_rows)
                cells = {key: _fmt(values.get(key)) for key in quick_columns[3:]}
            quick_rows.append(
                [
                    file_name,
                    str(_sample_count(file_rows)),
                    str(len(seed_groups)) if seed_groups else "NA",
                    *[cells[key] for key in quick_columns[3:]],
                ]
            )

        quick_part_groups: dict[tuple[str, str], list[dict]] = {}
        for file_name, file_rows in quick_by_file.items():
            for row in file_rows:
                target_part = _target_part_from_text(row.get("text", ""))
                if not target_part:
                    continue
                quick_part_groups.setdefault((file_name, target_part), []).append(row)
        lines = [
            "# Quick Interaction Metric Summary",
            "",
            f"- CSV: `{csv_path}`",
            f"- Samples: {sum(_sample_count(value) for value in quick_by_file.values())}",
            "- Computed metrics only: Part Acc., PCP, G2C, GSR",
            (
                f"- GSR: final {max(1, int(gsr_contact_frames))} frames all satisfy "
                f"CR>0, ID<=1cm, and lift>={max(0.0, float(gsr_lift_threshold_m))*1000.0:g}mm"
            ),
            "- G2C gaze target: mean gaze before the first contact frame",
        ]
        if recovered_seed_models:
            recovery_text = "; ".join(
                f"{name} ({method})"
                for name, method in sorted(recovered_seed_models.items())
            )
            lines.append(f"- Recovered blank seed values from CSV: {recovery_text}")
        lines.extend(["", _markdown_table(quick_columns, quick_rows), ""])
        if quick_part_groups:
            lines.extend(
                [
                    "## Target Grounding Summary by Part",
                    "",
                    (
                        "Each table compares models for one requested target part. "
                        "Parts are ordered by per-model sample count (descending). "
                        "Part Acc. and PCP are measured at the final valid frame; "
                        "G2C is in cm."
                    ),
                    "",
                ]
            )
            quick_part_summary_columns = [
                "model",
                "samples",
                "seeds",
                "Part Acc. (%) ↑",
                "PCP (%) ↑",
                "G2C (cm) ↓",
            ]
            for target_part, sample_count in _ordered_target_parts(quick_part_groups):
                part_summary_rows = []
                for file_name in sorted(quick_by_file, key=_model_sort_key):
                    part_rows = quick_part_groups.get((file_name, target_part))
                    if not part_rows:
                        continue
                    seed_groups = _seed_groups(part_rows)
                    if seed_groups:
                        values_by_seed = [
                            _quick_values(seed_rows)
                            for _seed, seed_rows in sorted(seed_groups.items())
                        ]
                        cells = {
                            key: _fmt_mean_sd(
                                [values.get(key) for values in values_by_seed]
                            )
                            for key in quick_columns[3:6]
                        }
                    else:
                        values = _quick_values(part_rows)
                        cells = {
                            key: _fmt(values.get(key)) for key in quick_columns[3:6]
                        }
                    part_summary_rows.append(
                        [
                            file_name,
                            str(_sample_count(part_rows)),
                            str(len(seed_groups)) if seed_groups else "NA",
                            cells["Part Acc. (%) ↑"],
                            cells["PCP (%) ↑"],
                            cells["G2C (cm) ↓"],
                        ]
                    )
                lines.extend(
                    [
                        f"### {target_part} ({sample_count} samples/model)",
                        "",
                        _markdown_table(quick_part_summary_columns, part_summary_rows),
                        "",
                    ]
                )
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return

    columns = [
        "model",
        "samples",
        "frames",
        "seeds",
        "contact_frames(CR>0) ↑",
        "ID_frame_max_mean_mm ↓",
        "ID_frame_max_max_mm ↓",
        "IV_volume_cm3_mean ↓",
        "CR_percent ↑",
        "Con_pass_percent ↑",
        "Pen_1cm_percent ↑",
        "Success_Rate_percent_last ↑",
        "Accel_m_mean ↓",
        "Off_ground_percent ↑",
        "Phy_percent ↑",
        "SD_m ↑",
        "OD_m ↑",
        "Part_Acc_percent ↑",
        "PCP_percent ↑",
        "G2C_cm ↓",
        "GSR_percent ↑",
    ]
    metric_categories = [
        (
            "Requested Metrics (ordered)",
            "요청된 모든 지표를 seed별로 먼저 계산한 뒤 Mean +- SD로 집계합니다. "
            "현재 입력 CSV에 없는 지표는 --로 표시합니다.",
            [
                "ID_frame_max_mean_mm ↓", "IV_volume_cm3_mean ↓",
                "Pen_1cm_percent ↑", "Part_Acc_percent ↑", "PCP_percent ↑",
                "GSR_percent ↑", "G2C_cm ↓", "Con_pass_percent ↑",
                "SD_m ↑", "OD_m ↑", "Off_ground_percent ↑",
                "Phy_percent ↑", "Accel_m_mean ↓",
            ],
        ),
        (
            "Penetration & Collision",
            (
                "손과 물체의 비현실적인 관통 정도를 비교합니다. "
                "각 프레임의 ID는 내부 손 정점에서 물체 표면까지 거리 중 "
                "최댓값입니다. ID_frame_max_mean은 샘플 안에서 이 프레임 "
                "최댓값들을 평균하고, ID_frame_max_max는 그중 최댓값을 "
                "취한 뒤 유효 샘플들끼리 평균냅니다. "
                "ID/IV가 발생하지 않은 no-contact 또는 zero-only 샘플은 "
                "0으로 보상하지 않고 해당 지표 평균에서 제외합니다. "
                "깊이와 부피는 낮을수록 좋고 1 cm 이내 관통 비율은 높을수록 좋습니다."
            ),
            [
                "contact_frames(CR>0) ↑",
                "ID_frame_max_mean_mm ↓",
                "ID_frame_max_max_mm ↓",
                "IV_volume_cm3_mean ↓",
                "Pen_1cm_percent ↑",
            ],
        ),
        (
            "Contact & Interaction",
            (
                "손-물체 접촉의 형성과 접촉 영역 품질을 비교합니다. "
                "모든 연속 지표는 샘플별 평균을 먼저 낸 뒤 유효 샘플 평균으로 집계합니다. "
                "CR과 접촉 통과율은 높을수록 좋습니다."
            ),
            [
                "CR_percent ↑",
                "Con_pass_percent ↑",
            ],
        ),
        (
            "Physical Plausibility & Motion",
            (
                "Off-ground는 초기 프레임의 물체 바닥을 지면으로 두고 물체가 "
                f"{OFF_GROUND_THRESHOLD_M * 1000.0:.0f}mm 이상 지면 위에 있는 "
                "전체 프레임 비율입니다. Phy는 CR>0인 접촉 프레임 중 "
                "off-ground인 프레임 비율입니다. 가속도는 낮을수록, "
                "나머지 비율은 높을수록 좋습니다."
            ),
            [
                "Accel_m_mean ↓",
                "Off_ground_percent ↑",
                "Phy_percent ↑",
            ],
        ),
        (
            "Diversity",
            (
                "같은 조건 안의 샘플 다양성(SD)과 전체 동작 다양성(OD)을 "
                "비교하며, 두 지표 모두 높을수록 좋습니다."
            ),
            [
                "SD_m ↑",
                "OD_m ↑",
            ],
        ),
        (
            "Overall Task Success",
            (
                f"마지막 {max(1, int(gsr_contact_frames))}개 프레임에서 contact(CR>0), "
                "최대 ID 1cm 이하, 그리고 물체 lift "
                f"{max(0.0, float(gsr_lift_threshold_m)) * 1000.0:g}mm 이상을 "
                "모두 연속으로 만족한 샘플 비율입니다."
            ),
            [
                "Success_Rate_percent_last ↑",
                "GSR_percent ↑",
            ],
        ),
    ]
    lines = [
        "# New Interaction Metric Summary",
        "",
        f"- CSV: `{csv_path}`",
        f"- Rows: {len(rows)}",
        "",
        "## Metric Categories",
        "",
        "- **Penetration & Collision**: 관통 및 충돌 안정성",
        "- **Contact & Interaction**: 접촉 형성과 상호작용 품질",
        "- **Physical Plausibility & Motion**: 물리적 타당성과 동작 안정성",
        "- **Diversity**: 샘플 및 전체 동작 다양성",
        "- **Overall Task Success**: 마지막 연속 프레임 기준 종합 성공률",
        "- **Target Grounding by Part**: prompt target part별 Part Acc., PCP, G2C",
        "",
        "표의 `↑`는 높을수록 좋고, `↓`는 낮을수록 좋음을 의미합니다.",
        "CSV에 `seed` 컬럼이 있으면 지표는 seed별 결과의 `mean +- SD`로 표기합니다.",
    ]
    if recovered_seed_models:
        recovery_text = "; ".join(
            f"{name} ({method})"
            for name, method in sorted(recovered_seed_models.items())
        )
        lines.append(f"빈 `seed` 값은 CSV 반복 구조로 복원했습니다: {recovery_text}.")
    lines.append("")

    by_file: dict[str, list[dict]] = {}
    for row in rows:
        by_file.setdefault(row.get("file_name", ""), []).append(row)

    if by_file:
        model_rows = []
        for file_name, file_rows in by_file.items():
            if os.path.basename(str(file_name)) in excluded_model_files:
                continue
            model_rows.append(_summary_row(str(file_name), file_rows))
        model_rows = sorted(model_rows, key=_model_sort_key)
        column_indices = {column: idx for idx, column in enumerate(columns)}
        lines.extend(["## By File", ""])
        for title, description, category_metric_columns in metric_categories:
            category_columns = [
                "model",
                "samples",
                "frames",
                "seeds",
                *category_metric_columns,
            ]
            category_rows = [
                [row[column_indices[column]] for column in category_columns]
                for row in model_rows
            ]
            category_rows = _bold_best_model_values(
                category_columns,
                category_rows,
                excluded_names={
                    str(row[0])
                    for row in category_rows
                    if row and _exclude_from_best_values(str(row[0]))
                },
            )
            lines.extend(
                [
                    f"### {title}",
                    "",
                    description,
                    "",
                    _markdown_table(category_columns, category_rows),
                    "",
                ]
            )

        # Target parts are encoded in prompts as "Grab {part} of {object} ...".
        # Report a model comparison for each requested part instead of a single
        # model-major table that interleaves unrelated part geometries.
        by_model_and_part: dict[tuple[str, str], list[dict]] = {}
        for file_name, file_rows in by_file.items():
            for row in file_rows:
                target_part = _target_part_from_text(row.get("text", ""))
                if not target_part:
                    continue
                by_model_and_part.setdefault(
                    (str(file_name), target_part), []
                ).append(row)
        if by_model_and_part:
            part_columns = [
                "model",
                "samples",
                "seeds",
                "Part Acc. (%) ↑",
                "PCP (%) ↑",
                "G2C (cm) ↓",
            ]
            lines.extend(
                [
                    "### Target Grounding Summary by Part",
                    "",
                    (
                        "Each table compares models for one requested target part. "
                        "Parts are ordered by per-model sample count (descending). "
                        "Part Acc. and PCP are measured at the final valid frame; "
                        "G2C is in cm."
                    ),
                    "",
                ]
            )
            for target_part, sample_count in _ordered_target_parts(by_model_and_part):
                part_rows = []
                for model in sorted(by_file, key=_model_sort_key):
                    metric_rows = by_model_and_part.get((str(model), target_part))
                    if not metric_rows:
                        continue
                    # Drop the target-part cell: it is now the table heading.
                    row = _target_grounding_part_row(
                        str(model), target_part, metric_rows
                    )
                    part_rows.append([row[0], *row[2:]])
                lines.extend(
                    [
                        f"#### {target_part} ({sample_count} samples/model)",
                        "",
                        _markdown_table(part_columns, part_rows),
                        "",
                    ]
                )

    if seed_metric_records:
        def _markdown_seed_sort_key(seed: str):
            try:
                return (0, int(seed))
            except (TypeError, ValueError):
                return (1, str(seed))

        ordered_md_seeds = sorted(seed_labels, key=_markdown_seed_sort_key)
        seed_md_columns = [f"seed_{seed}" for seed in ordered_md_seeds]
        lines.extend(
            [
                "## Seed-wise Values, Mean, and SD",
                "",
                (
                    "각 지표는 seed별로 먼저 계산합니다. `Mean`은 seed 값의 "
                    "산술평균이고, `SD`는 10개 seed에 대한 표본표준편차"
                    "(`ddof=1`)입니다."
                ),
                "",
            ]
        )
        records_by_model_section: dict[tuple[str, str], list[dict]] = {}
        for record in seed_metric_records:
            key = (str(record["model"]), str(record["section"]))
            records_by_model_section.setdefault(key, []).append(record)
        for (model, section), records in records_by_model_section.items():
            lines.extend([f"### {model}", "", f"#### {section}", ""])
            table_columns = ["metric", *seed_md_columns, "Mean", "SD"]
            table_rows = []
            for record in records:
                values_by_seed = record["values_by_seed"]
                table_rows.append(
                    [
                        str(record["metric"]),
                        *[
                            _fmt(values_by_seed.get(seed))
                            for seed in ordered_md_seeds
                        ],
                        _fmt(record["mean"]),
                        _fmt(record["sd"]),
                    ]
                )
            lines.extend([_markdown_table(table_columns, table_rows), ""])

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))

    if seed_csv_path:
        def _seed_sort_key(seed: str):
            try:
                return (0, int(seed))
            except (TypeError, ValueError):
                return (1, str(seed))

        ordered_seeds = sorted(seed_labels, key=_seed_sort_key)
        seed_columns = [f"seed_{seed}" for seed in ordered_seeds]
        fieldnames = [
            "section",
            "model",
            "metric",
            "n_seeds",
            *seed_columns,
            "Mean",
            "SD",
        ]
        os.makedirs(
            os.path.dirname(os.path.abspath(seed_csv_path)) or ".", exist_ok=True
        )
        with open(seed_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in seed_metric_records:
                values_by_seed = record["values_by_seed"]
                out = {
                    "section": record["section"],
                    "model": record["model"],
                    "metric": record["metric"],
                    "n_seeds": sum(
                        value is not None
                        and np.isfinite(float(value))
                        for value in values_by_seed.values()
                    ),
                    "Mean": _format_csv_float(record["mean"]),
                    "SD": _format_csv_float(record["sd"]),
                }
                for seed, column in zip(ordered_seeds, seed_columns):
                    out[column] = _format_csv_float(values_by_seed.get(seed))
                writer.writerow(out)


def _format_csv_float(value, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _split_tag_from_file_name(file_name: str) -> str:
    name = os.path.basename(str(file_name)).lower()
    if name.startswith("us_") or "_us_" in name:
        return "unseen"
    if name.startswith("s_") or "_s_" in name:
        return "seen"
    return "other"


def _object_bbox_diag_scales(obj_pc_by_key: dict) -> dict[str, float]:
    out = {}
    for key, points in obj_pc_by_key.items():
        pts = _to_numpy(points).astype(np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            continue
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            continue
        extent = np.max(pts, axis=0) - np.min(pts, axis=0)
        scale = float(np.linalg.norm(extent))
        if np.isfinite(scale):
            out[str(key).lower()] = scale
    return out


def _parse_sample_idx_filter(sample_indices) -> Optional[set[int]]:
    if sample_indices is None:
        return None
    out: set[int] = set()
    for group in sample_indices:
        values = group if isinstance(group, (list, tuple)) else [group]
        for value in values:
            for part in str(value).split(","):
                part = part.strip()
                if part:
                    out.add(int(part))
    return out


def _object_keys_used_in_inputs(
    input_paths: list[str],
    obj_pc_by_key: dict,
    sample_idx_filter: Optional[set[int]] = None,
) -> set[str]:
    keys = set()
    global_sample_idx = 0
    available = {str(key).lower() for key in obj_pc_by_key.keys()}
    for path in input_paths:
        for record in _load_items_from_path(path):
            for sample in _iter_samples_from_record(record):
                sample_idx = int(global_sample_idx)
                global_sample_idx += 1
                if (
                    sample_idx_filter is not None
                    and sample_idx not in sample_idx_filter
                ):
                    continue
                text_key = _extract_object_key(str(sample.get("text", "")))
                if text_key in available:
                    keys.add(text_key)
                object_meta = sample.get("object_meta")
                if isinstance(object_meta, dict):
                    meta_names = []
                    if object_meta.get("object_name") is not None:
                        meta_names.append(object_meta.get("object_name"))
                    object_names = object_meta.get("object_names")
                    if isinstance(object_names, (list, tuple)):
                        meta_names.extend(object_names)
                    for meta_name in meta_names:
                        meta_key = str(meta_name).strip().lower()
                        if meta_key in available:
                            keys.add(meta_key)
    return keys


def _center_mesh_vertices_at_origin(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        return vertices
    finite = np.all(np.isfinite(vertices), axis=1)
    if not np.any(finite):
        return vertices
    bbox_min = vertices[finite].min(axis=0)
    bbox_max = vertices[finite].max(axis=0)
    center = (bbox_min + bbox_max) * 0.5
    return vertices - center.reshape(1, 3)


def _object_color_from_index(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            [230, 80, 70],
            [60, 160, 240],
            [70, 190, 120],
            [245, 180, 65],
            [180, 110, 230],
            [70, 210, 210],
            [235, 100, 170],
            [150, 150, 150],
        ],
        dtype=np.uint8,
    )
    return palette[int(index) % int(palette.shape[0])]


def visualize_objects_centered_at_origin(
    input_paths: list[str],
    obj_pc_by_key: dict,
    obj_mesh_by_key: dict,
    sample_idx_filter: Optional[set[int]] = None,
) -> None:
    if rr is None:
        raise RuntimeError("rerun is not installed; cannot visualize object scales")
    _init_rerun_viewer("HOT3D object scale comparison")

    scale_by_key = _object_bbox_diag_scales(obj_pc_by_key)
    used_keys = _object_keys_used_in_inputs(
        input_paths,
        obj_pc_by_key,
        sample_idx_filter=sample_idx_filter,
    )
    if not used_keys:
        used_keys = set(obj_mesh_by_key.keys())
    ordered_keys = sorted(
        (key for key in used_keys if key in obj_mesh_by_key),
        key=lambda key: (scale_by_key.get(str(key).lower(), float("inf")), str(key)),
    )
    if not ordered_keys:
        print("[WARN] no object meshes available for object scale visualization.")
        return

    max_scale = max(scale_by_key.get(str(key).lower(), 0.0) for key in ordered_keys)
    axis_len = max(float(max_scale) * 0.7, 0.05)
    _log_xyz_axes(
        "object_scale/origin_overlay",
        np.zeros(3, dtype=np.float32),
        axis_len,
        static=True,
    )

    for idx, key in enumerate(ordered_keys):
        mesh = obj_mesh_by_key.get(key)
        if mesh is None:
            continue
        vertices = _center_mesh_vertices_at_origin(mesh)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2:
            continue
        color = _object_color_from_index(idx)
        rr.log(
            f"object_scale/origin_overlay/{_safe_path_token(key)}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
                vertex_colors=np.tile(color.reshape(1, 3), (vertices.shape[0], 1)),
            ),
            static=True,
        )
        scale = scale_by_key.get(str(key).lower())
        label = (
            f"{key}\nbbox_diag {float(scale):.4f} m" if scale is not None else str(key)
        )
        label_pos = np.asarray(
            [[0.0, float(idx) * 0.012, axis_len + 0.012]], dtype=np.float32
        )
        rr.log(
            f"object_scale/origin_overlay/{_safe_path_token(key)}/label",
            rr.Points3D(
                positions=label_pos,
                radii=[max(axis_len * 0.015, 0.002)],
                colors=color.reshape(1, 3),
                labels=[label],
                show_labels=True,
            ),
            static=True,
        )
    print(
        "[OBJECT_SCALE] visualized "
        f"{len(ordered_keys)} objects centered at the origin in Rerun."
    )

    # Relative-rotation debug print sections removed by request.


def _load_hand_sample_indices_100(path: str) -> np.ndarray:
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            indices = pickle.load(f)
    else:
        indices = np.load(path, allow_pickle=True)
        if isinstance(indices, np.ndarray) and indices.shape == ():
            indices = indices.item()
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    indices = indices[(indices >= 0) & (indices < 778)]
    if indices.shape[0] != 100:
        raise ValueError(
            f"expected 100 hand sample indices in {path}, got {indices.shape[0]}"
        )
    return indices


def _resolve_hand_sample_index_path(path: Optional[str]) -> str:
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            os.path.join(os.getcwd(), "part_fps_hand_index_100.pkl"),
            os.path.join(os.getcwd(), "part_fps_hand_index_100.npy"),
            os.path.join(HOT3D_ROOT, "part_fps_hand_index_100.pkl"),
            os.path.join(HOT3D_ROOT, "part_fps_hand_index_100.npy"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "could not find part_fps_hand_index_100.pkl/.npy; "
        "pass --hand-sample-index-path"
    )


def _zero_mano_vertices(layer) -> np.ndarray:
    with torch.no_grad():
        output = layer(
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            betas=torch.zeros((1, 10), dtype=torch.float32),
        )
    vertices = _to_numpy(output.vertices[0]).astype(np.float32)
    joints = _to_numpy(output.joints[0]).astype(np.float32)
    return vertices - joints[0].reshape(1, 3)


def visualize_zero_mano_hand_pointclouds(
    sample_index_path: Optional[str],
    *,
    rerun_port: int = 9876,
    rerun_web_port: int = 9090,
    viewer: str = "native",
) -> None:
    if rr is None:
        raise RuntimeError("rerun is not installed; cannot visualize MANO hands")
    index_path = _resolve_hand_sample_index_path(sample_index_path)
    sampled_indices = _load_hand_sample_indices_100(index_path)

    _init_rerun_viewer(
        "HOT3D zero MANO hand point clouds",
        rerun_port=rerun_port,
        rerun_web_port=rerun_web_port,
        viewer=viewer,
    )

    l_layer = build_mano_aa(is_rhand=False, flat_hand=True)
    r_layer = build_mano_aa(is_rhand=True, flat_hand=True)
    hands = [
        (
            "left_hand",
            _zero_mano_vertices(l_layer),
            np.asarray([70, 150, 245], dtype=np.uint8),
        ),
        (
            "right_hand",
            _zero_mano_vertices(r_layer),
            np.asarray([245, 120, 70], dtype=np.uint8),
        ),
    ]
    panel_offsets = {
        "all_778_vertices": np.asarray([-0.18, 0.0, 0.0], dtype=np.float32),
        "sampled_100_vertices": np.asarray([0.18, 0.0, 0.0], dtype=np.float32),
    }
    hand_offsets = {
        "left_hand": np.asarray([0.0, 0.06, 0.0], dtype=np.float32),
        "right_hand": np.asarray([0.0, -0.06, 0.0], dtype=np.float32),
    }

    rr.log(
        "zero_mano/notes",
        rr.TextDocument(
            "MANO global_orient, hand_pose, and betas are all zeros. "
            f"Sampled indices: {os.path.basename(index_path)}"
        ),
        static=True,
    )
    for panel_name, panel_offset in panel_offsets.items():
        for hand_name, vertices, color in hands:
            points = vertices + panel_offset.reshape(1, 3) + hand_offsets[hand_name]
            if panel_name == "sampled_100_vertices":
                points = points[sampled_indices]
                radii = np.full((100,), 0.004, dtype=np.float32)
            else:
                radii = np.full((points.shape[0],), 0.0025, dtype=np.float32)
            rr.log(
                f"zero_mano/{panel_name}/{hand_name}",
                rr.Points3D(
                    positions=points,
                    radii=radii,
                    colors=np.tile(color.reshape(1, 3), (points.shape[0], 1)),
                ),
                static=True,
            )
    print(
        "[MANO_ZERO] visualized zero-pose MANO hands: "
        f"778 vertices per hand and 100 sampled vertices from {index_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        dest="input_files",
        action="append",
        help="Restrict evaluation to this input file. Repeatable.",
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(PACKAGE_ROOT, "inputs"),
        help="Base directory for relative --input paths (default: package inputs/).",
    )
    parser.add_argument(
        "--obj-pkl",
        default=os.path.join(PACKAGE_ROOT, "data", "obj.pkl"),
        help="Object metadata pickle (default: package data/obj.pkl).",
    )
    parser.add_argument(
        "--part-labels",
        default=os.path.abspath(os.path.join(PACKAGE_ROOT, "..", "assets", "labels_merged.json")),
        help="Object-part annotation JSON used for PCP and Part Acc.",
    )
    parser.add_argument(
        "--target-part",
        dest="target_parts",
        action="append",
        default=None,
        help=(
            "Restrict metric evaluation to samples whose prompt specifies this "
            "target part. Repeat the option or provide comma-separated parts."
        ),
    )
    parser.add_argument(
        "--gaze-sigma-m",
        type=float,
        default=0.05,
        help="Gaussian forward-ray scale used for G2C (default: 0.05 m).",
    )
    parser.add_argument(
        "--sample-idx",
        dest="sample_indices",
        action="append",
        nargs="+",
        help=(
            "Restrict evaluation to sample indices. Supports repeated flags, "
            "space-separated values, or comma-separated values."
        ),
    )
    parser.add_argument(
        "--new-metric-visualize",
        action="store_true",
        help="Visualize New Metric geometry for a few samples in Rerun.",
    )
    parser.add_argument(
        "--new-metric-top10-visualize",
        action="store_true",
        help=(
            "Visualize per-metric top/bottom samples in Rerun using a generated "
            "new_metrics_per_frame_*.csv ranking."
        ),
    )
    parser.add_argument(
        "--top10-csv",
        default=None,
        help=(
            "Per-frame CSV used by --new-metric-top10-visualize. Defaults to "
            "--new-metric-csv-output or the newest new_metrics_per_frame_*.csv."
        ),
    )
    parser.add_argument(
        "--top10-k",
        type=int,
        default=10,
        help="Number of high and low ranked samples per metric.",
    )
    parser.add_argument(
        "--top10-direction",
        choices=("high", "low", "both"),
        default="both",
        help="Choose whether top10 visualization shows high, low, or both directions.",
    )
    parser.add_argument(
        "--top10-per-model",
        action="store_true",
        help="Rank top/bottom samples independently for each file_name in the CSV.",
    )
    parser.add_argument(
        "--top10-metric",
        dest="top10_metrics",
        action="append",
        default=None,
        help=(
            "Restrict top/bottom visualization to this metric name or source "
            "CSV column. Repeatable."
        ),
    )
    parser.add_argument(
        "--top10-model",
        dest="top10_models",
        action="append",
        default=None,
        help="Restrict top/bottom ranking to this CSV file_name. Repeatable.",
    )
    parser.add_argument(
        "--new-metric-pairdiff-visualize",
        action="store_true",
        help=(
            "Visualize paired samples where a base model has much larger metric "
            "values than a comparison model."
        ),
    )
    parser.add_argument(
        "--pairdiff-csv",
        default=None,
        help=(
            "Per-frame CSV used by --new-metric-pairdiff-visualize. Defaults to "
            "--top10-csv, --new-metric-csv-output, or the newest CSV."
        ),
    )
    parser.add_argument(
        "--pairdiff-metric",
        default="ID_mean_mm_contact_mean",
        help="Metric name or CSV column used for paired-difference ranking.",
    )
    parser.add_argument(
        "--pairdiff-k",
        type=int,
        default=10,
        help="Number of largest paired differences to visualize.",
    )
    parser.add_argument(
        "--pairdiff-base-model",
        default="s_bps.pkl",
        help="CSV file_name treated as the base model in base-minus-compare.",
    )
    parser.add_argument(
        "--pairdiff-compare-model",
        default="s_bps_9000.pkl",
        help="CSV file_name treated as the comparison model.",
    )
    parser.add_argument(
        "--new-metric-count",
        type=int,
        default=None,
        help="Optional maximum number of samples to visualize for --new-metric-visualize.",
    )
    parser.add_argument(
        "--new-metric-output-dir",
        default=None,
        help="Optional directory for GLB exports from --new-metric-visualize.",
    )
    parser.add_argument(
        "--fast-penetration-metric",
        action="store_true",
        help=(
            "Use the faster KD-tree/vertex-normal penetration approximation "
            "instead of mesh contains/signed-distance checks."
        ),
    )
    parser.add_argument(
        "--include-no-contact-penetration",
        action="store_true",
        help=(
            "Also compute ID/IV on frames with CR == 0. By default these "
            "frames are skipped because summary penetration metrics use only CR>0 frames."
        ),
    )
    parser.add_argument(
        "--show-id-values",
        action="store_true",
        help=(
            "Show explicit ID distance labels in Rerun when using "
            "--new-metric-visualize. Enabled automatically for "
            "--new-metric-top10-visualize."
        ),
    )
    parser.add_argument(
        "--id-label-count",
        type=int,
        default=20,
        help="Maximum number of visible ID value labels per hand.",
    )
    parser.add_argument(
        "--rerun-wait",
        type=float,
        default=-1.0,
        help=(
            "Seconds to keep the Python process alive after Rerun logging. "
            "Use 0 to exit immediately, or a negative value to keep it alive "
            "until Ctrl-C."
        ),
    )
    parser.add_argument(
        "--rerun-port",
        type=int,
        default=9876,
        help=(
            "Port used by the spawned Rerun viewer gRPC proxy. "
            "Defaults to 9876 so repeated runs reuse the same viewer. "
            "Use 0 to reuse the default fixed port."
        ),
    )
    parser.add_argument(
        "--rerun-web-port",
        type=int,
        default=9090,
        help="Port used by the Rerun web viewer when --viewer web is selected.",
    )
    parser.add_argument(
        "--viewer",
        choices=("web", "native"),
        default="native",
        help=(
            "Viewer frontend to launch. 'web' opens the Rerun browser viewer; "
            "'native' uses the local native Rerun window."
        ),
    )
    parser.add_argument(
        "--new-metric-csv-output",
        default=None,
        help="Output CSV path for New Metric per-frame results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Compute only Part Acc., PCP, G2C, and GSR. Contact is scanned over "
            "the sequence, while penetration ID is evaluated only for the final "
            "GSR window; IV, acceleration, and diversity are skipped."
        ),
    )
    parser.add_argument(
        "--id-pen-only",
        action="store_true",
        help=(
            "Compute only ID and Pen_1cm. Contact (CR) is retained because "
            "Pen_1cm is defined only on contact frames; IV, PCP, Part Acc., "
            "G2C, acceleration, grounding, and wrist metrics are skipped. "
            "Uses the fast penetration approximation and evaluates every frame."
        ),
    )
    parser.add_argument(
        "--new-metric-resume",
        action="store_true",
        help=(
            "Resume New Metric CSV generation from --new-metric-csv-output. "
            "Completed samples are kept, partial samples are dropped and recomputed."
        ),
    )
    parser.add_argument(
        "--new-metric-md-output",
        default=None,
        help="Output Markdown summary path for new metric evaluation.",
    )
    parser.add_argument(
        "--summary-from-csv",
        default=None,
        help=(
            "Regenerate a Markdown summary directly from an existing per-frame "
            "CSV without rerunning geometry evaluation. Blank seed values are "
            "recovered only when a complete repeated-sample pattern is verified."
        ),
    )
    parser.add_argument(
        "--seed-summary-csv-output",
        default=None,
        help=(
            "Optional CSV containing every seed-wise metric, its mean, and "
            "sample SD when using --summary-from-csv."
        ),
    )
    parser.add_argument(
        "--gsr-contact-frames",
        type=int,
        default=5,
        help=(
            "Number of consecutive final frames that must have contact (CR>0) "
            "for GSR (default: 5)."
        ),
    )
    parser.add_argument(
        "--gsr-lift-cm",
        type=float,
        default=5.0,
        help="Minimum final object lift in cm required for GSR (default: 5.0).",
    )
    parser.add_argument(
        "--visualize-object-scales",
        action="store_true",
        help=(
            "Visualize all objects used by the input files centered at the origin "
            "for size comparison."
        ),
    )
    parser.add_argument(
        "--visualize-zero-mano-hands",
        action="store_true",
        help=(
            "Visualize zero-valued flat MANO left/right hands as 778-point and "
            "100-sampled point clouds in Rerun."
        ),
    )
    parser.add_argument(
        "--hand-sample-index-path",
        default=None,
        help=(
            "Path to part_fps_hand_index_100.pkl/.npy used by "
            "--visualize-zero-mano-hands."
        ),
    )
    parser.add_argument(
        "--diversity-visualize",
        action="store_true",
        help="Visualize SD/OD canonical wrist trajectory diversity from a per-frame CSV.",
    )
    parser.add_argument(
        "--diversity-csv",
        default=None,
        help=(
            "Per-frame CSV for --diversity-visualize. Defaults to --new-metric-csv-output "
            "or the newest new_metrics_per_frame_*.csv in the current directory."
        ),
    )
    parser.add_argument(
        "--diversity-model",
        dest="diversity_models",
        action="append",
        default=None,
        help="Restrict SD/OD visualization to this model file name. Repeatable.",
    )
    parser.add_argument(
        "--diversity-text",
        default=None,
        help="Restrict SD/OD visualization to rows whose text contains this string.",
    )
    parser.add_argument(
        "--diversity-max-groups",
        type=int,
        default=3,
        help="Maximum number of SD text groups to visualize.",
    )
    parser.add_argument(
        "--diversity-max-samples",
        type=int,
        default=8,
        help="Maximum trajectories shown per SD/OD group.",
    )
    args = parser.parse_args()

    if args.summary_from_csv:
        summary_csv_path = os.path.abspath(
            os.path.expanduser(str(args.summary_from_csv))
        )
        if not os.path.exists(summary_csv_path):
            raise FileNotFoundError(
                f"Per-frame metric CSV not found: {summary_csv_path}"
            )
        summary_md_path = (
            os.path.abspath(os.path.expanduser(str(args.new_metric_md_output)))
            if args.new_metric_md_output
            else os.path.splitext(summary_csv_path)[0] + "_summary.md"
        )
        seed_summary_path = (
            os.path.abspath(
                os.path.expanduser(str(args.seed_summary_csv_output))
            )
            if args.seed_summary_csv_output
            else None
        )
        write_new_metric_summary_markdown(
            summary_md_path,
            summary_csv_path,
            seed_csv_path=seed_summary_path,
            gsr_contact_frames=max(1, int(args.gsr_contact_frames)),
            gsr_lift_threshold_m=max(0.0, float(args.gsr_lift_cm)) / 100.0,
            quick=bool(args.quick),
        )
        print(f"Saved metric Markdown summary: {summary_md_path}")
        if seed_summary_path:
            print(f"Saved seed-wise metric CSV: {seed_summary_path}")
        return

    if args.visualize_zero_mano_hands:
        visualize_zero_mano_hand_pointclouds(
            args.hand_sample_index_path,
            rerun_port=int(args.rerun_port),
            rerun_web_port=int(args.rerun_web_port),
            viewer=str(args.viewer),
        )
        _finish_rerun_viewer(args.rerun_wait)
        return

    if args.diversity_visualize:
        diversity_csv_path = (
            args.diversity_csv
            or args.new_metric_csv_output
            or _find_latest_new_metric_csv()
        )
        if not diversity_csv_path:
            raise RuntimeError(
                "No per-frame CSV found for --diversity-visualize. "
                "Pass --diversity-csv or generate one with --new-metric-csv-output."
            )
        visualize_diversity_from_csv(
            diversity_csv_path,
            model_filters=args.diversity_models,
            text_filter=args.diversity_text,
            max_groups=args.diversity_max_groups,
            max_samples=args.diversity_max_samples,
        )
        _finish_rerun_viewer(args.rerun_wait)
        return

    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    input_files = (
        list(args.input_files)
        if args.input_files
        else [
            # Pass prediction pickles with --input; there is no default.
            # "s_bps_init_gaze.pkl",
            # "us_1024bps_dir.pkl",
            # "abl_gage_alignment.pkl",
            # "abl_gage_averaged.pkl",
            # "abl_gage_closeness.pkl",

        ]
    )
    sample_idx_filter = _parse_sample_idx_filter(args.sample_indices)
    target_parts = None
    if args.target_parts:
        target_parts = {
            part.strip().lower()
            for value in args.target_parts
            for part in str(value).split(",")
            if part.strip()
        }
        print(f"[INFO] target-part filter: {', '.join(sorted(target_parts))}")

    def _resolve_input_path(file_name: str, input_dir: str) -> str:
        expanded = os.path.expanduser(file_name)
        if os.path.isabs(expanded):
            return expanded
        candidate = os.path.join(os.path.expanduser(input_dir), expanded)
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(expanded)

    resolved_input_paths = []
    for file_name in input_files:
        path = _resolve_input_path(file_name, input_dir)
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}")
            continue
        resolved_input_paths.append(path)

    if not resolved_input_paths:
        print("[WARN] no valid input files found.")
        return

    obj_pkl_path = os.path.abspath(os.path.expanduser(args.obj_pkl))
    if not os.path.exists(obj_pkl_path):
        raise FileNotFoundError(f"Object metadata not found: {obj_pkl_path}")
    object_model = ObjectModel(obj_pkl_path)
    part_labels_path = os.path.abspath(os.path.expanduser(args.part_labels))
    if os.path.exists(part_labels_path):
        with open(part_labels_path) as handle:
            part_labels = json.load(handle)
        part_labels = {
            str(key).lower(): value for key, value in part_labels.items()
        }
    else:
        print(
            f"[WARN] part labels not found: {part_labels_path}; "
            "PCP and Part Acc. will be empty."
        )
        part_labels = None

    # Use lowercased keys for robust text matching.
    obj_pc_by_key = {}
    obj_mesh_by_key = {}
    proxy_mesh_cache = {}
    for k, pc in object_model.obj_pcs.items():
        key = str(k).lower()
        original_pc = np.asarray(pc, dtype=np.float32)
        pc_to_use = original_pc
        obj_path_value = object_model.obj_path.get(k)
        resolved_mesh = None
        if obj_path_value is None:
            print(f"[WARN] missing obj_path entry for '{key}' in obj.pkl")
        else:
            resolved_mesh = _load_object_mesh(
                obj_pkl_path, obj_path_value, pc_to_use, object_key=key
            )
        proxy_metric_mesh = None
        if resolved_mesh is None:
            proxy_metric_mesh = _get_or_build_proxy_mesh_from_object_pc(
                pc_to_use, object_key=key, proxy_cache=proxy_mesh_cache
            )
        if resolved_mesh is not None:
            obj_mesh_by_key[key] = resolved_mesh
        elif proxy_metric_mesh is not None:
            print(
                f"[WARN] using point-cloud proxy mesh for metrics on '{key}' because "
                "resolved mesh was unavailable."
            )
            obj_mesh_by_key[key] = proxy_metric_mesh
        obj_pc_by_key[key] = torch.as_tensor(pc_to_use, dtype=torch.float32)

    if args.visualize_object_scales:
        visualize_objects_centered_at_origin(
            resolved_input_paths,
            obj_pc_by_key,
            obj_mesh_by_key,
            sample_idx_filter=sample_idx_filter,
        )
        _finish_rerun_viewer(args.rerun_wait)
        return

    print("[INFO] initializing MANO hand layers...")
    l_hand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    r_hand_layer = build_mano_aa(is_rhand=True, flat_hand=False)
    print("[INFO] MANO hand layers ready.")

    top10_contexts_by_sample = None
    if args.new_metric_top10_visualize:
        top10_csv_path = (
            args.top10_csv
            or args.new_metric_csv_output
            or _find_latest_new_metric_csv()
        )
        if not top10_csv_path:
            raise RuntimeError(
                "No per-frame CSV found for --new-metric-top10-visualize. "
                "Pass --top10-csv or generate one with --new-metric-csv-output."
            )
        top10_model_filters = args.top10_models
        if top10_model_filters is None and len(resolved_input_paths) > 1:
            top10_model_filters = [
                os.path.basename(str(path)) for path in resolved_input_paths
            ]
        top10_per_model = (
            bool(args.top10_per_model) or len(set(top10_model_filters or [])) > 1
        )
        if top10_per_model and not args.top10_per_model:
            print(
                "[TOP10] multiple input files detected; "
                "ranking top-k independently per file_name."
            )
        top10_contexts_by_sample = build_new_metric_top10_contexts_from_csv(
            top10_csv_path,
            top_k=max(1, int(args.top10_k)),
            model_filters=top10_model_filters,
            metric_filters=args.top10_metrics,
            directions=(
                ("high", "low")
                if args.top10_direction == "both"
                else (str(args.top10_direction),)
            ),
            per_model=top10_per_model,
        )
        if not top10_contexts_by_sample:
            raise RuntimeError(f"No top/bottom samples found in CSV: {top10_csv_path}")
        selected_top10_indices = set(top10_contexts_by_sample.keys())
        if sample_idx_filter is not None:
            selected_top10_indices &= set(sample_idx_filter)
            top10_contexts_by_sample = {
                key: value
                for key, value in top10_contexts_by_sample.items()
                if key in selected_top10_indices
            }
        sample_idx_filter = selected_top10_indices
        print(
            "[TOP10] "
            f"csv={top10_csv_path} unique_samples={len(sample_idx_filter)} "
            f"contexts={sum(len(v) for v in top10_contexts_by_sample.values())}"
        )

    if args.new_metric_pairdiff_visualize:
        pairdiff_csv_path = (
            args.pairdiff_csv
            or args.top10_csv
            or args.new_metric_csv_output
            or _find_latest_new_metric_csv()
        )
        if not pairdiff_csv_path:
            raise RuntimeError(
                "No per-frame CSV found for --new-metric-pairdiff-visualize. "
                "Pass --pairdiff-csv or generate one with --new-metric-csv-output."
            )
        pairdiff_contexts = build_new_metric_pairdiff_contexts_from_csv(
            pairdiff_csv_path,
            metric_name=str(args.pairdiff_metric),
            top_k=max(1, int(args.pairdiff_k)),
            base_model=os.path.basename(str(args.pairdiff_base_model)),
            compare_model=os.path.basename(str(args.pairdiff_compare_model)),
        )
        if not pairdiff_contexts:
            raise RuntimeError(
                f"No paired-difference samples found in CSV: {pairdiff_csv_path}"
            )
        if top10_contexts_by_sample:
            for sample_idx, contexts in pairdiff_contexts.items():
                top10_contexts_by_sample.setdefault(sample_idx, []).extend(contexts)
        else:
            top10_contexts_by_sample = pairdiff_contexts
        selected_pairdiff_indices = set(pairdiff_contexts.keys())
        if sample_idx_filter is not None:
            selected_pairdiff_indices &= set(sample_idx_filter)
            top10_contexts_by_sample = {
                key: value
                for key, value in top10_contexts_by_sample.items()
                if key in selected_pairdiff_indices
            }
        sample_idx_filter = selected_pairdiff_indices
        print(
            "[PAIRDIFF] "
            f"csv={pairdiff_csv_path} unique_samples={len(sample_idx_filter)} "
            f"contexts={sum(len(v) for v in top10_contexts_by_sample.values())}"
        )

    if (
        not args.new_metric_visualize
        and not args.new_metric_top10_visualize
        and not args.new_metric_pairdiff_visualize
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = args.new_metric_csv_output
        if args.new_metric_resume and not csv_path:
            csv_path = _find_latest_new_metric_csv()
            if csv_path:
                print(f"[RESUME] using latest per-frame CSV: {csv_path}")
        if not csv_path:
            csv_path = (
                f"quick_metrics_per_frame_{timestamp}.csv"
                if args.quick
                else f"new_metrics_per_frame_{timestamp}.csv"
            )
        md_path = args.new_metric_md_output
        if not md_path:
            md_path = (
                f"quick_metrics_summary_{timestamp}.md"
                if args.quick
                else f"new_metrics_summary_{timestamp}.md"
            )
        rows_written = write_new_metric_per_frame_csv(
            csv_path,
            resolved_input_paths,
            obj_pc_by_key,
            obj_mesh_by_key,
            l_hand_layer,
            r_hand_layer,
            sample_idx_filter,
            max_samples=(
                None
                if args.new_metric_count is None
                else max(1, int(args.new_metric_count))
            ),
            fast_penetration_metric=bool(args.fast_penetration_metric),
            contact_only_penetration=not bool(args.include_no_contact_penetration),
            resume=bool(args.new_metric_resume),
            part_labels=part_labels,
            target_parts=target_parts,
            gaze_sigma_m=float(args.gaze_sigma_m),
            quick=bool(args.quick),
            gsr_contact_frames=max(1, int(args.gsr_contact_frames)),
            id_pen_only=bool(args.id_pen_only),
        )
        write_new_metric_summary_markdown(
            md_path,
            csv_path,
            object_scale_by_key=_object_bbox_diag_scales(obj_pc_by_key),
            gsr_contact_frames=max(1, int(args.gsr_contact_frames)),
            gsr_lift_threshold_m=max(0.0, float(args.gsr_lift_cm)) / 100.0,
            quick=bool(args.quick),
        )
        print(
            "Saved new metric per-frame CSV: "
            f"{os.path.abspath(csv_path)} ({rows_written} rows)"
        )
        print(f"Saved new metric Markdown summary: {os.path.abspath(md_path)}")
        return

    rows = run_new_metric_visualization(
        resolved_input_paths,
        obj_pc_by_key,
        obj_mesh_by_key,
        l_hand_layer,
        r_hand_layer,
        sample_idx_filter,
        max_samples=(
            None
            if args.new_metric_count is None
            else max(1, int(args.new_metric_count))
        ),
        output_dir=args.new_metric_output_dir,
        show_id_values=bool(args.show_id_values or args.new_metric_top10_visualize),
        id_label_count=max(1, int(args.id_label_count)),
        rerun_port=int(args.rerun_port),
        rerun_web_port=int(args.rerun_web_port),
        viewer=str(args.viewer),
        fast_penetration_metric=bool(args.fast_penetration_metric),
        contact_only_penetration=not bool(args.include_no_contact_penetration),
        top10_contexts_by_sample=top10_contexts_by_sample,
    )
    if not rows:
        print("[WARN] no samples were visualized for the new metric.")
    else:
        success_values = [
            bool(row.get("success")) for row in rows if row.get("success") is not None
        ]
        if success_values:
            success_rate = 100.0 * float(np.mean(success_values))
            print(
                f"[NEW_METRIC] visualized_samples={len(rows)} "
                f"Success_Rate={success_rate:.6f}%"
            )
        if args.new_metric_output_dir:
            print(
                "Saved first-frame GLB visualizations: "
                f"{os.path.abspath(args.new_metric_output_dir)}"
            )
    _finish_rerun_viewer(args.rerun_wait)


if __name__ == "__main__":
    main()
