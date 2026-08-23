import os
import pickle
import re
import sys

import numpy as np
import torch
import trimesh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HOT3D_ROOT = SCRIPT_DIR
if HOT3D_ROOT not in sys.path:
    sys.path.insert(0, HOT3D_ROOT)

from data_loaders.pytorch3d_rotation.rotation_conversions import matrix_to_axis_angle
from rot import rot6d_to_axis_angle, rot6d_to_rotmat


class ObjectModel:
    def __init__(self, pkl_file: str):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        self.object_name = data["object_name"]
        self.obj_pcs = data["obj_pcs"]
        self.obj_path = data["obj_path"]
        self.obj_pc_normals = data.get("obj_pc_normals")
        self.point_sets = data.get("point_sets")

    def __call__(self, object_name):
        if isinstance(object_name, int):
            object_name = self.object_name[object_name]
        if self.point_sets is None or self.obj_pc_normals is None:
            raise KeyError("obj.pkl is missing point_sets or obj_pc_normals")
        return (
            self.point_sets[object_name].copy(),
            self.obj_pcs[object_name].copy(),
            self.obj_pc_normals[object_name].copy(),
            self.obj_path[object_name],
        )


def _to_torch(data, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype)


def _to_tensor(data, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(data):
        return data.to(dtype=dtype)
    if isinstance(data, (list, tuple)) and data and torch.is_tensor(data[0]):
        return torch.stack(list(data), dim=0).to(dtype=dtype)
    return torch.as_tensor(data, dtype=dtype)


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _coerce_text(text) -> str:
    if isinstance(text, (list, tuple)):
        return _coerce_text(text[0]) if text else ""
    if text is None:
        return ""
    return text if isinstance(text, str) else str(text)


def _extract_object_key(text: str) -> str:
    text = _coerce_text(text)
    patterns = [
        r"\bof\s+(.+?)\s+with\b",
        r"\bgrab\s+(.+?)\s+with\b",
        r"\bpick(?:\s+up)?\s+(.+?)\s+with\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            target = re.sub(r"\s+", " ", match.group(1)).strip()
            if target:
                return target.lower()
    return text.strip().lower()


def _pose9_sequence(params):
    params = _to_tensor(params)
    if params.ndim == 1:
        if params.shape[0] < 9:
            raise ValueError(f"Invalid pose shape: {tuple(params.shape)}")
        return params[:9].unsqueeze(0)
    if params.ndim == 2:
        if params.shape[1] < 9:
            raise ValueError(f"Invalid pose shape: {tuple(params.shape)}")
        return params[:, :9]
    raise ValueError(f"Unsupported pose shape: {tuple(params.shape)}")


def _sequence_length(params) -> int:
    t = _to_tensor(params)
    if t.ndim == 1:
        return 1 if t.shape[0] > 0 else 0
    if t.ndim >= 2:
        return int(t.shape[0])
    return 0


def _slice_last_frames(params, nframes: int):
    t = _to_tensor(params)
    if t.ndim == 1:
        return t.unsqueeze(0)
    if nframes <= 0:
        return t[:0]
    return t[-nframes:]


def _text2hoi_rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    rot6d = rot6d.reshape(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=1)
    proj = torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1)
    b2 = torch.nn.functional.normalize(a2 - proj * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def _text2hoi_rot6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(_text2hoi_rot6d_to_matrix(rot6d))


def _standard_rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    rot6d = rot6d.reshape(-1, 6)
    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=1)
    proj = torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1)
    b2 = torch.nn.functional.normalize(a2 - proj * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def process_hand_result_standard(hand_layer, hand_params):
    hand_params = _to_torch(hand_params)
    hand_pose = hand_params[:, 3:]
    hand_pose = rot6d_to_axis_angle(hand_pose).reshape(-1, 48)
    hand_trans = hand_params[:, :3]
    duration = hand_trans.shape[0]
    out = hand_layer(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:48],
        betas=torch.zeros((duration, 10), dtype=torch.float32),
    )
    hand_vertices = out.vertices + hand_trans.unsqueeze(1)
    hand_joints = (
        out.joints_w_tip if getattr(out, "joints_w_tip", None) is not None else out.joints
    )
    hand_joints = hand_joints + hand_trans.unsqueeze(1)
    hand_faces = torch.as_tensor(hand_layer.faces.copy().astype(np.int64))
    return hand_vertices, hand_joints, hand_faces


def process_hand_result_text2hoi(hand_layer, hand_params):
    hand_params = _to_tensor(hand_params)
    if not torch.isfinite(hand_params).all():
        hand_params = torch.nan_to_num(hand_params, nan=0.0, posinf=0.0, neginf=0.0)
    hand_trans = hand_params[:, :3]
    global_rot6d = hand_params[:, 3:9]
    local_rot6d = hand_params[:, 9:].reshape(-1, 6)
    global_orient = _text2hoi_rot6d_to_axis_angle(global_rot6d).reshape(-1, 3)
    local_hand_pose = matrix_to_axis_angle(_standard_rot6d_to_matrix(local_rot6d)).reshape(
        -1, 45
    )
    out = hand_layer(
        global_orient=global_orient,
        hand_pose=local_hand_pose,
        transl=hand_trans,
        betas=torch.zeros((hand_trans.shape[0], 10), dtype=torch.float32),
    )
    hand_vertices = out.vertices
    hand_joints = (
        out.joints_w_tip if getattr(out, "joints_w_tip", None) is not None else out.joints
    )
    if getattr(out, "transl", None) is None:
        hand_vertices = hand_vertices + hand_trans.unsqueeze(1)
        hand_joints = hand_joints + hand_trans.unsqueeze(1)
    hand_faces = torch.as_tensor(hand_layer.faces.copy().astype(np.int16), dtype=torch.long)
    return hand_vertices, hand_joints, hand_faces


def process_obj_result_standard(obj_verts, obj_params):
    obj_verts = _to_torch(obj_verts)
    obj_params = _pose9_sequence(obj_params)
    obj_trans = obj_params[:, :3]
    obj_rot6d = obj_params[:, 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
    obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    return obj_pc_rotated + obj_trans.unsqueeze(1)


def process_obj_result_text2hoi(obj_verts, obj_params):
    obj_verts = _to_tensor(obj_verts)
    obj_params = _to_tensor(obj_params)
    obj_rotmat = _text2hoi_rot6d_to_matrix(obj_params[:, 3:9]).reshape(-1, 3, 3)
    obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
    return obj_pc_rotated + obj_params[:, :3].unsqueeze(1)


def _safe_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return float(abs(mesh.volume))
    except Exception:
        return 0.0
