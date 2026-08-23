import os
import os.path as osp
import hashlib

home = os.path.expanduser("~")

import glob
import argparse
import numpy as np
from tqdm import tqdm
import time
import json
import trimesh
import pickle
from collections import Counter, defaultdict

import sys

sys.path.append(home + "/dir/Text2HOI")
os.chdir(osp.dirname(osp.abspath(osp.dirname(__file__))))

import torch
import torch.nn.functional as F
from constants.hot3d_constants import (
    hot3d_obj_name,
    action_list,
    present_participle,
    third_verb,
    passive_verb,
)

from lib.models.mano import build_mano_aa
from lib.models.object import build_object_model
from lib.utils.file import load_config
from lib.utils.proc_h2o import (
    process_hand_pose_h2o,
    process_hand_trans_h2o,
    process_text,
)
from lib.utils.proc import (
    get_contact_info,
    transform_hand_to_xdata,
    transform_xdata_to_joints,
    transform_obj_to_xdata,
    farthest_point_sample,
)


def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d


def proc_torch_frame(l):
    if isinstance(l, list) or isinstance(l, np.ndarray):
        l = [torch.FloatTensor(_l).unsqueeze(0) for _l in l]
        l = torch.cat(l)
    return l


import torch
import torch.nn.functional as F


def compute_distance_to_ray(pc, origin, direction):
    """
    pc: (N, 3)
    origin: (3,)
    direction: (3,) - normalized
    Return:
        dists: (N,) - shortest distance from each point to the ray
    """
    pc = torch.as_tensor(pc)
    origin = torch.as_tensor(origin, dtype=pc.dtype, device=pc.device)
    direction = torch.as_tensor(direction, dtype=pc.dtype, device=pc.device)

    vec = pc - origin  # (N, 3)
    t = torch.clamp((vec * direction).sum(dim=-1), min=0.0)  # projection length
    proj = origin + t.unsqueeze(-1) * direction  # closest point on ray
    dists = F.pairwise_distance(pc, proj)  # (N,)
    return dists


def get_accumulated_contact_score(point_cloud, ray_origin, ray_direction, sigma=0.01):
    """
    point_cloud: (T, N, 3)
    ray_origin: (T, 3)
    ray_direction: (T, 3)
    sigma: float - Gaussian spread
    Returns:
        contact_score: (N,) - normalized accumulated contact score over time
    """
    point_cloud = torch.as_tensor(point_cloud)
    ray_origin = torch.as_tensor(
        ray_origin, dtype=point_cloud.dtype, device=point_cloud.device
    )
    ray_direction = torch.as_tensor(
        ray_direction, dtype=point_cloud.dtype, device=point_cloud.device
    )

    T, N, _ = point_cloud.shape
    acc_score = torch.zeros(N, device=point_cloud.device, dtype=point_cloud.dtype)

    for t in range(T):
        pc = point_cloud[t]  # (N, 3)
        origin = ray_origin[t]  # (3,)
        direction = ray_direction[t]  # (3,)

        dists = compute_distance_to_ray(pc, origin, direction)  # (N,)
        frame_score = torch.exp(-(dists**2) / (2 * sigma**2))  # (N,)

        acc_score += frame_score  # 누적

    contact_score = acc_score / 5
    contact_score[contact_score > 1] = 1

    return contact_score  # (N,)


def get_points_near_ray(point_cloud, ray_origin, ray_direction, max_distance=0.1):
    """
    point_cloud: (N, 3)
    ray_origin: (3,)
    ray_direction: (3,) - normalized
    max_distance: float - threshold distance from ray
    Returns:
        matched_points: (M, 3) points within threshold distance
        matched_indices: (M,) indices of those points
    """
    point_cloud = torch.as_tensor(point_cloud)
    ray_origin = torch.as_tensor(
        ray_origin, dtype=point_cloud.dtype, device=point_cloud.device
    )
    ray_direction = torch.as_tensor(
        ray_direction, dtype=point_cloud.dtype, device=point_cloud.device
    )

    T, N, _ = point_cloud.shape
    matched_points_all = []
    matched_indices_all = []

    for t in range(T):
        pc = point_cloud[t]  # (N, 3)
        origin = ray_origin[t]  # (3,)
        direction = ray_direction[t]  # (3,)

        vec_to_points = pc - origin  # (N, 3)
        t_val = torch.clamp((vec_to_points * direction).sum(dim=-1), min=0.0)  # (N,)
        proj_points = origin + t_val.unsqueeze(1) * direction  # (N, 3)
        dists = F.pairwise_distance(pc, proj_points)  # (N,)

        mask = dists < max_distance
        matched_points = pc[mask]
        matched_indices = torch.where(mask)[0]  # global index

        matched_points_all.append(matched_points)
        matched_indices_all.append(matched_indices)

    # 최종 누적 결과
    matched_points_all = (
        torch.cat(matched_points_all, dim=0)
        if matched_points_all
        else torch.empty((0, 3), dtype=point_cloud.dtype, device=point_cloud.device)
    )
    matched_indices_all = (
        torch.cat(matched_indices_all, dim=0)
        if matched_indices_all
        else torch.empty((0,), dtype=torch.long, device=point_cloud.device)
    )

    return matched_points_all, matched_indices_all


def smooth_affordance_map(dmin, obj_pc, k=32):
    """
    Gaussian smoothing over object surface neighborhood.
    """
    pts = torch.as_tensor(obj_pc, dtype=dmin.dtype, device=dmin.device)
    dist = torch.cdist(pts, pts)
    dist.fill_diagonal_(float("inf"))
    nn_dist = dist.min(dim=1).values
    sigma = nn_dist.mean().clamp_min(1e-6)
    k = min(k, dist.shape[1])
    knn_idx = dist.topk(k, largest=False).indices
    knn_dist = dist.gather(1, knn_idx)
    weights = torch.exp(-(knn_dist**2) / (2 * sigma**2))
    weights = weights / weights.sum(dim=1, keepdim=True)
    neighbor_vals = dmin[knn_idx]
    return (weights * neighbor_vals).sum(dim=1)


def build_contact_frequency_map(v_num, lcov_ver_idx, rcov_ver_idx):
    """
    Build a per-grasp binary contact map from thresholded contact vertex indices.
    """
    contact_map = np.zeros(v_num, dtype=np.float32)
    contact_indices = []
    if lcov_ver_idx is not None and len(lcov_ver_idx) > 0:
        contact_indices.append(np.asarray(lcov_ver_idx, dtype=np.int64))
    if rcov_ver_idx is not None and len(rcov_ver_idx) > 0:
        contact_indices.append(np.asarray(rcov_ver_idx, dtype=np.int64))
    if contact_indices:
        merged = np.concatenate(contact_indices, axis=0)
        if merged.size > 0:
            contact_map[np.unique(merged)] = 1.0
    return contact_map


def build_affordance_key(instance_name, action_label, part_label, grasp_side, method):
    if method == "min_distance":
        return (
            str(instance_name),
            str(action_label),
            str(part_label),
            str(grasp_side),
        )
    return (str(instance_name), str(action_label))


def compute_grasp_min_distance_map(
    obj_pc,
    object_rotmat_list,
    lhand_pose_list,
    lhand_beta_list,
    lhand_trans_list,
    rhand_pose_list,
    rhand_beta_list,
    rhand_trans_list,
    lhand_layer,
    rhand_layer,
):
    """
    Compute per-grasp minimum distance map d_i = min_j ||o_i - h_j|| over
    hand surface vertices and all frames in the grasp.
    """
    if len(object_rotmat_list) == 0:
        return None

    device = next(lhand_layer.parameters()).device
    obj_pc_t = torch.as_tensor(obj_pc, dtype=torch.float32, device=device)
    obj_tf = torch.as_tensor(object_rotmat_list, dtype=torch.float32, device=device)
    obj_rot = obj_tf[:, :3, :3]
    obj_trans = obj_tf[:, :3, 3]
    sampled_obj_verts = torch.einsum("tij,kj->tki", obj_rot, obj_pc_t) + obj_trans.unsqueeze(1)

    dist_candidates = []

    if len(lhand_pose_list) > 0:
        lpose = torch.as_tensor(lhand_pose_list, dtype=torch.float32, device=device)
        lbeta = torch.as_tensor(lhand_beta_list, dtype=torch.float32, device=device)
        ltrans = torch.as_tensor(lhand_trans_list, dtype=torch.float32, device=device)
        out_l = lhand_layer(lbeta, lpose[:, :3], lpose[:, 3:])
        lhand_vertices = out_l.vertices + ltrans.unsqueeze(1)
        ldist = torch.cdist(sampled_obj_verts, lhand_vertices)
        dist_candidates.append(ldist.amin(dim=2).amin(dim=0))

    if len(rhand_pose_list) > 0:
        rpose = torch.as_tensor(rhand_pose_list, dtype=torch.float32, device=device)
        rbeta = torch.as_tensor(rhand_beta_list, dtype=torch.float32, device=device)
        rtrans = torch.as_tensor(rhand_trans_list, dtype=torch.float32, device=device)
        out_r = rhand_layer(rbeta, rpose[:, :3], rpose[:, 3:])
        rhand_vertices = out_r.vertices + rtrans.unsqueeze(1)
        rdist = torch.cdist(sampled_obj_verts, rhand_vertices)
        dist_candidates.append(rdist.amin(dim=2).amin(dim=0))

    if not dist_candidates:
        return None

    dmin = torch.stack(dist_candidates, dim=0).amin(dim=0)
    return dmin.detach().cpu().numpy()


def transform_gaze_world_to_camera(gaze_data, extrinsic_matrix):
    gaze = np.asarray(gaze_data)
    if gaze.ndim < 2 or gaze.shape[0] != 2:
        return gaze_data
    squeeze_last = False
    if gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
        squeeze_last = True
    if gaze.shape[-1] != 3:
        return gaze_data

    extrinsic = np.asarray(extrinsic_matrix)
    if extrinsic.shape != (4, 4):
        return gaze_data

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    origin = gaze[0]
    direction = gaze[1]

    origin_cam = R @ origin + t
    direction_cam = R @ direction
    norm = np.linalg.norm(direction_cam)
    if norm > 0:
        direction_cam = direction_cam / norm

    transformed = np.stack([origin_cam, direction_cam], axis=0)
    if squeeze_last:
        transformed = transformed[..., None]
    return transformed


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def process_obj_result(obj_verts, obj_params):
    if obj_params.dim() == 2:
        # obj_params: (T, 9)
        obj_trans = obj_params[:, :3]
        obj_rot6d = obj_params[:, 3:9]
        obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
        if obj_verts.dim() == 2:
            # obj_verts: (K, 3)
            obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(1)
            return obj_verts_transformed, obj_pc_rotated
        if obj_verts.dim() == 3:
            # obj_verts: (B, K, 3), broadcast over batch
            obj_pc_rotated = torch.einsum("tij,bkj->btki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(0).unsqueeze(2)
            return obj_verts_transformed, obj_pc_rotated
    elif obj_params.dim() == 3:
        # obj_params: (B, T, 9)
        obj_trans = obj_params[..., :3]
        obj_rot6d = obj_params[..., 3:9]
        bsz, nframes = obj_params.shape[:2]
        obj_rotmat = rot6d_to_rotmat(obj_rot6d.reshape(-1, 6)).reshape(
            bsz, nframes, 3, 3
        )
        if obj_verts.dim() == 2:
            # obj_verts: (K, 3), broadcast over batch
            obj_pc_rotated = torch.einsum("btij,kj->btki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
            return obj_verts_transformed, obj_pc_rotated
        if obj_verts.dim() == 3:
            # obj_verts: (B, K, 3)
            obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_verts)
            obj_verts_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)
            return obj_verts_transformed, obj_pc_rotated
    raise ValueError(
        f"Unsupported shapes: obj_params {tuple(obj_params.shape)}, obj_verts {tuple(obj_verts.shape)}"
    )


def _deterministic_rng(key):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    return np.random.RandomState(seed)


def _subset_field(value, idx_array, list_indices):
    if isinstance(value, np.ndarray):
        return value[idx_array]
    if isinstance(value, list):
        return [value[i] for i in list_indices]
    return value


def _subset_object_data(obj_data, indices):
    idx_array = np.array(indices, dtype=int) if indices else np.array([], dtype=int)
    list_indices = idx_array.tolist()
    return {
        key: _subset_field(value, idx_array, list_indices)
        for key, value in obj_data.items()
    }


def normalize_action_text(action):
    text = action if isinstance(action, str) else str(action)
    text = text.lower().strip()
    if text.endswith("."):
        text = text[:-1]
    return text


def parse_action_label(action_text):
    text = normalize_action_text(action_text)
    side = "both"
    for candidate in ("left", "right", "both"):
        if f" with {candidate} hand" in text:
            side = candidate
            break

    if " of " in text:
        part_section = text.split(" of ", 1)[0]
    else:
        part_section = text
    tokens = part_section.split()

    part = "unknown"
    if tokens:
        if (
            tokens[-1] == "edge"
            and len(tokens) >= 2
            and tokens[-2] in ("long", "short")
        ):
            part = f"{tokens[-2]}_{tokens[-1]}"
        else:
            part = tokens[-1]

    if not part:
        part = "unknown"

    if part == "head":
        part = "unknown"

    return part, side


def split_dataset_by_part(
    dataset, val_ratio=0.1, special_obj_names=None, special_val_ratio=None
):
    special_obj_names = set(special_obj_names or [])
    part_samples = defaultdict(list)
    for obj_name, obj_data in dataset.items():
        actions = obj_data.get("action", [])
        for idx, action in enumerate(actions):
            part, _ = parse_action_label(action)
            part_samples[part].append((obj_name, idx))

    per_object_indices = defaultdict(lambda: {"train": [], "val": []})
    for part, samples in part_samples.items():
        if not samples or part == "unknown":
            continue

        normal_samples = [s for s in samples if s[0] not in special_obj_names]
        special_samples = [s for s in samples if s[0] in special_obj_names]

        def _assign_samples(split_samples, rng_key, ratio):
            total = len(split_samples)
            if total == 0:
                return
            rng = _deterministic_rng(rng_key)
            order = rng.permutation(total)
            val_count = max(1, int(np.ceil(total * ratio))) if total > 1 else 0
            val_count = min(val_count, total - 1) if total > 1 else 0
            train_count = total - val_count
            for position, sample_index in enumerate(order):
                obj_name, idx = split_samples[sample_index]
                dest = "train" if position < train_count else "val"
                per_object_indices[obj_name][dest].append(idx)

        _assign_samples(normal_samples, part, val_ratio)
        if special_val_ratio is not None:
            _assign_samples(special_samples, f"{part}|special", special_val_ratio)
        else:
            _assign_samples(special_samples, f"{part}|special", val_ratio)

    def _build_split(split_type):
        split_data = defaultdict()
        for obj_name, splits in per_object_indices.items():
            indices = splits.get(split_type, [])
            if not indices:
                continue
            split_data[obj_name] = _subset_object_data(dataset[obj_name], indices)
        return split_data

    return _build_split("train"), _build_split("val")


with open(osp.join(home, "dir/Text2HOI/data/hot3d/previous/instance.json"), "r") as f:
    instance_ = json.load(f)


def make_json(save_name="per_object", afford_map_method="frequency"):
    if afford_map_method not in {"frequency", "min_distance"}:
        raise ValueError(
            f"Unsupported afford_map_method '{afford_map_method}'. "
            "Use one of: 'frequency', 'min_distance'."
        )

    print(f"[make_json] afford_map_method={afford_map_method}")
    data_root = home + "/dir/Text2HOI/data/hot3d/pre-process_dataset"

    object_model = build_object_model(
        osp.join(home, "dir/Text2HOI/data/hot3d/dataset/obj.pkl")
    )

    data_path = osp.join(data_root, f"acc_ori.pkl")

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False)
    lhand_layer = lhand_layer.cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False)
    rhand_layer = rhand_layer.cuda()
    """
        subject, background, object class, cam, 
    """

    dataset_dir = home + "/dir/Text2HOI/data/hot3d/dataset"

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    """
        hand_pose_manos -> 124 x 1 (0, 62 -> 1 값으로 고정 as dummy value)
        obj_pose_rt_data -> 17 x 1 (1 -> object idx, 1: -> 4 x 4 matrix)
        extrinsic_matrix -> 4 x 4
    """

    def _init_totals():
        return {
            "x_lhand": [],
            "x_rhand": [],
            "j_lhand": [],
            "j_rhand": [],
            "x_obj": [],
            "x_lhand_org": [],  # root position of left hand
            "x_rhand_org": [],  # root position of right hand
            "lcf_idx": [],
            "lcov_idx": [],
            "lchj_idx": [],
            "lcf_ver_idx": [],
            "lcov_ver_idx": [],
            "lchj_ver_idx": [],
            "ldist_value": [],
            "ldist_value_vertex": [],
            "rcf_idx": [],
            "rcov_idx": [],
            "rchj_idx": [],
            "rcf_ver_idx": [],
            "rcov_ver_idx": [],
            "rchj_ver_idx": [],
            "rdist_value": [],
            "rdist_value_vertex": [],
            "is_lhand": [],
            "is_rhand": [],
            "lhand_beta": [],
            "rhand_beta": [],
            "object_idx": [],
            "action": [],
            "action_name": [],
            "nframes": [],
            "subject": [],
            "background": [],
            "cam_pose": [],
            "gaze_map": [],
            "gaze": [],
            "post_obj": [],
            "post_j_lhand": [],
            "post_j_rhand": [],
            "score_map": [],
            "act_id": [],
            "afford_key": [],
            "afford_map": [],
            "part": [],
            "instance_name": [],
        }

    def _append_sample(totals, gaze_map, score, afford_key, part_label, instance_name):
        totals["gaze_map"].append(gaze_map)
        totals["x_lhand"].append(x_lhand)
        totals["x_rhand"].append(x_rhand)
        totals["j_lhand"].append(j_lhand)
        totals["j_rhand"].append(j_rhand)
        totals["x_obj"].append(x_obj)
        totals["x_lhand_org"].append(x_lhand_org_list)
        totals["x_rhand_org"].append(x_rhand_org_list)
        totals["lcf_idx"].append(lcf_idx)
        totals["lcov_idx"].append(lcov_idx)
        totals["lchj_idx"].append(lchj_idx)
        totals["lcf_ver_idx"].append(lcf_ver_idx)
        totals["lcov_ver_idx"].append(lcov_ver_idx)
        totals["lchj_ver_idx"].append(lchj_ver_idx)
        totals["ldist_value"].append(ldist_value)
        totals["ldist_value_vertex"].append(ldist_value_vertex)
        totals["rcf_idx"].append(rcf_idx)
        totals["rcov_idx"].append(rcov_idx)
        totals["rchj_idx"].append(rchj_idx)
        totals["rcf_ver_idx"].append(rcf_ver_idx)
        totals["rcov_ver_idx"].append(rcov_ver_idx)
        totals["rchj_ver_idx"].append(rchj_ver_idx)
        totals["rdist_value"].append(rdist_value)
        totals["rdist_value_vertex"].append(rdist_value_vertex)
        totals["is_lhand"].append(is_lhand)
        totals["is_rhand"].append(is_rhand)
        totals["lhand_beta"].append(lhand_beta_list)
        totals["rhand_beta"].append(rhand_beta_list)
        totals["object_idx"].append(int(obj_idx))
        totals["nframes"].append(len(object_rotmat_list))
        totals["subject"].append(data_path.split("/")[3])
        totals["background"].append(data_path.split("/")[4])
        totals["cam_pose"].append(cam_pose_list)
        totals["gaze"].append(gaze_datas)
        totals["post_obj"].append(sampled_obj_verts.detach().cpu())
        totals["score_map"].append(score)
        totals["post_j_lhand"].append(lhand_joints.detach().cpu())
        totals["post_j_rhand"].append(rhand_joints.detach().cpu())
        totals["action"].append(action_label)
        totals["action_name"].append(action_label)
        totals["act_id"].append(act_id)
        totals["afford_key"].append(afford_key)
        totals["afford_map"].append(None)
        totals["part"].append(part_label)
        totals["instance_name"].append(instance_name)

    totals_nonempty = _init_totals()
    totals_empty = _init_totals()
    afford_contact_sum = {}
    afford_sample_count = {}
    afford_dist_maps = defaultdict(list)
    afford_obj_pc = {}

    # motion_length = defaultdict()
    for object_name, value_list in tqdm(data.items(), total=len(data)):
        if object_name in ["cellphone", "keyboard"]:
            continue

        for i in tqdm(
            range(len(value_list["hand_pose_manos"]))
        ):  ## 하나의 object에 해당하는 모든 모션
            lhand_pose_list = []
            lhand_beta_list = []
            lhand_trans_list = []
            x_lhand_org_list = []
            rhand_pose_list = []
            rhand_beta_list = []
            rhand_trans_list = []
            x_rhand_org_list = []
            object_rotmat_list = []
            cam_pose_list = []
            obj_ext_list = []

            hand_pose_mano_datas = value_list["hand_pose_manos"][i]
            obj_pose_rt_datas = value_list["obj_pose_rts"][i]
            extrinsic_matrixs = value_list["cam_poses"][i]
            gaze_datas = value_list["gaze"][i]
            gaze_datas_cam = []
            action_label = value_list["action_labels"][i][0]
            act_id = value_list["act_id"][i][0]

            def replace_verb_with_grab(prompt):
                base = prompt.strip()
                suffix = ""
                with_idx = base.lower().rfind(" with ")
                if with_idx != -1:
                    suffix = base[with_idx:]
                    base = base[:with_idx].strip()
                words = base.split()
                if not words:
                    return prompt
                # handle multi-word verb like "palmar grasp"
                if (
                    len(words) >= 2
                    and " ".join(w.lower() for w in words[:2]) == "palmar grasp"
                ):
                    words = ["Grab"] + words[2:]
                else:
                    words[0] = "Grab"
                return " ".join(words) + suffix

            action_label = replace_verb_with_grab(action_label)
            part_label, grasp_side = parse_action_label(action_label)

            for idx in range(len(gaze_datas)):  ## 모션에서 하나의 frame씩
                hand_pose_mano_data = hand_pose_mano_datas[idx]
                obj_pose_rt_data = obj_pose_rt_datas[idx]
                extrinsic_matrix = extrinsic_matrixs[idx]
                gaze_data = gaze_datas[idx]
                gaze_data_cam = transform_gaze_world_to_camera(gaze_data, extrinsic_matrix)
                if torch.isnan(torch.tensor(gaze_data_cam)).any():
                    continue
                gaze_datas_cam.append(gaze_data_cam)

                lhand_trans = hand_pose_mano_data[1:4]
                lhand_pose = hand_pose_mano_data[4:52]
                lhand_beta = hand_pose_mano_data[52:62]

                left_rotvec = process_hand_pose_h2o(
                    lhand_pose, lhand_trans, extrinsic_matrix
                )
                lhand_pose[:3] = left_rotvec

                new_left_trans, lhand_origin = process_hand_trans_h2o(
                    lhand_pose,
                    lhand_beta,
                    lhand_trans,
                    extrinsic_matrix,
                    lhand_layer,
                )
                lhand_trans_list.append(new_left_trans)
                lhand_pose_list.append(lhand_pose)
                lhand_beta_list.append(lhand_beta)
                x_lhand_org_list.append(lhand_origin)

                rhand_trans = hand_pose_mano_data[63:66]
                rhand_pose = hand_pose_mano_data[66:114]
                rhand_beta = hand_pose_mano_data[114:124]

                right_rotvec = process_hand_pose_h2o(
                    rhand_pose, rhand_trans, extrinsic_matrix
                )
                rhand_pose[:3] = right_rotvec

                new_right_trans, rhand_origin = process_hand_trans_h2o(
                    rhand_pose,
                    rhand_beta,
                    rhand_trans,
                    extrinsic_matrix,
                    rhand_layer,
                )
                rhand_trans_list.append(new_right_trans)
                rhand_pose_list.append(rhand_pose)
                rhand_beta_list.append(rhand_beta)
                x_rhand_org_list.append(rhand_origin)

                obj_idx = obj_pose_rt_data[0]
                object_ext = obj_pose_rt_data[1:].reshape(4, 4)

                new_object_matrix = np.dot(extrinsic_matrix, object_ext)
                object_rotmat_list.append(new_object_matrix)

                cam_pose_list.append(extrinsic_matrix)
                obj_ext_list.append(object_ext)

            _, obj_pc, _, _ = object_model(
                instance_[str(int(obj_idx))]["instance_name"]
            )

            (
                lcf_idx,
                lcov_idx,
                lchj_idx,
                ldist_value,
                rcf_idx,
                rcov_idx,
                rchj_idx,
                rdist_value,
                is_lhand,
                is_rhand,
                lcf_ver_idx,
                lcov_ver_idx,
                lchj_ver_idx,
                ldist_value_vertex,
                rcf_ver_idx,
                rcov_ver_idx,
                rchj_ver_idx,
                rdist_value_vertex,
                sampled_obj_verts,
                lhand_joints,
                rhand_joints,
            ) = get_contact_info(
                lhand_pose_list,
                lhand_beta_list,
                lhand_trans_list,
                rhand_pose_list,
                rhand_beta_list,
                rhand_trans_list,
                object_rotmat_list,
                lhand_layer,
                rhand_layer,
                obj_pc,
            )

            x_lhand = transform_hand_to_xdata(lhand_trans_list, lhand_pose_list)
            x_rhand = transform_hand_to_xdata(rhand_trans_list, rhand_pose_list)
            j_lhand = transform_xdata_to_joints(x_lhand, lhand_layer)
            j_rhand = transform_xdata_to_joints(x_rhand, rhand_layer)
            x_obj = transform_obj_to_xdata(object_rotmat_list)
            obj_mat = transform_obj_to_xdata(obj_ext_list)

            post_pc, _ = process_obj_result(torch.tensor(obj_pc), torch.tensor(obj_mat))
            lhand_trans_np = np.asarray(lhand_trans_list, dtype=np.float32)
            rhand_trans_np = np.asarray(rhand_trans_list, dtype=np.float32)
            post_j_lhand = j_lhand.copy()
            post_j_rhand = j_rhand.copy()

            if lhand_trans_np.shape[0] == post_j_lhand.shape[0]:
                post_j_lhand = post_j_lhand + lhand_trans_np[:, None, :]
            if rhand_trans_np.shape[0] == post_j_rhand.shape[0]:
                post_j_rhand = post_j_rhand + rhand_trans_np[:, None, :]

            index_list = list()
            gaze_datas_world = np.array(gaze_datas)
            gaze_datas = np.array(gaze_datas_cam)
            index = torch.empty(0, dtype=torch.long)
            score_map = None
            afford_key = None

            obj_meta = instance_.get(str(int(obj_idx)))
            if obj_meta and obj_meta.get("instance_name"):
                afford_key = build_affordance_key(
                    obj_meta["instance_name"],
                    action_label,
                    part_label,
                    grasp_side,
                    afford_map_method,
                )
                afford_obj_pc[afford_key] = np.asarray(obj_pc)
                contact_map = build_contact_frequency_map(
                    np.asarray(obj_pc).shape[0],
                    lcov_ver_idx,
                    rcov_ver_idx,
                )
                if afford_key not in afford_contact_sum:
                    afford_contact_sum[afford_key] = contact_map
                    afford_sample_count[afford_key] = 1
                else:
                    afford_contact_sum[afford_key] += contact_map
                    afford_sample_count[afford_key] += 1

            for frame_idx in range(1, len(gaze_datas_world) + 1):
                _, index = get_points_near_ray(
                    post_pc[:frame_idx],
                    gaze_datas_world[:frame_idx, 0].squeeze(-1),
                    gaze_datas_world[:frame_idx, 1].squeeze(-1),
                    max_distance=0.01,
                )
                score_map = get_accumulated_contact_score(
                    post_pc[:frame_idx],
                    gaze_datas_world[:frame_idx, 0].squeeze(-1),
                    gaze_datas_world[:frame_idx, 1].squeeze(-1),
                    sigma=0.01,
                )
                index_list.append(torch.unique(index))

            if len(index) == 0:
                _append_sample(
                    totals_empty, index_list, score_map, afford_key, part_label, object_name
                )
                continue

            _append_sample(
                totals_nonempty, index_list, score_map, afford_key, part_label, object_name
            )

    afford_map_by_key = {}
    for key, contact_sum_np in afford_contact_sum.items():
        sample_count = max(afford_sample_count.get(key, 1), 1)
        freq_map_np = contact_sum_np / float(sample_count)
        obj_pc = afford_obj_pc[key]
        freq_map_t = torch.as_tensor(freq_map_np, dtype=torch.float32)
        obj_pc_t = torch.as_tensor(obj_pc, dtype=freq_map_t.dtype)
        afford_map = smooth_affordance_map(freq_map_t, obj_pc_t, k=32)
        afford_map_by_key[key] = 1.0 - afford_map.detach().cpu().numpy()


    for totals in (totals_nonempty, totals_empty):
        totals["afford_map"] = [
            afford_map_by_key.get(key) if key is not None else None
            for key in totals["afford_key"]
        ]

    def ragged_array(seq):
        return np.array(seq, dtype=object)

    def _build_final_dict(totals):
        total_dict = {
            "x_lhand": totals["x_lhand"],
            "x_rhand": totals["x_rhand"],
            "j_lhand": totals["j_lhand"],
            "j_rhand": totals["j_rhand"],
            "x_obj": totals["x_obj"],
            "lhand_beta": totals["lhand_beta"],
            "rhand_beta": totals["rhand_beta"],
            "lhand_org": totals["x_lhand_org"],
            "rhand_org": totals["x_rhand_org"],
        }
        return {
            **total_dict,
            "lcf_idx": ragged_array(totals["lcf_idx"]),
            "lcov_idx": ragged_array(totals["lcov_idx"]),
            "lchj_idx": ragged_array(totals["lchj_idx"]),
            "lcf_ver_idx": ragged_array(totals["lcf_ver_idx"]),
            "lcov_ver_idx": ragged_array(totals["lcov_ver_idx"]),
            "lchj_ver_idx": ragged_array(totals["lchj_ver_idx"]),
            "ldist_value": ragged_array(totals["ldist_value"]),
            "ldist_value_vertex": ragged_array(totals["ldist_value_vertex"]),
            "rcf_idx": ragged_array(totals["rcf_idx"]),
            "rcov_idx": ragged_array(totals["rcov_idx"]),
            "rchj_idx": ragged_array(totals["rchj_idx"]),
            "rcf_ver_idx": ragged_array(totals["rcf_ver_idx"]),
            "rcov_ver_idx": ragged_array(totals["rcov_ver_idx"]),
            "rchj_ver_idx": ragged_array(totals["rchj_ver_idx"]),
            "rdist_value": ragged_array(totals["rdist_value"]),
            "rdist_value_vertex": ragged_array(totals["rdist_value_vertex"]),
            "is_lhand": np.array(totals["is_lhand"]),
            "is_rhand": np.array(totals["is_rhand"]),
            "object_idx": np.array(totals["object_idx"]),
            "action": ragged_array(totals["action"]),
            "act_id": ragged_array(totals["act_id"]),
            "gaze_map": totals["gaze_map"],
            "gaze": totals["gaze"],
            "post_obj": totals["post_obj"],
            "score_map": totals["score_map"],
            "action_name": np.array(totals["action_name"]),
            "nframes": np.array(totals["nframes"]),
            "subject": np.array(totals["subject"]),
            "background": np.array(totals["background"]),
            "cam_pose": ragged_array(totals["cam_pose"]),
            "post_j_lhand": totals["post_j_lhand"],
            "post_j_rhand": totals["post_j_rhand"],
            "afford_map": totals["afford_map"],
            "part": np.array(totals["part"]),
            "instance_name": np.array(totals["instance_name"]),
        }


    def _subset_totals(totals, indices):
        idx_list = list(indices)
        return {k: [v[i] for i in idx_list] for k, v in totals.items()}

    def _save_pair(nonempty_totals, empty_totals, out_name):
        final_dict = _build_final_dict(nonempty_totals)
        with open(osp.join(dataset_dir, f"{out_name}.pkl"), "wb") as f:
            pickle.dump(final_dict, f)

        empty_dataset_name = f"{out_name}_gaze_empty"
        empty_dict = _build_final_dict(empty_totals)
        with open(osp.join(dataset_dir, f"{empty_dataset_name}.pkl"), "wb") as f:
            pickle.dump(empty_dict, f)

    def _split_by_instance_name(totals, test_ratio=0.1):
        per_instance_indices = defaultdict(list)
        for idx, instance_name in enumerate(totals["instance_name"]):
            per_instance_indices[str(instance_name)].append(idx)

        train_index_set = set()
        test_index_set = set()

        for instance_name, indices in per_instance_indices.items():
            total = len(indices)
            if total <= 1:
                train_index_set.update(indices)
                continue

            rng = _deterministic_rng(f"instance_split|{instance_name}")
            order = rng.permutation(total)
            test_count = max(1, int(np.ceil(total * test_ratio)))
            test_count = min(test_count, total - 1)
            train_count = total - test_count

            for position, local_idx in enumerate(order):
                global_idx = indices[local_idx]
                if position < train_count:
                    train_index_set.add(global_idx)
                else:
                    test_index_set.add(global_idx)

        train_indices = []
        test_indices = []
        for idx in range(len(totals["instance_name"])):
            if idx in train_index_set:
                train_indices.append(idx)
            elif idx in test_index_set:
                test_indices.append(idx)

        return train_indices, test_indices

    os.makedirs(dataset_dir, exist_ok=True)

    train_idx_nonempty, test_idx_nonempty = _split_by_instance_name(totals_nonempty)
    train_idx_empty, test_idx_empty = _split_by_instance_name(totals_empty)

    _save_pair(
        _subset_totals(totals_nonempty, train_idx_nonempty),
        _subset_totals(totals_empty, train_idx_empty),
        "gaze_train",
    )
    _save_pair(
        _subset_totals(totals_nonempty, test_idx_nonempty),
        _subset_totals(totals_empty, test_idx_empty),
        "gaze_test",
    )

def preprocessing_text():
    text_json = "data/hot3d/text.json"
    with open("text.json", "r") as f:
        action_list = json.load(f)

    text_description = defaultdict(dict)
    for action in action_list:
        action = action.lower()

        action_v, action_o, action_h = (
            action.split(" ")[0],
            " ".join(action.split(" ")[1:]).split(" with ")[0],
            " ".join(action.split(" ")[1:]).split(" with ")[1][:-1],
        )
        action_ving = present_participle[action_v]

        text = action.capitalize()

        text1 = f"{action_ving} {action_o} with {action_h}.".capitalize()

        action_3rd_v = third_verb[action_v]
        text2 = f"{action_h.capitalize()} {action_3rd_v} {action_o}."

        action_passive = passive_verb[action_v]
        text3 = f"{action_o} {action_passive} with {action_h}.".capitalize()

        text_description[action] = [text, text1, text2, text3]

    with open(text_json, "w") as f:
        json.dump(text_description, f)


def preprocessing_balance_weights():
    data_path = home + "/dir/Text2HOI/data/hot3d/dataset/acc_ori.pkl"
    balance_weights_path = home + "/dir/Text2HOI/data/hot3d/balance_weights.pkl"

    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        act_id = data["action"]

    text_list = []
    text_act = []
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i],
            is_lhand[i],
            is_rhand[i],
            text_descriptions=None,
            return_key=True,
        )
        text_list.append(text_key)
        text_act.append(act_id[i])

    text_counter = Counter(text_list)
    text_dict = dict(text_counter)
    text_prob = {k: 1 / v for k, v in text_dict.items()}
    balance_weights = {
        text_act[idx]: text_prob[text] for idx, text in enumerate(text_list)
    }
    with open(balance_weights_path, "wb") as f:
        pickle.dump(balance_weights, f)


def preprocessing_text2length():
    data_path = osp.join(home, "dir/Text2HOI/data/hot3d/patterned.npz")
    t2l_json = home + "/dir/Text2HOI/data/hot3d/text_length_patterned.json"

    with np.load(data_path, allow_pickle=True) as data:
        is_lhand = data["is_lhand"]
        is_rhand = data["is_rhand"]
        action_name = data["action_name"]
        nframes = data["nframes"]

    text_dict = {}
    for i in range(len(action_name)):
        text_key = process_text(
            action_name[i],
            is_lhand[i],
            is_rhand[i],
            text_descriptions=None,
            return_key=True,
        )

        num_frames = int(nframes[i])
        if num_frames > 150:
            num_frames = 150
        if text_key not in text_dict:
            text_dict[text_key] = [num_frames]
        else:
            text_dict[text_key].append(num_frames)
    with open(t2l_json, "w") as f:
        json.dump(text_dict, f)


def print_text_data_num():
    hot3d_config = load_config("configs/dataset/hot3d.yaml")
    data_path = hot3d_config.data_path
    t2l_json_path = hot3d_config.t2l_json

    with np.load(data_path, allow_pickle=True) as data:
        action_name = data["action_name"]
    print(f"data num: {len(action_name)}")

    with open(t2l_json_path, "r") as f:
        text = json.load(f)
    print(f"text num: {len(text)}")


def preprocessing_object():
    h2o_config = load_config("configs/dataset/hot3d.yaml")

    obj_pcs = {}
    obj_pc_normals = {}
    point_sets = {}
    obj_path = {}
    object_paths = glob.glob(osp.join(h2o_config.obj_root, "*.ply"))

    for object_path in tqdm(object_paths):
        mesh = trimesh.load(object_path, process=False)
        verts = torch.FloatTensor(mesh.vertices).unsqueeze(0).cuda()
        normal = torch.FloatTensor(mesh.vertex_normals).unsqueeze(0).cuda()
        normal = normal / torch.norm(normal, dim=2, keepdim=True)
        point_set = farthest_point_sample(verts, 1024)
        sampled_pc = verts[0, point_set[0]].cpu().numpy()
        sampled_normal = normal[0, point_set[0]].cpu().numpy()
        with open("/".join(object_path.split("/")[:-1]) + "/instance.json", "r") as f:
            instance = json.load(f)
        object_name = instance[str(object_path.split("/")[-1].split(".")[0])][
            "instance_name"
        ]
        key = f"{object_name}"
        obj_pcs[key] = sampled_pc
        obj_pc_normals[key] = sampled_normal
        point_sets[key] = point_set[0].cpu().numpy()
        obj_path[key] = "/".join(object_path.split("/")[-2:])

    os.makedirs("data/hot3d", exist_ok=True)
    with open("data/hot3d/obj.pkl", "wb") as f:
        pickle.dump(
            {
                "object_name": hot3d_obj_name,
                "obj_pcs": obj_pcs,
                "obj_pc_normals": obj_pc_normals,
                "point_sets": point_sets,
                "obj_path": obj_path,
            },
            f,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default="grab_afford_train")
    parser.add_argument(
        "--afford_map_method",
        type=str,
        default="frequency",
        choices=["frequency", "min_distance"],
    )
    args = parser.parse_args()

    make_json(save_name=args.save_name, afford_map_method=args.afford_map_method)
    # preprocessing_balance_weights()
    # preprocessing_text()
