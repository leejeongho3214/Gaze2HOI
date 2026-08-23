import os
import os.path as osp
import sys

# isort: off
# Keep this bootstrap before project imports (formatters may run isort separately).
PROJECT_ROOT = osp.dirname(osp.abspath(osp.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# isort: on

# isort: split

import json
import pickle
import random
from copy import deepcopy

import hydra
import numpy as np
import torch
import tqdm
from easydict import EasyDict as edict
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from lib.datasets.datasets import get_dataloader
from lib.models.mano import build_mano_aa
from lib.utils.eval import (
    get_valid_mask_bunch,
)
from lib.utils.model_utils import (
    build_model_and_diffusion,
    build_pointnetfeat,
)
from lib.utils.motion_normalizer import MotionNormalizer
from lib.utils.proc import (
    load_bps_basis,
    proc_obj_feat_final_train,
)
from lib.utils.proc_output import (
    get_hand_joints_w_tip,
    get_NN,
    get_interior,
    get_hand_verts,
    get_transformed_obj_pc,
)
from lib.utils.rot import rot6d_to_rotmat
from lib.utils.seed import seed_everything
from lib.utils.gaze2hoi_train_helpers import (
    _compute_last_frame_mesh_penetration_metrics_per_hand,
    _is_relative_gaze_mlp_mode,
    _is_temporal_gaze_token_mode,
    build_gaze_alignment_temporal_mask,
    apply_null_gaze_condition,
    build_bimart_refinement_mesh_cache,
    build_gaze_condition_feature_for_gaze2hoi,
    build_bps_correspondence_cache,
    build_relative_gaze_mlp_for_gaze2hoi,
    compute_bps_feature_from_mesh_cache_for_gaze2hoi,
    configure_gaze_token_fusion_for_mode,
    compute_bps_gaze_ray_closeness_map_from_mesh_cache_for_gaze2hoi,
    compute_tail_mask,
    get_bps_correspondence_source,
    get_gaze2hoi_gaze_condition_dim,
    gaze_condition_requires_bps,
    load_hand_sample_indices_for_gaze2hoi,
    load_object_mesh_bps_cache,
    repeat_initial_object_pose_for_gaze_condition,
    restore_point_token_outputs_for_gaze2hoi,
    restore_hand_point_token_outputs_with_object_pose_for_gaze2hoi,
    load_object_mesh_cache,
    resolve_partwise_bps_context,
)


def _resolve_eval_hands(text_entry, is_lhand_value, is_rhand_value):
    text_lower = str(text_entry).lower()
    if "both hands" in text_lower:
        return True, True
    if "left hand" in text_lower and "right hand" not in text_lower:
        return True, False
    if "right hand" in text_lower and "left hand" not in text_lower:
        return False, True
    return bool(is_lhand_value), bool(is_rhand_value)


def _resolve_test_data_name(config, task_overrides):
    data_name_was_overridden = any(
        str(override).lstrip("+").startswith("dataset.data_name=")
        for override in task_overrides
    )
    if data_name_was_overridden:
        return str(config.dataset.data_name)
    return str(
        getattr(config.gaze2hoi.exp, "test_data_name", "ori_dataset/gaze_test")
    )


def _get_target_object_indices(item, batch_size, nobj, device):
    raw_idx = item.get("target_obj_idx", item.get("target_object_idx"))
    if raw_idx is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if torch.is_tensor(raw_idx):
        idx = raw_idx.to(device=device, dtype=torch.long).view(-1)
    else:
        idx = torch.as_tensor(raw_idx, dtype=torch.long, device=device).view(-1)
    if idx.numel() == 1:
        idx = idx.expand(batch_size)
    return idx[:batch_size].clamp(0, nobj - 1)


def _gather_target_object_slot(tensor, target_indices):
    if not torch.is_tensor(tensor) or tensor.dim() < 2:
        return tensor
    batch_idx = torch.arange(tensor.shape[0], device=tensor.device)
    return tensor[batch_idx, target_indices.to(device=tensor.device)]


def _object_names_per_sample(item, batch_size, nobj):
    names = item.get("candidate_obj_names")
    if names is None:
        obj_name = item.get("obj_name", item.get("object_name"))
        context_name = item.get("context_obj_name")
        if nobj == 2 and context_name is not None:
            names = list(zip(obj_name, context_name))
        else:
            names = obj_name
    if names is None:
        return [[None] * nobj for _ in range(batch_size)]

    if (
        nobj > 1
        and isinstance(names, (list, tuple))
        and len(names) == nobj
        and all(isinstance(col, (list, tuple)) and len(col) == batch_size for col in names)
    ):
        names = list(zip(*names))

    rows = []
    for b in range(batch_size):
        sample_names = names[b] if isinstance(names, (list, tuple)) else names
        if isinstance(sample_names, (list, tuple)):
            row = list(sample_names)
        else:
            row = [sample_names] * nobj
        if len(row) < nobj:
            row = row + [row[-1] if row else None] * (nobj - len(row))
        rows.append([str(name) if name is not None else None for name in row[:nobj]])
    return rows


def _target_object_names(item, batch_size, nobj, target_indices):
    rows = _object_names_per_sample(item, batch_size, nobj)
    target_cpu = target_indices.detach().cpu().tolist()
    return [
        rows[b][int(target_cpu[b])] if rows[b] and int(target_cpu[b]) < len(rows[b]) else None
        for b in range(batch_size)
    ]


def _rotate_obj_vectors(obj_params, vectors, dataset_name):
    bs, nframes = obj_params.shape[:2]
    obj_rot6d = obj_params[..., 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(bs, nframes, 3, 3)
    if dataset_name == "grab":
        return torch.einsum("btij,bki->btkj", obj_rotmat, vectors)
    return torch.einsum("btij,bkj->btki", obj_rotmat, vectors)


def _normalize_gaze_tensor(gaze):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        return gaze.squeeze(-1), True
    return gaze, False


def _active_gaze_target_obj(obj_pc, gaze_map_frame, origin, direction, obj_points_cam):
    active = gaze_map_frame > 0
    if bool(active.any().item()):
        return obj_pc[active].mean(dim=0)

    direction = direction / direction.norm().clamp_min(1e-8)
    rel = obj_points_cam - origin.unsqueeze(0)
    ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
    closest = origin.unsqueeze(0) + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
    dist = (obj_points_cam - closest).norm(dim=-1)
    return obj_pc[dist.argmin()]


def _build_gaze_down_variants(
    gaze, gaze_map, x_obj, obj_pc, down_ratios, shift_origin=False
):
    """Lower the object-local gaze target y and rebuild gaze directions."""
    if gaze_map is None:
        raise ValueError("gaze_down_sweep requires item['gaze_map'] to rebuild shifted ray maps.")

    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    ratios = torch.as_tensor(down_ratios, device=gaze.device, dtype=gaze.dtype)
    variant_gaze = []
    variant_gaze_map = []
    batch_size, nframes = x_obj.shape[:2]

    for ratio in ratios:
        gaze_out = gaze_base.clone()
        map_out = torch.zeros_like(gaze_map)
        for b in range(batch_size):
            y_min = obj_pc[b, :, 1].min()
            y_max = obj_pc[b, :, 1].max()
            height = (y_max - y_min).clamp_min(1e-8)
            for t in range(nframes):
                rot = rot6d_to_rotmat(x_obj[b, t, 3:9].reshape(1, 6)).reshape(3, 3)
                trans = x_obj[b, t, :3]
                obj_points_cam = torch.matmul(obj_pc[b], rot.transpose(0, 1)) + trans
                origin = gaze_base[b, t, 0]
                direction = gaze_base[b, t, 1]
                direction = direction / direction.norm().clamp_min(1e-8)

                # Choose a target on the original raw ray at the depth of the
                # nearest object point. This keeps ratio=0 exactly equal to the
                # input gaze direction; larger ratios change only target local-y.
                rel = obj_points_cam - origin.unsqueeze(0)
                ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
                closest = (
                    origin.unsqueeze(0)
                    + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
                )
                dist = (obj_points_cam - closest).norm(dim=-1)
                forward = ray_t > 0
                nearest_idx = torch.where(
                    forward, dist, torch.full_like(dist, torch.inf)
                ).argmin() if bool(forward.any().item()) else dist.argmin()
                target_t = ray_t[nearest_idx].clamp_min(1e-4)
                target_cam = origin + target_t * direction
                target_obj = torch.matmul(rot.transpose(0, 1), target_cam - trans)
                shifted_target_obj = target_obj.clone()
                shifted_target_obj[1] = target_obj[1] * (1.0 - ratio) + y_min * ratio
                shifted_origin = origin
                if shift_origin:
                    origin_obj = torch.matmul(rot.transpose(0, 1), origin - trans)
                    shifted_origin_obj = origin_obj.clone()
                    shifted_origin_obj[1] = origin_obj[1] - ratio * height
                    shifted_origin = torch.matmul(rot, shifted_origin_obj) + trans
                shifted_target = torch.matmul(rot, shifted_target_obj) + trans
                shifted_direction = shifted_target - shifted_origin
                shifted_direction = shifted_direction / shifted_direction.norm().clamp_min(1e-8)
                gaze_out[b, t, 0] = shifted_origin
                gaze_out[b, t, 1] = shifted_direction

                rel = obj_points_cam - shifted_origin.unsqueeze(0)
                ray_t = (rel * shifted_direction.unsqueeze(0)).sum(dim=-1)
                closest = (
                    shifted_origin.unsqueeze(0)
                    + ray_t.clamp_min(0.0).unsqueeze(-1) * shifted_direction
                )
                dist = (obj_points_cam - closest).norm(dim=-1)
                active = (dist <= 0.01) & (ray_t > 0)
                if not bool(active.any().item()):
                    topk = min(64, dist.numel())
                    active_idx = torch.topk(dist, k=topk, largest=False).indices
                    map_out[b, t, active_idx] = 1.0
                else:
                    map_out[b, t, active] = 1.0
        if had_last_dim:
            gaze_out = gaze_out.unsqueeze(-1)
        variant_gaze.append(gaze_out)
        variant_gaze_map.append(map_out)

    return torch.cat(variant_gaze, dim=0), torch.cat(variant_gaze_map, dim=0)


def _repeat_tensor_for_variants(tensor, repeat_count):
    if tensor is None:
        return None
    return tensor.repeat_interleave(repeat_count, dim=0)


def _repeat_list_for_variants(values, repeat_count, suffixes=None):
    out = []
    for value in values:
        for idx in range(repeat_count):
            suffix = "" if suffixes is None else suffixes[idx]
            out.append(f"{value}{suffix}")
    return out


def _normalize_gaze_noise_mode(mode):
    mode = str(mode).strip().lower()
    aliases = {
        "": "none",
        "clean": "none",
        "off": "none",
        "rotate": "rotation",
        "rot": "rotation",
        "translate": "translation",
        "trans": "translation",
        "frame_drop": "frame_hold",
        "drop_frame": "frame_hold",
        "temporal_drop": "frame_hold",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("none", "rotation", "translation", "frame_hold"):
        raise ValueError(
            f"Unknown gaze2hoi.exp.gaze_noise_mode={mode!r}; expected "
            "'none', 'rotation', 'translation', or 'frame_hold'."
        )
    return mode


def _hold_every_nth_frame(tensor, interval):
    if tensor is None:
        return None
    interval = int(interval)
    if interval < 2:
        raise ValueError("gaze_frame_hold_interval must be at least 2.")
    output = tensor.clone()
    frame_indices = torch.arange(
        interval - 1,
        tensor.shape[1],
        interval,
        device=tensor.device,
    )
    if frame_indices.numel() > 0:
        output[:, frame_indices] = output[:, frame_indices - 1]
    return output


def _apply_gaze_noise(
    gaze,
    mode,
    generator=None,
    rotation_std_deg=2.0,
    translation_std_m=0.005,
    frame_hold_interval=5,
):
    """Apply test-only corruption to camera-frame gaze rays."""
    mode = _normalize_gaze_noise_mode(mode)
    if mode == "none":
        return gaze
    if mode == "frame_hold":
        return _hold_every_nth_frame(gaze, frame_hold_interval)

    gaze_base, had_last_dim = _normalize_gaze_tensor(gaze)
    if gaze_base.dim() != 4 or gaze_base.shape[2:] != (2, 3):
        raise ValueError(
            f"Expected gaze shape (B,T,2,3) or (B,T,2,3,1), got "
            f"{tuple(gaze.shape)}."
        )
    output = gaze_base.clone()

    if mode == "rotation":
        rotation_std_deg = float(rotation_std_deg)
        if rotation_std_deg < 0:
            raise ValueError("gaze_rotation_std_deg must be non-negative.")
        direction = output[:, :, 1]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis = torch.randn(
            direction.shape,
            device=direction.device,
            dtype=direction.dtype,
            generator=generator,
        )
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angle = torch.randn(
            (*direction.shape[:2], 1),
            device=direction.device,
            dtype=direction.dtype,
            generator=generator,
        )
        angle = angle * np.deg2rad(rotation_std_deg)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotated = (
            direction * cos_angle
            + torch.cross(axis, direction, dim=-1) * sin_angle
            + axis
            * (axis * direction).sum(dim=-1, keepdim=True)
            * (1.0 - cos_angle)
        )
        output[:, :, 1] = rotated / rotated.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    else:
        translation_std_m = float(translation_std_m)
        if translation_std_m < 0:
            raise ValueError("gaze_translation_std_m must be non-negative.")
        translation = torch.randn(
            output[:, :, 0].shape,
            device=output.device,
            dtype=output.dtype,
            generator=generator,
        )
        # Parallel ray shift: move the attended region without another rotation.
        output[:, :, 0] += translation * translation_std_m

    return output.unsqueeze(-1) if had_last_dim else output


def _rebuild_gaze_map_for_rays(gaze, x_obj, obj_pc, template):
    """Rebuild the raw object-point gaze map after ray corruption."""
    if template is None:
        return None
    gaze_base, _ = _normalize_gaze_tensor(gaze)
    rebuilt = torch.zeros_like(template)
    batch_size, frame_count = x_obj.shape[:2]
    for b in range(batch_size):
        for t in range(frame_count):
            rot = rot6d_to_rotmat(x_obj[b, t, 3:9].reshape(1, 6)).reshape(3, 3)
            obj_points_cam = torch.matmul(obj_pc[b], rot.transpose(0, 1))
            obj_points_cam = obj_points_cam + x_obj[b, t, :3]
            origin = gaze_base[b, t, 0]
            direction = gaze_base[b, t, 1]
            direction = direction / direction.norm().clamp_min(1e-8)
            rel = obj_points_cam - origin.unsqueeze(0)
            ray_t = (rel * direction.unsqueeze(0)).sum(dim=-1)
            closest = (
                origin.unsqueeze(0)
                + ray_t.clamp_min(0.0).unsqueeze(-1) * direction
            )
            distance = (obj_points_cam - closest).norm(dim=-1)
            active = (distance <= 0.01) & (ray_t > 0)
            if bool(active.any().item()):
                rebuilt[b, t, active] = 1.0
            else:
                nearest = torch.topk(
                    distance,
                    k=min(64, distance.numel()),
                    largest=False,
                ).indices
                rebuilt[b, t, nearest] = 1.0
    return rebuilt


def _config_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ("true", "1", "yes", "y", "on"):
            return True
        if value_lower in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)


def _compute_exterior_contact_frame_mask_per_hand(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    obj_pc_normal_org,
    dataset_name,
    valid_mask_hand,
    valid_mask_obj,
    contact_threshold=0.01,
    contact_min_keypoints=2,
):
    bs, nframes = pred_obj.shape[:2]
    hand_joints = get_hand_joints_w_tip(pred_hand, hand_layer)
    transf_obj_pc = get_transformed_obj_pc(pred_obj, obj_pc_org, dataset_name)

    flat_hand_joints = hand_joints.reshape(bs * nframes, -1, 3)
    flat_obj_pc = transf_obj_pc.reshape(bs * nframes, -1, 3)

    nn_dist, _ = get_NN(flat_hand_joints, flat_obj_pc)
    nn_dist = nn_dist.sqrt().view(bs, nframes, -1)

    close_contact_mask = nn_dist < contact_threshold
    return (close_contact_mask.sum(dim=2) >= contact_min_keypoints) & torch.logical_and(
        valid_mask_obj, valid_mask_hand
    )


def _resolve_output_representation(config):
    output_representation = str(
        getattr(config.gaze2hoi.model, "output_representation", "point")
    ).lower()
    aliases = {
        "points": "point",
        "point_token": "point",
        "point_tokens": "point",
        "mano": "pose",
        "mano_pose": "pose",
        "param": "pose",
        "params": "pose",
    }
    output_representation = aliases.get(output_representation, output_representation)
    if output_representation not in ("point", "pose"):
        raise ValueError(
            f"Unknown gaze2hoi.model.output_representation={output_representation!r}; "
            "expected 'point' or 'pose'."
        )
    config.gaze2hoi.model.output_representation = output_representation
    config.gaze2hoi.model.use_point_token_output = output_representation == "point"
    return output_representation


def _max_tail_joint_penetration_depth_per_hand(
    hand_joints,
    obj_pc,
    obj_pc_normal,
    tail_mask,
):
    batch_size, nframes = tail_mask.shape
    depths = torch.zeros(
        batch_size,
        nframes,
        device=hand_joints.device,
        dtype=hand_joints.dtype,
    )
    if not tail_mask.any():
        return depths

    flat_tail_mask = tail_mask.reshape(-1)
    flat_hand_joints = hand_joints.reshape(batch_size * nframes, -1, 3)
    flat_obj_pc = obj_pc.reshape(batch_size * nframes, -1, 3)
    flat_obj_pc_normal = obj_pc_normal.reshape(batch_size * nframes, -1, 3)

    tail_hand_joints = flat_hand_joints[flat_tail_mask]
    tail_obj_pc = flat_obj_pc[flat_tail_mask]
    tail_obj_pc_normal = flat_obj_pc_normal[flat_tail_mask]

    joint_nn_dist, joint_nn_idx = get_NN(tail_hand_joints, tail_obj_pc)
    joint_is_interior = get_interior(
        tail_obj_pc_normal,
        tail_obj_pc,
        tail_hand_joints,
        joint_nn_idx,
    )
    joint_depth = torch.where(
        joint_is_interior,
        joint_nn_dist.sqrt(),
        torch.zeros_like(joint_nn_dist),
    )
    depths[tail_mask] = joint_depth.max(dim=1).values
    return depths


def _run_single_seed(config):
    checkpoint_path = getattr(config.gaze2hoi.exp, "weight_path", None)
    if checkpoint_path and osp.exists(checkpoint_path):
        checkpoint_metadata = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_overrides = {
            "gaze_condition_mode": checkpoint_metadata.get("gaze_condition_mode"),
            "null_gaze_condition": checkpoint_metadata.get("null_gaze_condition"),
            "gaze_token_fusion": checkpoint_metadata.get("gaze_token_fusion"),
            "cross_attn_order": checkpoint_metadata.get("cross_attn_order"),
            "gaze_token_target": checkpoint_metadata.get("gaze_token_target"),
            "use_obj_scale": checkpoint_metadata.get("use_obj_scale"),
            "use_obj_centroid": checkpoint_metadata.get("use_obj_centroid"),
        }
        for key, value in checkpoint_overrides.items():
            if value is not None:
                config.gaze2hoi.model[key] = value
        del checkpoint_metadata

    test_seed = seed_everything(
        getattr(config.gaze2hoi.exp, "seed", 0),
        deterministic=bool(getattr(config.gaze2hoi.exp, "deterministic", False)),
    )
    print(
        f"Using seed={test_seed} "
        f"deterministic={bool(getattr(config.gaze2hoi.exp, 'deterministic', False))}"
    )

    print(f"Test dataset split: {config.dataset.data_name}")
    config.batch_size = int(getattr(config.gaze2hoi.exp, "test_batch_size", 64))
    save_pre_post_comparison = _config_bool(
        getattr(config.gaze2hoi.exp, "save_pre_post_comparison", False),
        default=False,
    )
    max_test_batches = int(getattr(config.gaze2hoi.exp, "max_test_batches", 0))
    max_comparison_samples = int(
        getattr(config.gaze2hoi.exp, "max_comparison_samples", 0)
    )
    pre_post_comparison_records = []
    gaze_noise_mode = _normalize_gaze_noise_mode(
        getattr(config.gaze2hoi.exp, "gaze_noise_mode", "none")
    )
    gaze_rotation_std_deg = float(
        getattr(config.gaze2hoi.exp, "gaze_rotation_std_deg", 2.0)
    )
    gaze_translation_std_m = float(
        getattr(config.gaze2hoi.exp, "gaze_translation_std_m", 0.005)
    )
    gaze_frame_hold_interval = int(
        getattr(config.gaze2hoi.exp, "gaze_frame_hold_interval", 5)
    )
    gaze_noise_generator = torch.Generator(device="cuda")
    gaze_noise_generator.manual_seed(int(test_seed) + 104729)
    print(
        "Test gaze noise: "
        f"mode={gaze_noise_mode}, rotation_std={gaze_rotation_std_deg:g}deg, "
        f"translation_std={gaze_translation_std_m:g}m, "
        f"frame_hold_interval={gaze_frame_hold_interval}"
    )

    config.gaze2hoi.exp.mano_fit_use_post_opt_losses = _config_bool(
        getattr(config.gaze2hoi.exp, "mano_fit_use_post_opt_losses", True),
        default=True,
    )
    print(
        "MANO post-optimization: "
        f"{'ON' if config.gaze2hoi.exp.mano_fit_use_post_opt_losses else 'OFF'}"
    )
    
    target_object_name = getattr(config.gaze2hoi.exp, "target_object_name", None)

    if target_object_name is not None:
        target_object_name = str(target_object_name)

    data_config = config.dataset
    dataset_name = data_config.name

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False).cuda()
    _, data_loader = get_dataloader(
        "Motion" + dataset_name, config, data_config, test=True
    )
    eval_dataset_indices = None
    eval_dataset_indices_path = getattr(
        config.gaze2hoi.exp, "eval_dataset_indices_path", None
    )
    if eval_dataset_indices_path:
        with open(eval_dataset_indices_path, encoding="utf-8") as handle:
            eval_dataset_indices = [int(index) for index in json.load(handle)]
        dataset_size = len(data_loader.dataset)
        if (
            not eval_dataset_indices
            or min(eval_dataset_indices) < 0
            or max(eval_dataset_indices) >= dataset_size
            or len(set(eval_dataset_indices)) != len(eval_dataset_indices)
        ):
            raise ValueError(
                "eval_dataset_indices_path must contain unique valid dataset "
                f"indices for a dataset of size {dataset_size}."
            )
        subset = Subset(data_loader.dataset, eval_dataset_indices)
        data_loader = DataLoader(
            subset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=False,
        )
        print(
            "Evaluation subset: "
            f"{len(eval_dataset_indices)}/{dataset_size} samples from "
            f"{eval_dataset_indices_path} (test_flag=True)."
        )

    object_feature_type = str(
        getattr(config.gaze2hoi.model, "object_feature_type", "bps")
    ).lower()
    use_pointnet_object_feature = object_feature_type == "pointnet"
    if object_feature_type not in ("bps", "pointnet"):
        raise ValueError(
            f"Unknown gaze2hoi.model.object_feature_type={object_feature_type!r}; "
            "expected 'bps' or 'pointnet'."
        )
    gaze_condition_mode = str(
        getattr(config.gaze2hoi.model, "gaze_condition_mode", "alignment")
    ).lower()
    configure_gaze_token_fusion_for_mode(config)
    needs_bps = (
        not use_pointnet_object_feature
        or gaze_condition_requires_bps(gaze_condition_mode)
    )
    if use_pointnet_object_feature and gaze_condition_requires_bps(gaze_condition_mode):
        raise ValueError(
            "PointNet object ablation cannot use the BPS-dependent gaze mode "
            f"{gaze_condition_mode!r}. Use "
            "gaze_condition_mode=object_ray_closeness_temporal so gaze is "
            "computed directly from the gaze ray and object point cloud."
        )
    if needs_bps:
        bps_path = getattr(
            config.gaze2hoi.exp,
            "bps_path",
            "assets/grab_bps_1024.pt",
        )
        if not osp.isabs(bps_path):
            bps_path = osp.abspath(osp.join(PROJECT_ROOT, bps_path))
        bps_basis = load_bps_basis(bps_path, device="cuda")
        part_label_map, num_bps_parts = resolve_partwise_bps_context(config)
        bps_count = int(bps_basis.shape[0])
    else:
        bps_basis = None
        part_label_map = None
        num_bps_parts = 1
        bps_count = int(getattr(config.gaze2hoi.model, "gaze_map_dim", 1024))
        print("BPS disabled: object and gaze conditions use PointNet/object PC only.")
    object_bps_feature_mode = str(
        getattr(config.gaze2hoi.model, "object_bps_feature_mode", "displacement")
    ).lower()
    if object_bps_feature_mode in ("distance", "dist", "scalar", "norm"):
        object_bps_dim_per_token = 1
    elif object_bps_feature_mode in ("displacement", "xyz", "vector", "nearest_delta", "delta"):
        object_bps_dim_per_token = 3
    else:
        raise ValueError(
            f"Unknown gaze2hoi.model.object_bps_feature_mode={object_bps_feature_mode!r}; "
            "expected 'distance' or 'displacement'."
        )
    use_relative_gaze_mlp = _is_relative_gaze_mlp_mode(gaze_condition_mode)
    use_contact_condition = gaze_condition_mode in (
        "contact_map",
        "cov_map",
        "gt_contact_map",
        "bps_contact_map",
        "bps_cov_map",
        "raw_contact_map",
        "raw_cov_map",
    )
    use_gaze_closeness = bool(getattr(config.gaze2hoi.exp, "use_gaze_closeness", False))
    if (
        use_contact_condition
        or _is_temporal_gaze_token_mode(gaze_condition_mode)
        or gaze_condition_mode in (
            "direction_with_origin_xyz",
            "direction_alignment_origin_xyz",
            "alignment_direction_origin_xyz",
            "bps_direction_origin_xyz",
            "gaze_direction_origin_xyz",
            "alignment_combined",
            "combined_alignment",
            "origin_direction_alignment",
            "direction_origin_alignment",
            "bps_alignment_combined",
            "gaze_alignment_combined",
            "method1",
            "gaze_method1",
            "ray_distance_map",
            "object_ray_distance",
            "object_point_ray_distance",
            "ray_closeness_map",
            "object_ray_closeness_map",
            "method2",
            "gaze_method2",
            "bps_ray_distance_map",
            "bps_ray_closeness",
            "bps_ray_closeness_map",
            "bps_gaze_ray_closeness",
            "bps_gaze_ray_closeness_map",
        )
    ):
        use_gaze_closeness = False
    config.gaze2hoi.model.use_gaze_alignment = gaze_condition_mode in (
        "alignment",
        "bps_alignment",
        "gaze_alignment",
        "direction_with_origin_xyz",
        "direction_alignment_origin_xyz",
        "alignment_direction_origin_xyz",
        "bps_direction_origin_xyz",
        "gaze_direction_origin_xyz",
        "alignment_combined",
        "combined_alignment",
        "origin_direction_alignment",
        "direction_origin_alignment",
        "bps_alignment_combined",
        "gaze_alignment_combined",
        "method1",
        "gaze_method1",
        "ray_distance_map",
        "object_ray_distance",
        "object_point_ray_distance",
        "ray_closeness_map",
        "object_ray_closeness_map",
        "method2",
        "gaze_method2",
        "bps_ray_distance_map",
        "bps_ray_closeness",
        "bps_ray_closeness_map",
        "bps_gaze_ray_closeness",
        "bps_gaze_ray_closeness_map",
    ) or _is_temporal_gaze_token_mode(gaze_condition_mode)
    gaze_condition_dim = get_gaze2hoi_gaze_condition_dim(
        config, num_bps_parts, bps_count
    )
    object_feature_dim = (
        int(getattr(config.gaze2hoi.model, "pointnet_obj_dim", 1024))
        if use_pointnet_object_feature
        else num_bps_parts * bps_count * object_bps_dim_per_token
    )
    config.gaze2hoi.model.obj_global_dim = object_feature_dim
    if _is_temporal_gaze_token_mode(gaze_condition_mode):
        config.gaze2hoi.model.obj_dim = object_feature_dim
        config.gaze2hoi.model.gaze_token_dim = gaze_condition_dim
    else:
        config.gaze2hoi.model.obj_dim = (
            object_feature_dim
            + gaze_condition_dim
            + (num_bps_parts * bps_count if use_gaze_closeness else 0)
        )
        config.gaze2hoi.model.gaze_token_dim = 0
    print(
        "Using object/gaze condition dims: "
        f"num_bps_parts={num_bps_parts}, bps_count={bps_count}, "
        f"object_bps_feature_mode={object_bps_feature_mode}, "
        f"object_feature_dim={object_feature_dim}, "
        f"gaze_condition_dim={gaze_condition_dim}, "
        f"obj_dim={config.gaze2hoi.model.obj_dim}."
    )
    output_representation = _resolve_output_representation(config)
    use_point_token_output = output_representation == "point"
    predict_object_pose = bool(
        getattr(config.gaze2hoi.model, "predict_object_pose", True)
    )
    config.gaze2hoi.model.predict_object_pose = predict_object_pose
    hand_sample_indices = None
    if use_point_token_output:
        hand_sample_indices = load_hand_sample_indices_for_gaze2hoi(
            config, device="cuda"
        )
        include_hand_object_dirvec = True
        config.gaze2hoi.model.include_hand_object_dirvec = True
        canonicalize_point_targets = bool(
            getattr(config.gaze2hoi.model, "canonicalize_point_targets", True)
        )
        config.gaze2hoi.model.canonicalize_hand_point_targets = canonicalize_point_targets
        config.gaze2hoi.model.canonicalize_object_point_targets = canonicalize_point_targets
        use_obj_scale = bool(getattr(config.gaze2hoi.model, "use_obj_scale", True))
        use_obj_centroid = bool(
            getattr(config.gaze2hoi.model, "use_obj_centroid", False)
        )
        config.gaze2hoi.model.use_obj_scale = use_obj_scale
        config.gaze2hoi.model.use_obj_centroid = use_obj_centroid
        point_coord_dim = int(hand_sample_indices.numel()) * 3
        config.gaze2hoi.model.hand_nfeats = point_coord_dim * 2
        config.gaze2hoi.model.obj_nfeats = 9 if predict_object_pose else point_coord_dim
    else:
        use_obj_scale = bool(getattr(config.gaze2hoi.model, "use_obj_scale", True))
        use_obj_centroid = bool(
            getattr(config.gaze2hoi.model, "use_obj_centroid", False)
        )
    gaze2hoi, diffusion = build_model_and_diffusion(
        config, lhand_layer, rhand_layer, test=True
    )
    motion_normalizer = None
    if use_point_token_output and predict_object_pose:
        normalization_checkpoint = torch.load(
            config.gaze2hoi.exp.weight_path, map_location="cpu"
        )
        normalization_state = normalization_checkpoint.get("motion_normalization")
        if normalization_state is None:
            raise RuntimeError(
                "Hybrid hand-point/object-pose checkpoint is missing "
                "motion_normalization statistics. Retrain or use a compatible checkpoint."
            )
        motion_normalizer = MotionNormalizer.from_state_dict(normalization_state)
    use_obj_scale = bool(getattr(config.gaze2hoi.model, "use_obj_scale", True))
    use_obj_centroid = bool(
        getattr(config.gaze2hoi.model, "use_obj_centroid", False)
    )
    relative_gaze_mlp = (
        build_relative_gaze_mlp_for_gaze2hoi(config, output_dim=gaze_condition_dim).cuda()
        if use_relative_gaze_mlp
        else None
    )
    if relative_gaze_mlp is not None:
        ckpt = torch.load(config.gaze2hoi.exp.weight_path, map_location="cuda")
        relative_gaze_ema_state = ckpt.get("relative_gaze_ema")
        relative_gaze_state = (
            relative_gaze_ema_state.get("shadow")
            if bool(getattr(config.gaze2hoi.exp, "use_ema_for_eval", True))
            and isinstance(relative_gaze_ema_state, dict)
            else ckpt.get("relative_gaze_mlp")
        )
        if relative_gaze_state is None:
            raise RuntimeError(
                "Checkpoint does not contain `relative_gaze_mlp` state, "
                "but gaze_condition_mode uses relative_gaze_mlp."
            )
        relative_gaze_mlp.load_state_dict(relative_gaze_state)
        relative_gaze_mlp.eval()
        for param in relative_gaze_mlp.parameters():
            param.requires_grad = False
    pointnet = None
    if use_pointnet_object_feature:
        config.pointfeat.global_feat = True
        pointnet = build_pointnetfeat(config, test=False)
        ckpt = torch.load(config.gaze2hoi.exp.weight_path, map_location="cuda")
        pointnet_ema_state = ckpt.get("pointnet_ema")
        pointnet_state = (
            pointnet_ema_state.get("shadow")
            if bool(getattr(config.gaze2hoi.exp, "use_ema_for_eval", True))
            and isinstance(pointnet_ema_state, dict)
            else ckpt.get("pointnet_model")
        )
        if pointnet_state is None:
            raise RuntimeError(
                "PointNet object feature mode requires `pointnet_model` in the Gaze2HOI checkpoint."
            )
        pointnet.load_state_dict(pointnet_state)
        pointnet.eval()
        for param in pointnet.parameters():
            param.requires_grad = False
        print(
            "Using PointNet object feature instead of BPS: "
            f"dim {object_feature_dim}."
        )
    bps_correspondence_source = get_bps_correspondence_source(config)
    mesh_bps_cache = (
        load_object_mesh_bps_cache(
            config,
            device="cuda",
            align_to_obj_pc_norm=True,
        )
        if (not use_pointnet_object_feature and bps_correspondence_source == "object_mesh")
        else None
    )
    mesh_bps_correspondence_cache = (
        None
        if use_pointnet_object_feature
        else build_bps_correspondence_cache(
            config,
            bps_basis,
            mesh_cache=mesh_bps_cache,
            device="cuda",
        )
    )
    if part_label_map is not None:
        print("Part-wise BPS enabled; BPS correspondences remain dynamic.")
    obj_meshes = load_object_mesh_cache(data_config)
    refinement_obj_meshes = build_bimart_refinement_mesh_cache(
        obj_meshes,
        max_faces=int(
            getattr(config.gaze2hoi.exp, "mano_refine_mesh_max_faces", 2000)
        ),
    )
    hand_nfeats = config.gaze2hoi.model.hand_nfeats
    obj_nfeats = config.gaze2hoi.model.obj_nfeats
    save_list = []
    numerical_events = []
    evaluated_sample_count = 0
    dataset_cursor = 0
    gaze_down_sweep = bool(getattr(config.gaze2hoi.exp, "gaze_down_sweep", False))
    gaze_down_sample_idx = int(getattr(config.gaze2hoi.exp, "gaze_down_sample_idx", 29))
    gaze_down_ratios = list(
        getattr(config.gaze2hoi.exp, "gaze_down_ratios", [0.0, 0.25, 0.5, 0.75])
    )
    gaze_down_shift_origin = bool(
        getattr(config.gaze2hoi.exp, "gaze_down_shift_origin", False)
    )
    gaze_down_processed = False

    with torch.no_grad():
        for batch_idx, item in enumerate(
            tqdm.tqdm(data_loader, desc="LOADING DATA"),
            start=1,
        ):
            if max_test_batches > 0 and batch_idx > max_test_batches:
                break
            if (
                save_pre_post_comparison
                and max_comparison_samples > 0
                and len(pre_post_comparison_records) >= max_comparison_samples
            ):
                break
            original_batch_size = item["x_obj"].shape[0]
            if eval_dataset_indices is None:
                batch_dataset_indices = range(
                    dataset_cursor,
                    dataset_cursor + original_batch_size,
                )
            else:
                batch_dataset_indices = eval_dataset_indices[
                    dataset_cursor : dataset_cursor + original_batch_size
                ]
            dataset_indices = torch.as_tensor(
                list(batch_dataset_indices),
                dtype=torch.long,
                device="cuda",
            )
            dataset_cursor += original_batch_size
            x_obj = item["x_obj"].cuda()
            x_lhand_gt = item["x_lhand"].cuda()
            x_rhand_gt = item["x_rhand"].cuda()
            obj_pc_org = item["obj_pc"].cuda()
            obj_pc_normal_org = item["obj_pc_normal"].cuda()
            gaze = item["gaze"].cuda()
            ldist_map = item["ldist_map"].cuda()
            rdist_map = item["rdist_map"].cuda()
            source_gaze_map = item["gaze_map"].cuda()
            gaze_map = (
                source_gaze_map
                if gaze_condition_mode
                in ("gaze_map", "raw_gaze_map", "dataset_gaze_map")
                else None
            )
            contact_map = item["cov_map"].cuda() if use_contact_condition else None
            cam_pose = item.get("cam_pose")
            normalized_obj_pc = item["normalized_obj_pc"].cuda()
            obj_cent = item["obj_cent"].cuda()
            obj_scale = item["obj_scale"].cuda()
            duration = item["nframes"].cuda()
            is_lhand, is_rhand = item["is_lhand"].cuda(), item["is_rhand"].cuda()
            text = item["text"]
            batch_size = x_obj.shape[0]
            nobj = x_obj.shape[1] if x_obj.dim() == 4 else 1
            target_object_indices = _get_target_object_indices(
                item, batch_size, nobj, x_obj.device
            )
            object_names = _target_object_names(
                item, batch_size, nobj, target_object_indices
            )
            if x_obj.dim() == 4:
                x_obj = _gather_target_object_slot(x_obj, target_object_indices)
                obj_pc_org = _gather_target_object_slot(
                    obj_pc_org, target_object_indices
                )
                obj_pc_normal_org = _gather_target_object_slot(
                    obj_pc_normal_org, target_object_indices
                )
                normalized_obj_pc = _gather_target_object_slot(
                    normalized_obj_pc, target_object_indices
                )
                obj_cent = _gather_target_object_slot(
                    obj_cent, target_object_indices
                )
                obj_scale = _gather_target_object_slot(
                    obj_scale, target_object_indices
                )
            max_nframes = x_obj.shape[1]
            if target_object_name is not None:
                keep_indices = [
                    idx
                    for idx, object_name in enumerate(object_names)
                    if object_name == target_object_name
                ]
                if not keep_indices:
                    continue
                keep_tensor = torch.as_tensor(
                    keep_indices, dtype=torch.long, device=x_obj.device
                )
                x_obj = x_obj.index_select(0, keep_tensor)
                x_lhand_gt = x_lhand_gt.index_select(0, keep_tensor)
                x_rhand_gt = x_rhand_gt.index_select(0, keep_tensor)
                obj_pc_org = obj_pc_org.index_select(0, keep_tensor)
                obj_pc_normal_org = obj_pc_normal_org.index_select(0, keep_tensor)
                gaze = gaze.index_select(0, keep_tensor)
                ldist_map = ldist_map.index_select(0, keep_tensor)
                rdist_map = rdist_map.index_select(0, keep_tensor)
                if gaze_map is not None:
                    gaze_map = gaze_map.index_select(0, keep_tensor)
                if contact_map is not None:
                    contact_map = contact_map.index_select(0, keep_tensor)
                normalized_obj_pc = normalized_obj_pc.index_select(0, keep_tensor)
                obj_cent = obj_cent.index_select(0, keep_tensor)
                obj_scale = obj_scale.index_select(0, keep_tensor)
                duration = duration.index_select(0, keep_tensor)
                is_lhand = is_lhand.index_select(0, keep_tensor)
                is_rhand = is_rhand.index_select(0, keep_tensor)
                if torch.is_tensor(cam_pose):
                    cam_pose = cam_pose.index_select(0, keep_tensor.to(cam_pose.device))
                text = [text[idx] for idx in keep_indices]
                object_names = [object_names[idx] for idx in keep_indices]
                dataset_indices = dataset_indices.index_select(0, keep_tensor)
                source_gaze_map = source_gaze_map.index_select(0, keep_tensor)
            batch_size = x_obj.shape[0]

            gaze_down_ratio_values = None
            if gaze_down_sweep:
                keep_indices = torch.nonzero(
                    dataset_indices == gaze_down_sample_idx, as_tuple=False
                ).flatten()
                if keep_indices.numel() == 0:
                    continue
                if gaze_down_processed:
                    break
                keep_tensor = keep_indices.to(device=x_obj.device, dtype=torch.long)
                x_obj = x_obj.index_select(0, keep_tensor)
                x_lhand_gt = x_lhand_gt.index_select(0, keep_tensor)
                x_rhand_gt = x_rhand_gt.index_select(0, keep_tensor)
                obj_pc_org = obj_pc_org.index_select(0, keep_tensor)
                obj_pc_normal_org = obj_pc_normal_org.index_select(0, keep_tensor)
                normalized_obj_pc = normalized_obj_pc.index_select(0, keep_tensor)
                obj_cent = obj_cent.index_select(0, keep_tensor)
                obj_scale = obj_scale.index_select(0, keep_tensor)
                duration = duration.index_select(0, keep_tensor)
                is_lhand = is_lhand.index_select(0, keep_tensor)
                is_rhand = is_rhand.index_select(0, keep_tensor)
                gaze = gaze.index_select(0, keep_tensor)
                ldist_map = ldist_map.index_select(0, keep_tensor)
                rdist_map = rdist_map.index_select(0, keep_tensor)
                source_gaze_map = source_gaze_map.index_select(0, keep_tensor)
                if gaze_map is not None:
                    gaze_map = gaze_map.index_select(0, keep_tensor)
                if contact_map is not None:
                    contact_map = contact_map.index_select(0, keep_tensor)
                if torch.is_tensor(cam_pose):
                    cam_pose = cam_pose.index_select(0, keep_tensor.to(cam_pose.device))
                text = [text[int(idx)] for idx in keep_indices.detach().cpu().tolist()]
                object_names = [
                    object_names[int(idx)] for idx in keep_indices.detach().cpu().tolist()
                ]
                dataset_indices = dataset_indices.index_select(0, keep_tensor)

                repeat_count = len(gaze_down_ratios)
                gaze, shifted_gaze_map = _build_gaze_down_variants(
                    gaze,
                    source_gaze_map,
                    x_obj,
                    obj_pc_org,
                    gaze_down_ratios,
                    shift_origin=gaze_down_shift_origin,
                )
                if gaze_map is not None:
                    gaze_map = shifted_gaze_map
                ldist_map = _repeat_tensor_for_variants(ldist_map, repeat_count)
                rdist_map = _repeat_tensor_for_variants(rdist_map, repeat_count)
                x_obj = _repeat_tensor_for_variants(x_obj, repeat_count)
                x_lhand_gt = _repeat_tensor_for_variants(x_lhand_gt, repeat_count)
                x_rhand_gt = _repeat_tensor_for_variants(x_rhand_gt, repeat_count)
                obj_pc_org = _repeat_tensor_for_variants(obj_pc_org, repeat_count)
                obj_pc_normal_org = _repeat_tensor_for_variants(
                    obj_pc_normal_org, repeat_count
                )
                normalized_obj_pc = _repeat_tensor_for_variants(
                    normalized_obj_pc, repeat_count
                )
                obj_cent = _repeat_tensor_for_variants(obj_cent, repeat_count)
                obj_scale = _repeat_tensor_for_variants(obj_scale, repeat_count)
                duration = _repeat_tensor_for_variants(duration, repeat_count)
                is_lhand = _repeat_tensor_for_variants(is_lhand, repeat_count)
                is_rhand = _repeat_tensor_for_variants(is_rhand, repeat_count)
                if contact_map is not None:
                    contact_map = _repeat_tensor_for_variants(contact_map, repeat_count)
                if torch.is_tensor(cam_pose):
                    cam_pose = _repeat_tensor_for_variants(cam_pose, repeat_count)
                ratio_suffixes = [
                    f" [gaze_down={float(ratio):.2f}]" for ratio in gaze_down_ratios
                ]
                text = _repeat_list_for_variants(text, repeat_count, ratio_suffixes)
                object_names = _repeat_list_for_variants(object_names, repeat_count)
                dataset_indices = _repeat_tensor_for_variants(dataset_indices, repeat_count)
                gaze_down_ratio_values = [
                    float(ratio)
                    for _ in range(keep_indices.numel())
                    for ratio in gaze_down_ratios
                ]
                gaze_down_processed = True
                batch_size = x_obj.shape[0]

            if gaze_noise_mode != "none":
                gaze = _apply_gaze_noise(
                    gaze,
                    gaze_noise_mode,
                    generator=gaze_noise_generator,
                    rotation_std_deg=gaze_rotation_std_deg,
                    translation_std_m=gaze_translation_std_m,
                    frame_hold_interval=gaze_frame_hold_interval,
                )
                if gaze_noise_mode == "frame_hold":
                    source_gaze_map = _hold_every_nth_frame(
                        source_gaze_map, gaze_frame_hold_interval
                    )
                    if gaze_map is not None:
                        gaze_map = source_gaze_map
                elif gaze_map is not None:
                    gaze_map = _rebuild_gaze_map_for_rays(
                        gaze,
                        x_obj,
                        obj_pc_org,
                        source_gaze_map,
                    )
                    source_gaze_map = gaze_map

            (
                valid_mask_lhand,
                valid_mask_rhand,
                valid_mask_obj,
            ) = get_valid_mask_bunch(is_lhand, is_rhand, max_nframes, duration)
            if gaze_condition_mode in (
                "alignment",
                "bps_alignment",
                "gaze_alignment",
                "direction_with_origin_xyz",
                "direction_alignment_origin_xyz",
                "alignment_direction_origin_xyz",
                "bps_direction_origin_xyz",
                "gaze_direction_origin_xyz",
                "alignment_combined",
                "combined_alignment",
                "origin_direction_alignment",
                "direction_origin_alignment",
                "bps_alignment_combined",
                "gaze_alignment_combined",
                "method1",
                "gaze_method1",
                "ray_distance_map",
                "object_ray_distance",
                "object_point_ray_distance",
                "ray_closeness_map",
                "object_ray_closeness_map",
                "method2",
                "gaze_method2",
                "bps_ray_distance_map",
                "bps_ray_closeness",
                "bps_ray_closeness_map",
                "bps_gaze_ray_closeness",
                "bps_gaze_ray_closeness_map",
            ) or _is_temporal_gaze_token_mode(gaze_condition_mode):
                gaze_condition_valid_mask = build_gaze_alignment_temporal_mask(
                    valid_mask_obj,
                    ldist_map=ldist_map,
                    rdist_map=rdist_map,
                    temporal_scope=getattr(
                        config.gaze2hoi.model,
                        "gaze_alignment_temporal_scope",
                        "all",
                    ),
                )
            else:
                gaze_condition_valid_mask = valid_mask_obj

            with torch.no_grad():
                if use_pointnet_object_feature:
                    obj_feat = pointnet(normalized_obj_pc)
                else:
                    obj_feat = compute_bps_feature_from_mesh_cache_for_gaze2hoi(
                        normalized_obj_pc,
                        bps_basis,
                        object_names=object_names,
                        mesh_cache=mesh_bps_cache,
                        mesh_bps_correspondence_cache=mesh_bps_correspondence_cache,
                        part_label_map=part_label_map,
                        bbox_margin=float(
                            getattr(config.dataset, "mesh_part_bbox_margin", 0.03)
                        ),
                        feature_mode=object_bps_feature_mode,
                    )
                gaze_condition_x_obj = (
                    repeat_initial_object_pose_for_gaze_condition(x_obj)
                )
                gaze_score = build_gaze_condition_feature_for_gaze2hoi(
                    config,
                    gaze,
                    gaze_map,
                    gaze_condition_x_obj,
                    normalized_obj_pc,
                    bps_basis,
                    obj_cent,
                    obj_scale,
                    gaze_condition_valid_mask,
                    object_names=object_names,
                    mesh_cache=mesh_bps_cache,
                    part_label_map=part_label_map,
                    bbox_margin=float(
                        getattr(config.dataset, "mesh_part_bbox_margin", 0.03)
                    ),
                    target_dim=gaze_condition_dim,
                    contact_map=contact_map,
                    relative_gaze_mlp=relative_gaze_mlp,
                )
                gaze_score = apply_null_gaze_condition(config, gaze_score)
                if use_gaze_closeness:
                    gaze_closeness = compute_bps_gaze_ray_closeness_map_from_mesh_cache_for_gaze2hoi(
                        gaze,
                        gaze_condition_x_obj,
                        normalized_obj_pc,
                        bps_basis,
                        obj_cent,
                        obj_scale,
                        valid_mask_obj,
                        object_names=object_names,
                        mesh_cache=mesh_bps_cache,
                        part_label_map=part_label_map,
                        bbox_margin=float(
                            getattr(config.dataset, "mesh_part_bbox_margin", 0.03)
                        ),
                        sigma=float(getattr(config.gaze2hoi.exp, "gaze_distance_sigma", 0.05)),
                    )
                    gaze_closeness = apply_null_gaze_condition(config, gaze_closeness)
                    contact_feat = torch.cat([gaze_score, gaze_closeness], dim=1)
                else:
                    contact_feat = gaze_score

            obj_feat_final = proc_obj_feat_final_train(
                contact_feat,
                obj_scale,
                obj_cent,
                obj_feat,
                use_obj_scale=use_obj_scale,
                use_obj_centroid=use_obj_centroid,
            )

            coarse_x_lhand, coarse_x_rhand, coarse_x_obj = diffusion.sampling(
                gaze2hoi,
                obj_feat_final,
                max_nframes,
                hand_nfeats,
                obj_nfeats,
                valid_mask_lhand,
                valid_mask_rhand,
                valid_mask_obj,
                device=torch.device("cuda"),
                enc_text=None,
                shared_noise_across_batch=gaze_down_sweep,
            )
            if motion_normalizer is not None:
                coarse_x_lhand, coarse_x_rhand, coarse_x_obj = (
                    motion_normalizer.denormalize(
                        coarse_x_lhand, coarse_x_rhand, coarse_x_obj
                    )
                )
            batch_diagnostics = []
            post_optimization_applied = False
            if use_point_token_output:
                point_token_lhand = coarse_x_lhand
                point_token_rhand = coarse_x_rhand
                point_token_obj = coarse_x_obj

                def _restore_point_token_variant(
                    refine_mode,
                    diagnostics_target,
                    return_pre_post,
                ):
                    fit_hands = [
                        _resolve_eval_hands(
                            text_entry,
                            left_value.item(),
                            right_value.item(),
                        )
                        for text_entry, left_value, right_value in zip(
                            text, is_lhand, is_rhand
                        )
                    ]
                    fit_is_lhand = torch.tensor(
                        [
                            left_hand for left_hand, _ in fit_hands
                        ],
                        device=is_lhand.device,
                        dtype=torch.bool,
                    )
                    fit_is_rhand = torch.tensor(
                        [
                            right_hand for _, right_hand in fit_hands
                        ],
                        device=is_rhand.device,
                        dtype=torch.bool,
                    )
                    return restore_hand_point_token_outputs_with_object_pose_for_gaze2hoi(
                        point_token_lhand,
                        point_token_rhand,
                        point_token_obj,
                        obj_pc_org,
                        lhand_layer,
                        rhand_layer,
                        hand_sample_indices,
                        dataset_name,
                        obj_pc_normal=obj_pc_normal_org,
                        include_hand_object_dirvec=True,
                        mano_fit_iters=int(
                            getattr(config.gaze2hoi.exp, "mano_fit_iters", 1500)
                        ),
                        mano_fit_lr=float(
                            getattr(config.gaze2hoi.exp, "mano_fit_lr", 0.05)
                        ),
                        canonicalize_hand_targets=bool(
                            getattr(config.gaze2hoi.model, "canonicalize_point_targets", True)
                        ),
                        canonicalize_object_targets=bool(
                            getattr(config.gaze2hoi.model, "canonicalize_point_targets", True)
                        ),
                        initial_obj_pose=x_obj[:, :1],
                        object_pose_is_relative=predict_object_pose,
                        mano_fit_pose_reg_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_pose_reg_weight",
                                1e-4,
                            )
                        ),
                        mano_fit_temporal_reg_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_temporal_reg_weight",
                                1e-3,
                            )
                        ),
                        mano_fit_shape_reg_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_shape_reg_weight",
                                1e-4,
                            )
                        ),
                        mano_fit_trans_temporal_reg_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_trans_temporal_reg_weight",
                                1e-3,
                            )
                        ),
                        mano_fit_projection_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_projection_weight",
                                0.0,
                            )
                        ),
                        mano_fit_projection_obj_points=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_projection_obj_points",
                                256,
                            )
                        ),
                        mano_fit_penetration_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_penetration_weight",
                                0.0,
                            )
                        ),
                        mano_fit_penetration_obj_points=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_penetration_obj_points",
                                256,
                            )
                        ),
                        mano_fit_acceleration_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_acceleration_weight",
                                0.0,
                            )
                        ),
                        mano_fit_post_opt_steps=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_post_opt_steps",
                                0,
                            )
                        ),
                        mano_fit_use_post_opt_losses=_config_bool(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_use_post_opt_losses",
                                True,
                            ),
                            default=True,
                        ),
                        mano_fit_optimize_shape=bool(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_optimize_shape",
                                True,
                            )
                        ),
                        mano_fit_use_root_alignment=bool(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_use_root_alignment",
                                True,
                            )
                        ),
                        diagnostics=diagnostics_target,
                        object_names=object_names,
                        object_meshes=refinement_obj_meshes,
                        is_lhand=fit_is_lhand,
                        is_rhand=fit_is_rhand,
                        valid_mask_lhand=valid_mask_lhand,
                        valid_mask_rhand=valid_mask_rhand,
                        valid_mask_obj=valid_mask_obj,
                        mano_refine_steps=int(
                            getattr(config.gaze2hoi.exp, "mano_refine_steps", 100)
                        ),
                        mano_refine_lr=float(
                            getattr(config.gaze2hoi.exp, "mano_refine_lr", 5e-4)
                        ),
                        mano_refine_penetration_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_penetration_weight",
                                10.0,
                            )
                        ),
                        mano_refine_projection_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_projection_weight",
                                100.0,
                            )
                        ),
                        mano_refine_acceleration_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_acceleration_weight",
                                1000.0,
                            )
                        ),
                        mano_refine_contact_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_contact_weight",
                                100.0,
                            )
                        ),
                        mano_refine_contact_threshold=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_contact_threshold",
                                0.02,
                            )
                        ),
                        mano_refine_fallback_contact_frames=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_fallback_contact_frames",
                                10,
                            )
                        ),
                        mano_refine_transition_frames=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_transition_frames",
                                5,
                            )
                        ),
                        mano_refine_velocity_preserve_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_velocity_preserve_weight",
                                0.0,
                            )
                        ),
                        mano_refine_hand_object_penetration_weight=float(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_hand_object_penetration_weight",
                                0.0,
                            )
                        ),
                        mano_refine_mode=str(refine_mode),
                        mano_refine_max_frames=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_max_frames",
                                20,
                            )
                        ),
                        return_pre_post_opt=return_pre_post,
                        progress_prefix=f"batch {batch_idx}/{len(data_loader)}",
                    )
                restore_result = _restore_point_token_variant(
                    str(
                        getattr(
                            config.gaze2hoi.exp,
                            "mano_refine_mode",
                            "contact_ramp",
                        )
                    ),
                    batch_diagnostics,
                    save_pre_post_comparison,
                )
                if save_pre_post_comparison:
                    (
                        coarse_x_lhand,
                        coarse_x_rhand,
                        coarse_x_obj,
                        pre_post_x_lhand,
                        pre_post_x_rhand,
                        pre_post_x_obj,
                    ) = restore_result
                    for sample_idx in range(coarse_x_obj.shape[0]):
                        sample_duration = int(duration[sample_idx].item())
                        pre_post_comparison_records.append(
                            {
                                "dataset_index": int(dataset_indices[sample_idx].item()),
                                "text": str(text[sample_idx]),
                                "object_name": str(object_names[sample_idx]),
                                "duration": sample_duration,
                                "is_lhand": bool(is_lhand[sample_idx].item()),
                                "is_rhand": bool(is_rhand[sample_idx].item()),
                                "object_points": obj_pc_org[sample_idx].detach().cpu(),
                                "object_pose": pre_post_x_obj[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "pre_lhand": pre_post_x_lhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "pre_rhand": pre_post_x_rhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "post_lhand": coarse_x_lhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                                "post_rhand": coarse_x_rhand[
                                    sample_idx, :sample_duration
                                ].detach().cpu(),
                            }
                        )
                else:
                    coarse_x_lhand, coarse_x_rhand, coarse_x_obj = restore_result
                post_optimization_applied = bool(
                    config.gaze2hoi.exp.mano_fit_use_post_opt_losses
                )
                if batch_diagnostics:
                    for event in batch_diagnostics:
                        event.update(
                            {
                                "sample_start": evaluated_sample_count + 1,
                                "sample_end": evaluated_sample_count + batch_size,
                                "object_names": list(object_names),
                                "texts": list(text),
                            }
                        )
                    numerical_events.extend(batch_diagnostics)

            finite_outputs = {
                "left_hand": torch.isfinite(coarse_x_lhand)
                .reshape(coarse_x_lhand.shape[0], -1)
                .all(dim=1),
                "right_hand": torch.isfinite(coarse_x_rhand)
                .reshape(coarse_x_rhand.shape[0], -1)
                .all(dim=1),
                "object": torch.isfinite(coarse_x_obj)
                .reshape(coarse_x_obj.shape[0], -1)
                .all(dim=1),
            }
            invalid_samples = []
            for local_idx in range(coarse_x_obj.shape[0]):
                invalid_parts = [
                    name
                    for name, finite_mask in finite_outputs.items()
                    if not bool(finite_mask[local_idx].item())
                ]
                if invalid_parts:
                    invalid_samples.append(
                        {
                            "sample": evaluated_sample_count + local_idx + 1,
                            "local_index": local_idx,
                            "object_name": object_names[local_idx],
                            "text": text[local_idx],
                            "invalid_parts": invalid_parts,
                        }
                    )
            if invalid_samples:
                failure_event = {
                    "stage": "final_output_validation",
                    "reason": "non_finite_output",
                    "action": "abort_without_saving_incomplete_results",
                    "samples": invalid_samples,
                }
                numerical_events.append(failure_event)
                save_name = (
                    config.gaze2hoi.exp.save_name
                    if config.gaze2hoi.exp.save_name
                    else config.gaze2hoi.exp.name
                )
                info_path = f"{save_name}_numerical_info.json"
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "test_seed": test_seed,
                            "status": "failed",
                            "evaluated_samples_before_failure": evaluated_sample_count,
                            "events": numerical_events,
                        },
                        f,
                        indent=2,
                    )
                raise RuntimeError(
                    "Non-finite evaluation output detected. "
                    f"Details were written to {info_path}; incomplete PKL was not saved."
                )

            def _append_batch_result(
                target_save_list,
                pred_lhand,
                pred_rhand,
                pred_obj,
                variant_diagnostics,
                post_optimization_applied,
            ):
                contact_threshold = 0.02
                contact_min_keypoints = int(
                    getattr(config.gaze2hoi.exp, "eval_contact_min_keypoints", 2)
                )
                last_k = 5
                penetration_max_depth = 0.02
                lhand_contact_frame_mask = _compute_exterior_contact_frame_mask_per_hand(
                    pred_lhand,
                    lhand_layer,
                    pred_obj,
                    obj_pc_org,
                    obj_pc_normal_org,
                    dataset_name,
                    valid_mask_lhand,
                    valid_mask_obj,
                    contact_threshold=contact_threshold,
                    contact_min_keypoints=contact_min_keypoints,
                )
                rhand_contact_frame_mask = _compute_exterior_contact_frame_mask_per_hand(
                    pred_rhand,
                    rhand_layer,
                    pred_obj,
                    obj_pc_org,
                    obj_pc_normal_org,
                    dataset_name,
                    valid_mask_rhand,
                    valid_mask_obj,
                    contact_threshold=contact_threshold,
                    contact_min_keypoints=contact_min_keypoints,
                )
                tail_mask_obj = compute_tail_mask(valid_mask_obj, last_k=last_k)
                has_full_tail_obj = tail_mask_obj.sum(dim=1) == last_k

                end_contact_list = []
                eval_hands_list = []
                lhand_joints = get_hand_joints_w_tip(pred_lhand, lhand_layer)
                rhand_joints = get_hand_joints_w_tip(pred_rhand, rhand_layer)
                transf_obj_pc = get_transformed_obj_pc(
                    pred_obj, obj_pc_org, dataset_name
                )
                transf_obj_pc_normal = _rotate_obj_vectors(
                    pred_obj, obj_pc_normal_org, dataset_name
                )
                left_tail_penetration_depth = _max_tail_joint_penetration_depth_per_hand(
                    lhand_joints,
                    transf_obj_pc,
                    transf_obj_pc_normal,
                    tail_mask_obj,
                )
                right_tail_penetration_depth = _max_tail_joint_penetration_depth_per_hand(
                    rhand_joints,
                    transf_obj_pc,
                    transf_obj_pc_normal,
                    tail_mask_obj,
                )
                left_last_id_max = None
                right_last_id_max = None
                left_last_id_valid = None
                right_last_id_valid = None
                if obj_meshes is not None:
                    _, left_last_id_max, _, _, left_last_id_valid = (
                        _compute_last_frame_mesh_penetration_metrics_per_hand(
                            pred_lhand,
                            lhand_layer,
                            pred_obj,
                            obj_meshes,
                            object_names,
                            valid_mask_lhand,
                            valid_mask_obj=valid_mask_obj,
                        )
                    )
                    _, right_last_id_max, _, _, right_last_id_valid = (
                        _compute_last_frame_mesh_penetration_metrics_per_hand(
                            pred_rhand,
                            rhand_layer,
                            pred_obj,
                            obj_meshes,
                            object_names,
                            valid_mask_rhand,
                            valid_mask_obj=valid_mask_obj,
                        )
                    )
                joint_penetration_depth_list = []
                for i in range(pred_obj.shape[0]):
                    use_left_eval, use_right_eval = _resolve_eval_hands(
                        text[i], is_lhand[i].item(), is_rhand[i].item()
                    )
                    eval_hands_list.append(
                        {"left": bool(use_left_eval), "right": bool(use_right_eval)}
                    )
                    sample_contact_frame_mask = torch.zeros_like(
                        lhand_contact_frame_mask[i], dtype=torch.bool
                    )
                    if use_left_eval:
                        sample_contact_frame_mask |= lhand_contact_frame_mask[i]
                    if use_right_eval:
                        sample_contact_frame_mask |= rhand_contact_frame_mask[i]
                    sample_end_contact = bool(
                        ((sample_contact_frame_mask | (~tail_mask_obj[i])).all().item())
                        and bool(has_full_tail_obj[i].item())
                    )
                    end_contact_list.append(sample_end_contact)
                    tail_indices = torch.nonzero(
                        tail_mask_obj[i], as_tuple=False
                    ).flatten()
                    max_penetration_depth = torch.zeros(
                        last_k, dtype=pred_obj.dtype
                    )
                    used_mesh_penetration = False
                    if obj_meshes is not None and tail_indices.numel() > 0:
                        sample_depth = torch.tensor(0.0, dtype=pred_obj.dtype)
                        if (
                            use_left_eval
                            and left_last_id_max is not None
                            and bool(left_last_id_valid[i].item())
                        ):
                            sample_depth = torch.maximum(
                                sample_depth, left_last_id_max[i].detach().cpu()
                            )
                            used_mesh_penetration = True
                        if (
                            use_right_eval
                            and right_last_id_max is not None
                            and bool(right_last_id_valid[i].item())
                        ):
                            sample_depth = torch.maximum(
                                sample_depth, right_last_id_max[i].detach().cpu()
                            )
                            used_mesh_penetration = True
                    if used_mesh_penetration and tail_indices.numel() > 0:
                        max_penetration_depth[: tail_indices.numel()] = sample_depth
                    elif tail_indices.numel() > 0:
                        if use_left_eval:
                            max_penetration_depth[: tail_indices.numel()] = torch.maximum(
                                max_penetration_depth[: tail_indices.numel()],
                                left_tail_penetration_depth[
                                    i, tail_indices
                                ].detach().cpu(),
                            )
                        if use_right_eval:
                            max_penetration_depth[: tail_indices.numel()] = torch.maximum(
                                max_penetration_depth[: tail_indices.numel()],
                                right_tail_penetration_depth[
                                    i, tail_indices
                                ].detach().cpu(),
                            )
                    joint_penetration_depth_list.append(
                        [float(v) for v in max_penetration_depth.tolist()]
                    )

                target_save_list.append(
                    [
                        [
                            pred_obj[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_obj.shape[0])
                        ],
                        [
                            pred_lhand[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_lhand.shape[0])
                        ],
                        [
                            pred_rhand[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_rhand.shape[0])
                        ],
                        text,
                        [
                            {
                                "contact_success": bool(end_contact_list[i]),
                                "contact_fail": not bool(end_contact_list[i]),
                                "success": bool(
                                    end_contact_list[i]
                                    and all(
                                        v <= penetration_max_depth
                                        for v in joint_penetration_depth_list[i]
                                    )
                                ),
                                "eval_hands": eval_hands_list[i],
                                "post_optimization_applied": bool(
                                    post_optimization_applied
                                ),
                                "numerical_events_applied": bool(variant_diagnostics),
                                "numerical_events": variant_diagnostics,
                                "numerical_fallback_applied": False,
                                "numerical_fallback_events": [],
                                "joint_penetration_max_depth_threshold_m": penetration_max_depth,
                                "joint_penetration_max_depth_last_frames": joint_penetration_depth_list[
                                    i
                                ],
                            }
                            for i in range(pred_obj.shape[0])
                        ],
                        gaze.detach().cpu(),
                        cam_pose.detach().cpu(),
                        [
                            x_obj[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_obj.shape[0])
                        ],
                        [
                            x_lhand_gt[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_lhand.shape[0])
                        ],
                        [
                            x_rhand_gt[i, : int(duration[i].item())].detach().cpu()
                            for i in range(pred_rhand.shape[0])
                        ],
                        [
                            {
                                "object_name": (
                                    object_names[i] if i < len(object_names) else None
                                ),
                                "dataset_name": dataset_name,
                                "obj_pc_org": obj_pc_org[i].detach().cpu(),
                                "dataset_sample_idx": int(dataset_indices[i].item())
                                if i < dataset_indices.shape[0]
                                else None,
                                "gaze_down_ratio": (
                                    gaze_down_ratio_values[i]
                                    if gaze_down_ratio_values is not None
                                    and i < len(gaze_down_ratio_values)
                                    else None
                                ),
                                "gaze_down_shift_origin": gaze_down_shift_origin,
                                "gaze_noise_mode": gaze_noise_mode,
                                "gaze_rotation_std_deg": gaze_rotation_std_deg,
                                "gaze_translation_std_m": gaze_translation_std_m,
                                "gaze_frame_hold_interval": gaze_frame_hold_interval,
                                "test_seed": test_seed,
                            }
                            for i in range(pred_obj.shape[0])
                        ],
                    ]
                )

            _append_batch_result(
                save_list,
                coarse_x_lhand,
                coarse_x_rhand,
                coarse_x_obj,
                batch_diagnostics,
                post_optimization_applied,
            )
            evaluated_sample_count += coarse_x_obj.shape[0]

    if gaze_down_sweep and not gaze_down_processed:
        raise RuntimeError(
            f"gaze_down_sweep requested sample index {gaze_down_sample_idx}, "
            f"but it was not found in dataset split {config.dataset.data_name}."
        )

    return save_list, pre_post_comparison_records


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    task_overrides = HydraConfig.get().overrides.task
    config = edict(OmegaConf.to_object(config))
    gaze_noise_mode = _normalize_gaze_noise_mode(
        getattr(config.gaze2hoi.exp, "gaze_noise_mode", "none")
    )
    config.dataset.data_name = _resolve_test_data_name(config, task_overrides)
    print(
        "Null gaze condition: "
        f"{'ON' if bool(getattr(config.gaze2hoi.model, 'null_gaze_condition', False)) else 'OFF'}"
    )

    base_seed = int(getattr(config.gaze2hoi.exp, "seed", 0))
    num_test_seeds = int(getattr(config.gaze2hoi.exp, "num_test_seeds", 10))

    test_seeds = list(range(base_seed, base_seed + num_test_seeds))
    print(f"Running {num_test_seeds} diffusion samples with seeds: {test_seeds}")
    save_list = []
    pre_post_comparison_records = []
    save_pre_post_comparison = bool(
        getattr(config.gaze2hoi.exp, "save_pre_post_comparison", False)
    )

    for run_idx, test_seed in enumerate(test_seeds, start=1):
        print(f"\n=== Test seed {run_idx}/{num_test_seeds}: {test_seed} ===")
        run_config = deepcopy(config)
        run_config.gaze2hoi.exp.seed = test_seed
        seed_save_list, seed_comparison_records = _run_single_seed(run_config)
        save_list.extend(seed_save_list)
        pre_post_comparison_records.extend(seed_comparison_records)

    save_name = (
        config.gaze2hoi.exp.save_name
        if config.gaze2hoi.exp.save_name
        else (
            f"{config.gaze2hoi.exp.name}_gaze_noise_{gaze_noise_mode}"
            if gaze_noise_mode != "none"
            else config.gaze2hoi.exp.name
        )
    )
    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(save_list, f)

    if save_pre_post_comparison:
        comparison_path = f"{save_name}_pre_post_comparison.pkl"
        with open(comparison_path, "wb") as f:
            pickle.dump(pre_post_comparison_records, f)
        print(f"Saved pre/post comparison at {comparison_path}")

    print(f"Saved at {save_name}.pkl")


if __name__ == "__main__":
    main()
