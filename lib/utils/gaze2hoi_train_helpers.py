import os
import os.path as osp
import json
import random
import re

import numpy as np
import torch
import torch.nn as nn
import trimesh
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError
from tqdm.auto import tqdm

from lib.utils.proc import (
    _get_bps_nearest_data,
    compute_bps_contact_feature_map,
    compute_bps_contact_map,
    compute_bps_contact_feature_sequence,
    compute_bps_distance_map,
    compute_bps_distance_feature,
    compute_bps_feature,
    compute_bps_gaze_alignment_map,
    compute_bps_gaze_alignment_sequence,
    farthest_point_sample,
    build_gaze_map_from_arrow,
    build_gaze_sequence_from_arrow,
    get_contact_frame,
    normalize_bps_distance_map,
    pc_normalize,
    proc_obj_feat_final_train,
)
from lib.utils.proc_output import (
    get_pytorch3d_meshes,
    get_hand_verts,
    get_transformed_obj_pc,
    get_NN,
    get_interior,
    get_hand_obj_dist_map as get_hand_obj_dist_map_eval,
    get_hand_joints_w_tip,
)
from lib.utils.rot import (
    axis_angle_to_rot6d,
    rot6d_to_axis_angle,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
)
from lib.utils.gaze2hoi_config import move_batch_to_cuda
from lib.utils.gaze2hoi_viz import (
    apply_display_rotation,
    apply_display_rotation_to_dirs,
    contact_colorbar_spec,
    contact_values_for_display,
    display_rotation_matrix,
    flip_vertical,
    flip_xy_plane,
    gaze_colorbar_spec,
    gaze_values_for_display,
    plot_gaze_vector,
    plot_hand_skeleton,
    scatter_sparse_values_to_vertices,
    set_hand_focused_view,
    transform_dirs_to_canonical,
    transform_points_to_initial_object_canonical,
    transform_points_to_canonical,
)
from lib.networks.cvae import CTCVAE

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))


def repeat_initial_object_pose_for_gaze_condition(x_obj):
    """Repeat frame-0 object pose over time for gaze/object conditioning."""
    if not torch.is_tensor(x_obj) or x_obj.dim() < 3:
        raise ValueError(
            "Expected object pose tensor shaped (..., T, D), got "
            f"{type(x_obj).__name__} with shape "
            f"{getattr(x_obj, 'shape', None)}."
        )
    if x_obj.shape[-2] < 1:
        raise ValueError("Object pose trajectory must contain at least one frame.")
    return x_obj[..., :1, :].expand(*x_obj.shape)


def apply_null_gaze_condition(config, gaze_condition):
    """Replace a gaze-derived condition with a shape-preserving null value.

    Keeping the tensor shape unchanged preserves the gaze projection and
    attention architecture, so this ablation isolates the information carried
    by gaze rather than changing model capacity.
    """
    if bool(getattr(config.gaze2hoi.model, "null_gaze_condition", False)):
        return torch.zeros_like(gaze_condition)
    return gaze_condition


def load_hand_sample_indices_for_gaze2hoi(config, device):
    index_path = getattr(
        config.gaze2hoi.exp,
        "hand_index_path",
        "assets/part_fps_hand_index_100.npy",
    )
    if not osp.isabs(index_path):
        index_path = osp.abspath(osp.join(PROJECT_ROOT, index_path))
    indices = np.load(index_path).astype(np.int64)
    point_count = int(getattr(config.gaze2hoi.model, "point_token_count", len(indices)))
    if point_count > len(indices):
        raise ValueError(
            f"point_token_count={point_count} exceeds hand index count={len(indices)}"
        )
    return torch.as_tensor(indices[:point_count], dtype=torch.long, device=device)


def _sample_bps_basis_subset(bps_basis, count):
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if bps_basis.dim() != 2 or bps_basis.shape[1] != 3:
        raise ValueError(f"Expected bps_basis shape (K,3), got {tuple(bps_basis.shape)}")
    if bps_basis.shape[0] <= count:
        return bps_basis
    sample_idx = farthest_point_sample(bps_basis.unsqueeze(0), count)[0]
    return bps_basis.index_select(0, sample_idx)


def get_sparse_object_indices_for_gaze2hoi(
    obj_pc,
    bps_basis,
    point_count,
    object_names=None,
    part_label_map=None,
):
    if obj_pc.dim() != 3 or obj_pc.shape[-1] != 3:
        raise ValueError(f"Expected obj_pc shape (B, N, 3), got {tuple(obj_pc.shape)}")

    batch_indices = []
    for b in range(obj_pc.shape[0]):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            obj_pc.shape[1],
            obj_pc.device,
        )
        num_parts = len(part_groups)
        base_quota = int(point_count) // num_parts
        remainder = int(point_count) % num_parts
        selected_parts = []
        for part_idx, part_indices in enumerate(part_groups):
            quota = base_quota + (1 if part_idx < remainder else 0)
            part_bps_basis = _sample_bps_basis_subset(bps_basis, quota)
            _, _, nearest_idx_local = _get_bps_nearest_data(
                obj_pc[b : b + 1, part_indices, :],
                part_bps_basis,
            )
            sparse_part = part_indices[nearest_idx_local[0]]
            selected_parts.append(sparse_part)

        sparse_indices = torch.cat(selected_parts, dim=0).unsqueeze(0)
        if sparse_indices.shape[1] == int(point_count):
            batch_indices.append(sparse_indices)
        else:
            raise RuntimeError(
                f"Failed to build sparse object indices with point_count={point_count}"
            )
    return torch.cat(batch_indices, dim=0)


def gather_sparse_object_points_for_gaze2hoi(
    obj_points_all,
    obj_pc,
    bps_basis,
    point_count,
    object_names=None,
    part_label_map=None,
):
    sparse_indices = get_sparse_object_indices_for_gaze2hoi(
        obj_pc,
        bps_basis,
        point_count,
        object_names=object_names,
        part_label_map=part_label_map,
    )
    gather_idx = sparse_indices[:, None, :, None].expand(
        -1,
        obj_points_all.shape[1],
        -1,
        obj_points_all.shape[-1],
    )
    obj_points = torch.gather(obj_points_all, 2, gather_idx)
    return obj_points, sparse_indices


def build_point_token_motion_targets_for_gaze2hoi(
    x_lhand,
    x_rhand,
    x_obj,
    obj_pc,
    lhand_layer,
    rhand_layer,
    dataset_name,
    hand_indices,
    obj_pc_top_idx=None,
    include_hand_object_dirvec=True,
    canonicalize_hand_targets=True,
    canonicalize_object_targets=True,
    bps_basis=None,
    object_names=None,
    part_label_map=None,
):
    point_count = int(hand_indices.numel())
    lhand_verts_world = get_hand_verts(x_lhand, lhand_layer).index_select(2, hand_indices)
    rhand_verts_world = get_hand_verts(x_rhand, rhand_layer).index_select(2, hand_indices)
    obj_points_all = get_transformed_obj_pc(
        x_obj, obj_pc, dataset_name, obj_pc_top_idx
    )
    if bps_basis is not None:
        obj_points, _ = gather_sparse_object_points_for_gaze2hoi(
            obj_points_all,
            obj_pc,
            bps_basis,
            point_count,
            object_names=object_names,
            part_label_map=part_label_map,
        )
    else:
        obj_points = obj_points_all[:, :, :point_count, :]
    if canonicalize_object_targets:
        to_target_space = transform_points_to_initial_object_canonical
    else:
        to_target_space = transform_points_to_canonical

    if canonicalize_hand_targets:
        lhand_verts = to_target_space(lhand_verts_world, x_obj)
        rhand_verts = to_target_space(rhand_verts_world, x_obj)
        obj_points_all_for_dir = to_target_space(obj_points_all, x_obj)
    else:
        lhand_verts = lhand_verts_world
        rhand_verts = rhand_verts_world
        obj_points_all_for_dir = obj_points_all
    if canonicalize_object_targets:
        obj_points = transform_points_to_initial_object_canonical(obj_points, x_obj)
    if include_hand_object_dirvec:
        bs, nframes = x_obj.shape[:2]
        flat_obj = obj_points_all_for_dir.reshape(
            bs * nframes, obj_points_all_for_dir.shape[2], 3
        )
        flat_lhand = lhand_verts.reshape(bs * nframes, point_count, 3)
        flat_rhand = rhand_verts.reshape(bs * nframes, point_count, 3)
        lhand_nn = torch.cdist(flat_lhand, flat_obj).argmin(dim=2)
        rhand_nn = torch.cdist(flat_rhand, flat_obj).argmin(dim=2)
        lhand_nearest = torch.gather(
            flat_obj, 1, lhand_nn.unsqueeze(-1).expand(-1, -1, 3)
        ).reshape(bs, nframes, point_count, 3)
        rhand_nearest = torch.gather(
            flat_obj, 1, rhand_nn.unsqueeze(-1).expand(-1, -1, 3)
        ).reshape(bs, nframes, point_count, 3)
        lhand_verts = torch.cat([lhand_verts, lhand_nearest - lhand_verts], dim=-1)
        rhand_verts = torch.cat([rhand_verts, rhand_nearest - rhand_verts], dim=-1)

    bs, nframes = x_obj.shape[:2]
    return (
        lhand_verts.reshape(bs, nframes, -1),
        rhand_verts.reshape(bs, nframes, -1),
        obj_points.reshape(bs, nframes, point_count * 3),
    )


def canonicalize_object_pose_to_initial_object(x_obj):
    """Express every rigid object pose in its first-frame coordinate system."""
    if x_obj.shape[-1] < 9:
        raise ValueError(f"Expected object pose with at least 9 dims, got {x_obj.shape[-1]}")
    obj_trans = x_obj[..., :3]
    obj_rotmat = rot6d_to_rotmat(x_obj[..., 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], x_obj.shape[1], 3, 3
    )
    init_trans = obj_trans[:, :1]
    init_rotmat = obj_rotmat[:, :1]
    rel_trans = torch.einsum(
        "btji,btj->bti",
        init_rotmat.expand(-1, x_obj.shape[1], -1, -1),
        obj_trans - init_trans,
    )
    rel_rotmat = torch.matmul(init_rotmat.transpose(-1, -2), obj_rotmat)
    rel_rot6d = rotmat_to_rot6d(rel_rotmat.reshape(-1, 3, 3)).reshape(
        x_obj.shape[0], x_obj.shape[1], 6
    )
    return torch.cat([rel_trans, rel_rot6d], dim=-1)


def compose_relative_object_pose_with_initial(relative_obj, initial_obj):
    """Convert an initial-frame-relative 9D pose sequence back to world space."""
    if relative_obj.shape[-1] != 9 or initial_obj.shape[-1] < 9:
        raise ValueError(
            f"Expected relative (...,9) and initial (...,>=9), got "
            f"{relative_obj.shape[-1]} and {initial_obj.shape[-1]}"
        )
    rel_rotmat = rot6d_to_rotmat(relative_obj[..., 3:9].reshape(-1, 6)).reshape(
        relative_obj.shape[0], relative_obj.shape[1], 3, 3
    )
    init_pose = initial_obj[:, :1]
    init_trans = init_pose[..., :3]
    init_rotmat = rot6d_to_rotmat(init_pose[..., 3:9].reshape(-1, 6)).reshape(
        initial_obj.shape[0], 1, 3, 3
    )
    world_trans = torch.einsum(
        "btij,btj->bti",
        init_rotmat.expand(-1, relative_obj.shape[1], -1, -1),
        relative_obj[..., :3],
    ) + init_trans
    world_rotmat = torch.matmul(init_rotmat, rel_rotmat)
    world_rot6d = rotmat_to_rot6d(world_rotmat.reshape(-1, 3, 3)).reshape(
        relative_obj.shape[0], relative_obj.shape[1], 6
    )
    return torch.cat([world_trans, world_rot6d], dim=-1)


def build_hybrid_motion_targets_for_gaze2hoi(
    x_lhand,
    x_rhand,
    x_obj,
    obj_pc,
    lhand_layer,
    rhand_layer,
    dataset_name,
    hand_indices,
    obj_pc_top_idx=None,
    include_hand_object_dirvec=True,
    canonicalize_hand_targets=True,
):
    """Build hand point tokens and a first-frame-relative rigid object pose."""
    point_lhand, point_rhand, _ = build_point_token_motion_targets_for_gaze2hoi(
        x_lhand,
        x_rhand,
        x_obj,
        obj_pc,
        lhand_layer,
        rhand_layer,
        dataset_name,
        hand_indices,
        obj_pc_top_idx=obj_pc_top_idx,
        include_hand_object_dirvec=include_hand_object_dirvec,
        canonicalize_hand_targets=canonicalize_hand_targets,
        canonicalize_object_targets=True,
    )
    relative_obj = canonicalize_object_pose_to_initial_object(x_obj)
    return point_lhand, point_rhand, relative_obj


def _split_point_tokens(
    point_tokens,
    point_count,
    include_hand_object_dirvec=True,
):
    coord_dim = 6 if include_hand_object_dirvec else 3
    if point_tokens.shape[-1] != point_count * coord_dim:
        raise ValueError(
            f"Expected point token dim {point_count * coord_dim}, got {point_tokens.shape[-1]}"
        )
    point_tokens = point_tokens.reshape(
        *point_tokens.shape[:2], point_count, coord_dim
    )
    vertices = point_tokens[..., :3]
    dirvec = point_tokens[..., 3:6] if include_hand_object_dirvec else None
    return vertices, dirvec


def _uncanonicalize_points_to_world(points, x_obj):
    obj_trans = x_obj[..., :3]
    obj_rotmat = rot6d_to_rotmat(x_obj[..., 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], x_obj.shape[1], 3, 3
    )
    return torch.einsum("btij,btnj->btni", obj_rotmat, points) + obj_trans.unsqueeze(2)


def _uncanonicalize_dirs_to_world(dirs, x_obj):
    obj_rotmat = rot6d_to_rotmat(x_obj[..., 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], x_obj.shape[1], 3, 3
    )
    return torch.einsum("btij,btnj->btni", obj_rotmat, dirs)


def _uncanonicalize_points_from_initial_object(points, x_obj):
    obj_trans = x_obj[:, :1, :3].expand(-1, points.shape[1], -1)
    obj_rotmat = rot6d_to_rotmat(x_obj[:, :1, 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], 1, 3, 3
    )
    obj_rotmat = obj_rotmat.expand(-1, points.shape[1], -1, -1)
    return torch.einsum("btij,btnj->btni", obj_rotmat, points) + obj_trans.unsqueeze(2)


def _uncanonicalize_dirs_from_initial_object(dirs, x_obj):
    obj_rotmat = rot6d_to_rotmat(x_obj[:, :1, 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], 1, 3, 3
    )
    obj_rotmat = obj_rotmat.expand(-1, dirs.shape[1], -1, -1)
    return torch.einsum("btij,btnj->btni", obj_rotmat, dirs)


def _rigid_align_points_to_obj_params(source_points, target_points, dataset_name):
    bs, nframes, npts, _ = target_points.shape
    source = source_points.unsqueeze(1).expand(-1, nframes, -1, -1).reshape(-1, npts, 3)
    target = target_points.reshape(-1, npts, 3)

    source_mean = source.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.transpose(1, 2) @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rot_row = u @ vh

    det = torch.det(rot_row)
    if (det < 0).any():
        vh = vh.clone()
        vh[det < 0, -1, :] *= -1
        rot_row = u @ vh

    trans = target_mean.squeeze(1) - torch.bmm(source_mean, rot_row).squeeze(1)
    if dataset_name == "grab":
        rotmat = rot_row
    else:
        rotmat = rot_row.transpose(1, 2)
    rot6d = rotmat_to_rot6d(rotmat)
    return torch.cat([trans, rot6d], dim=1).reshape(bs, nframes, 9)


def _estimate_rigid_transform(source_points, target_points):
    bs, nframes, npts, _ = target_points.shape
    source = source_points.reshape(-1, npts, 3)
    target = target_points.reshape(-1, npts, 3)

    source_mean = source.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.transpose(1, 2) @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotmat = vh.transpose(1, 2) @ u.transpose(1, 2)

    det = torch.det(rotmat)
    if (det < 0).any():
        vh = vh.clone()
        vh[det < 0, -1, :] *= -1
        rotmat = vh.transpose(1, 2) @ u.transpose(1, 2)

    trans = target_mean.squeeze(1) - torch.bmm(source_mean, rotmat).squeeze(1)
    return rotmat.reshape(bs, nframes, 3, 3), trans.reshape(bs, nframes, 3)


def _subsample_object_points_for_projection(obj_points, max_points):
    if max_points is None or int(max_points) <= 0 or obj_points.shape[2] <= int(max_points):
        return obj_points
    sample_idx = torch.linspace(
        0,
        obj_points.shape[2] - 1,
        steps=int(max_points),
        device=obj_points.device,
    ).round().long()
    return obj_points.index_select(2, sample_idx)


def _uniform_frame_indices(nframes, max_frames, device):
    if max_frames is None or int(max_frames) <= 0 or nframes <= int(max_frames):
        return torch.arange(nframes, device=device, dtype=torch.long)
    return torch.linspace(
        0, nframes - 1, steps=int(max_frames), device=device
    ).round().long()


def _mano_params_to_trans_pose(hand_params):
    bs, nframes = hand_params.shape[:2]
    trans = hand_params[..., :3]
    pose = rot6d_to_axis_angle(hand_params[..., 3:].reshape(-1, 6)).reshape(
        bs, nframes, 16, 3
    )
    return trans, pose


def _mano_forward_vertices(trans, pose, hand_layer):
    bs, nframes = trans.shape[:2]
    flat_count = bs * nframes
    flat_pose = pose.reshape(flat_count, 48)
    flat_betas = torch.zeros(flat_count, 10, device=trans.device, dtype=trans.dtype)
    vertices = hand_layer(
        flat_betas,
        flat_pose[:, :3],
        flat_pose[:, 3:],
    ).vertices.reshape(bs, nframes, 778, 3)
    return vertices + trans.unsqueeze(2)


def _vertices_normals_from_vertices(vertices, hand_layer):
    from pytorch3d.structures import Meshes

    flat_vertices = vertices.reshape(-1, 778, 3)
    faces = torch.as_tensor(
        hand_layer.faces.astype(np.int64),
        dtype=torch.long,
        device=vertices.device,
    )
    faces = faces.unsqueeze(0).expand(flat_vertices.shape[0], -1, -1)
    mesh = Meshes(flat_vertices, faces)
    normals = mesh.verts_normals_packed().reshape(flat_vertices.shape[0], 778, 3)
    return flat_vertices, normals


def _interior_object_penetration_loss(hand_vertices, hand_layer, obj_vertices_world):
    flat_hand, flat_normals = _vertices_normals_from_vertices(hand_vertices, hand_layer)
    flat_obj = obj_vertices_world.reshape(-1, obj_vertices_world.shape[2], 3)
    nn_dist, nn_idx = get_NN(flat_obj, flat_hand)
    interior = get_interior(flat_normals, flat_hand, flat_obj, nn_idx)
    if bool(interior.any().item()):
        return nn_dist.sqrt()[interior].mean()
    return hand_vertices.new_zeros(())


def _acceleration_loss(values):
    if values.shape[1] < 3:
        return values.new_zeros(())
    accel = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
    return accel.square().mean()


def _contact_to_end_mask(
    hand_vertices,
    obj_vertices_world,
    hand_indices,
    valid_mask,
    contact_threshold,
    fallback_frames,
):
    """Select each sample from its first predicted contact through its valid end."""
    bs, nframes = hand_vertices.shape[:2]
    device = hand_vertices.device
    selected_mask = torch.zeros((bs, nframes), dtype=torch.bool, device=device)
    contact_starts = []
    fallback_used = []
    sampled_hand = hand_vertices.index_select(2, hand_indices)

    with torch.no_grad():
        for sample_idx in range(bs):
            valid_idx = torch.nonzero(valid_mask[sample_idx], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                contact_starts.append(None)
                fallback_used.append(False)
                continue
            sample_hand = sampled_hand[sample_idx].index_select(0, valid_idx)
            sample_obj = obj_vertices_world[sample_idx].index_select(0, valid_idx)
            nn_dist_sq, _ = get_NN(sample_hand, sample_obj)
            min_dist = nn_dist_sq.amin(dim=1).sqrt()
            contact_local = torch.nonzero(
                min_dist <= float(contact_threshold), as_tuple=False
            ).flatten()
            if contact_local.numel() > 0:
                start_local = int(contact_local[0].item())
                fallback = False
            else:
                start_local = max(0, int(valid_idx.numel()) - int(fallback_frames))
                fallback = True
            selected_idx = valid_idx[start_local:]
            selected_mask[sample_idx, selected_idx] = True
            contact_starts.append(int(selected_idx[0].item()))
            fallback_used.append(fallback)
    return selected_mask, contact_starts, fallback_used


def _build_refine_blend_alpha(segment_mask, transition_frames, dtype):
    """Build a per-sample cosine ramp over the start of each refine segment."""
    alpha = torch.zeros(
        (*segment_mask.shape, 1), device=segment_mask.device, dtype=dtype
    )
    transition_frames = max(1, int(transition_frames))
    for sample_idx in range(segment_mask.shape[0]):
        frame_idx = torch.nonzero(
            segment_mask[sample_idx], as_tuple=False
        ).flatten()
        if frame_idx.numel() == 0:
            continue
        ramp_count = min(transition_frames, int(frame_idx.numel()))
        if ramp_count == 1:
            ramp = torch.ones(1, device=segment_mask.device, dtype=dtype)
        else:
            phase = torch.linspace(
                0.0,
                torch.pi,
                steps=ramp_count,
                device=segment_mask.device,
                dtype=dtype,
            )
            ramp = 0.5 - 0.5 * torch.cos(phase)
        alpha[sample_idx, frame_idx[:ramp_count], 0] = ramp
        alpha[sample_idx, frame_idx[ramp_count:], 0] = 1.0
    return alpha


def _segment_acceleration_loss(values, segment_mask, context_frames=2):
    losses = []
    for sample_idx in range(values.shape[0]):
        frame_idx = torch.nonzero(segment_mask[sample_idx], as_tuple=False).flatten()
        if frame_idx.numel() == 0:
            continue
        first_frame = int(frame_idx[0].item())
        context_start = max(0, first_frame - int(context_frames))
        context_idx = torch.arange(
            context_start,
            first_frame,
            device=values.device,
            dtype=torch.long,
        )
        loss_idx = torch.cat([context_idx, frame_idx])
        if loss_idx.numel() < 3:
            continue
        segment = values[sample_idx : sample_idx + 1].index_select(1, loss_idx)
        losses.append(_acceleration_loss(segment))
    if not losses:
        return values.new_zeros(())
    return torch.stack(losses).mean()


def _segment_velocity_preservation_loss(
    values, reference_values, segment_mask, context_frames=1
):
    losses = []
    for sample_idx in range(values.shape[0]):
        frame_idx = torch.nonzero(
            segment_mask[sample_idx], as_tuple=False
        ).flatten()
        if frame_idx.numel() == 0:
            continue
        first_frame = int(frame_idx[0].item())
        context_start = max(0, first_frame - int(context_frames))
        context_idx = torch.arange(
            context_start,
            first_frame,
            device=values.device,
            dtype=torch.long,
        )
        loss_idx = torch.cat([context_idx, frame_idx])
        if loss_idx.numel() < 2:
            continue
        segment = values[sample_idx].index_select(0, loss_idx)
        reference = reference_values[sample_idx].index_select(0, loss_idx)
        velocity = segment[1:] - segment[:-1]
        reference_velocity = reference[1:] - reference[:-1]
        losses.append((velocity - reference_velocity).square().mean())
    if not losses:
        return values.new_zeros(())
    return torch.stack(losses).mean()


def _hand_to_object_penetration_loss(
    hand_vertices, obj_vertices_world, obj_normals_world, hand_indices
):
    sampled_hand = hand_vertices.index_select(2, hand_indices)
    flat_hand = sampled_hand.reshape(-1, sampled_hand.shape[2], 3)
    flat_obj = obj_vertices_world.reshape(-1, obj_vertices_world.shape[2], 3)
    flat_normals = obj_normals_world.reshape(-1, obj_normals_world.shape[2], 3)
    distance_sq, nearest_idx = get_NN(flat_hand, flat_obj)
    interior = get_interior(
        flat_normals, flat_obj, flat_hand, nearest_idx
    )
    if bool(interior.any().item()):
        return distance_sq[interior].mean()
    return hand_vertices.new_zeros(())


def _refine_mano_params_with_uniform_post_opt(
    hand_params,
    target_vertices,
    target_dirvec,
    obj_vertices_world,
    obj_normals_world,
    hand_layer,
    hand_indices,
    active_samples=None,
    num_steps=0,
    lr=5e-4,
    penetration_weight=0.0,
    projection_weight=0.0,
    acceleration_weight=0.0,
    anchor_weight=1.0,
    max_obj_points=256,
    max_refine_frames=20,
    progress_desc=None,
):
    """Original post-opt baseline using at most 20 uniform sequence frames."""
    if int(num_steps) <= 0:
        return hand_params, None
    if (
        float(penetration_weight) <= 0.0
        and float(projection_weight) <= 0.0
        and float(acceleration_weight) <= 0.0
    ):
        return hand_params, None

    bs, nframes = hand_params.shape[:2]
    device = hand_params.device
    if active_samples is None:
        active_samples = torch.ones(bs, dtype=torch.bool, device=device)
    else:
        active_samples = active_samples.to(device=device, dtype=torch.bool).view(-1)[:bs]
    if not bool(active_samples.any().item()):
        return hand_params, None

    frame_idx = _uniform_frame_indices(nframes, max_refine_frames, device)
    obj_vertices_ref = _subsample_object_points_for_projection(
        obj_vertices_world.index_select(1, frame_idx), max_obj_points
    )
    target_vertices_ref = target_vertices.index_select(1, frame_idx)
    target_dirvec_ref = (
        target_dirvec.index_select(1, frame_idx)
        if target_dirvec is not None
        else None
    )

    init_trans, init_pose = _mano_params_to_trans_pose(hand_params)
    with torch.enable_grad():
        trans = init_trans.detach().clone().requires_grad_(True)
        pose = init_pose.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([trans, pose], lr=float(lr))
        last_losses = {}

        step_iter = range(int(num_steps))
        step_iter = tqdm(
            step_iter,
            desc=progress_desc or "MANO post-opt",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in step_iter:
            optimizer.zero_grad()
            vertices = _mano_forward_vertices(trans, pose, hand_layer)
            vertices_ref = vertices.index_select(1, frame_idx)
            active_vertices = vertices_ref[active_samples]
            active_obj = obj_vertices_ref[active_samples]

            anchor_loss = (
                active_vertices.index_select(2, hand_indices)
                - target_vertices_ref[active_samples]
            ).square().mean() * float(anchor_weight)

            penetration_loss = vertices.new_zeros(())
            if float(penetration_weight) > 0.0:
                penetration_loss = _interior_object_penetration_loss(
                    active_vertices, hand_layer, active_obj
                )

            projection_loss = vertices.new_zeros(())
            if (
                float(projection_weight) > 0.0
                and target_dirvec_ref is not None
                and active_obj.numel() > 0
            ):
                sampled_vertices = active_vertices.index_select(2, hand_indices)
                projected_points = (
                    sampled_vertices + target_dirvec_ref[active_samples]
                )
                flat_projected = projected_points.reshape(
                    -1, projected_points.shape[2], 3
                )
                flat_obj = active_obj.reshape(-1, active_obj.shape[2], 3)
                projection_loss = torch.cdist(flat_projected, flat_obj).amin(
                    dim=2
                ).square().mean()

            acceleration_loss = vertices.new_zeros(())
            if float(acceleration_weight) > 0.0:
                active_count = int(active_samples.sum().item())
                acceleration_loss = _acceleration_loss(trans[active_samples])
                acceleration_loss = acceleration_loss + _acceleration_loss(
                    pose[active_samples].reshape(active_count, nframes, -1)
                )

            loss = (
                anchor_loss
                + float(penetration_weight) * penetration_loss
                + float(projection_weight) * projection_loss
                + float(acceleration_weight) * acceleration_loss
            )
            loss.backward()
            optimizer.step()
            last_losses = {
                "anchor_loss": float(anchor_loss.detach().cpu().item()),
                "penetration_loss": float(penetration_loss.detach().cpu().item()),
                "projection_loss": float(projection_loss.detach().cpu().item()),
                "acceleration_loss": float(acceleration_loss.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
            }

    last_losses.update(
        {
            "frame_selection": "uniform_max_frames",
            "selected_frame_count": int(frame_idx.numel()),
            "max_refine_frames": int(max_refine_frames),
        }
    )
    refined_rot6d = axis_angle_to_rot6d(pose.detach().reshape(-1, 3)).reshape(
        bs, nframes, 96
    )
    return torch.cat([trans.detach(), refined_rot6d], dim=2), last_losses


def _refine_mano_params_with_post_opt(
    hand_params,
    target_vertices,
    target_dirvec,
    obj_vertices_world,
    obj_normals_world,
    hand_layer,
    hand_indices,
    active_samples=None,
    valid_mask=None,
    num_steps=0,
    lr=5e-4,
    penetration_weight=0.0,
    projection_weight=0.0,
    acceleration_weight=0.0,
    contact_weight=100.0,
    contact_threshold=0.02,
    fallback_contact_frames=10,
    transition_frames=5,
    velocity_preserve_weight=0.0,
    hand_object_penetration_weight=0.0,
    anchor_weight=1.0,
    max_obj_points=256,
    progress_desc=None,
):
    if int(num_steps) <= 0:
        return hand_params, None
    if (
        float(penetration_weight) <= 0.0
        and float(projection_weight) <= 0.0
        and float(acceleration_weight) <= 0.0
        and float(contact_weight) <= 0.0
        and float(velocity_preserve_weight) <= 0.0
        and float(hand_object_penetration_weight) <= 0.0
    ):
        return hand_params, None

    bs, nframes = hand_params.shape[:2]
    device = hand_params.device
    dtype = hand_params.dtype
    if active_samples is None:
        active_samples = torch.ones(bs, dtype=torch.bool, device=device)
    else:
        active_samples = active_samples.to(device=device, dtype=torch.bool).view(-1)[:bs]
    if not bool(active_samples.any().item()):
        return hand_params, None

    init_trans, init_pose = _mano_params_to_trans_pose(hand_params)
    obj_vertices_ref = _subsample_object_points_for_projection(
        obj_vertices_world, max_obj_points
    )
    obj_normals_ref = (
        _subsample_object_points_for_projection(
            obj_normals_world, max_obj_points
        )
        if obj_normals_world is not None
        else None
    )
    if valid_mask is None:
        valid_mask = torch.ones((bs, nframes), dtype=torch.bool, device=device)
    else:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
    valid_mask = valid_mask & active_samples[:, None]
    with torch.no_grad():
        init_vertices = _mano_forward_vertices(init_trans, init_pose, hand_layer)
        refine_mask, contact_starts, fallback_used = _contact_to_end_mask(
            init_vertices,
            obj_vertices_ref,
            hand_indices,
            valid_mask,
            contact_threshold,
            fallback_contact_frames,
        )
        blend_alpha = _build_refine_blend_alpha(
            refine_mask, transition_frames, init_trans.dtype
        )
    if not bool(refine_mask.any().item()):
        return hand_params, None

    with torch.enable_grad():
        trans = init_trans.detach().clone().requires_grad_(True)
        pose = init_pose.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([trans, pose], lr=float(lr))
        last_losses = {}

        step_iter = range(int(num_steps))
        step_iter = tqdm(
            step_iter,
            desc=progress_desc or "MANO post-opt",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in step_iter:
            optimizer.zero_grad()
            effective_trans = init_trans + blend_alpha * (trans - init_trans)
            pose_alpha = blend_alpha.unsqueeze(-1)
            effective_pose = init_pose + pose_alpha * (pose - init_pose)
            vertices = _mano_forward_vertices(
                effective_trans, effective_pose, hand_layer
            )
            active_vertices = vertices[refine_mask].unsqueeze(1)
            active_obj = obj_vertices_ref[refine_mask].unsqueeze(1)
            active_obj_normals = (
                obj_normals_ref[refine_mask].unsqueeze(1)
                if obj_normals_ref is not None
                else None
            )
            active_target = target_vertices[refine_mask].unsqueeze(1)

            anchor_loss = (
                active_vertices.index_select(2, hand_indices)
                - active_target
            ).square().mean() * float(anchor_weight)

            penetration_loss = vertices.new_zeros(())
            if float(penetration_weight) > 0.0:
                penetration_loss = _interior_object_penetration_loss(
                    active_vertices,
                    hand_layer,
                    active_obj,
                )

            projection_loss = vertices.new_zeros(())
            if (
                float(projection_weight) > 0.0
                and target_dirvec is not None
                and active_obj.numel() > 0
            ):
                sampled_vertices = active_vertices.index_select(2, hand_indices)
                active_dirvec = target_dirvec[refine_mask].unsqueeze(1)
                projected_points = sampled_vertices + active_dirvec
                flat_projected = projected_points.reshape(-1, projected_points.shape[2], 3)
                flat_obj = active_obj.reshape(-1, active_obj.shape[2], 3)
                projection_dist_sq, _ = get_NN(flat_projected, flat_obj)
                projection_loss = projection_dist_sq.mean()

            contact_loss = vertices.new_zeros(())
            if float(contact_weight) > 0.0 and active_obj.numel() > 0:
                sampled_vertices = active_vertices.index_select(2, hand_indices)
                flat_sampled = sampled_vertices.reshape(-1, sampled_vertices.shape[2], 3)
                flat_obj = active_obj.reshape(-1, active_obj.shape[2], 3)
                contact_dist_sq, _ = get_NN(flat_sampled, flat_obj)
                contact_distance = contact_dist_sq.amin(dim=1).sqrt()
                contact_loss = torch.relu(
                    contact_distance - float(contact_threshold)
                ).square().mean()

            hand_object_penetration_loss = vertices.new_zeros(())
            if (
                float(hand_object_penetration_weight) > 0.0
                and active_obj_normals is not None
            ):
                hand_object_penetration_loss = (
                    _hand_to_object_penetration_loss(
                        active_vertices,
                        active_obj,
                        active_obj_normals,
                        hand_indices,
                    )
                )

            acceleration_loss = vertices.new_zeros(())
            if float(acceleration_weight) > 0.0:
                acceleration_loss = _segment_acceleration_loss(
                    effective_trans, refine_mask
                )
                acceleration_loss = acceleration_loss + _segment_acceleration_loss(
                    effective_pose.reshape(bs, nframes, -1), refine_mask
                )

            velocity_preserve_loss = vertices.new_zeros(())
            if float(velocity_preserve_weight) > 0.0:
                velocity_preserve_loss = _segment_velocity_preservation_loss(
                    effective_trans, init_trans, refine_mask
                )
                velocity_preserve_loss = (
                    velocity_preserve_loss
                    + _segment_velocity_preservation_loss(
                        effective_pose.reshape(bs, nframes, -1),
                        init_pose.reshape(bs, nframes, -1),
                        refine_mask,
                    )
                )

            loss = (
                anchor_loss
                + float(penetration_weight) * penetration_loss
                + float(projection_weight) * projection_loss
                + float(acceleration_weight) * acceleration_loss
                + float(contact_weight) * contact_loss
                + float(velocity_preserve_weight) * velocity_preserve_loss
                + float(hand_object_penetration_weight)
                * hand_object_penetration_loss
            )
            loss.backward()
            optimizer.step()
            last_losses = {
                "anchor_loss": float(anchor_loss.detach().cpu().item()),
                "penetration_loss": float(penetration_loss.detach().cpu().item()),
                "projection_loss": float(projection_loss.detach().cpu().item()),
                "contact_loss": float(contact_loss.detach().cpu().item()),
                "velocity_preserve_loss": float(
                    velocity_preserve_loss.detach().cpu().item()
                ),
                "hand_object_penetration_loss": float(
                    hand_object_penetration_loss.detach().cpu().item()
                ),
                "acceleration_loss": float(acceleration_loss.detach().cpu().item()),
                "total_loss": float(loss.detach().cpu().item()),
            }

    last_losses.update(
        {
            "frame_selection": "first_contact_to_valid_end",
            "selected_frame_counts": refine_mask.sum(dim=1).detach().cpu().tolist(),
            "contact_start_frames": contact_starts,
            "fallback_used": fallback_used,
            "fallback_contact_frames": int(fallback_contact_frames),
            "transition_frames": int(transition_frames),
            "contact_threshold": float(contact_threshold),
        }
    )

    with torch.no_grad():
        refined_trans = init_trans + blend_alpha * (trans.detach() - init_trans)
        refined_pose = init_pose + blend_alpha.unsqueeze(-1) * (
            pose.detach() - init_pose
        )
    refined_rot6d = axis_angle_to_rot6d(refined_pose.reshape(-1, 3)).reshape(
        bs, nframes, 96
    )
    refined_params = torch.cat([refined_trans, refined_rot6d], dim=2)
    return refined_params, last_losses


def _fit_mano_params_to_sampled_vertices(
    target_vertices,
    hand_layer,
    hand_indices,
    num_iters=80,
    lr=0.05,
    pose_reg_weight=1e-4,
    temporal_reg_weight=1e-3,
    shape_reg_weight=1e-4,
    trans_temporal_reg_weight=1e-3,
    projection_weight=0.0,
    target_dirvec=None,
    obj_vertices_world=None,
    optimize_shape=True,
    use_root_alignment=True,
    progress_desc=None,
    active_samples=None,
):
    bs, nframes, point_count, _ = target_vertices.shape
    device = target_vertices.device
    dtype = target_vertices.dtype

    if active_samples is not None:
        active_samples = torch.as_tensor(
            active_samples, device=device, dtype=torch.bool
        ).reshape(-1)
        if active_samples.numel() != bs:
            raise ValueError(
                "active_samples must have one entry per batch element: "
                f"got {active_samples.numel()} for batch size {bs}"
            )
        if not bool(active_samples.all()):
            # Inactive hands are omitted from rendering and evaluation.  Keep a
            # shape-compatible neutral MANO parameter sequence for them without
            # spending optimization steps on an unconstrained fit.
            restored = target_vertices.new_zeros(bs, nframes, 99)
            if not bool(active_samples.any()):
                return restored
            active_indices = torch.nonzero(active_samples, as_tuple=False).squeeze(1)
            active_dirvec = (
                target_dirvec.index_select(0, active_indices)
                if target_dirvec is not None
                else None
            )
            active_obj_vertices = (
                obj_vertices_world.index_select(0, active_indices)
                if obj_vertices_world is not None
                else None
            )
            restored.index_copy_(
                0,
                active_indices,
                _fit_mano_params_to_sampled_vertices(
                    target_vertices.index_select(0, active_indices),
                    hand_layer,
                    hand_indices,
                    num_iters=num_iters,
                    lr=lr,
                    pose_reg_weight=pose_reg_weight,
                    temporal_reg_weight=temporal_reg_weight,
                    shape_reg_weight=shape_reg_weight,
                    trans_temporal_reg_weight=trans_temporal_reg_weight,
                    projection_weight=projection_weight,
                    target_dirvec=active_dirvec,
                    obj_vertices_world=active_obj_vertices,
                    optimize_shape=optimize_shape,
                    use_root_alignment=use_root_alignment,
                    progress_desc=progress_desc,
                ),
            )
            return restored
    flat_count = bs * nframes

    with torch.enable_grad():
        with torch.no_grad():
            zero_pose = torch.zeros(flat_count, 48, device=device, dtype=dtype)
            zero_betas = torch.zeros(bs, 10, device=device, dtype=dtype)
            template_full = hand_layer(
                zero_betas.repeat_interleave(nframes, dim=0),
                zero_pose[:, :3],
                zero_pose[:, 3:],
            ).vertices.reshape(bs, nframes, 778, 3)
            template = template_full.index_select(2, hand_indices)

            init_pose = torch.zeros(bs, nframes, 16, 3, device=device, dtype=dtype)
            if use_root_alignment:
                init_rotmat, init_trans = _estimate_rigid_transform(template, target_vertices)
                init_root_aa = rot6d_to_axis_angle(
                    rotmat_to_rot6d(init_rotmat.reshape(-1, 3, 3))
                ).reshape(bs, nframes, 3)
                init_pose[:, :, 0, :] = init_root_aa
            else:
                init_trans = target_vertices.mean(dim=2) - template.mean(dim=2)

        trans = init_trans.detach().clone().requires_grad_(True)
        pose = init_pose.detach().clone().requires_grad_(True)
        shape = torch.zeros(bs, 10, device=device, dtype=dtype, requires_grad=optimize_shape)
        optim_params = [trans, pose]
        if optimize_shape:
            optim_params.append(shape)
        optimizer = torch.optim.Adam(optim_params, lr=lr)

        fit_iter = range(int(num_iters))
        fit_iter = tqdm(
            fit_iter,
            desc=progress_desc or "MANO fitting",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in fit_iter:
            optimizer.zero_grad()
            flat_pose = pose.reshape(flat_count, 48)
            flat_betas = (
                shape[:, None, :]
                .expand(bs, nframes, 10)
                .reshape(flat_count, 10)
            )
            pred_vertices = hand_layer(
                flat_betas,
                flat_pose[:, :3],
                flat_pose[:, 3:],
            ).vertices.reshape(bs, nframes, 778, 3).index_select(2, hand_indices)
            pred_vertices = pred_vertices + trans.unsqueeze(2)
            fit_loss = (pred_vertices - target_vertices).square().mean()
            reg_loss = pose[:, :, 1:].square().mean() * pose_reg_weight
            if optimize_shape:
                reg_loss = reg_loss + shape.square().mean() * shape_reg_weight
            if nframes > 1:
                reg_loss = reg_loss + (
                    (pose[:, 1:, 1:] - pose[:, :-1, 1:]).square().mean()
                    * temporal_reg_weight
                )
                reg_loss = reg_loss + (
                    (trans[:, 1:] - trans[:, :-1]).square().mean()
                    * trans_temporal_reg_weight
                )

            proj_loss = torch.zeros((), device=device, dtype=dtype)
            if (
                projection_weight > 0.0
                and target_dirvec is not None
                and obj_vertices_world is not None
            ):
                projected_points = pred_vertices + target_dirvec
                flat_projected = projected_points.reshape(flat_count, point_count, 3)
                flat_obj = obj_vertices_world.reshape(flat_count, obj_vertices_world.shape[2], 3)
                proj_loss = (
                    torch.cdist(flat_projected, flat_obj).amin(dim=2).square().mean()
                    * projection_weight
                )

            loss = fit_loss + reg_loss + proj_loss
            loss.backward()
            optimizer.step()

        rot6d = axis_angle_to_rot6d(pose.detach().reshape(-1, 3)).reshape(bs, nframes, 96)
        return torch.cat([trans.detach(), rot6d], dim=2)


def restore_point_token_outputs_for_gaze2hoi(
    point_lhand,
    point_rhand,
    point_obj,
    obj_pc,
    lhand_layer,
    rhand_layer,
    hand_indices,
    dataset_name,
    include_hand_object_dirvec=True,
    mano_fit_iters=80,
    mano_fit_lr=0.05,
    canonicalize_hand_targets=True,
    canonicalize_object_targets=False,
    mano_fit_pose_reg_weight=1e-4,
    mano_fit_temporal_reg_weight=1e-3,
    mano_fit_shape_reg_weight=1e-4,
    mano_fit_trans_temporal_reg_weight=1e-3,
    mano_fit_projection_weight=0.0,
    mano_fit_projection_obj_points=256,
    mano_fit_optimize_shape=True,
    mano_fit_use_root_alignment=True,
    bps_basis=None,
    object_names=None,
    part_label_map=None,
    progress_prefix=None,
):
    point_count = int(hand_indices.numel())
    lhand_vertices, lhand_dirvec = _split_point_tokens(
        point_lhand, point_count, include_hand_object_dirvec
    )
    rhand_vertices, rhand_dirvec = _split_point_tokens(
        point_rhand, point_count, include_hand_object_dirvec
    )
    obj_vertices = point_obj.reshape(*point_obj.shape[:2], point_count, 3)

    if bps_basis is not None:
        source_obj_points, _ = gather_sparse_object_points_for_gaze2hoi(
            obj_pc.unsqueeze(1),
            obj_pc,
            bps_basis,
            point_count,
            object_names=object_names,
            part_label_map=part_label_map,
        )
        source_obj_points = source_obj_points[:, 0]
    else:
        source_obj_points = obj_pc[:, :point_count]

    restored_obj = _rigid_align_points_to_obj_params(
        source_obj_points, obj_vertices, dataset_name
    )
    if canonicalize_hand_targets and not canonicalize_object_targets:
        lhand_vertices = _uncanonicalize_points_to_world(lhand_vertices, restored_obj)
        rhand_vertices = _uncanonicalize_points_to_world(rhand_vertices, restored_obj)
        if lhand_dirvec is not None:
            lhand_dirvec = _uncanonicalize_dirs_to_world(lhand_dirvec, restored_obj)
            rhand_dirvec = _uncanonicalize_dirs_to_world(rhand_dirvec, restored_obj)
    restored_obj_points_world = get_transformed_obj_pc(restored_obj, obj_pc, dataset_name)
    restored_obj_points_world = _subsample_object_points_for_projection(
        restored_obj_points_world,
        mano_fit_projection_obj_points,
    )
    restored_lhand = _fit_mano_params_to_sampled_vertices(
        lhand_vertices,
        lhand_layer,
        hand_indices,
        num_iters=mano_fit_iters,
        lr=mano_fit_lr,
        pose_reg_weight=mano_fit_pose_reg_weight,
        temporal_reg_weight=mano_fit_temporal_reg_weight,
        shape_reg_weight=mano_fit_shape_reg_weight,
        trans_temporal_reg_weight=mano_fit_trans_temporal_reg_weight,
        projection_weight=mano_fit_projection_weight,
        target_dirvec=lhand_dirvec,
        obj_vertices_world=restored_obj_points_world,
        optimize_shape=mano_fit_optimize_shape,
        use_root_alignment=mano_fit_use_root_alignment,
        progress_desc=(
            f"{progress_prefix} MANO fit left" if progress_prefix else "MANO fit left"
        ),
    )
    restored_rhand = _fit_mano_params_to_sampled_vertices(
        rhand_vertices,
        rhand_layer,
        hand_indices,
        num_iters=mano_fit_iters,
        lr=mano_fit_lr,
        pose_reg_weight=mano_fit_pose_reg_weight,
        temporal_reg_weight=mano_fit_temporal_reg_weight,
        shape_reg_weight=mano_fit_shape_reg_weight,
        trans_temporal_reg_weight=mano_fit_trans_temporal_reg_weight,
        projection_weight=mano_fit_projection_weight,
        target_dirvec=rhand_dirvec,
        obj_vertices_world=restored_obj_points_world,
        optimize_shape=mano_fit_optimize_shape,
        use_root_alignment=mano_fit_use_root_alignment,
        progress_desc=(
            f"{progress_prefix} MANO fit right" if progress_prefix else "MANO fit right"
        ),
    )
    return restored_lhand, restored_rhand, restored_obj


def restore_hand_point_token_outputs_with_object_pose_for_gaze2hoi(
    point_lhand,
    point_rhand,
    pred_obj,
    obj_pc,
    lhand_layer,
    rhand_layer,
    hand_indices,
    dataset_name,
    obj_pc_normal=None,
    include_hand_object_dirvec=True,
    mano_fit_iters=80,
    mano_fit_lr=0.05,
    canonicalize_hand_targets=True,
    canonicalize_object_targets=False,
    initial_obj_pose=None,
    object_pose_is_relative=False,
    mano_fit_pose_reg_weight=1e-4,
    mano_fit_temporal_reg_weight=1e-3,
    mano_fit_shape_reg_weight=1e-4,
    mano_fit_trans_temporal_reg_weight=1e-3,
    mano_fit_projection_weight=0.0,
    mano_fit_projection_obj_points=256,
    mano_fit_optimize_shape=True,
    mano_fit_use_root_alignment=True,
    mano_fit_penetration_weight=0.0,
    mano_fit_penetration_obj_points=256,
    mano_fit_acceleration_weight=0.0,
    mano_fit_post_opt_steps=0,
    mano_fit_use_post_opt_losses=True,
    diagnostics=None,
    object_names=None,
    object_meshes=None,
    bps_basis=None,
    part_label_map=None,
    is_lhand=None,
    is_rhand=None,
    valid_mask_lhand=None,
    valid_mask_rhand=None,
    valid_mask_obj=None,
    mano_refine_steps=100,
    mano_refine_lr=5e-4,
    mano_refine_penetration_weight=10.0,
    mano_refine_projection_weight=100.0,
    mano_refine_acceleration_weight=1000.0,
    mano_refine_contact_weight=100.0,
    mano_refine_contact_threshold=0.02,
    mano_refine_fallback_contact_frames=10,
    mano_refine_transition_frames=5,
    mano_refine_velocity_preserve_weight=0.0,
    mano_refine_hand_object_penetration_weight=0.0,
    mano_refine_mode="contact_ramp",
    mano_refine_max_frames=20,
    progress_prefix=None,
    return_pre_post_opt=False,
    **_unused_kwargs,
):
    point_count = int(hand_indices.numel())
    lhand_vertices, lhand_dirvec = _split_point_tokens(
        point_lhand, point_count, include_hand_object_dirvec
    )
    rhand_vertices, rhand_dirvec = _split_point_tokens(
        point_rhand, point_count, include_hand_object_dirvec
    )
    if pred_obj.shape[-1] == 9:
        relative_obj = pred_obj[..., :9]
        if object_pose_is_relative:
            if initial_obj_pose is None:
                raise ValueError(
                    "initial_obj_pose is required when object_pose_is_relative=True"
                )
            restored_obj = compose_relative_object_pose_with_initial(
                relative_obj, initial_obj_pose
            )
        else:
            restored_obj = relative_obj
    else:
        if pred_obj.shape[-1] != point_count * 3:
            raise ValueError(
                f"Expected object pose dim 9 or object point dim {point_count * 3}, "
                f"got {pred_obj.shape[-1]}"
            )
        obj_vertices = pred_obj.reshape(*pred_obj.shape[:2], point_count, 3)
        if bps_basis is not None:
            source_obj_points, _ = gather_sparse_object_points_for_gaze2hoi(
                obj_pc.unsqueeze(1),
                obj_pc,
                bps_basis,
                point_count,
                object_names=object_names,
                part_label_map=part_label_map,
            )
            source_obj_points = source_obj_points[:, 0]
        else:
            source_obj_points = obj_pc[:, :point_count]
        restored_obj = _rigid_align_points_to_obj_params(
            source_obj_points, obj_vertices, dataset_name
        )

    if canonicalize_hand_targets:
        if canonicalize_object_targets:
            hand_initial_pose = (
                initial_obj_pose
                if object_pose_is_relative and initial_obj_pose is not None
                else restored_obj
            )
            lhand_vertices = _uncanonicalize_points_from_initial_object(
                lhand_vertices, hand_initial_pose
            )
            rhand_vertices = _uncanonicalize_points_from_initial_object(
                rhand_vertices, hand_initial_pose
            )
            if lhand_dirvec is not None:
                lhand_dirvec = _uncanonicalize_dirs_from_initial_object(
                    lhand_dirvec, hand_initial_pose
                )
                rhand_dirvec = _uncanonicalize_dirs_from_initial_object(
                    rhand_dirvec, hand_initial_pose
                )
        else:
            lhand_vertices = _uncanonicalize_points_to_world(lhand_vertices, restored_obj)
            rhand_vertices = _uncanonicalize_points_to_world(rhand_vertices, restored_obj)
            if lhand_dirvec is not None:
                lhand_dirvec = _uncanonicalize_dirs_to_world(lhand_dirvec, restored_obj)
                rhand_dirvec = _uncanonicalize_dirs_to_world(rhand_dirvec, restored_obj)

    restored_obj_points_world = get_transformed_obj_pc(restored_obj, obj_pc, dataset_name)
    restored_obj_points_world = _subsample_object_points_for_projection(
        restored_obj_points_world,
        mano_fit_projection_obj_points,
    )
    restored_lhand = _fit_mano_params_to_sampled_vertices(
        lhand_vertices,
        lhand_layer,
        hand_indices,
        num_iters=mano_fit_iters,
        lr=mano_fit_lr,
        pose_reg_weight=mano_fit_pose_reg_weight,
        temporal_reg_weight=mano_fit_temporal_reg_weight,
        shape_reg_weight=mano_fit_shape_reg_weight,
        trans_temporal_reg_weight=mano_fit_trans_temporal_reg_weight,
        projection_weight=mano_fit_projection_weight,
        target_dirvec=lhand_dirvec,
        obj_vertices_world=restored_obj_points_world,
        optimize_shape=mano_fit_optimize_shape,
        use_root_alignment=mano_fit_use_root_alignment,
        progress_desc=(
            f"{progress_prefix} MANO fit left" if progress_prefix else "MANO fit left"
        ),
        active_samples=is_lhand,
    )
    restored_rhand = _fit_mano_params_to_sampled_vertices(
        rhand_vertices,
        rhand_layer,
        hand_indices,
        num_iters=mano_fit_iters,
        lr=mano_fit_lr,
        pose_reg_weight=mano_fit_pose_reg_weight,
        temporal_reg_weight=mano_fit_temporal_reg_weight,
        shape_reg_weight=mano_fit_shape_reg_weight,
        trans_temporal_reg_weight=mano_fit_trans_temporal_reg_weight,
        projection_weight=mano_fit_projection_weight,
        target_dirvec=rhand_dirvec,
        obj_vertices_world=restored_obj_points_world,
        optimize_shape=mano_fit_optimize_shape,
        use_root_alignment=mano_fit_use_root_alignment,
        progress_desc=(
            f"{progress_prefix} MANO fit right" if progress_prefix else "MANO fit right"
        ),
        active_samples=is_rhand,
    )
    pre_post_lhand = restored_lhand
    pre_post_rhand = restored_rhand
    pre_post_obj = restored_obj
    if bool(mano_fit_use_post_opt_losses):
        post_steps = int(mano_fit_post_opt_steps)
        if post_steps <= 0:
            post_steps = int(mano_refine_steps)
        pen_weight = float(mano_fit_penetration_weight)
        if pen_weight <= 0.0:
            pen_weight = float(mano_refine_penetration_weight)
        proj_weight = float(mano_refine_projection_weight)
        acc_weight = float(mano_fit_acceleration_weight)
        if acc_weight <= 0.0:
            acc_weight = float(mano_refine_acceleration_weight)
        refine_obj_points_world = _subsample_object_points_for_projection(
            get_transformed_obj_pc(restored_obj, obj_pc, dataset_name),
            mano_fit_penetration_obj_points,
        )
        refine_obj_normals_world = None
        if obj_pc_normal is not None:
            refine_obj_normals_world = _subsample_object_points_for_projection(
                _rotate_obj_vectors(restored_obj, obj_pc_normal, dataset_name),
                mano_fit_penetration_obj_points,
            )
        l_active = is_lhand if is_lhand is not None else None
        r_active = is_rhand if is_rhand is not None else None
        l_valid = valid_mask_lhand
        r_valid = valid_mask_rhand
        if valid_mask_obj is not None:
            l_valid = (
                valid_mask_obj
                if l_valid is None
                else (l_valid.to(dtype=torch.bool) & valid_mask_obj.to(dtype=torch.bool))
            )
            r_valid = (
                valid_mask_obj
                if r_valid is None
                else (r_valid.to(dtype=torch.bool) & valid_mask_obj.to(dtype=torch.bool))
            )
        refine_mode = str(mano_refine_mode).lower()
        if refine_mode not in ("contact_ramp", "uniform20_legacy"):
            raise ValueError(
                f"Unknown mano_refine_mode={mano_refine_mode!r}; expected "
                "'contact_ramp' or 'uniform20_legacy'."
            )
        refine_fn = (
            _refine_mano_params_with_uniform_post_opt
            if refine_mode == "uniform20_legacy"
            else _refine_mano_params_with_post_opt
        )
        refine_extra = (
            {"max_refine_frames": mano_refine_max_frames}
            if refine_mode == "uniform20_legacy"
            else {
                "valid_mask": l_valid,
                "contact_weight": mano_refine_contact_weight,
                "contact_threshold": mano_refine_contact_threshold,
                "fallback_contact_frames": mano_refine_fallback_contact_frames,
                "transition_frames": mano_refine_transition_frames,
                "velocity_preserve_weight": mano_refine_velocity_preserve_weight,
                "hand_object_penetration_weight": (
                    mano_refine_hand_object_penetration_weight
                ),
            }
        )
        restored_lhand, l_diag = refine_fn(
            restored_lhand,
            lhand_vertices,
            lhand_dirvec,
            refine_obj_points_world,
            refine_obj_normals_world,
            lhand_layer,
            hand_indices,
            active_samples=l_active,
            num_steps=post_steps,
            lr=mano_refine_lr,
            penetration_weight=pen_weight,
            projection_weight=proj_weight,
            acceleration_weight=acc_weight,
            max_obj_points=mano_fit_penetration_obj_points,
            progress_desc=(
                f"{progress_prefix} post-opt left"
                if progress_prefix
                else "post-opt left"
            ),
            **refine_extra,
        )
        refine_extra = (
            {"max_refine_frames": mano_refine_max_frames}
            if refine_mode == "uniform20_legacy"
            else {
                "valid_mask": r_valid,
                "contact_weight": mano_refine_contact_weight,
                "contact_threshold": mano_refine_contact_threshold,
                "fallback_contact_frames": mano_refine_fallback_contact_frames,
                "transition_frames": mano_refine_transition_frames,
                "velocity_preserve_weight": mano_refine_velocity_preserve_weight,
                "hand_object_penetration_weight": (
                    mano_refine_hand_object_penetration_weight
                ),
            }
        )
        restored_rhand, r_diag = refine_fn(
            restored_rhand,
            rhand_vertices,
            rhand_dirvec,
            refine_obj_points_world,
            refine_obj_normals_world,
            rhand_layer,
            hand_indices,
            active_samples=r_active,
            num_steps=post_steps,
            lr=mano_refine_lr,
            penetration_weight=pen_weight,
            projection_weight=proj_weight,
            acceleration_weight=acc_weight,
            max_obj_points=mano_fit_penetration_obj_points,
            progress_desc=(
                f"{progress_prefix} post-opt right"
                if progress_prefix
                else "post-opt right"
            ),
            **refine_extra,
        )
        if diagnostics is not None and (l_diag is not None or r_diag is not None):
            diagnostics.append(
                {
                    "event": "bimart_post_optimization",
                    "progress_prefix": progress_prefix,
                    "steps": int(post_steps),
                    "lr": float(mano_refine_lr),
                    "penetration_weight": float(pen_weight),
                    "projection_weight": float(proj_weight),
                    "acceleration_weight": float(acc_weight),
                    "contact_weight": float(mano_refine_contact_weight),
                    "contact_threshold": float(mano_refine_contact_threshold),
                    "fallback_contact_frames": int(
                        mano_refine_fallback_contact_frames
                    ),
                    "transition_frames": int(mano_refine_transition_frames),
                    "velocity_preserve_weight": float(
                        mano_refine_velocity_preserve_weight
                    ),
                    "hand_object_penetration_weight": float(
                        mano_refine_hand_object_penetration_weight
                    ),
                    "refine_mode": refine_mode,
                    "frame_selection": (
                        "uniform_max_frames"
                        if refine_mode == "uniform20_legacy"
                        else "first_contact_to_valid_end"
                    ),
                    "left": l_diag,
                    "right": r_diag,
                }
            )
    if bool(return_pre_post_opt):
        return (
            restored_lhand,
            restored_rhand,
            restored_obj,
            pre_post_lhand,
            pre_post_rhand,
            pre_post_obj,
        )
    return restored_lhand, restored_rhand, restored_obj


def _compute_bps_object_feature(obj_pc, bps_basis, feature_mode="displacement"):
    mode = str(feature_mode).lower()
    if mode in ("displacement", "xyz", "vector", "nearest_delta", "delta"):
        return compute_bps_feature(obj_pc, bps_basis)
    if mode in ("distance", "dist", "scalar", "norm"):
        return compute_bps_distance_feature(obj_pc, bps_basis)
    raise ValueError(
        f"Unknown object_bps_feature_mode={feature_mode!r}; "
        "expected 'displacement' or 'distance'."
    )


def _bps_feature_from_precomputed_entry(cache_entry, feature_mode):
    mode = str(feature_mode).lower()
    if mode in ("displacement", "xyz", "vector", "nearest_delta", "delta"):
        return cache_entry["displacement"]
    if mode in ("distance", "dist", "scalar", "norm"):
        return cache_entry["distance"]
    raise ValueError(
        f"Unknown object_bps_feature_mode={feature_mode!r}; "
        "expected 'displacement' or 'distance'."
    )


def get_bps_correspondence_source(config):
    source = str(
        getattr(config.gaze2hoi.exp, "bps_correspondence_source", "normalized_obj_pc")
    ).lower()
    aliases = {
        "1024": "normalized_obj_pc",
        "obj_pc": "normalized_obj_pc",
        "sampled": "normalized_obj_pc",
        "sampled_pc": "normalized_obj_pc",
        "normalized_pc": "normalized_obj_pc",
        "mesh": "object_mesh",
        "raw_mesh": "object_mesh",
        "raw_vertex": "object_mesh",
        "raw_vertices": "object_mesh",
    }
    source = aliases.get(source, source)
    if source not in ("normalized_obj_pc", "object_mesh"):
        raise ValueError(
            f"Unknown gaze2hoi.exp.bps_correspondence_source={source!r}; "
            "expected 'normalized_obj_pc' or 'object_mesh'."
        )
    return source


def should_precompute_bps_correspondence(config):
    return bool(getattr(config.gaze2hoi.exp, "precompute_bps_correspondence", True))


def _resolve_bps_correspondence_cache_path(config, source, bps_basis):
    explicit_path = getattr(config.gaze2hoi.exp, "bps_correspondence_cache_path", None)
    if explicit_path:
        if not osp.isabs(explicit_path):
            explicit_path = osp.join(PROJECT_ROOT, explicit_path)
        return explicit_path
    dataset_name = str(getattr(config.dataset, "name", "dataset"))
    bps_count = int(bps_basis.shape[0])
    return osp.join(
        PROJECT_ROOT,
        "cache",
        "bps_cache",
        f"{dataset_name}_{source}_{bps_count}.pt",
    )


def _bps_basis_signature(bps_basis):
    basis = bps_basis.detach().float().cpu()
    return {
        "shape": tuple(int(v) for v in basis.shape),
        "sum": float(basis.sum().item()),
        "abs_sum": float(basis.abs().sum().item()),
    }


def _move_bps_correspondence_cache(cache, device):
    moved = {}
    for object_name, entry in cache.items():
        moved[object_name] = {
            key: value.to(device=device) if torch.is_tensor(value) else value
            for key, value in entry.items()
        }
    return moved


def _build_bps_correspondence_entry(candidate_pc, bps_basis):
    basis_batch, nearest_points, nearest_idx = _get_bps_nearest_data(
        candidate_pc.unsqueeze(0),
        bps_basis,
    )
    displacement = nearest_points - basis_batch
    return {
        "nearest_points": nearest_points,
        "nearest_idx": nearest_idx,
        "displacement": displacement.reshape(1, -1),
        "distance": displacement.norm(dim=-1),
    }


def _load_normalized_obj_pc_candidates(config, device):
    from lib.models.object import build_object_model

    data_obj_pc_path = config.dataset.data_obj_pc_path
    if not osp.isabs(data_obj_pc_path):
        data_obj_pc_path = osp.join(PROJECT_ROOT, data_obj_pc_path)
    object_model = build_object_model(data_obj_pc_path)
    candidates = {}
    for object_name, obj_pc in object_model.obj_pcs.items():
        normalized_pc = pc_normalize(np.asarray(obj_pc, dtype=np.float32))
        candidates[str(object_name)] = torch.as_tensor(
            normalized_pc,
            dtype=torch.float32,
            device=device,
        )
    return candidates


def build_bps_correspondence_cache(
    config,
    bps_basis,
    mesh_cache=None,
    device="cuda",
):
    """
    Precompute fixed BPS-to-object correspondences and persist them on disk.

    - normalized_obj_pc: reproduces older models that used the 1024-point
      normalized object point cloud as the BPS candidate set.
    - object_mesh: uses normalized raw mesh vertices as the BPS candidate set.

    Part-wise BPS is intentionally excluded because its candidate mesh subset is
    sample-dependent.
    """
    if not should_precompute_bps_correspondence(config):
        return None
    if bool(getattr(config.gaze2hoi.exp, "use_partwise_bps", False)):
        return None

    source = get_bps_correspondence_source(config)
    cache_path = _resolve_bps_correspondence_cache_path(config, source, bps_basis)
    signature = _bps_basis_signature(bps_basis)
    metadata = {
        "source": source,
        "dataset": str(getattr(config.dataset, "name", "dataset")),
        "data_obj_pc_path": str(getattr(config.dataset, "data_obj_pc_path", "")),
        "obj_root": str(getattr(config.dataset, "obj_root", "")),
        "object_mesh_normalization": (
            "aligned_to_normalized_obj_pc" if source == "object_mesh" else "normalized_obj_pc"
        ),
        "bps": signature,
    }

    if osp.exists(cache_path):
        payload = torch.load(cache_path, map_location=device)
        if payload.get("metadata") == metadata:
            cache = _move_bps_correspondence_cache(payload["cache"], device)
            print(
                f"Loaded precomputed BPS correspondences from {cache_path} "
                f"({source}, {len(cache)} objects)."
            )
            return cache
        print(f"Ignoring stale BPS correspondence cache: {cache_path}")

    if source == "normalized_obj_pc":
        candidates = _load_normalized_obj_pc_candidates(config, device=device)
    else:
        if mesh_cache is None:
            mesh_cache = load_object_mesh_bps_cache(
                config,
                device=device,
                align_to_obj_pc_norm=True,
            )
        candidates = mesh_cache

    cache = {
        object_name: _build_bps_correspondence_entry(candidate_pc, bps_basis)
        for object_name, candidate_pc in candidates.items()
    }
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "cache": _move_bps_correspondence_cache(cache, "cpu"),
        },
        cache_path,
    )
    print(
        f"Saved precomputed BPS correspondences to {cache_path} "
        f"({source}, {len(cache)} objects, {int(bps_basis.shape[0])} queries)."
    )
    return _move_bps_correspondence_cache(cache, device)


def _resolve_part_items(part_dict):
    def _part_sort_key(item):
        key = str(item[0])
        if key.startswith("part_"):
            suffix = key.split("_")[-1]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, key)

    return sorted(part_dict.items(), key=_part_sort_key)


def resolve_partwise_bps_context(config):
    use_partwise = bool(getattr(config.gaze2hoi.exp, "use_partwise_bps", False))
    if not use_partwise:
        return None, 1

    labels_path = getattr(config.gaze2hoi.exp, "part_labels_json", "assets/label_merged_3parts.json")
    if not osp.isabs(labels_path):
        labels_path = osp.join(PROJECT_ROOT, labels_path)
    if not osp.exists(labels_path):
        raise FileNotFoundError(f"Part labels file not found: {labels_path}")

    with open(labels_path, "r") as f:
        part_label_map = json.load(f)
    if not part_label_map:
        raise ValueError(f"Part labels file is empty: {labels_path}")

    first_items = next(iter(part_label_map.values()))
    num_parts = len(_resolve_part_items(first_items))
    return part_label_map, num_parts


def _get_part_index_groups(object_name, part_label_map, num_points, device):
    if part_label_map is None:
        return [torch.arange(num_points, device=device, dtype=torch.long)]
    if object_name is None:
        raise KeyError("Object name is required for part-wise BPS")
    if object_name not in part_label_map:
        raise KeyError(f"Missing part labels for object '{object_name}'")

    part_groups = []
    for _, part_indices in _resolve_part_items(part_label_map[object_name]):
        part_indices = np.asarray(part_indices, dtype=np.int64)
        part_indices = part_indices[(part_indices >= 0) & (part_indices < num_points)]
        if part_indices.size == 0:
            raise ValueError(f"Empty part after filtering for object '{object_name}'")
        part_groups.append(torch.as_tensor(part_indices, device=device, dtype=torch.long))
    return part_groups


def _resolve_hot3d_mesh_name_map(obj_root):
    instance_path = osp.join(obj_root, "instance.json")
    if not osp.exists(instance_path):
        return {}
    with open(instance_path, "r") as f:
        instance_meta = json.load(f)
    return {
        str(value["instance_name"]): str(value["instance_id"])
        for value in instance_meta.values()
        if value.get("instance_type") == "object"
    }


def _load_object_normalization_params_from_obj_pkl(config):
    from lib.models.object import build_object_model

    data_obj_pc_path = config.dataset.data_obj_pc_path
    if not osp.isabs(data_obj_pc_path):
        data_obj_pc_path = osp.join(PROJECT_ROOT, data_obj_pc_path)
    object_model = build_object_model(data_obj_pc_path)
    norm_params = {}
    target_max_norm = 0.85
    for object_name, obj_pc in object_model.obj_pcs.items():
        pc = np.asarray(obj_pc, dtype=np.float32)
        centroid = np.mean(pc, axis=0)
        centered = pc - centroid
        scale = np.max(np.sqrt(np.sum(centered**2, axis=1)))
        effective_scale = scale / target_max_norm
        norm_params[str(object_name)] = (centroid, effective_scale)
    return norm_params


def load_object_mesh_bps_cache(config, device="cuda", align_to_obj_pc_norm=False):
    if getattr(config.dataset, "bps_obj_source", "object_mesh") != "object_mesh":
        return None

    obj_root = getattr(config.dataset, "obj_root", None)
    if obj_root is None:
        return None
    if not osp.isabs(obj_root):
        obj_root = osp.join(PROJECT_ROOT, obj_root)
    if not osp.isdir(obj_root):
        raise FileNotFoundError(f"Object mesh directory not found: {obj_root}")

    name_to_id = _resolve_hot3d_mesh_name_map(obj_root)
    norm_params = (
        _load_object_normalization_params_from_obj_pkl(config)
        if align_to_obj_pc_norm
        else None
    )
    mesh_cache = {}
    for object_name, instance_id in name_to_id.items():
        mesh_path = osp.join(obj_root, f"{instance_id}.ply")
        if not osp.exists(mesh_path):
            continue
        mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            continue
        if norm_params is not None and object_name in norm_params:
            centroid, effective_scale = norm_params[object_name]
            normalized_vertices = (vertices - centroid[None]) / effective_scale
        else:
            normalized_vertices = pc_normalize(vertices)
        mesh_cache[object_name] = torch.as_tensor(
            normalized_vertices,
            dtype=torch.float32,
            device=device,
        )
    if not mesh_cache:
        raise RuntimeError(f"No usable PLY object meshes loaded from {obj_root}")
    norm_desc = "aligned to 1024 obj_pc normalization" if align_to_obj_pc_norm else "self-normalized"
    print(f"Loaded {len(mesh_cache)} object meshes for BPS from {obj_root} ({norm_desc})")
    return mesh_cache


def _select_mesh_points_for_sample_part(
    mesh_pc,
    reference_pc,
    part_indices=None,
    bbox_margin=0.03,
):
    if part_indices is None:
        return mesh_pc

    ref_part = reference_pc[part_indices]
    if ref_part.numel() == 0:
        return mesh_pc
    part_min = ref_part.min(dim=0).values - bbox_margin
    part_max = ref_part.max(dim=0).values + bbox_margin
    in_box = ((mesh_pc >= part_min) & (mesh_pc <= part_max)).all(dim=1)
    if int(in_box.sum().item()) >= 8:
        return mesh_pc[in_box]

    nearest_dist = torch.cdist(mesh_pc.unsqueeze(0), ref_part.unsqueeze(0))[0].min(dim=1).values
    keep_count = min(mesh_pc.shape[0], max(32, ref_part.shape[0] * 4))
    keep_idx = torch.topk(nearest_dist, keep_count, largest=False).indices
    return mesh_pc[keep_idx]


def compute_bps_feature_from_mesh_cache_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    object_names,
    mesh_cache=None,
    mesh_bps_correspondence_cache=None,
    part_label_map=None,
    bbox_margin=0.03,
    feature_mode="displacement",
):
    if (
        mesh_bps_correspondence_cache is not None
        and object_names is not None
        and part_label_map is None
    ):
        cached_features = []
        all_found = True
        for object_name in object_names:
            cache_entry = mesh_bps_correspondence_cache.get(object_name)
            if cache_entry is None:
                all_found = False
                break
            cached_features.append(
                _bps_feature_from_precomputed_entry(cache_entry, feature_mode)
            )
        if all_found:
            return torch.cat(cached_features, dim=0).to(
                device=normalized_obj_pc.device,
                dtype=normalized_obj_pc.dtype,
            )

    if mesh_cache is None or object_names is None:
        return compute_bps_feature_for_gaze2hoi(
            normalized_obj_pc,
            bps_basis,
            object_names=object_names,
            part_label_map=part_label_map,
            feature_mode=feature_mode,
        )

    batch_features = []
    for b, object_name in enumerate(object_names):
        mesh_pc = mesh_cache.get(object_name)
        if mesh_pc is None:
            batch_features.append(
                compute_bps_feature_for_gaze2hoi(
                    normalized_obj_pc[b : b + 1],
                    bps_basis,
                    object_names=[object_name],
                    part_label_map=part_label_map,
                    feature_mode=feature_mode,
                )
            )
            continue

        if part_label_map is None:
            if (
                mesh_bps_correspondence_cache is not None
                and object_name in mesh_bps_correspondence_cache
            ):
                batch_features.append(
                    _bps_feature_from_precomputed_entry(
                        mesh_bps_correspondence_cache[object_name],
                        feature_mode,
                    )
                )
                continue
            batch_features.append(
                _compute_bps_object_feature(
                    mesh_pc.unsqueeze(0),
                    bps_basis,
                    feature_mode=feature_mode,
                )
            )
            continue

        part_groups = _get_part_index_groups(
            object_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_features = []
        for part_indices in part_groups:
            mesh_part_pc = _select_mesh_points_for_sample_part(
                mesh_pc,
                normalized_obj_pc[b],
                part_indices=part_indices,
                bbox_margin=bbox_margin,
            )
            part_features.append(
                _compute_bps_object_feature(
                    mesh_part_pc.unsqueeze(0),
                    bps_basis,
                    feature_mode=feature_mode,
                )
            )
        batch_features.append(torch.cat(part_features, dim=1))
    return torch.cat(batch_features, dim=0)


def compute_bps_gaze_alignment_map_from_mesh_cache_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    obj_cent,
    obj_scale,
    valid_mask=None,
    object_names=None,
    mesh_cache=None,
    part_label_map=None,
    bbox_margin=0.03,
    alignment_method="direction",
):
    if mesh_cache is None or object_names is None:
        return compute_bps_gaze_alignment_map_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            object_names=object_names,
            part_label_map=part_label_map,
            alignment_method=alignment_method,
        )

    outputs = []
    for b, object_name in enumerate(object_names):
        mesh_pc = mesh_cache.get(object_name)
        if mesh_pc is None:
            outputs.append(
                compute_bps_gaze_alignment_map_for_gaze2hoi(
                    gaze[b : b + 1],
                    x_obj[b : b + 1],
                    normalized_obj_pc[b : b + 1],
                    bps_basis,
                    obj_cent[b : b + 1],
                    obj_scale[b : b + 1],
                    valid_mask=valid_mask[b : b + 1] if valid_mask is not None else None,
                    object_names=[object_name],
                    part_label_map=part_label_map,
                    alignment_method=alignment_method,
                )
            )
            continue

        if part_label_map is None:
            outputs.append(
                compute_bps_gaze_alignment_map(
                    gaze[b : b + 1],
                    x_obj[b : b + 1],
                    mesh_pc.unsqueeze(0),
                    bps_basis,
                    obj_cent[b : b + 1],
                    obj_scale[b : b + 1],
                    valid_mask=valid_mask[b : b + 1] if valid_mask is not None else None,
                    alignment_method=alignment_method,
                )
            )
            continue

        part_groups = _get_part_index_groups(
            object_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            mesh_part_pc = _select_mesh_points_for_sample_part(
                mesh_pc,
                normalized_obj_pc[b],
                part_indices=part_indices,
                bbox_margin=bbox_margin,
            )
            part_outputs.append(
                compute_bps_gaze_alignment_map(
                    gaze[b : b + 1],
                    x_obj[b : b + 1],
                    mesh_part_pc.unsqueeze(0),
                    bps_basis,
                    obj_cent[b : b + 1],
                    obj_scale[b : b + 1],
                    valid_mask=valid_mask[b : b + 1] if valid_mask is not None else None,
                    alignment_method=alignment_method,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def _valid_mask_to_nframes(valid_mask, batch_size, frame_count, device):
    if valid_mask is None:
        return torch.full(
            (batch_size,),
            frame_count,
            device=device,
            dtype=torch.long,
        )
    return valid_mask.to(device=device).long().sum(dim=1).clamp(min=0, max=frame_count)


def compute_bps_gaze_ray_closeness_map_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    valid_mask=None,
    object_names=None,
    part_label_map=None,
    sigma=0.05,
    obj_cent=None,
    obj_scale=None,
):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    batch_size, frame_count = x_obj.shape[:2]
    nframes = _valid_mask_to_nframes(
        valid_mask,
        batch_size,
        frame_count,
        x_obj.device,
    )

    if part_label_map is None:
        _, nearest_points, _ = _get_bps_nearest_data(normalized_obj_pc, bps_basis)
        return build_gaze_map_from_arrow(
            gaze,
            nframes,
            x_obj,
            nearest_points,
            sigma=sigma,
            obj_cent=obj_cent,
            obj_scale=obj_scale,
        )

    outputs = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            _, nearest_points, _ = _get_bps_nearest_data(
                normalized_obj_pc[b : b + 1, part_indices, :],
                bps_basis,
            )
            part_outputs.append(
                build_gaze_map_from_arrow(
                    gaze[b : b + 1],
                    nframes[b : b + 1],
                    x_obj[b : b + 1],
                    nearest_points,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def compute_bps_gaze_ray_closeness_sequence_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    valid_mask=None,
    object_names=None,
    part_label_map=None,
    sigma=0.05,
    obj_cent=None,
    obj_scale=None,
):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    batch_size, frame_count = x_obj.shape[:2]
    nframes = _valid_mask_to_nframes(
        valid_mask,
        batch_size,
        frame_count,
        x_obj.device,
    )

    if part_label_map is None:
        _, nearest_points, _ = _get_bps_nearest_data(normalized_obj_pc, bps_basis)
        return build_gaze_sequence_from_arrow(
            gaze,
            nframes,
            x_obj,
            nearest_points,
            sigma=sigma,
            obj_cent=obj_cent,
            obj_scale=obj_scale,
        )

    outputs = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            _, nearest_points, _ = _get_bps_nearest_data(
                normalized_obj_pc[b : b + 1, part_indices, :],
                bps_basis,
            )
            part_outputs.append(
                build_gaze_sequence_from_arrow(
                    gaze[b : b + 1],
                    nframes[b : b + 1],
                    x_obj[b : b + 1],
                    nearest_points,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=2))
    return torch.cat(outputs, dim=0)


def compute_bps_gaze_ray_closeness_map_from_mesh_cache_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    obj_cent=None,
    obj_scale=None,
    valid_mask=None,
    object_names=None,
    mesh_cache=None,
    part_label_map=None,
    bbox_margin=0.03,
    sigma=0.05,
):
    if mesh_cache is None or object_names is None:
        return compute_bps_gaze_ray_closeness_map_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            valid_mask=valid_mask,
            object_names=object_names,
            part_label_map=part_label_map,
            sigma=sigma,
            obj_cent=obj_cent,
            obj_scale=obj_scale,
        )

    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    batch_size, frame_count = x_obj.shape[:2]
    nframes = _valid_mask_to_nframes(
        valid_mask,
        batch_size,
        frame_count,
        x_obj.device,
    )

    outputs = []
    for b, object_name in enumerate(object_names):
        mesh_pc = mesh_cache.get(object_name)
        if mesh_pc is None:
            outputs.append(
                compute_bps_gaze_ray_closeness_map_for_gaze2hoi(
                    gaze[b : b + 1],
                    x_obj[b : b + 1],
                    normalized_obj_pc[b : b + 1],
                    bps_basis,
                    valid_mask=valid_mask[b : b + 1] if valid_mask is not None else None,
                    object_names=[object_name],
                    part_label_map=part_label_map,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
            continue

        if part_label_map is None:
            _, nearest_points, _ = _get_bps_nearest_data(mesh_pc.unsqueeze(0), bps_basis)
            outputs.append(
                build_gaze_map_from_arrow(
                    gaze[b : b + 1],
                    nframes[b : b + 1],
                    x_obj[b : b + 1],
                    nearest_points,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
            continue

        part_groups = _get_part_index_groups(
            object_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            mesh_part_pc = _select_mesh_points_for_sample_part(
                mesh_pc,
                normalized_obj_pc[b],
                part_indices=part_indices,
                bbox_margin=bbox_margin,
            )
            _, nearest_points, _ = _get_bps_nearest_data(
                mesh_part_pc.unsqueeze(0),
                bps_basis,
            )
            part_outputs.append(
                build_gaze_map_from_arrow(
                    gaze[b : b + 1],
                    nframes[b : b + 1],
                    x_obj[b : b + 1],
                    nearest_points,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def compute_bps_gaze_ray_closeness_sequence_from_mesh_cache_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    obj_cent=None,
    obj_scale=None,
    valid_mask=None,
    object_names=None,
    mesh_cache=None,
    part_label_map=None,
    bbox_margin=0.03,
    sigma=0.05,
):
    if mesh_cache is None or object_names is None:
        return compute_bps_gaze_ray_closeness_sequence_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            valid_mask=valid_mask,
            object_names=object_names,
            part_label_map=part_label_map,
            sigma=sigma,
            obj_cent=obj_cent,
            obj_scale=obj_scale,
        )

    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    batch_size, frame_count = x_obj.shape[:2]
    nframes = _valid_mask_to_nframes(
        valid_mask,
        batch_size,
        frame_count,
        x_obj.device,
    )

    outputs = []
    for b, object_name in enumerate(object_names):
        mesh_pc = mesh_cache.get(object_name)
        if mesh_pc is None:
            outputs.append(
                compute_bps_gaze_ray_closeness_sequence_for_gaze2hoi(
                    gaze[b : b + 1],
                    x_obj[b : b + 1],
                    normalized_obj_pc[b : b + 1],
                    bps_basis,
                    valid_mask=valid_mask[b : b + 1] if valid_mask is not None else None,
                    object_names=[object_name],
                    part_label_map=part_label_map,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
            continue

        if part_label_map is None:
            _, nearest_points, _ = _get_bps_nearest_data(mesh_pc.unsqueeze(0), bps_basis)
            outputs.append(
                build_gaze_sequence_from_arrow(
                    gaze[b : b + 1],
                    nframes[b : b + 1],
                    x_obj[b : b + 1],
                    nearest_points,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
            continue

        part_groups = _get_part_index_groups(
            object_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            mesh_part_pc = _select_mesh_points_for_sample_part(
                mesh_pc,
                normalized_obj_pc[b],
                part_indices=part_indices,
                bbox_margin=bbox_margin,
            )
            _, nearest_points, _ = _get_bps_nearest_data(
                mesh_part_pc.unsqueeze(0),
                bps_basis,
            )
            part_outputs.append(
                build_gaze_sequence_from_arrow(
                    gaze[b : b + 1],
                    nframes[b : b + 1],
                    x_obj[b : b + 1],
                    nearest_points,
                    sigma=sigma,
                    obj_cent=obj_cent[b : b + 1] if obj_cent is not None else None,
                    obj_scale=obj_scale[b : b + 1] if obj_scale is not None else None,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=2))
    return torch.cat(outputs, dim=0)


def compute_object_gaze_ray_closeness_map_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    obj_cent,
    obj_scale,
    valid_mask=None,
    sigma=0.05,
):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    batch_size, frame_count = x_obj.shape[:2]
    nframes = _valid_mask_to_nframes(
        valid_mask,
        batch_size,
        frame_count,
        x_obj.device,
    )
    obj_points = (
        normalized_obj_pc
        * obj_scale.to(device=normalized_obj_pc.device, dtype=normalized_obj_pc.dtype).view(
            batch_size, 1, 1
        )
        + obj_cent.to(device=normalized_obj_pc.device, dtype=normalized_obj_pc.dtype).view(
            batch_size, 1, 3
        )
    )
    return build_gaze_map_from_arrow(
        gaze,
        nframes,
        x_obj,
        obj_points,
        sigma=sigma,
    )


def compute_object_gaze_ray_closeness_sequence_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    obj_cent,
    obj_scale,
    valid_mask=None,
    sigma=0.05,
):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    batch_size, frame_count = x_obj.shape[:2]
    nframes = _valid_mask_to_nframes(
        valid_mask,
        batch_size,
        frame_count,
        x_obj.device,
    )
    obj_points = (
        normalized_obj_pc
        * obj_scale.to(device=normalized_obj_pc.device, dtype=normalized_obj_pc.dtype).view(
            batch_size, 1, 1
        )
        + obj_cent.to(device=normalized_obj_pc.device, dtype=normalized_obj_pc.dtype).view(
            batch_size, 1, 3
        )
    )
    return build_gaze_sequence_from_arrow(
        gaze,
        nframes,
        x_obj,
        obj_points,
        sigma=sigma,
    )


def compute_bps_feature_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    object_names=None,
    part_label_map=None,
    feature_mode="displacement",
):
    if part_label_map is None:
        return _compute_bps_object_feature(
            normalized_obj_pc,
            bps_basis,
            feature_mode=feature_mode,
        )

    batch_size = normalized_obj_pc.shape[0]
    features = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_features = []
        for part_indices in part_groups:
            part_features.append(
                _compute_bps_object_feature(
                    normalized_obj_pc[b : b + 1, part_indices, :],
                    bps_basis,
                    feature_mode=feature_mode,
                )
            )
        features.append(torch.cat(part_features, dim=1))
    return torch.cat(features, dim=0)


def compute_bps_contact_feature_sequence_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    gt_ldist_map,
    gt_rdist_map,
    valid_mask_lhand,
    valid_mask_rhand,
    config,
    object_names=None,
    part_label_map=None,
):
    contact_mode = getattr(config.gaze2hoi.exp, "contact_feature_mode", "distance_raw")
    contact_threshold = float(getattr(config.gaze2hoi.exp, "contact_threshold", 0.01))
    distance_scale = float(getattr(config.gaze2hoi.exp, "contact_distance_scale", 0.02))

    if part_label_map is None:
        return compute_bps_contact_feature_sequence(
            normalized_obj_pc,
            bps_basis,
            gt_ldist_map,
            gt_rdist_map,
            valid_mask_lhand,
            valid_mask_rhand,
            mode=contact_mode,
            contact_threshold=contact_threshold,
            distance_scale=distance_scale,
        )

    batch_size = normalized_obj_pc.shape[0]
    outputs = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            part_outputs.append(
                compute_bps_contact_feature_sequence(
                    normalized_obj_pc[b : b + 1, part_indices, :],
                    bps_basis,
                    gt_ldist_map[b : b + 1, :, part_indices, :],
                    gt_rdist_map[b : b + 1, :, part_indices, :],
                    valid_mask_lhand[b : b + 1],
                    valid_mask_rhand[b : b + 1],
                    mode=contact_mode,
                    contact_threshold=contact_threshold,
                    distance_scale=distance_scale,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=-1))
    return torch.cat(outputs, dim=0)


def compute_bps_contact_feature_map_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    gt_ldist_map,
    gt_rdist_map,
    valid_mask_lhand,
    valid_mask_rhand,
    config,
    object_names=None,
    part_label_map=None,
):
    if part_label_map is None:
        contact_mode = getattr(config.gaze2hoi.exp, "contact_feature_mode", "distance_raw")
        contact_threshold = float(getattr(config.gaze2hoi.exp, "contact_threshold", 0.01))
        distance_scale = float(getattr(config.gaze2hoi.exp, "contact_distance_scale", 0.02))
        return compute_bps_contact_feature_map(
            normalized_obj_pc,
            bps_basis,
            gt_ldist_map,
            gt_rdist_map,
            valid_mask_lhand,
            valid_mask_rhand,
            mode=contact_mode,
            contact_threshold=contact_threshold,
            distance_scale=distance_scale,
        )

    contact_seq = compute_bps_contact_feature_sequence_for_gaze2hoi(
        normalized_obj_pc,
        bps_basis,
        gt_ldist_map,
        gt_rdist_map,
        valid_mask_lhand,
        valid_mask_rhand,
        config,
        object_names=object_names,
        part_label_map=part_label_map,
    )
    if valid_mask_lhand is not None or valid_mask_rhand is not None:
        if valid_mask_lhand is not None and valid_mask_rhand is not None:
            valid_mask = valid_mask_lhand | valid_mask_rhand
        elif valid_mask_lhand is not None:
            valid_mask = valid_mask_lhand
        else:
            valid_mask = valid_mask_rhand
        valid_mask = valid_mask.to(device=contact_seq.device, dtype=contact_seq.dtype)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (contact_seq * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
    return contact_seq.mean(dim=1)


def compute_bps_gaze_alignment_sequence_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    obj_cent,
    obj_scale,
    valid_mask=None,
    object_names=None,
    part_label_map=None,
    alignment_method="direction",
):
    if part_label_map is None:
        return compute_bps_gaze_alignment_sequence(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            alignment_method=alignment_method,
        )

    batch_size = normalized_obj_pc.shape[0]
    outputs = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            part_outputs.append(
                compute_bps_gaze_alignment_sequence(
                    gaze[b : b + 1],
                    x_obj[b : b + 1],
                    normalized_obj_pc[b : b + 1, part_indices, :],
                    bps_basis,
                    obj_cent[b : b + 1],
                    obj_scale[b : b + 1],
                    valid_mask=valid_mask[b : b + 1] if valid_mask is not None else None,
                    alignment_method=alignment_method,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=-1))
    return torch.cat(outputs, dim=0)


def compute_bps_gaze_alignment_map_for_gaze2hoi(
    gaze,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    obj_cent,
    obj_scale,
    valid_mask=None,
    object_names=None,
    part_label_map=None,
    alignment_method="direction",
):
    gaze_seq = compute_bps_gaze_alignment_sequence_for_gaze2hoi(
        gaze,
        x_obj,
        normalized_obj_pc,
        bps_basis,
        obj_cent,
        obj_scale,
        valid_mask=valid_mask,
        object_names=object_names,
        part_label_map=part_label_map,
        alignment_method=alignment_method,
    )
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=gaze_seq.device, dtype=gaze_seq.dtype)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (gaze_seq * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
    return gaze_seq.mean(dim=1)


RAW_GAZE_MAP_CONDITION_DIM = 1024
RAW_CONTACT_MAP_CONDITION_DIM = 1024
RELATIVE_GAZE_MLP_CONDITION_DIM = 128
RELATIVE_GAZE_MLP_SEQUENCE_LENGTH = 100


def _is_relative_gaze_mlp_mode(mode):
    return mode in (
        "relative_gaze_mlp",
        "object_relative_gaze_mlp",
        "raw_relative_gaze_mlp",
    )


def _is_temporal_gaze_token_mode(mode):
    mode = str(mode).lower()
    return (
        _is_direction_temporal_gaze_token_mode(mode)
        or _is_object_ray_distance_temporal_mode(mode)
        or _is_bps_ray_distance_temporal_mode(mode)
        or _is_bps_ray_alignment_tokens_temporal_mode(mode)
        or _is_bps_ray_alignment_tokens_averaged_mode(mode)
    )


def gaze_condition_requires_bps(mode):
    """Return whether a gaze condition needs BPS basis/correspondences.

    Point-cloud ray-closeness, raw gaze/contact maps, and the relative-gaze MLP
    are deliberately BPS-free.  All remaining supported modes encode gaze on
    BPS correspondences and therefore require the basis.
    """
    mode = str(mode).lower()
    if _is_object_ray_distance_temporal_mode(mode):
        return False
    if mode in (
        "method1",
        "gaze_method1",
        "ray_distance_map",
        "object_ray_distance",
        "object_point_ray_distance",
        "ray_closeness_map",
        "object_ray_closeness_map",
        "gaze_map",
        "raw_gaze_map",
        "dataset_gaze_map",
        "raw_contact_map",
        "raw_cov_map",
    ):
        return False
    if _is_relative_gaze_mlp_mode(mode):
        return False
    return True


def _is_direction_temporal_gaze_token_mode(mode):
    mode = str(mode).lower()
    return mode in (
        "direction_temporal",
        "temporal_direction",
        "direction_alignment_temporal",
        "temporal_direction_alignment",
        "bps_direction_temporal",
        "gaze_direction_temporal",
        "gage_alignment_temporal",
        "gega_alignment_temporal",
        "alignment_only_temporal",
    )


def _is_object_ray_distance_temporal_mode(mode):
    mode = str(mode).lower()
    return mode in (
        "ray_distance_temporal",
        "temporal_ray_distance",
        "ray_distance_map_temporal",
        "temporal_ray_distance_map",
        "object_ray_distance_temporal",
        "object_ray_distance_map_temporal",
        "object_point_ray_distance_temporal",
        "object_ray_closeness_temporal",
        "object_ray_closeness_map_temporal",
    )


def _is_bps_ray_distance_temporal_mode(mode):
    mode = str(mode).lower()
    return mode in (
        "bps_ray_distance_temporal",
        "temporal_bps_ray_distance",
        "bps_ray_distance_map_temporal",
        "temporal_bps_ray_distance_map",
        "bps_ray_closeness_temporal",
        "bps_ray_closeness_map_temporal",
        "bps_gaze_ray_closeness_temporal",
        "bps_gaze_ray_closeness_map_temporal",
        "gage_closeness_temporal",
        "gega_closeness_temporal",
        "closeness_only_temporal",
    )


def _is_bps_ray_alignment_tokens_temporal_mode(mode):
    mode = str(mode).lower()
    return mode in (
        "bps_ray_alignment_tokens_temporal",
        "bps_ray_alignment_token_temporal",
        "bps_ray_direction_tokens_temporal",
        "bps_ray_direction_token_temporal",
        "bps_direction_ray_tokens_temporal",
        "bps_direction_ray_token_temporal",
        "ray_alignment_tokens_temporal",
        "ray_direction_tokens_temporal",
    )


def _is_bps_ray_alignment_tokens_averaged_mode(mode):
    mode = str(mode).lower()
    return mode in (
        "bps_ray_alignment_tokens_averaged",
        "bps_ray_alignment_tokens_average",
        "averaged_ray_alignment_tokens",
        "gage_averaged_tokens",
        "gage_average_tokens",
        "gega_averaged_tokens",
        "gega_average_tokens",
    )


def configure_gaze_token_fusion_for_mode(config):
    """Make the averaged GAGE ablation select its matching model path."""
    mode = str(
        getattr(config.gaze2hoi.model, "gaze_condition_mode", "alignment")
    ).lower()
    if not _is_bps_ray_alignment_tokens_averaged_mode(mode):
        return
    append_aliases = ("append", "append_tokens", "sequence_append", "global_tokens")
    configured = str(
        getattr(config.gaze2hoi.model, "gaze_token_fusion", "cross_attn")
    ).lower()
    if configured not in append_aliases:
        print(
            "Override gaze2hoi.model.gaze_token_fusion from "
            f"{configured} to append for averaged GAGE tokens."
        )
        config.gaze2hoi.model.gaze_token_fusion = "append"


def _is_contact_condition_mode(mode):
    return mode in (
        "contact_map",
        "cov_map",
        "gt_contact_map",
        "bps_contact_map",
        "bps_cov_map",
    )


def build_gaze_alignment_temporal_mask(
    valid_mask_obj,
    ldist_map=None,
    rdist_map=None,
    temporal_scope="all",
):
    """
    Build the frame mask used to average frame-wise gaze alignment.

    `all` keeps every valid object frame. `until_contact` keeps frames from the
    start through the first GT contact frame, inclusive. Samples with no contact
    fall back to all valid frames.
    """
    scope = str(temporal_scope).lower()
    if scope in ("all", "full", "whole", "entire", "sequence"):
        return valid_mask_obj
    if scope not in ("until_contact", "contact_until", "contact", "pre_contact"):
        raise ValueError(
            f"Unknown gaze_alignment_temporal_scope={temporal_scope!r}; "
            "expected 'all' or 'until_contact'."
        )
    if ldist_map is None and rdist_map is None:
        raise ValueError(
            "gaze_alignment_temporal_scope='until_contact' requires ldist_map or rdist_map."
        )

    valid_bool = valid_mask_obj.to(dtype=torch.bool)
    contact_mask = torch.zeros_like(valid_bool)
    for dist_map in (ldist_map, rdist_map):
        if dist_map is None:
            continue
        dist = dist_map.to(device=valid_bool.device)
        while dist.dim() > 2:
            dist = torch.any(dist.abs() > 0, dim=-1)
        contact_mask = contact_mask | dist.to(dtype=torch.bool)

    contact_mask = contact_mask & valid_bool
    has_contact = contact_mask.any(dim=1)
    first_contact = torch.argmax(contact_mask.long(), dim=1)
    last_valid = valid_bool.long().sum(dim=1).clamp_min(1) - 1
    end_frame = torch.where(has_contact, first_contact, last_valid)
    frame_idx = torch.arange(valid_bool.shape[1], device=valid_bool.device).unsqueeze(0)
    return valid_bool & (frame_idx <= end_frame.unsqueeze(1))


def build_object_relative_gaze_sequence(
    gaze,
    x_obj,
    valid_mask=None,
    target_length=RELATIVE_GAZE_MLP_SEQUENCE_LENGTH,
    eps=1e-8,
):
    """
    Convert camera-frame gaze rays to object-relative rays.

    Output shape is (B, target_length, 6): [origin_obj, direction_obj].
    The transform matches the dataset object pose convention used for object
    points: p_cam = p_obj @ R^T + t.
    """
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    if gaze.dim() != 4 or gaze.shape[2:] != (2, 3):
        raise ValueError(f"Expected gaze shape (B,T,2,3), got {tuple(gaze.shape)}")
    if x_obj.dim() != 3 or x_obj.shape[-1] < 9:
        raise ValueError(f"Expected x_obj shape (B,T,9+), got {tuple(x_obj.shape)}")

    batch_size, nframes = x_obj.shape[:2]
    use_frames = min(int(target_length), int(nframes), int(gaze.shape[1]))
    device = x_obj.device
    dtype = x_obj.dtype

    gaze = gaze[:, :use_frames].to(device=device, dtype=dtype)
    x_obj = x_obj[:, :use_frames].to(device=device, dtype=dtype)

    gaze_origin = gaze[:, :, 0]
    gaze_dir = gaze[:, :, 1]
    gaze_dir = gaze_dir / gaze_dir.norm(dim=-1, keepdim=True).clamp_min(eps)

    obj_trans = x_obj[..., :3]
    obj_rotmat = rot6d_to_rotmat(x_obj[..., 3:9].reshape(-1, 6)).reshape(
        batch_size, use_frames, 3, 3
    )

    origin_obj = torch.einsum("btj,btjk->btk", gaze_origin - obj_trans, obj_rotmat)
    dir_obj = torch.einsum("btj,btjk->btk", gaze_dir, obj_rotmat)
    dir_obj = dir_obj / dir_obj.norm(dim=-1, keepdim=True).clamp_min(eps)
    relative_gaze = torch.cat([origin_obj, dir_obj], dim=-1)

    if valid_mask is not None:
        valid = valid_mask[:, :use_frames].to(device=device, dtype=torch.bool)
        relative_gaze = relative_gaze * valid.unsqueeze(-1).to(dtype=dtype)

    if use_frames < int(target_length):
        padded = relative_gaze.new_zeros(batch_size, int(target_length), 6)
        padded[:, :use_frames] = relative_gaze
        relative_gaze = padded

    return relative_gaze


class RelativeGazeMLP(nn.Module):
    def __init__(
        self,
        output_dim=RELATIVE_GAZE_MLP_CONDITION_DIM,
        sequence_length=RELATIVE_GAZE_MLP_SEQUENCE_LENGTH,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.sequence_length = int(sequence_length)
        input_dim = self.sequence_length * 6
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, relative_gaze_seq):
        if relative_gaze_seq.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected relative gaze length {self.sequence_length}, "
                f"got {relative_gaze_seq.shape[1]}"
            )
        return self.net(relative_gaze_seq.reshape(relative_gaze_seq.shape[0], -1))


def build_relative_gaze_mlp_for_gaze2hoi(config, output_dim=None):
    output_dim = int(
        output_dim
        if output_dim is not None
        else getattr(
            config.gaze2hoi.model,
            "relative_gaze_feat_dim",
            RELATIVE_GAZE_MLP_CONDITION_DIM,
        )
    )
    return RelativeGazeMLP(
        output_dim=output_dim,
        sequence_length=int(
            getattr(
                config.gaze2hoi.model,
                "relative_gaze_sequence_length",
                RELATIVE_GAZE_MLP_SEQUENCE_LENGTH,
            )
        ),
        hidden_dim=int(getattr(config.gaze2hoi.model, "relative_gaze_hidden_dim", 256)),
        dropout=float(getattr(config.gaze2hoi.model, "relative_gaze_dropout", 0.1)),
    )


def get_gaze2hoi_gaze_condition_dim(config, num_bps_parts, bps_count):
    mode = str(
        getattr(config.gaze2hoi.model, "gaze_condition_mode", "alignment")
    ).lower()
    if mode in ("alignment", "bps_alignment", "gaze_alignment"):
        return int(num_bps_parts) * int(bps_count)
    if _is_object_ray_distance_temporal_mode(mode):
        return int(
            getattr(
                config.gaze2hoi.model,
                "gaze_map_dim",
                RAW_GAZE_MAP_CONDITION_DIM,
            )
        )
    if (
        _is_direction_temporal_gaze_token_mode(mode)
        or _is_bps_ray_distance_temporal_mode(mode)
        or _is_bps_ray_alignment_tokens_temporal_mode(mode)
        or _is_bps_ray_alignment_tokens_averaged_mode(mode)
    ):
        return int(num_bps_parts) * int(bps_count)
    if mode in (
        "direction_with_origin_xyz",
        "direction_alignment_origin_xyz",
        "alignment_direction_origin_xyz",
        "bps_direction_origin_xyz",
        "gaze_direction_origin_xyz",
    ):
        return int(num_bps_parts) * int(bps_count) + 3
    if mode in (
        "alignment_combined",
        "combined_alignment",
        "origin_direction_alignment",
        "direction_origin_alignment",
        "bps_alignment_combined",
        "gaze_alignment_combined",
    ):
        return 2 * int(num_bps_parts) * int(bps_count)
    if mode in (
        "method1",
        "gaze_method1",
        "ray_distance_map",
        "object_ray_distance",
        "object_point_ray_distance",
        "ray_closeness_map",
        "object_ray_closeness_map",
    ):
        return int(
            getattr(
                config.gaze2hoi.model,
                "gaze_map_dim",
                RAW_GAZE_MAP_CONDITION_DIM,
            )
        )
    if mode in (
        "method2",
        "gaze_method2",
        "bps_ray_distance_map",
        "bps_ray_closeness",
        "bps_ray_closeness_map",
        "bps_gaze_ray_closeness",
        "bps_gaze_ray_closeness_map",
    ):
        return int(num_bps_parts) * int(bps_count)
    if mode in ("gaze_map", "raw_gaze_map", "dataset_gaze_map"):
        return int(
            getattr(
                config.gaze2hoi.model,
                "gaze_map_dim",
                RAW_GAZE_MAP_CONDITION_DIM,
            )
        )
    if _is_relative_gaze_mlp_mode(mode):
        return int(
            getattr(
                config.gaze2hoi.model,
                "relative_gaze_feat_dim",
                RELATIVE_GAZE_MLP_CONDITION_DIM,
            )
        )
    if mode in ("contact_map", "cov_map", "gt_contact_map", "bps_contact_map", "bps_cov_map"):
        return int(num_bps_parts) * int(bps_count)
    if mode in ("raw_contact_map", "raw_cov_map"):
        return int(
            getattr(
                config.gaze2hoi.model,
                "contact_map_dim",
                RAW_CONTACT_MAP_CONDITION_DIM,
            )
        )
    raise ValueError(
        f"Unknown gaze2hoi.model.gaze_condition_mode={mode!r}; "
        "expected 'alignment', 'alignment_combined', 'direction_with_origin_xyz', "
        "'ray_distance_map', 'bps_ray_distance_map', 'gaze_map', "
        "'relative_gaze_mlp', or 'contact_map'."
    )


def build_gaze_origin_xyz_condition_for_gaze2hoi(gaze, x_obj, valid_mask=None):
    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)

    if gaze.dim() == 4 and gaze.shape[2:] == (2, 3):
        gaze_origin = gaze[:, :, 0, :]
    elif gaze.dim() == 3 and gaze.shape[-1] == 3:
        gaze_origin = x_obj[..., :3]
    else:
        raise ValueError(
            f"Expected gaze shape (B,T,2,3) or (B,T,3), got {tuple(gaze.shape)}"
        )

    if valid_mask is None:
        return gaze_origin.mean(dim=1)

    valid = valid_mask.to(device=gaze_origin.device, dtype=gaze_origin.dtype)
    denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (gaze_origin * valid.unsqueeze(-1)).sum(dim=1) / denom


def build_raw_gaze_map_condition_for_gaze2hoi(
    gaze_map,
    valid_mask=None,
    target_dim=None,
):
    """
    Convert dataset gaze_map conditioning to one raw per-object-point vector.
    """
    if gaze_map.dim() == 3:
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=gaze_map.device, dtype=torch.bool)
            last_idx = valid_mask.long().sum(dim=1).sub(1).clamp_min(0)
            batch_idx = torch.arange(gaze_map.shape[0], device=gaze_map.device)
            gaze_feat = gaze_map[batch_idx, last_idx]
        else:
            gaze_feat = gaze_map[:, -1]
    elif gaze_map.dim() == 2:
        gaze_feat = gaze_map
    else:
        raise ValueError(
            f"Expected gaze_map shape (B,T,N) or (B,N), got {tuple(gaze_map.shape)}"
        )

    if target_dim is None:
        return gaze_feat

    target_dim = int(target_dim)
    feat_dim = int(gaze_feat.shape[1])
    if feat_dim == target_dim:
        return gaze_feat
    raise ValueError(
        f"Cannot adapt gaze_map dim {feat_dim} to target dim {target_dim}. "
        "Raw gaze_map conditioning keeps the dataset point dimension."
    )


def build_bps_gaze_map_condition_for_gaze2hoi(
    gaze_map,
    normalized_obj_pc,
    bps_basis,
    valid_mask=None,
    object_names=None,
    part_label_map=None,
):
    gaze_feat = build_raw_gaze_map_condition_for_gaze2hoi(
        gaze_map,
        valid_mask=valid_mask,
    )
    if gaze_feat.shape[1] != normalized_obj_pc.shape[1]:
        raise ValueError(
            f"Expected gaze_map point dim {gaze_feat.shape[1]} to match "
            f"normalized_obj_pc points {normalized_obj_pc.shape[1]} before BPS projection."
        )

    if part_label_map is None:
        _, _, nearest_idx = _get_bps_nearest_data(normalized_obj_pc, bps_basis)
        return torch.gather(gaze_feat, 1, nearest_idx)

    outputs = []
    batch_size = normalized_obj_pc.shape[0]
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            part_pc = normalized_obj_pc[b : b + 1, part_indices, :]
            _, _, nearest_part_idx = _get_bps_nearest_data(part_pc, bps_basis)
            nearest_obj_idx = part_indices[nearest_part_idx.squeeze(0)]
            part_outputs.append(gaze_feat[b : b + 1, nearest_obj_idx])
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def build_raw_contact_map_condition_for_gaze2hoi(
    contact_map,
    target_dim=None,
):
    """
    Convert dataset cov_map/contact_map conditioning to one per-object-point vector.
    """
    if contact_map.dim() == 3:
        contact_feat = contact_map.squeeze(-1)
    elif contact_map.dim() == 2:
        contact_feat = contact_map
    else:
        raise ValueError(
            f"Expected contact_map shape (B,N) or (B,N,1), got {tuple(contact_map.shape)}"
        )

    if target_dim is None:
        return contact_feat

    target_dim = int(target_dim)
    feat_dim = int(contact_feat.shape[1])
    if feat_dim == target_dim:
        return contact_feat
    raise ValueError(
        f"Cannot adapt contact_map dim {feat_dim} to target dim {target_dim}. "
        "Raw contact_map conditioning keeps the dataset point dimension."
    )


def build_bps_contact_map_condition_for_gaze2hoi(
    contact_map,
    normalized_obj_pc,
    bps_basis,
    object_names=None,
    part_label_map=None,
):
    contact_feat = build_raw_contact_map_condition_for_gaze2hoi(contact_map)
    if contact_feat.shape[1] != normalized_obj_pc.shape[1]:
        raise ValueError(
            f"Expected contact_map point dim {contact_feat.shape[1]} to match "
            f"normalized_obj_pc points {normalized_obj_pc.shape[1]} before BPS projection."
        )

    if part_label_map is None:
        _, _, nearest_idx = _get_bps_nearest_data(normalized_obj_pc, bps_basis)
        return torch.gather(contact_feat, 1, nearest_idx)

    outputs = []
    batch_size = normalized_obj_pc.shape[0]
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            part_pc = normalized_obj_pc[b : b + 1, part_indices, :]
            _, _, nearest_part_idx = _get_bps_nearest_data(part_pc, bps_basis)
            nearest_obj_idx = part_indices[nearest_part_idx.squeeze(0)]
            part_outputs.append(contact_feat[b : b + 1, nearest_obj_idx])
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def build_gaze_condition_feature_for_gaze2hoi(
    config,
    gaze,
    gaze_map,
    x_obj,
    normalized_obj_pc,
    bps_basis,
    obj_cent,
    obj_scale,
    valid_mask=None,
    object_names=None,
    mesh_cache=None,
    part_label_map=None,
    bbox_margin=0.03,
    target_dim=None,
    contact_map=None,
    relative_gaze_mlp=None,
):
    mode = str(
        getattr(config.gaze2hoi.model, "gaze_condition_mode", "alignment")
    ).lower()
    fallback_ray_sigma = float(
        getattr(config.gaze2hoi.model, "gaze_ray_distance_sigma", 0.05)
    )
    if mode in ("alignment", "bps_alignment", "gaze_alignment"):
        alignment_method = getattr(
            config.gaze2hoi.model,
            "gaze_alignment_method",
            "direction",
        )
        return compute_bps_gaze_alignment_map_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            alignment_method=alignment_method,
        )
    if _is_direction_temporal_gaze_token_mode(mode):
        align_feat = compute_bps_gaze_alignment_sequence_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            part_label_map=part_label_map,
            alignment_method="direction",
        )
        # Preserve the legacy direction_temporal checkpoint convention, while
        # the explicitly named GEGA ablation follows the paper's [0,1] score.
        if mode in (
            "gage_alignment_temporal",
            "gega_alignment_temporal",
            "alignment_only_temporal",
        ):
            align_feat = ((align_feat + 1.0) * 0.5).clamp(0.0, 1.0)
        return align_feat
    if _is_object_ray_distance_temporal_mode(mode):
        gaze_feat = compute_object_gaze_ray_closeness_sequence_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            sigma=float(
                getattr(
                    config.gaze2hoi.model,
                    "gaze_ray_distance_sigma_method1",
                    fallback_ray_sigma,
                )
            ),
        )
        if target_dim is not None and gaze_feat.shape[2] != int(target_dim):
            raise ValueError(
                f"ray_distance_temporal dim {gaze_feat.shape[2]} does not match "
                f"target dim {int(target_dim)}."
            )
        return gaze_feat
    if _is_bps_ray_distance_temporal_mode(mode):
        gaze_feat = compute_bps_gaze_ray_closeness_sequence_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            sigma=float(
                getattr(
                    config.gaze2hoi.model,
                    "gaze_ray_distance_sigma_method2",
                    fallback_ray_sigma,
                )
            ),
        )
        if target_dim is not None and gaze_feat.shape[2] != int(target_dim):
            raise ValueError(
                f"bps_ray_distance_temporal dim {gaze_feat.shape[2]} does not match "
                f"target dim {int(target_dim)}."
            )
        return gaze_feat
    if _is_bps_ray_alignment_tokens_temporal_mode(mode):
        ray_feat = compute_bps_gaze_ray_closeness_sequence_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            sigma=float(
                getattr(
                    config.gaze2hoi.model,
                    "gaze_ray_distance_sigma_method2",
                    fallback_ray_sigma,
                )
            ),
        )
        align_feat = compute_bps_gaze_alignment_sequence_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            part_label_map=part_label_map,
            alignment_method="direction",
        )
        align_feat = ((align_feat + 1.0) * 0.5).clamp(0.0, 1.0)
        if target_dim is not None:
            if ray_feat.shape[2] != int(target_dim):
                raise ValueError(
                    f"ray token dim {ray_feat.shape[2]} does not match "
                    f"target dim {int(target_dim)}."
                )
            if align_feat.shape[2] != int(target_dim):
                raise ValueError(
                    f"alignment token dim {align_feat.shape[2]} does not match "
                    f"target dim {int(target_dim)}."
                )
        return torch.stack((ray_feat, align_feat), dim=2)
    if _is_bps_ray_alignment_tokens_averaged_mode(mode):
        ray_feat = compute_bps_gaze_ray_closeness_sequence_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            sigma=float(
                getattr(
                    config.gaze2hoi.model,
                    "gaze_ray_distance_sigma_method2",
                    fallback_ray_sigma,
                )
            ),
        )
        align_feat = compute_bps_gaze_alignment_sequence_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            part_label_map=part_label_map,
            alignment_method="direction",
        )
        align_feat = ((align_feat + 1.0) * 0.5).clamp(0.0, 1.0)
        if target_dim is not None:
            if ray_feat.shape[2] != int(target_dim):
                raise ValueError(
                    f"ray token dim {ray_feat.shape[2]} does not match "
                    f"target dim {int(target_dim)}."
                )
            if align_feat.shape[2] != int(target_dim):
                raise ValueError(
                    f"alignment token dim {align_feat.shape[2]} does not match "
                    f"target dim {int(target_dim)}."
                )
        if valid_mask is None:
            ray_avg = ray_feat.mean(dim=1)
            align_avg = align_feat.mean(dim=1)
        else:
            weights = valid_mask.to(device=ray_feat.device, dtype=ray_feat.dtype)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            ray_avg = (ray_feat * weights.unsqueeze(-1)).sum(dim=1) / denom
            align_avg = (align_feat * weights.unsqueeze(-1)).sum(dim=1) / denom
        # Two sequence-level tokens: [ray closeness, directional alignment].
        return torch.stack((ray_avg, align_avg), dim=1)
    if mode in (
        "direction_with_origin_xyz",
        "direction_alignment_origin_xyz",
        "alignment_direction_origin_xyz",
        "bps_direction_origin_xyz",
        "gaze_direction_origin_xyz",
    ):
        direction_alignment = compute_bps_gaze_alignment_map_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            alignment_method="direction",
        )
        gaze_origin_xyz = build_gaze_origin_xyz_condition_for_gaze2hoi(
            gaze,
            x_obj,
            valid_mask=valid_mask,
        ).to(device=direction_alignment.device, dtype=direction_alignment.dtype)
        return torch.cat([direction_alignment, gaze_origin_xyz], dim=1)
    if mode in (
        "alignment_combined",
        "combined_alignment",
        "origin_direction_alignment",
        "direction_origin_alignment",
        "bps_alignment_combined",
        "gaze_alignment_combined",
    ):
        origin_alignment = compute_bps_gaze_alignment_map_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            alignment_method="origin",
        )
        direction_alignment = compute_bps_gaze_alignment_map_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            alignment_method="direction",
        )
        return torch.cat([origin_alignment, direction_alignment], dim=1)
    if mode in (
        "method1",
        "gaze_method1",
        "ray_distance_map",
        "object_ray_distance",
        "object_point_ray_distance",
        "ray_closeness_map",
        "object_ray_closeness_map",
    ):
        gaze_feat = compute_object_gaze_ray_closeness_map_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            sigma=float(
                getattr(
                    config.gaze2hoi.model,
                    "gaze_ray_distance_sigma_method1",
                    fallback_ray_sigma,
                )
            ),
        )
        if target_dim is not None and gaze_feat.shape[1] != int(target_dim):
            raise ValueError(
                f"ray_distance_map dim {gaze_feat.shape[1]} does not match "
                f"target dim {int(target_dim)}."
            )
        return gaze_feat
    if mode in (
        "method2",
        "gaze_method2",
        "bps_ray_distance_map",
        "bps_ray_closeness",
        "bps_ray_closeness_map",
        "bps_gaze_ray_closeness",
        "bps_gaze_ray_closeness_map",
    ):
        return compute_bps_gaze_ray_closeness_map_from_mesh_cache_for_gaze2hoi(
            gaze,
            x_obj,
            normalized_obj_pc,
            bps_basis,
            obj_cent,
            obj_scale,
            valid_mask=valid_mask,
            object_names=object_names,
            mesh_cache=mesh_cache,
            part_label_map=part_label_map,
            bbox_margin=bbox_margin,
            sigma=float(
                getattr(
                    config.gaze2hoi.model,
                    "gaze_ray_distance_sigma_method2",
                    fallback_ray_sigma,
                )
            ),
        )
    if mode in ("gaze_map", "raw_gaze_map", "dataset_gaze_map"):
        if gaze_map is None:
            raise ValueError("gaze_condition_mode='gaze_map' requires item['gaze_map'].")
        return build_raw_gaze_map_condition_for_gaze2hoi(
            gaze_map,
            valid_mask=valid_mask,
            target_dim=target_dim,
        )
    if _is_relative_gaze_mlp_mode(mode):
        if relative_gaze_mlp is None:
            raise ValueError(
                f"gaze_condition_mode={mode!r} requires a RelativeGazeMLP module."
            )
        relative_gaze = build_object_relative_gaze_sequence(
            gaze,
            x_obj,
            valid_mask=valid_mask,
            target_length=int(
                getattr(
                    config.gaze2hoi.model,
                    "relative_gaze_sequence_length",
                    RELATIVE_GAZE_MLP_SEQUENCE_LENGTH,
                )
            ),
        )
        gaze_feat = relative_gaze_mlp(relative_gaze)
        if target_dim is not None and gaze_feat.shape[1] != int(target_dim):
            raise ValueError(
                f"RelativeGazeMLP output dim {gaze_feat.shape[1]} does not match "
                f"target dim {int(target_dim)}."
            )
        return gaze_feat
    if mode in ("contact_map", "cov_map", "gt_contact_map", "bps_contact_map", "bps_cov_map"):
        if contact_map is None:
            raise ValueError(
                f"gaze_condition_mode={mode!r} requires item['cov_map'] or item['contact_map']."
            )
        return build_bps_contact_map_condition_for_gaze2hoi(
            contact_map,
            normalized_obj_pc,
            bps_basis,
            object_names=object_names,
            part_label_map=part_label_map,
        )
    if mode in ("raw_contact_map", "raw_cov_map"):
        if contact_map is None:
            raise ValueError(
                f"gaze_condition_mode={mode!r} requires item['cov_map'] or item['contact_map']."
            )
        return build_raw_contact_map_condition_for_gaze2hoi(
            contact_map,
            target_dim=target_dim,
        )
    raise ValueError(
        f"Unknown gaze2hoi.model.gaze_condition_mode={mode!r}; "
        "expected 'alignment', 'alignment_combined', 'direction_with_origin_xyz', "
        "'ray_distance_map', 'bps_ray_distance_map', 'gaze_map', "
        "'relative_gaze_mlp', or 'contact_map'."
    )


def compute_bps_distance_map_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    gt_ldist_map,
    gt_rdist_map,
    valid_mask_lhand,
    valid_mask_rhand,
    object_names=None,
    part_label_map=None,
):
    if part_label_map is None:
        return compute_bps_distance_map(
            normalized_obj_pc,
            bps_basis,
            gt_ldist_map,
            gt_rdist_map,
            valid_mask_lhand,
            valid_mask_rhand,
        )

    batch_size = normalized_obj_pc.shape[0]
    outputs = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            part_outputs.append(
                compute_bps_distance_map(
                    normalized_obj_pc[b : b + 1, part_indices, :],
                    bps_basis,
                    gt_ldist_map[b : b + 1, :, part_indices, :],
                    gt_rdist_map[b : b + 1, :, part_indices, :],
                    valid_mask_lhand[b : b + 1],
                    valid_mask_rhand[b : b + 1],
                )
            )
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def _resolve_eval_hands(text_entry, is_lhand_value, is_rhand_value):
    text_lower = str(text_entry).lower()
    if "both hands" in text_lower:
        return True, True
    if "left hand" in text_lower and "right hand" not in text_lower:
        return True, False
    if "right hand" in text_lower and "left hand" not in text_lower:
        return False, True
    return bool(is_lhand_value), bool(is_rhand_value)


def _rotate_obj_vectors(obj_params, vectors, dataset_name):
    bs, nframes = obj_params.shape[:2]
    obj_rot6d = obj_params[..., 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(bs, nframes, 3, 3)
    if dataset_name == "grab":
        return torch.einsum("btij,bki->btkj", obj_rotmat, vectors)
    return torch.einsum("btij,bkj->btki", obj_rotmat, vectors)


def _compute_iv_per_frame_per_hand(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    obj_pc_normal_org,
    dataset_name,
    valid_mask_hand,
    obj_pc_top_idx=None,
):
    bs, nframes = pred_obj.shape[:2]
    hand_verts = get_pytorch3d_meshes(pred_hand, hand_layer)[1].view(bs, nframes, 778, 3)
    transf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    )
    transf_obj_pc_normal = _rotate_obj_vectors(pred_obj, obj_pc_normal_org, dataset_name)

    flat_hand = hand_verts.reshape(bs * nframes, 778, 3)
    flat_obj = transf_obj_pc.reshape(bs * nframes, -1, 3)
    flat_obj_normal = transf_obj_pc_normal.reshape(bs * nframes, -1, 3)
    flat_valid = valid_mask_hand.reshape(bs * nframes)
    iv_frame = torch.zeros(bs * nframes, device=pred_obj.device, dtype=pred_obj.dtype)

    for idx in range(bs * nframes):
        if not bool(flat_valid[idx].item()):
            continue
        hand_xyz = flat_hand[idx : idx + 1]
        obj_xyz = flat_obj[idx : idx + 1]
        obj_normal = flat_obj_normal[idx : idx + 1]
        hand_nn_dist, hand_nn_idx = get_NN(hand_xyz, obj_xyz)
        hand_interior = get_interior(obj_normal, obj_xyz, hand_xyz, hand_nn_idx)[0]
        if int(hand_interior.sum().item()) < 4:
            continue
        interior_hand_vertices = hand_xyz[0, hand_interior].detach().cpu().numpy()
        try:
            iv_frame[idx] = float(ConvexHull(interior_hand_vertices).volume * 1e6)
        except QhullError:
            continue

    return iv_frame.view(bs, nframes)


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
    obj_pc_top_idx=None,
):
    bs, nframes = pred_obj.shape[:2]
    hand_joints = get_hand_joints_w_tip(pred_hand, hand_layer)
    transf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    )

    flat_hand_joints = hand_joints.reshape(bs * nframes, -1, 3)
    flat_obj_pc = transf_obj_pc.reshape(bs * nframes, -1, 3)

    nn_dist, _ = get_NN(flat_hand_joints, flat_obj_pc)
    nn_dist = nn_dist.sqrt().view(bs, nframes, -1)

    close_contact_mask = nn_dist < contact_threshold
    return (
        close_contact_mask.sum(dim=2) >= contact_min_keypoints
    ) & torch.logical_and(valid_mask_obj, valid_mask_hand)


def _compute_contact_joint_mask_per_hand(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    dataset_name,
    valid_mask_hand,
    valid_mask_obj,
    contact_threshold=0.01,
    obj_pc_top_idx=None,
):
    bs, nframes = pred_obj.shape[:2]
    hand_joints = get_hand_joints_w_tip(pred_hand, hand_layer)
    transf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    )

    flat_hand_joints = hand_joints.reshape(bs * nframes, -1, 3)
    flat_obj_pc = transf_obj_pc.reshape(bs * nframes, -1, 3)
    nn_dist, _ = get_NN(flat_hand_joints, flat_obj_pc)
    joint_contact_mask = nn_dist.sqrt().view(bs, nframes, -1) < contact_threshold
    valid_frame_mask = torch.logical_and(valid_mask_obj, valid_mask_hand).unsqueeze(-1)
    return joint_contact_mask & valid_frame_mask


def _sanitize_failure_name(value, default="sample"):
    if value is None:
        return default
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = value.strip("._")
    return value[:80] or default


def _plot_failure_hand(ax, joints, joint_contact_mask, base_color, label_prefix):
    joints = np.asarray(joints, dtype=np.float32)
    plot_hand_skeleton(ax, joints, base_color)
    joint_contact_mask = np.asarray(joint_contact_mask, dtype=bool)
    contact_points = joints[joint_contact_mask]
    non_contact_points = joints[~joint_contact_mask]
    if non_contact_points.size > 0:
        ax.scatter(
            non_contact_points[:, 0],
            non_contact_points[:, 1],
            non_contact_points[:, 2],
            s=28,
            c="#ff9f1c",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.95,
        )
    if contact_points.size > 0:
        ax.scatter(
            contact_points[:, 0],
            contact_points[:, 1],
            contact_points[:, 2],
            s=34,
            c="#00d084",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.98,
        )


def _save_failure_visualization(
    output_path,
    sample_title,
    object_name,
    obj_points,
    lhand_joints,
    rhand_joints,
    lhand_contact_joint_mask,
    rhand_contact_joint_mask,
    use_left_eval,
    use_right_eval,
    end_contact_ok,
    penetration_ok,
    penetration_mm,
    penetration_threshold_mm,
    penetration_segments=None,
):
    obj_points = np.asarray(obj_points, dtype=np.float32)
    lhand_joints = None if lhand_joints is None else np.asarray(lhand_joints, dtype=np.float32)
    rhand_joints = None if rhand_joints is None else np.asarray(rhand_joints, dtype=np.float32)
    reason_labels = []
    if not end_contact_ok:
        reason_labels.append("CR fail")
    if not penetration_ok:
        reason_labels.append(f"Penetration fail ({penetration_mm:.1f} mm > {penetration_threshold_mm:.1f} mm)")
    if not reason_labels:
        reason_labels.append("Failure")

    fig = plt.figure(figsize=(13, 6))
    axes = [
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    ]
    view_specs = [(18.0, -60.0), (18.0, 35.0)]
    for ax, (elev, azim) in zip(axes, view_specs):
        ax.scatter(
            obj_points[:, 0],
            obj_points[:, 1],
            obj_points[:, 2],
            s=7,
            c="#b8bec7",
            edgecolors="none",
            alpha=0.22,
        )
        if use_left_eval and lhand_joints is not None:
            _plot_failure_hand(
                ax,
                lhand_joints,
                lhand_contact_joint_mask,
                "#1f77b4",
                "Left",
            )
        elif lhand_joints is not None:
            plot_hand_skeleton(ax, lhand_joints, "#9bb9d4")
        if use_right_eval and rhand_joints is not None:
            _plot_failure_hand(
                ax,
                rhand_joints,
                rhand_contact_joint_mask,
                "#2ca02c",
                "Right",
            )
        elif rhand_joints is not None:
            plot_hand_skeleton(ax, rhand_joints, "#a8cfaa")
        if penetration_segments:
            for segment in penetration_segments:
                hand_point = np.asarray(segment["hand_point"], dtype=np.float32)
                obj_point = np.asarray(segment["object_point"], dtype=np.float32)
                line_mid = 0.5 * (hand_point + obj_point)
                line_color = "#d62728" if segment.get("hand") == "left" else "#c2185b"
                end_color = "#7f1d1d" if segment.get("hand") == "left" else "#6a1b9a"
                ax.plot(
                    [hand_point[0], obj_point[0]],
                    [hand_point[1], obj_point[1]],
                    [hand_point[2], obj_point[2]],
                    color=line_color,
                    linewidth=2.6,
                    alpha=0.98,
                )
                ax.scatter(
                    [hand_point[0], obj_point[0]],
                    [hand_point[1], obj_point[1]],
                    [hand_point[2], obj_point[2]],
                    s=44,
                    c=[line_color, end_color],
                    edgecolors="black",
                    linewidths=0.45,
                    alpha=0.98,
                )
                ax.text(
                    line_mid[0],
                    line_mid[1],
                    line_mid[2],
                    f"{segment['hand']} {segment['depth_mm']:.1f} mm",
                    color=end_color,
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=line_color, alpha=0.85),
                )
        set_hand_focused_view(ax, obj_points, lhand_joints, rhand_joints)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    legend_handles = [
        Line2D([0], [0], color="#00d084", marker="o", linestyle="None", markersize=7, label="Contact joint"),
        Line2D([0], [0], color="#ff9f1c", marker="o", linestyle="None", markersize=7, label="Non-contact joint"),
    ]
    if penetration_segments:
        for segment in penetration_segments:
            line_color = "#d62728" if segment.get("hand") == "left" else "#c2185b"
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=line_color,
                    linewidth=2.6,
                    label=f"{segment['hand'].capitalize()} penetration: {segment['depth_mm']:.1f} mm",
                )
            )
    legend_handles.append(
        Line2D([0], [0], color="none", label="Reasons: " + " | ".join(reason_labels))
    )
    axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        frameon=True,
        handlelength=1.8,
    )
    fig.suptitle(
        f"Eval failure | object={object_name} | text={sample_title}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _count_tail_joint_penetrations_per_hand(
    hand_joints,
    obj_pc,
    obj_pc_normal,
    tail_mask,
    penetration_threshold=-0.01,
):
    batch_size, nframes = tail_mask.shape
    counts = torch.zeros(
        batch_size,
        nframes,
        device=hand_joints.device,
        dtype=torch.long,
    )
    if not tail_mask.any():
        return counts

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
    joint_min_dist = joint_nn_dist.sqrt()
    joint_signed_dist = torch.where(
        joint_is_interior,
        -joint_min_dist,
        joint_min_dist,
    )

    tail_counts = (joint_signed_dist < penetration_threshold).sum(dim=1)
    counts[tail_mask] = tail_counts
    return counts


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


def _max_tail_joint_penetration_detail_per_hand(
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
    hand_points = [[None for _ in range(nframes)] for _ in range(batch_size)]
    object_points = [[None for _ in range(nframes)] for _ in range(batch_size)]
    if not tail_mask.any():
        return depths, hand_points, object_points

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
    max_depth, max_joint_idx = joint_depth.max(dim=1)
    depths[tail_mask] = max_depth

    tail_flat_indices = torch.nonzero(flat_tail_mask, as_tuple=False).flatten()
    for tail_row, flat_idx in enumerate(tail_flat_indices.tolist()):
        if float(max_depth[tail_row].item()) <= 0.0:
            continue
        joint_idx = int(max_joint_idx[tail_row].item())
        obj_idx = int(joint_nn_idx[tail_row, joint_idx].item())
        batch_idx = flat_idx // nframes
        frame_idx = flat_idx % nframes
        hand_points[batch_idx][frame_idx] = (
            tail_hand_joints[tail_row, joint_idx].detach().cpu().numpy().astype(np.float32).tolist()
        )
        object_points[batch_idx][frame_idx] = (
            tail_obj_pc[tail_row, obj_idx].detach().cpu().numpy().astype(np.float32).tolist()
        )

    return depths, hand_points, object_points


def _max_tail_joint_penetration_detail_per_hand_from_mesh(
    hand_joints,
    pred_obj,
    obj_meshes,
    object_names,
    tail_mask,
):
    batch_size, nframes = tail_mask.shape
    depths = torch.zeros(
        batch_size,
        nframes,
        device=hand_joints.device,
        dtype=hand_joints.dtype,
    )
    hand_points = [[None for _ in range(nframes)] for _ in range(batch_size)]
    object_points = [[None for _ in range(nframes)] for _ in range(batch_size)]
    if obj_meshes is None or not object_names or not tail_mask.any():
        return depths, hand_points, object_points

    for b in range(batch_size):
        object_name = object_names[b] if b < len(object_names) else None
        obj_mesh = obj_meshes.get(object_name)
        if obj_mesh is None:
            continue
        tail_frames = torch.nonzero(tail_mask[b], as_tuple=False).flatten().tolist()
        for frame_idx in tail_frames:
            hand_world = hand_joints[b, frame_idx].detach().cpu().numpy().astype(np.float64)
            pose = pred_obj[b, frame_idx, :9].detach().cpu()
            obj_trans = pose[:3].numpy().astype(np.float64)
            obj_rot = (
                rot6d_to_rotmat(pose[3:9].reshape(1, 6))
                .reshape(3, 3)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            hand_local = np.einsum("ni,ij->nj", hand_world - obj_trans[None, :], obj_rot)

            try:
                signed_distance = np.asarray(
                    trimesh.proximity.signed_distance(obj_mesh, hand_local),
                    dtype=np.float64,
                )
                contains_mask = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
                inside_mask = np.isfinite(signed_distance) & (
                    contains_mask | (signed_distance > 0.0)
                )
                if not inside_mask.any():
                    continue

                inside_points = hand_local[inside_mask]
                closest_local, _, _ = trimesh.proximity.closest_point(
                    obj_mesh, inside_points
                )
                if closest_local is None:
                    continue
                closest_local = np.asarray(closest_local, dtype=np.float64)
                depth_values = np.linalg.norm(inside_points - closest_local, axis=1)
                if depth_values.size == 0:
                    continue

                max_local_idx = int(np.argmax(depth_values))
                inside_indices = np.flatnonzero(inside_mask)
                joint_idx = int(inside_indices[max_local_idx])
                closest_world = np.einsum(
                    "ni,ji->nj",
                    closest_local[max_local_idx : max_local_idx + 1],
                    obj_rot,
                ) + obj_trans[None, :]

                depths[b, frame_idx] = float(depth_values[max_local_idx])
                hand_points[b][frame_idx] = (
                    hand_world[joint_idx].astype(np.float32).tolist()
                )
                object_points[b][frame_idx] = (
                    closest_world[0].astype(np.float32).tolist()
                )
            except Exception:
                continue

    return depths, hand_points, object_points


def build_contact_feature_map_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    gt_ldist_map,
    gt_rdist_map,
    valid_mask_lhand,
    valid_mask_rhand,
    config,
    object_names=None,
    part_label_map=None,
    bps_distance_stats=None,
):
    contact_mode = getattr(config.gaze2hoi.exp, "contact_feature_mode", "distance_raw")
    contact_map = compute_bps_contact_feature_map_for_gaze2hoi(
        normalized_obj_pc,
        bps_basis,
        gt_ldist_map,
        gt_rdist_map,
        valid_mask_lhand,
        valid_mask_rhand,
        config,
        object_names=object_names,
        part_label_map=part_label_map,
    )
    if contact_mode == "distance_raw" and bps_distance_stats is not None:
        contact_map = normalize_bps_distance_map(
            contact_map,
            bps_distance_stats,
            use_per_dim=getattr(
                config.gaze2hoi.exp, "bps_distance_use_per_dim", True
            ),
        )
    return contact_map


def configure_bps_cvae_condition_dim(config, num_bps_parts, bps_count):
    bps_feat_dim = int(num_bps_parts) * int(bps_count) * 3
    config.contact.cond_dim = (
        bps_feat_dim
        + (1 if bool(getattr(config.contact, "use_scale", True)) else 0)
        + (1 if bool(getattr(config.contact, "use_gaze_map", True)) else 0)
    )
    return config.contact.cond_dim


def build_bps_cvae_condition(config, obj_scale, obj_feat, npts, gaze_map=None):
    batch_size = obj_feat.shape[0]
    cond_parts = []
    if bool(getattr(config.contact, "use_scale", True)):
        cond_parts.append(obj_scale.view(batch_size, 1, 1).expand(-1, npts, -1))
    cond_parts.append(obj_feat.unsqueeze(1).expand(-1, npts, -1))
    if bool(getattr(config.contact, "use_gaze_map", True)) and gaze_map is not None:
        if gaze_map.dim() == 3:
            cond_parts.append(gaze_map)
        else:
            cond_parts.append(gaze_map.unsqueeze(-1))
    return torch.cat(cond_parts, dim=2)


def dense_contact_to_bps_contact_map_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    dense_contact_map,
    object_names=None,
    part_label_map=None,
):
    dense_contact_map = dense_contact_map.to(
        device=normalized_obj_pc.device,
        dtype=normalized_obj_pc.dtype,
    )
    if dense_contact_map.dim() == 3:
        dense_contact_map = dense_contact_map.squeeze(-1)

    if part_label_map is None:
        return compute_bps_contact_map(
            normalized_obj_pc,
            bps_basis,
            dense_contact_map,
            dense_contact_map,
        )

    batch_size = normalized_obj_pc.shape[0]
    outputs = []
    for b in range(batch_size):
        sample_name = object_names[b] if object_names is not None and b < len(object_names) else None
        part_groups = _get_part_index_groups(
            sample_name,
            part_label_map,
            normalized_obj_pc.shape[1],
            normalized_obj_pc.device,
        )
        part_outputs = []
        for part_indices in part_groups:
            part_contact = dense_contact_map[b : b + 1, part_indices]
            part_outputs.append(
                compute_bps_contact_map(
                    normalized_obj_pc[b : b + 1, part_indices, :],
                    bps_basis,
                    part_contact,
                    part_contact,
                )
            )
        outputs.append(torch.cat(part_outputs, dim=1))
    return torch.cat(outputs, dim=0)


def predict_bps_contact_map_for_gaze2hoi(
    contact_estimator,
    config,
    obj_scale,
    obj_feat,
    normalized_obj_pc,
    bps_basis,
    gaze_map=None,
    object_names=None,
    part_label_map=None,
):
    npts = normalized_obj_pc.shape[1]
    condition = build_bps_cvae_condition(
        config,
        obj_scale,
        obj_feat,
        npts,
        gaze_map=gaze_map,
    )
    dense_contact = contact_estimator.decode(condition)
    return dense_contact_to_bps_contact_map_for_gaze2hoi(
        normalized_obj_pc,
        bps_basis,
        dense_contact,
        object_names=object_names,
        part_label_map=part_label_map,
    )


def resolve_contact_estimator_checkpoint_path(config):
    candidates = []
    weight_path = getattr(config.contact, "weight_path", None)
    if weight_path:
        candidates.append(weight_path)
    candidates.append(
        osp.join(config.contact.save_root, config.contact.name, "best_model.pth")
    )
    for candidate in candidates:
        if candidate and osp.exists(candidate):
            return candidate
    return candidates[0] if candidates else None


def load_frozen_bps_contact_estimator(config, device="cuda"):
    checkpoint_path = resolve_contact_estimator_checkpoint_path(config)
    if checkpoint_path is None or not osp.exists(checkpoint_path):
        fallback_path = osp.join(
            config.contact.save_root,
            config.contact.name,
            "best_model.pth",
        )
        raise FileNotFoundError(
            "BPS contact estimator checkpoint not found. "
            f"Tried: {checkpoint_path} and {fallback_path}. "
            "Train stage 1 with gaze2hoi/train_contact_estimator.py first."
        )
    contact_estimator = CTCVAE(**config.contact).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("contact_model", checkpoint)
    contact_estimator.load_state_dict(state_dict)
    contact_estimator.eval()
    for param in contact_estimator.parameters():
        param.requires_grad_(False)
    return contact_estimator, checkpoint_path


def build_contact_feature_sequence_for_gaze2hoi(
    normalized_obj_pc,
    bps_basis,
    gt_ldist_map,
    gt_rdist_map,
    valid_mask_lhand,
    valid_mask_rhand,
    config,
    object_names=None,
    part_label_map=None,
    bps_distance_stats=None,
):
    contact_mode = getattr(config.gaze2hoi.exp, "contact_feature_mode", "distance_raw")
    contact_seq = compute_bps_contact_feature_sequence_for_gaze2hoi(
        normalized_obj_pc,
        bps_basis,
        gt_ldist_map,
        gt_rdist_map,
        valid_mask_lhand,
        valid_mask_rhand,
        config,
        object_names=object_names,
        part_label_map=part_label_map,
    )
    if contact_mode == "distance_raw" and bps_distance_stats is not None:
        batch_size, nframes, feat_dim = contact_seq.shape
        contact_seq = normalize_bps_distance_map(
            contact_seq.reshape(batch_size * nframes, feat_dim),
            bps_distance_stats,
            use_per_dim=getattr(
                config.gaze2hoi.exp, "bps_distance_use_per_dim", True
            ),
        ).reshape(batch_size, nframes, feat_dim)
    return contact_seq


def save_first_batch_contact_gaze_gif(
    model_folder,
    item,
    batch_cuda,
    bps_basis,
    gt_ldist_map,
    gt_rdist_map,
    valid_mask_lhand,
    valid_mask_rhand,
    valid_mask_obj,
    data_config,
    config,
    lhand_layer,
    rhand_layer,
    object_names=None,
    part_label_map=None,
    bps_distance_stats=None,
):
    if part_label_map is not None:
        print("Skipping first-batch contact/gaze GIF for part-wise BPS")
        return
    gif_path = osp.join(model_folder, "first_train_batch_contact_gaze.gif")
    png_path = osp.join(model_folder, "first_train_batch_contact_gaze_used.png")
    normalized_obj_pc = batch_cuda["normalized_obj_pc"]
    obj_pc_org = batch_cuda["obj_pc"]
    x_obj = batch_cuda["x_obj"]
    x_lhand = batch_cuda["x_lhand"]
    x_rhand = batch_cuda["x_rhand"]
    obj_scale = batch_cuda["obj_scale"]
    obj_cent = batch_cuda["obj_cent"]
    gaze = batch_cuda["gaze"]

    contact_seq = build_contact_feature_sequence_for_gaze2hoi(
        normalized_obj_pc,
        bps_basis,
        gt_ldist_map,
        gt_rdist_map,
        valid_mask_lhand,
        valid_mask_rhand,
        config,
        object_names=object_names,
        part_label_map=part_label_map,
        bps_distance_stats=bps_distance_stats,
    )
    gaze_seq = compute_bps_gaze_alignment_sequence_for_gaze2hoi(
        gaze,
        x_obj,
        normalized_obj_pc,
        bps_basis,
        obj_cent,
        obj_scale,
        valid_mask=valid_mask_obj,
        object_names=object_names,
        part_label_map=part_label_map,
    )
    contact_map = build_contact_feature_map_for_gaze2hoi(
        normalized_obj_pc,
        bps_basis,
        gt_ldist_map,
        gt_rdist_map,
        valid_mask_lhand,
        valid_mask_rhand,
        config,
        object_names=object_names,
        part_label_map=part_label_map,
        bps_distance_stats=bps_distance_stats,
    )
    gaze_map = compute_bps_gaze_alignment_map_for_gaze2hoi(
        gaze,
        x_obj,
        normalized_obj_pc,
        bps_basis,
        obj_cent,
        obj_scale,
        valid_mask=valid_mask_obj,
        object_names=object_names,
        part_label_map=part_label_map,
    )

    sample_idx = 0
    nframes = int(valid_mask_obj[sample_idx].long().sum().item())
    if nframes <= 0:
        return

    _, _, nearest_idx = _get_bps_nearest_data(
        normalized_obj_pc[sample_idx : sample_idx + 1], bps_basis
    )
    nearest_idx = nearest_idx[0].detach().cpu().numpy()
    obj_canon = obj_pc_org[sample_idx].detach().cpu().numpy().astype(np.float32)
    lhand_joints_world = get_hand_joints_w_tip(
        x_lhand[sample_idx : sample_idx + 1], lhand_layer
    )
    rhand_joints_world = get_hand_joints_w_tip(
        x_rhand[sample_idx : sample_idx + 1], rhand_layer
    )
    lhand_joints = transform_points_to_canonical(
        lhand_joints_world, x_obj[sample_idx : sample_idx + 1]
    )[0, :nframes].detach().cpu().numpy()
    rhand_joints = transform_points_to_canonical(
        rhand_joints_world, x_obj[sample_idx : sample_idx + 1]
    )[0, :nframes].detach().cpu().numpy()

    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)
    gaze_origin_canon = None
    gaze_dir_canon = None
    if gaze.dim() == 4 and gaze.shape[2:] == (2, 3):
        gaze_origin_canon = transform_dirs_to_canonical(
            gaze[sample_idx : sample_idx + 1, :nframes, 0]
            - x_obj[sample_idx : sample_idx + 1, :nframes, :3],
            x_obj[sample_idx : sample_idx + 1, :nframes],
        )[0].detach().cpu().numpy()
        gaze_dir_canon = transform_dirs_to_canonical(
            gaze[sample_idx : sample_idx + 1, :nframes, 1],
            x_obj[sample_idx : sample_idx + 1, :nframes],
        )[0].detach().cpu().numpy()
    elif gaze.dim() == 3 and gaze.shape[-1] == 3:
        gaze_origin_canon = np.zeros((nframes, 3), dtype=np.float32)
        gaze_dir_canon = transform_dirs_to_canonical(
            gaze[sample_idx : sample_idx + 1, :nframes],
            x_obj[sample_idx : sample_idx + 1, :nframes],
        )[0].detach().cpu().numpy()

    display_rot = display_rotation_matrix()
    obj_canon = apply_display_rotation(obj_canon, display_rot)
    lhand_joints = apply_display_rotation(lhand_joints, display_rot)
    rhand_joints = apply_display_rotation(rhand_joints, display_rot)
    if gaze_origin_canon is not None:
        gaze_origin_canon = apply_display_rotation(gaze_origin_canon, display_rot)
    if gaze_dir_canon is not None:
        gaze_dir_canon = apply_display_rotation_to_dirs(gaze_dir_canon, display_rot)
    obj_canon = flip_vertical(obj_canon)
    lhand_joints = flip_vertical(lhand_joints)
    rhand_joints = flip_vertical(rhand_joints)
    if gaze_origin_canon is not None:
        gaze_origin_canon = flip_vertical(gaze_origin_canon)
    if gaze_dir_canon is not None:
        gaze_dir_canon = flip_vertical(gaze_dir_canon)
    obj_canon = flip_xy_plane(obj_canon)
    lhand_joints = flip_xy_plane(lhand_joints)
    rhand_joints = flip_xy_plane(rhand_joints)
    if gaze_origin_canon is not None:
        gaze_origin_canon = flip_xy_plane(gaze_origin_canon)
    if gaze_dir_canon is not None:
        gaze_dir_canon = flip_xy_plane(gaze_dir_canon)

    obj_extent = float(np.max(obj_canon.max(axis=0) - obj_canon.min(axis=0)))
    gaze_ray_length = max(obj_extent * 8.0, 1.5)

    bps_count = int(bps_basis.shape[0])
    contact_mode = getattr(config.gaze2hoi.exp, "contact_feature_mode", "distance_raw")
    left_contact_sparse = (
        contact_seq[sample_idx, :nframes, :bps_count].detach().cpu().numpy()
    )
    right_contact_sparse = (
        contact_seq[sample_idx, :nframes, bps_count:].detach().cpu().numpy()
    )
    gaze_sparse = gaze_seq[sample_idx, :nframes].detach().cpu().numpy()
    left_contact_used_sparse = (
        contact_map[sample_idx, :bps_count].detach().cpu().numpy()[None, :]
    )
    right_contact_used_sparse = (
        contact_map[sample_idx, bps_count:].detach().cpu().numpy()[None, :]
    )
    gaze_used_sparse = gaze_map[sample_idx].detach().cpu().numpy()[None, :]

    left_contact_vertex_raw, vertex_mask = scatter_sparse_values_to_vertices(
        left_contact_sparse, nearest_idx, obj_canon.shape[0]
    )
    right_contact_vertex_raw, _ = scatter_sparse_values_to_vertices(
        right_contact_sparse, nearest_idx, obj_canon.shape[0]
    )
    gaze_vertex_raw, _ = scatter_sparse_values_to_vertices(
        gaze_sparse, nearest_idx, obj_canon.shape[0]
    )
    left_contact_used_vertex_raw, _ = scatter_sparse_values_to_vertices(
        left_contact_used_sparse, nearest_idx, obj_canon.shape[0]
    )
    right_contact_used_vertex_raw, _ = scatter_sparse_values_to_vertices(
        right_contact_used_sparse, nearest_idx, obj_canon.shape[0]
    )
    gaze_used_vertex_raw, _ = scatter_sparse_values_to_vertices(
        gaze_used_sparse, nearest_idx, obj_canon.shape[0]
    )

    left_contact_vertex_vis = contact_values_for_display(
        left_contact_vertex_raw, contact_mode
    )
    right_contact_vertex_vis = contact_values_for_display(
        right_contact_vertex_raw, contact_mode
    )
    gaze_vertex_vis = gaze_values_for_display(gaze_vertex_raw)
    left_contact_used_vertex_vis = contact_values_for_display(
        left_contact_used_vertex_raw, contact_mode
    )
    right_contact_used_vertex_vis = contact_values_for_display(
        right_contact_used_vertex_raw, contact_mode
    )
    gaze_used_vertex_vis = gaze_values_for_display(gaze_used_vertex_raw)

    show_lhand = valid_mask_lhand[sample_idx, :nframes].detach().cpu().numpy().astype(bool)
    show_rhand = valid_mask_rhand[sample_idx, :nframes].detach().cpu().numpy().astype(bool)

    sample_name = "sample0"
    if "text" in item:
        text_value = item["text"]
        if isinstance(text_value, (list, tuple)) and len(text_value) > sample_idx:
            sample_name = str(text_value[sample_idx])[:60]

    fig = plt.figure(figsize=(18, 6))
    axes = [
        fig.add_subplot(131, projection="3d"),
        fig.add_subplot(132, projection="3d"),
        fig.add_subplot(133, projection="3d"),
    ]
    panel_specs = [
        ("Left Contact", left_contact_vertex_raw, left_contact_vertex_vis, "coolwarm"),
        ("Right Contact", right_contact_vertex_raw, right_contact_vertex_vis, "coolwarm"),
        ("Gaze Score", gaze_vertex_raw, gaze_vertex_vis, "coolwarm"),
    ]
    cbar_axes = [
        fig.add_axes([0.11, 0.08, 0.20, 0.025]),
        fig.add_axes([0.40, 0.08, 0.20, 0.025]),
        fig.add_axes([0.69, 0.08, 0.20, 0.025]),
    ]
    left_cbar = fig.colorbar(
        cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, 1.0)),
        cax=cbar_axes[0],
        orientation="horizontal",
    )
    left_cbar_spec = contact_colorbar_spec(left_contact_vertex_raw, contact_mode)
    left_cbar.set_ticks(left_cbar_spec["ticks"])
    left_cbar.set_ticklabels(left_cbar_spec["ticklabels"])
    left_cbar.set_label(left_cbar_spec["label"], fontsize=8)
    left_cbar.ax.tick_params(labelsize=8)

    right_cbar = fig.colorbar(
        cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, 1.0)),
        cax=cbar_axes[1],
        orientation="horizontal",
    )
    right_cbar_spec = contact_colorbar_spec(right_contact_vertex_raw, contact_mode)
    right_cbar.set_ticks(right_cbar_spec["ticks"])
    right_cbar.set_ticklabels(right_cbar_spec["ticklabels"])
    right_cbar.set_label(right_cbar_spec["label"], fontsize=8)
    right_cbar.ax.tick_params(labelsize=8)

    gaze_cbar = fig.colorbar(
        cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, 1.0)),
        cax=cbar_axes[2],
        orientation="horizontal",
    )
    gaze_cbar_spec = gaze_colorbar_spec()
    gaze_cbar.set_ticks(gaze_cbar_spec["ticks"])
    gaze_cbar.set_ticklabels(gaze_cbar_spec["ticklabels"])
    gaze_cbar.set_label(gaze_cbar_spec["label"], fontsize=8)
    gaze_cbar.ax.tick_params(labelsize=8)

    def _update(frame_idx):
        for ax in axes:
            ax.cla()
        for ax, (title, raw_values, vis_values, cmap_name) in zip(axes, panel_specs):
            ax.scatter(
                obj_canon[:, 0],
                obj_canon[:, 1],
                obj_canon[:, 2],
                s=7,
                c="#c6cbd3",
                edgecolors="none",
                alpha=0.16,
            )
            ax.scatter(
                obj_canon[vertex_mask, 0],
                obj_canon[vertex_mask, 1],
                obj_canon[vertex_mask, 2],
                s=24,
                c=vis_values[frame_idx, vertex_mask],
                cmap=cmap_name,
                vmin=0.0,
                vmax=1.0,
                edgecolors="black",
                linewidths=0.25,
                alpha=0.95,
            )
            if show_lhand[frame_idx]:
                plot_hand_skeleton(ax, lhand_joints[frame_idx], "#1f77b4")
            if show_rhand[frame_idx]:
                plot_hand_skeleton(ax, rhand_joints[frame_idx], "#2ca02c")
            if title == "Gaze Score" and gaze_origin_canon is not None and gaze_dir_canon is not None:
                plot_gaze_vector(
                    ax,
                    gaze_origin_canon[frame_idx],
                    gaze_dir_canon[frame_idx],
                    length=gaze_ray_length,
                )
            frame_lhand = lhand_joints[frame_idx] if show_lhand[frame_idx] else None
            frame_rhand = rhand_joints[frame_idx] if show_rhand[frame_idx] else None
            set_hand_focused_view(ax, obj_canon, frame_lhand, frame_rhand)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(
                f"{title}\n"
                f"mean={float(raw_values[frame_idx, vertex_mask].mean() if np.any(vertex_mask) else 0.0):.3f} "
                f"min={float(raw_values[frame_idx, vertex_mask].min() if np.any(vertex_mask) else 0.0):.3f} "
                f"max={float(raw_values[frame_idx, vertex_mask].max() if np.any(vertex_mask) else 0.0):.3f}"
            )
        fig.suptitle(
            f"First train batch | {data_config.name} | {sample_name} | frame {frame_idx}",
            fontsize=13,
        )

    anim = FuncAnimation(fig, _update, frames=nframes, interval=250)
    try:
        anim.save(gif_path, writer=PillowWriter(fps=4))
    finally:
        plt.close(fig)

    png_fig = plt.figure(figsize=(18, 6))
    png_axes = [
        png_fig.add_subplot(131, projection="3d"),
        png_fig.add_subplot(132, projection="3d"),
        png_fig.add_subplot(133, projection="3d"),
    ]
    png_specs = [
        ("Left Contact Used", left_contact_used_vertex_raw, left_contact_used_vertex_vis, "coolwarm"),
        ("Right Contact Used", right_contact_used_vertex_raw, right_contact_used_vertex_vis, "coolwarm"),
        ("Gaze Score Used", gaze_used_vertex_raw, gaze_used_vertex_vis, "coolwarm"),
    ]
    png_cbar_axes = [
        png_fig.add_axes([0.11, 0.08, 0.20, 0.025]),
        png_fig.add_axes([0.40, 0.08, 0.20, 0.025]),
        png_fig.add_axes([0.69, 0.08, 0.20, 0.025]),
    ]
    png_left_cbar = png_fig.colorbar(
        cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, 1.0)),
        cax=png_cbar_axes[0],
        orientation="horizontal",
    )
    png_left_spec = contact_colorbar_spec(left_contact_used_vertex_raw, contact_mode)
    png_left_cbar.set_ticks(png_left_spec["ticks"])
    png_left_cbar.set_ticklabels(png_left_spec["ticklabels"])
    png_left_cbar.set_label(png_left_spec["label"], fontsize=8)
    png_left_cbar.ax.tick_params(labelsize=8)
    png_right_cbar = png_fig.colorbar(
        cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, 1.0)),
        cax=png_cbar_axes[1],
        orientation="horizontal",
    )
    png_right_spec = contact_colorbar_spec(right_contact_used_vertex_raw, contact_mode)
    png_right_cbar.set_ticks(png_right_spec["ticks"])
    png_right_cbar.set_ticklabels(png_right_spec["ticklabels"])
    png_right_cbar.set_label(png_right_spec["label"], fontsize=8)
    png_right_cbar.ax.tick_params(labelsize=8)
    png_gaze_cbar = png_fig.colorbar(
        cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0.0, 1.0)),
        cax=png_cbar_axes[2],
        orientation="horizontal",
    )
    png_gaze_spec = gaze_colorbar_spec()
    png_gaze_cbar.set_ticks(png_gaze_spec["ticks"])
    png_gaze_cbar.set_ticklabels(png_gaze_spec["ticklabels"])
    png_gaze_cbar.set_label(png_gaze_spec["label"], fontsize=8)
    png_gaze_cbar.ax.tick_params(labelsize=8)
    last_frame_idx = max(nframes - 1, 0)
    for ax, (title, raw_values, vis_values, cmap_name) in zip(png_axes, png_specs):
        ax.scatter(
            obj_canon[:, 0],
            obj_canon[:, 1],
            obj_canon[:, 2],
            s=7,
            c="#c6cbd3",
            edgecolors="none",
            alpha=0.16,
        )
        ax.scatter(
            obj_canon[vertex_mask, 0],
            obj_canon[vertex_mask, 1],
            obj_canon[vertex_mask, 2],
            s=24,
            c=vis_values[0, vertex_mask],
            cmap=cmap_name,
            vmin=0.0,
            vmax=1.0,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.95,
        )
        if show_lhand[last_frame_idx]:
            plot_hand_skeleton(ax, lhand_joints[last_frame_idx], "#1f77b4")
        if show_rhand[last_frame_idx]:
            plot_hand_skeleton(ax, rhand_joints[last_frame_idx], "#2ca02c")
        if title == "Gaze Score Used" and gaze_origin_canon is not None and gaze_dir_canon is not None:
            plot_gaze_vector(
                ax,
                gaze_origin_canon[last_frame_idx],
                gaze_dir_canon[last_frame_idx],
                length=gaze_ray_length,
            )
        last_lhand = lhand_joints[last_frame_idx] if show_lhand[last_frame_idx] else None
        last_rhand = rhand_joints[last_frame_idx] if show_rhand[last_frame_idx] else None
        set_hand_focused_view(ax, obj_canon, last_lhand, last_rhand)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(
            f"{title}\n"
            f"mean={float(raw_values[0, vertex_mask].mean() if np.any(vertex_mask) else 0.0):.3f} "
            f"min={float(raw_values[0, vertex_mask].min() if np.any(vertex_mask) else 0.0):.3f} "
            f"max={float(raw_values[0, vertex_mask].max() if np.any(vertex_mask) else 0.0):.3f}"
        )
    png_fig.suptitle(
        f"First train batch used maps | {data_config.name} | {sample_name}",
        fontsize=13,
    )
    png_fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.96])
    png_fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(png_fig)
    print(f"Saved first-batch contact/gaze GIF to {gif_path}")
    print(f"Saved first-batch used contact/gaze PNG to {png_path}")


def compute_tail_mask(valid_mask, last_k=5):
    tail_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    lengths = valid_mask.long().sum(dim=1)
    for b in range(valid_mask.shape[0]):
        end = int(lengths[b].item())
        start = max(0, end - last_k)
        tail_mask[b, start:end] = True
    return tail_mask & valid_mask


def compute_penetration_per_frame(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    dataset_name,
    obj_pc_top_idx=None,
):
    bs, nframes = pred_obj.shape[:2]
    npts = obj_pc_org.shape[1]
    hand_mesh, hand_verts = get_pytorch3d_meshes(pred_hand, hand_layer)
    hand_normal = hand_mesh.verts_normals_packed().view(-1, 778, 3)
    transf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    ).reshape(-1, npts, 3)
    nn_dist, nn_idx = get_NN(transf_obj_pc, hand_verts)
    interior = get_interior(hand_normal, hand_verts, transf_obj_pc, nn_idx)
    nn_dist = nn_dist.sqrt()

    penet_frame = torch.zeros(bs * nframes, device=pred_obj.device, dtype=pred_obj.dtype)
    for idx in range(bs * nframes):
        if interior[idx].any():
            penet_frame[idx] = nn_dist[idx][interior[idx]].mean()
    return penet_frame.view(bs, nframes)


def compute_penetration_max_per_sample(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    valid_mask_hand,
    dataset_name,
    obj_pc_top_idx=None,
):
    bs, nframes = pred_obj.shape[:2]
    npts = obj_pc_org.shape[1]
    hand_mesh, hand_verts = get_pytorch3d_meshes(pred_hand, hand_layer)
    hand_normal = hand_mesh.verts_normals_packed().view(-1, 778, 3)
    transf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    ).reshape(-1, npts, 3)
    nn_dist, nn_idx = get_NN(transf_obj_pc, hand_verts)
    interior = get_interior(hand_normal, hand_verts, transf_obj_pc, nn_idx)
    nn_dist = nn_dist.sqrt()

    penet_max_frame = torch.zeros(bs * nframes, device=pred_obj.device, dtype=pred_obj.dtype)
    penet_max_obj_points = [None] * (bs * nframes)
    penet_max_hand_points = [None] * (bs * nframes)
    penet_max_obj_indices = [None] * (bs * nframes)
    penet_max_hand_indices = [None] * (bs * nframes)
    for idx in range(bs * nframes):
        if interior[idx].any():
            interior_idx = torch.nonzero(interior[idx], as_tuple=False).flatten()
            max_local_idx = torch.argmax(nn_dist[idx][interior[idx]])
            max_obj_idx = int(interior_idx[max_local_idx].item())
            max_hand_idx = int(nn_idx[idx][max_obj_idx].item())
            penet_max_frame[idx] = nn_dist[idx][max_obj_idx]
            flat_obj_points = transf_obj_pc[idx].detach().cpu().numpy().astype(np.float32)
            flat_hand_points = hand_verts[idx].detach().cpu().numpy().astype(np.float32)
            penet_max_obj_points[idx] = flat_obj_points[max_obj_idx].tolist()
            penet_max_hand_points[idx] = flat_hand_points[max_hand_idx].tolist()
            penet_max_obj_indices[idx] = max_obj_idx
            penet_max_hand_indices[idx] = max_hand_idx
    penet_max_frame = penet_max_frame.view(bs, nframes)

    last_valid_idx = valid_mask_hand.long().sum(dim=1) - 1
    last_valid_idx = last_valid_idx.clamp_min(0)
    batch_idx = torch.arange(bs, device=pred_obj.device)
    has_valid = valid_mask_hand.any(dim=1)
    penet_max = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
    penet_max[has_valid] = penet_max_frame[batch_idx[has_valid], last_valid_idx[has_valid]]
    final_obj_points = [None] * bs
    final_hand_points = [None] * bs
    final_obj_indices = [None] * bs
    final_hand_indices = [None] * bs
    for b in range(bs):
        if has_valid[b]:
            flat_idx = b * nframes + int(last_valid_idx[b].item())
            final_obj_points[b] = penet_max_obj_points[flat_idx]
            final_hand_points[b] = penet_max_hand_points[flat_idx]
            final_obj_indices[b] = penet_max_obj_indices[flat_idx]
            final_hand_indices[b] = penet_max_hand_indices[flat_idx]
    return (
        penet_max * 1000.0,
        final_hand_points,
        final_obj_points,
        final_hand_indices,
        final_obj_indices,
    )


def compute_contact_guided_penetration_max_per_sample(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_pc_org,
    valid_mask_hand,
    dataset_name,
    contact_threshold=0.02,
    obj_pc_top_idx=None,
):
    bs, nframes = pred_obj.shape[:2]
    npts = obj_pc_org.shape[1]

    hand_mesh, hand_verts = get_pytorch3d_meshes(pred_hand, hand_layer)
    hand_normal = hand_mesh.verts_normals_packed().view(-1, 778, 3)
    transf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    ).reshape(-1, npts, 3)
    hand_vert_dist = torch.cdist(
        transf_obj_pc, hand_verts
    )
    nn_dist, nn_idx = get_NN(transf_obj_pc, hand_verts)
    interior = get_interior(hand_normal, hand_verts, transf_obj_pc, nn_idx)
    nn_dist = nn_dist.sqrt()

    penet_max_frame = torch.zeros(bs * nframes, device=pred_obj.device, dtype=pred_obj.dtype)
    penet_max_obj_points = [None] * (bs * nframes)
    penet_max_hand_points = [None] * (bs * nframes)
    penet_max_obj_indices = [None] * (bs * nframes)
    penet_max_hand_indices = [None] * (bs * nframes)
    penet_contact_hand_vertex_indices = [None] * (bs * nframes)
    penet_contact_object_vertex_indices = [None] * (bs * nframes)

    for idx in range(bs * nframes):
        hand_vert_min_dist, hand_vert_min_idx = hand_vert_dist[idx].min(dim=1)
        contact_obj_mask = hand_vert_min_dist < contact_threshold
        candidate_mask = interior[idx] & contact_obj_mask
        if candidate_mask.any():
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            max_local_idx = torch.argmax(nn_dist[idx][candidate_mask])
            max_obj_idx = int(candidate_indices[max_local_idx].item())
            max_hand_idx = int(nn_idx[idx][max_obj_idx].item())
            penet_max_frame[idx] = nn_dist[idx][max_obj_idx]

            flat_obj_points = transf_obj_pc[idx].detach().cpu().numpy().astype(np.float32)
            flat_hand_points = hand_verts[idx].detach().cpu().numpy().astype(np.float32)
            penet_max_obj_points[idx] = flat_obj_points[max_obj_idx].tolist()
            penet_max_hand_points[idx] = flat_hand_points[max_hand_idx].tolist()
            penet_max_obj_indices[idx] = max_obj_idx
            penet_max_hand_indices[idx] = max_hand_idx
            penet_contact_hand_vertex_indices[idx] = torch.unique(
                hand_vert_min_idx[contact_obj_mask]
            ).detach().cpu().tolist()
            penet_contact_object_vertex_indices[idx] = torch.nonzero(
                contact_obj_mask, as_tuple=False
            ).flatten().detach().cpu().tolist()

    penet_max_frame = penet_max_frame.view(bs, nframes)
    last_valid_idx = valid_mask_hand.long().sum(dim=1) - 1
    last_valid_idx = last_valid_idx.clamp_min(0)
    batch_idx = torch.arange(bs, device=pred_obj.device)
    has_valid = valid_mask_hand.any(dim=1)
    penet_max = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
    penet_max[has_valid] = penet_max_frame[batch_idx[has_valid], last_valid_idx[has_valid]]

    final_obj_points = [None] * bs
    final_hand_points = [None] * bs
    final_obj_indices = [None] * bs
    final_hand_indices = [None] * bs
    final_contact_hand_vertex_indices = [None] * bs
    final_contact_object_vertex_indices = [None] * bs
    for b in range(bs):
        if has_valid[b]:
            flat_idx = b * nframes + int(last_valid_idx[b].item())
            final_obj_points[b] = penet_max_obj_points[flat_idx]
            final_hand_points[b] = penet_max_hand_points[flat_idx]
            final_obj_indices[b] = penet_max_obj_indices[flat_idx]
            final_hand_indices[b] = penet_max_hand_indices[flat_idx]
            final_contact_hand_vertex_indices[b] = penet_contact_hand_vertex_indices[flat_idx]
            final_contact_object_vertex_indices[b] = penet_contact_object_vertex_indices[flat_idx]

    return (
        penet_max * 1000.0,
        final_hand_points,
        final_obj_points,
        final_hand_indices,
        final_obj_indices,
        final_contact_hand_vertex_indices,
        final_contact_object_vertex_indices,
    )


def _resolve_object_mesh_path(obj_path, data_config):
    candidates = []
    if osp.isabs(obj_path):
        candidates.append(obj_path)
    root = getattr(data_config, "root", None)
    if root:
        candidates.append(osp.join(PROJECT_ROOT, root, obj_path))
    obj_root = getattr(data_config, "obj_root", None)
    if obj_root:
        candidates.append(osp.join(PROJECT_ROOT, obj_root, osp.basename(obj_path)))
    candidates.append(osp.join(PROJECT_ROOT, obj_path))

    for path in candidates:
        if path and osp.exists(path):
            return path
    return None


def build_bimart_refinement_mesh_cache(object_meshes, max_faces=2000):
    return object_meshes


def load_object_mesh_cache(data_config):
    data_obj_pc_path = data_config.data_obj_pc_path
    data_config_for_mesh = data_config
    if getattr(data_config, "name", None) == "h2o":
        data_config_for_mesh = data_config
        data_config_for_mesh = data_config_for_mesh.copy()
        data_config_for_mesh.obj_root = "data/hot3d/object_mesh"
        data_obj_pc_path = "data/hot3d/dataset/obj.pkl"

    from lib.models.object import build_object_model

    object_model = build_object_model(osp.join(PROJECT_ROOT, data_obj_pc_path))
    mesh_cache = {}
    for object_name in object_model.obj_pcs.keys():
        obj_meta = object_model(object_name)
        obj_path = obj_meta[3]
        mesh_path = _resolve_object_mesh_path(obj_path, data_config_for_mesh)
        if mesh_path is None:
            continue
        try:
            mesh_cache[object_name] = trimesh.load(mesh_path, force="mesh")
        except Exception:
            continue
    return mesh_cache


def _get_batch_object_names(item):
    names = item.get("object_name")
    if names is None:
        names = item.get("obj_name")
    if names is None:
        return None
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return [str(names)]


def _get_batch_text_entries(item, batch_size):
    texts = item.get("text")
    if texts is None:
        return [None] * batch_size
    if isinstance(texts, (list, tuple)):
        return list(texts)
    return [texts] * batch_size


def _reservoir_add_sample(samples, candidate, seen_count, max_keep, rng):
    if max_keep <= 0:
        return seen_count
    seen_count += 1
    if len(samples) < max_keep:
        samples.append(candidate)
        return seen_count
    replace_idx = rng.randint(0, seen_count - 1)
    if replace_idx < max_keep:
        samples[replace_idx] = candidate
    return seen_count


def _set_equal_axis_for_points(ax, points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.55, 0.08)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _render_hand_mesh(ax, vertices, faces, facecolor):
    tris = np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int32)]
    mesh = Poly3DCollection(
        tris,
        facecolor=facecolor,
        edgecolor=(0.15, 0.15, 0.15, 0.18),
        linewidth=0.05,
    )
    ax.add_collection3d(mesh)


def _plot_eval_hoi_panel(
    ax,
    obj_points,
    lhand_vertices,
    rhand_vertices,
    lhand_faces,
    rhand_faces,
    use_left,
    use_right,
    title,
):
    obj_points = np.asarray(obj_points, dtype=np.float32)
    ax.scatter(
        obj_points[:, 0],
        obj_points[:, 1],
        obj_points[:, 2],
        s=4,
        c="#aeb7c2",
        edgecolors="none",
        alpha=0.32,
    )
    all_points = [obj_points]
    if use_left and lhand_vertices is not None:
        _render_hand_mesh(ax, lhand_vertices, lhand_faces, (0.18, 0.50, 0.93, 0.82))
        all_points.append(np.asarray(lhand_vertices, dtype=np.float32))
    if use_right and rhand_vertices is not None:
        _render_hand_mesh(ax, rhand_vertices, rhand_faces, (0.91, 0.34, 0.26, 0.82))
        all_points.append(np.asarray(rhand_vertices, dtype=np.float32))
    _set_equal_axis_for_points(ax, np.concatenate(all_points, axis=0))
    ax.view_init(elev=22, azim=-58)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def _save_eval_qualitative_gif(
    output_path,
    sample,
    lhand_layer,
    rhand_layer,
    dataset_name,
):
    device = next(lhand_layer.parameters()).device
    pred_lhand = sample["pred_lhand"].unsqueeze(0).to(device)
    pred_rhand = sample["pred_rhand"].unsqueeze(0).to(device)
    pred_obj = sample["pred_obj"].unsqueeze(0).to(device)
    gt_lhand = sample["gt_lhand"].unsqueeze(0).to(device)
    gt_rhand = sample["gt_rhand"].unsqueeze(0).to(device)
    gt_obj = sample["gt_obj"].unsqueeze(0).to(device)
    obj_pc = sample["obj_pc"].unsqueeze(0).to(device)
    canonicalize_object_targets = bool(sample.get("canonicalize_object_targets", False))

    with torch.no_grad():
        pred_obj_world = get_transformed_obj_pc(pred_obj, obj_pc, dataset_name)
        gt_obj_world = get_transformed_obj_pc(gt_obj, obj_pc, dataset_name)
        pred_lverts_world = get_hand_verts(pred_lhand, lhand_layer)
        pred_rverts_world = get_hand_verts(pred_rhand, rhand_layer)
        gt_lverts_world = get_hand_verts(gt_lhand, lhand_layer)
        gt_rverts_world = get_hand_verts(gt_rhand, rhand_layer)

        if canonicalize_object_targets:
            pred_obj_points = (
                transform_points_to_initial_object_canonical(pred_obj_world, pred_obj)[0]
                .detach()
                .cpu()
                .numpy()
            )
            gt_obj_points = (
                transform_points_to_initial_object_canonical(gt_obj_world, gt_obj)[0]
                .detach()
                .cpu()
                .numpy()
            )
            pred_lverts = pred_lverts_world[0].detach().cpu().numpy()
            pred_rverts = pred_rverts_world[0].detach().cpu().numpy()
            gt_lverts = (
                transform_points_to_initial_object_canonical(gt_lverts_world, gt_obj)[0]
                .detach()
                .cpu()
                .numpy()
            )
            gt_rverts = (
                transform_points_to_initial_object_canonical(gt_rverts_world, gt_obj)[0]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            pred_obj_points = pred_obj_world[0].detach().cpu().numpy()
            gt_obj_points = gt_obj_world[0].detach().cpu().numpy()
            pred_lverts = pred_lverts_world[0].detach().cpu().numpy()
            pred_rverts = pred_rverts_world[0].detach().cpu().numpy()
            gt_lverts = gt_lverts_world[0].detach().cpu().numpy()
            gt_rverts = gt_rverts_world[0].detach().cpu().numpy()

    frame_count = min(
        pred_obj_points.shape[0],
        gt_obj_points.shape[0],
        pred_lverts.shape[0],
        pred_rverts.shape[0],
        gt_lverts.shape[0],
        gt_rverts.shape[0],
    )
    frame_count = max(frame_count, 1)

    fig = plt.figure(figsize=(10, 5))
    pred_ax = fig.add_subplot(121, projection="3d")
    gt_ax = fig.add_subplot(122, projection="3d")
    fig.suptitle(sample["title"], fontsize=12)

    def _update(frame_idx):
        pred_ax.cla()
        gt_ax.cla()
        _plot_eval_hoi_panel(
            pred_ax,
            pred_obj_points[frame_idx],
            pred_lverts[frame_idx] if sample["use_left"] else None,
            pred_rverts[frame_idx] if sample["use_right"] else None,
            lhand_layer.faces,
            rhand_layer.faces,
            sample["use_left"],
            sample["use_right"],
            f"Pred | frame {frame_idx + 1}/{frame_count}",
        )
        _plot_eval_hoi_panel(
            gt_ax,
            gt_obj_points[frame_idx],
            gt_lverts[frame_idx] if sample["use_left"] else None,
            gt_rverts[frame_idx] if sample["use_right"] else None,
            lhand_layer.faces,
            rhand_layer.faces,
            sample["use_left"],
            sample["use_right"],
            "GT",
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    anim = FuncAnimation(fig, _update, frames=frame_count, interval=140)
    try:
        anim.save(output_path, writer=PillowWriter(fps=6))
    finally:
        plt.close(fig)


def _save_eval_qualitative_report(
    output_dir,
    samples,
    lhand_layer,
    rhand_layer,
    dataset_name,
):
    if not samples:
        return
    os.makedirs(output_dir, exist_ok=True)
    gif_dir = osp.join(output_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    report_entries = []
    for sample_idx, sample in enumerate(samples, start=1):
        object_slug = _sanitize_failure_name(sample["object_name"], default="object")
        text_slug = _sanitize_failure_name(sample["text"], default=f"sample_{sample_idx:02d}")
        gif_path = osp.join(gif_dir, f"{sample_idx:02d}_{object_slug}_{text_slug}.gif")
        _save_eval_qualitative_gif(
            gif_path,
            sample,
            lhand_layer,
            rhand_layer,
            dataset_name,
        )
        report_entries.append(
            {
                "index": sample_idx,
                "text": sample["text"],
                "object_name": sample["object_name"],
                "success": sample["success"],
                "gif_path": gif_path,
            }
        )

    md_path = osp.join(output_dir, "report.md")
    with open(md_path, "w") as f:
        f.write("# Eval Qualitative Samples\n\n")
        f.write(f"Random qualitative samples saved from evaluation: {len(report_entries)}\n\n")
        for entry in report_entries:
            rel_gif = osp.relpath(entry["gif_path"], output_dir)
            status = "success" if entry["success"] else "fail"
            f.write(f"## Sample {entry['index']}\n\n")
            f.write(f"- object: `{entry['object_name']}`\n")
            f.write(f"- status: `{status}`\n")
            f.write(f"- text: `{entry['text']}`\n")
            f.write(f"- gif: [{osp.basename(entry['gif_path'])}]({rel_gif})\n\n")
            f.write(f"![sample_{entry['index']}]({rel_gif})\n\n")

    print(f"Saved eval qualitative report to {md_path}")


def _compute_last_frame_id_max_per_hand(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_meshes,
    object_names,
    valid_mask_hand,
):
    bs, nframes = pred_obj.shape[:2]
    if not object_names:
        zeros = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
        return zeros, [None] * bs, [None] * bs

    _, hand_verts = get_pytorch3d_meshes(pred_hand, hand_layer)
    hand_verts = hand_verts.view(bs, nframes, 778, 3)
    id_max = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
    hand_points = [None] * bs
    object_points = [None] * bs

    for b in range(bs):
        if not valid_mask_hand[b].any():
            continue
        object_name = object_names[b] if b < len(object_names) else None
        obj_mesh = obj_meshes.get(object_name)
        if obj_mesh is None:
            continue

        frame_idx = int(valid_mask_hand[b].long().sum().item()) - 1
        frame_idx = max(frame_idx, 0)
        hand_vertices = hand_verts[b, frame_idx].detach().cpu().numpy().astype(np.float64)
        pose = pred_obj[b, frame_idx, :9].detach().cpu()
        obj_trans = pose[:3].numpy().astype(np.float64)
        obj_rot = (
            rot6d_to_rotmat(pose[3:9].reshape(1, 6))
            .reshape(3, 3)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        hand_local = np.einsum("ni,ij->nj", hand_vertices - obj_trans[None, :], obj_rot)

        try:
            signed_distance = np.asarray(
                trimesh.proximity.signed_distance(obj_mesh, hand_local),
                dtype=np.float64,
            )
            inside_mask = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
            if inside_mask.any():
                inside_indices = np.flatnonzero(inside_mask)
                local_max_idx = int(np.argmax(np.abs(signed_distance[inside_mask])))
                max_idx = int(inside_indices[local_max_idx])
                id_max[b] = float(np.abs(signed_distance[max_idx]) * 1000.0)
                hand_points[b] = hand_vertices[max_idx].astype(np.float32).tolist()
                closest_local, _, _ = trimesh.proximity.closest_point(
                    obj_mesh,
                    hand_local[max_idx : max_idx + 1],
                )
                closest_world = np.einsum(
                    "ni,ji->nj", closest_local.astype(np.float64), obj_rot
                ) + obj_trans[None, :]
                object_points[b] = closest_world[0].astype(np.float32).tolist()
        except Exception:
            continue

    return id_max, hand_points, object_points


def _compute_last_frame_mesh_penetration_metrics_per_hand(
    pred_hand,
    hand_layer,
    pred_obj,
    obj_meshes,
    object_names,
    valid_mask_hand,
    valid_mask_obj=None,
):
    bs, nframes = pred_obj.shape[:2]
    zeros = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
    valid = torch.zeros(bs, device=pred_obj.device, dtype=torch.bool)
    if not object_names:
        return zeros, zeros.clone(), [None] * bs, [None] * bs, valid

    _, hand_verts = get_pytorch3d_meshes(pred_hand, hand_layer)
    hand_verts = hand_verts.view(bs, nframes, 778, 3)
    id_mean = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
    id_max = torch.zeros(bs, device=pred_obj.device, dtype=pred_obj.dtype)
    hand_points = [None] * bs
    object_points = [None] * bs

    for b in range(bs):
        if valid_mask_obj is not None and valid_mask_obj[b].any():
            frame_idx = int(valid_mask_obj[b].long().sum().item()) - 1
        elif valid_mask_hand[b].any():
            frame_idx = int(valid_mask_hand[b].long().sum().item()) - 1
        else:
            continue
        frame_idx = max(frame_idx, 0)
        if not bool(valid_mask_hand[b, frame_idx].item()):
            continue

        object_name = object_names[b] if b < len(object_names) else None
        obj_mesh = obj_meshes.get(object_name)
        if obj_mesh is None:
            continue

        hand_vertices = hand_verts[b, frame_idx].detach().cpu().numpy().astype(np.float64)
        pose = pred_obj[b, frame_idx, :9].detach().cpu()
        obj_trans = pose[:3].numpy().astype(np.float64)
        obj_rot = (
            rot6d_to_rotmat(pose[3:9].reshape(1, 6))
            .reshape(3, 3)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        hand_local = np.einsum("ni,ij->nj", hand_vertices - obj_trans[None, :], obj_rot)

        try:
            signed_distance = np.asarray(
                trimesh.proximity.signed_distance(obj_mesh, hand_local),
                dtype=np.float64,
            )
            contains_mask = np.asarray(obj_mesh.contains(hand_local), dtype=bool)
            inside_mask = np.isfinite(signed_distance) & (contains_mask | (signed_distance > 0.0))
            if not inside_mask.any():
                continue

            inside_points = hand_local[inside_mask]
            closest_local, _, _ = trimesh.proximity.closest_point(obj_mesh, inside_points)
            if closest_local is None:
                continue
            closest_local = np.asarray(closest_local, dtype=np.float64)
            depths = np.linalg.norm(inside_points - closest_local, axis=1)
            if depths.size == 0:
                continue

            id_mean[b] = float(depths.mean())
            max_local_idx = int(np.argmax(depths))
            id_max[b] = float(depths[max_local_idx])
            valid[b] = True

            inside_indices = np.flatnonzero(inside_mask)
            max_idx = int(inside_indices[max_local_idx])
            hand_points[b] = hand_vertices[max_idx].astype(np.float32).tolist()
            closest_world = np.einsum("ni,ji->nj", closest_local[max_local_idx:max_local_idx + 1], obj_rot)
            closest_world = closest_world + obj_trans[None, :]
            object_points[b] = closest_world[0].astype(np.float32).tolist()
        except Exception:
            continue

    return id_mean, id_max, hand_points, object_points, valid


def evaluate_generation_metrics(
    gaze2hoi,
    diffusion,
    bps_basis,
    eval_loader,
    data_config,
    config,
    lhand_layer,
    rhand_layer,
    failure_save_path=None,
    obj_meshes=None,
    part_label_map=None,
    bps_distance_stats=None,
    contact_estimator=None,
    num_bps_parts=None,
    bps_count=None,
    mesh_bps_cache=None,
    hand_sample_indices=None,
    qualitative_save_dir=None,
    qualitative_num_samples=5,
):
    # Success criterion:
    # selected hand(s) keep at least 2 joints within 0.02m
    # on each of the last 5 valid object frames, and max penetration depth is <= 0.02m.
    contact_threshold = 0.02
    contact_min_keypoints = int(getattr(config.gaze2hoi.exp, "eval_contact_min_keypoints", 2))
    last_k = 5
    penetration_max_depth = 0.02
    total_samples = 0
    success_sum = 0.0
    end_contact_sum = 0.0
    joint_penetration_ok_sum = 0.0
    sample_save_list = []
    failed_save_list = []
    success_image_dir = None
    failure_image_dir = None
    if failure_save_path:
        failure_root = osp.splitext(failure_save_path)[0]
        image_root = failure_root + "_images"
        success_image_dir = osp.join(image_root, "success")
        failure_image_dir = osp.join(image_root, "failure")

    gaze2hoi.eval()
    use_point_token_output = bool(
        getattr(config.gaze2hoi.model, "use_point_token_output", False)
    )
    if use_point_token_output and hand_sample_indices is None:
        hand_sample_indices = load_hand_sample_indices_for_gaze2hoi(config, device="cuda")
    qualitative_samples = []
    qualitative_seen = 0
    qualitative_rng = random.Random()

    with torch.no_grad():
        for item in eval_loader:
            object_names = _get_batch_object_names(item)
            obj_pc_top_idx = None
            cuda_keys = [
                "x_lhand",
                "x_rhand",
                "x_obj",
                "obj_pc",
                "obj_pc_normal",
                "gaze",
                "normalized_obj_pc",
                "obj_scale",
                "obj_cent",
                "ldist_map",
                "rdist_map",
                "valid_mask_lhand",
                "valid_mask_rhand",
                "valid_mask_obj",
            ]
            gaze_condition_mode = str(
                getattr(config.gaze2hoi.model, "gaze_condition_mode", "alignment")
            ).lower()
            use_contact_condition = _is_contact_condition_mode(gaze_condition_mode) or (
                gaze_condition_mode in ("raw_contact_map", "raw_cov_map")
            )
            use_contact_map = bool(getattr(config.gaze2hoi.model, "use_contact_map", False))
            if gaze_condition_mode in (
                "gaze_map",
                "raw_gaze_map",
                "dataset_gaze_map",
            ) or use_contact_map:
                cuda_keys.append("gaze_map")
            if use_contact_condition:
                cuda_keys.append("cov_map")
            batch_cuda = move_batch_to_cuda(item, cuda_keys)
            x_lhand = batch_cuda["x_lhand"]
            x_rhand = batch_cuda["x_rhand"]
            x_obj = batch_cuda["x_obj"]
            obj_pc_org = batch_cuda["obj_pc"]
            obj_pc_normal_org = batch_cuda["obj_pc_normal"]
            normalized_obj_pc = batch_cuda["normalized_obj_pc"]
            obj_scale = batch_cuda["obj_scale"]
            obj_cent = batch_cuda["obj_cent"]
            valid_mask_lhand = batch_cuda["valid_mask_lhand"]
            valid_mask_rhand = batch_cuda["valid_mask_rhand"]
            valid_mask_obj = batch_cuda["valid_mask_obj"]
            if _is_temporal_gaze_token_mode(gaze_condition_mode):
                gaze_condition_valid_mask = build_gaze_alignment_temporal_mask(
                    valid_mask_obj,
                    ldist_map=batch_cuda.get("ldist_map"),
                    rdist_map=batch_cuda.get("rdist_map"),
                    temporal_scope=getattr(
                        config.gaze2hoi.model,
                        "gaze_alignment_temporal_scope",
                        "all",
                    ),
                )
            else:
                gaze_condition_valid_mask = valid_mask_obj

            obj_feat = compute_bps_feature_from_mesh_cache_for_gaze2hoi(
                normalized_obj_pc,
                bps_basis,
                object_names=object_names,
                mesh_cache=mesh_bps_cache,
                part_label_map=part_label_map,
                bbox_margin=float(getattr(config.dataset, "mesh_part_bbox_margin", 0.03)),
            )
            gaze_condition_dim = get_gaze2hoi_gaze_condition_dim(
                config,
                num_bps_parts or 1,
                bps_count or int(bps_basis.shape[0]),
            )
            gaze_condition_x_obj = (
                repeat_initial_object_pose_for_gaze_condition(x_obj)
            )
            gaze_score = build_gaze_condition_feature_for_gaze2hoi(
                config,
                batch_cuda["gaze"],
                batch_cuda.get("gaze_map"),
                gaze_condition_x_obj,
                normalized_obj_pc,
                bps_basis,
                obj_cent,
                obj_scale,
                gaze_condition_valid_mask,
                object_names=object_names,
                mesh_cache=mesh_bps_cache,
                part_label_map=part_label_map,
                bbox_margin=float(getattr(config.dataset, "mesh_part_bbox_margin", 0.03)),
                target_dim=gaze_condition_dim,
                contact_map=batch_cuda.get("cov_map"),
            )
            gaze_score = apply_null_gaze_condition(config, gaze_score)
            if use_contact_map and contact_estimator is None:
                gt_ldist_map, gt_rdist_map = get_hand_obj_dist_map_eval(
                    x_lhand,
                    x_rhand,
                    x_obj,
                    obj_pc_org,
                    lhand_layer,
                    rhand_layer,
                    data_config.name,
                    obj_pc_top_idx,
                )
                contact_map = build_contact_feature_map_for_gaze2hoi(
                    normalized_obj_pc,
                    bps_basis,
                    gt_ldist_map,
                    gt_rdist_map,
                    valid_mask_lhand,
                    valid_mask_rhand,
                    config,
                    object_names=object_names,
                    part_label_map=part_label_map,
                    bps_distance_stats=bps_distance_stats,
                )
            elif use_contact_map:
                contact_map = predict_bps_contact_map_for_gaze2hoi(
                    contact_estimator,
                    config,
                    obj_scale,
                    obj_feat,
                    normalized_obj_pc,
                    bps_basis,
                    gaze_map=batch_cuda["gaze_map"][:, -1].unsqueeze(-1),
                    object_names=object_names,
                    part_label_map=part_label_map,
                )
            else:
                contact_map = obj_feat.new_zeros((obj_feat.shape[0], 0))
            contact_feat = torch.cat([contact_map, gaze_score], dim=1)
            obj_feat_final = proc_obj_feat_final_train(
                contact_feat,
                obj_scale,
                obj_cent,
                obj_feat,
            )

            coarse_x_lhand, coarse_x_rhand, coarse_x_obj = diffusion.sampling(
                gaze2hoi,
                obj_feat_final,
                valid_mask_obj.shape[1],
                config.gaze2hoi.model.hand_nfeats,
                config.gaze2hoi.model.obj_nfeats,
                valid_mask_lhand,
                valid_mask_rhand,
                valid_mask_obj,
                device=torch.device("cuda"),
                enc_text=None,
            )

            if use_point_token_output:
                coarse_x_lhand, coarse_x_rhand, coarse_x_obj = (
                    restore_hand_point_token_outputs_with_object_pose_for_gaze2hoi(
                        coarse_x_lhand,
                        coarse_x_rhand,
                        coarse_x_obj,
                        obj_pc_org,
                        lhand_layer,
                        rhand_layer,
                        hand_sample_indices,
                        data_config.name,
                        include_hand_object_dirvec=True,
                        mano_fit_iters=int(
                            getattr(config.gaze2hoi.exp, "mano_fit_iters", 80)
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
                        mano_fit_use_post_opt_losses=bool(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_fit_use_post_opt_losses",
                                True,
                            )
                        ),
                        is_lhand=valid_mask_lhand.any(dim=1),
                        is_rhand=valid_mask_rhand.any(dim=1),
                        valid_mask_lhand=valid_mask_lhand,
                        valid_mask_rhand=valid_mask_rhand,
                        valid_mask_obj=valid_mask_obj,
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
                        mano_refine_mode=str(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_mode",
                                "contact_ramp",
                            )
                        ),
                        mano_refine_max_frames=int(
                            getattr(
                                config.gaze2hoi.exp,
                                "mano_refine_max_frames",
                                20,
                            )
                        ),
                    )
                )

            lhand_contact_frame_mask = _compute_exterior_contact_frame_mask_per_hand(
                coarse_x_lhand,
                lhand_layer,
                coarse_x_obj,
                obj_pc_org,
                obj_pc_normal_org,
                data_config.name,
                valid_mask_lhand,
                valid_mask_obj,
                contact_threshold=contact_threshold,
                contact_min_keypoints=contact_min_keypoints,
                obj_pc_top_idx=obj_pc_top_idx,
            )
            lhand_contact_joint_mask = _compute_contact_joint_mask_per_hand(
                coarse_x_lhand,
                lhand_layer,
                coarse_x_obj,
                obj_pc_org,
                data_config.name,
                valid_mask_lhand,
                valid_mask_obj,
                contact_threshold=contact_threshold,
                obj_pc_top_idx=obj_pc_top_idx,
            )
            rhand_contact_frame_mask = _compute_exterior_contact_frame_mask_per_hand(
                coarse_x_rhand,
                rhand_layer,
                coarse_x_obj,
                obj_pc_org,
                obj_pc_normal_org,
                data_config.name,
                valid_mask_rhand,
                valid_mask_obj,
                contact_threshold=contact_threshold,
                contact_min_keypoints=contact_min_keypoints,
                obj_pc_top_idx=obj_pc_top_idx,
            )
            rhand_contact_joint_mask = _compute_contact_joint_mask_per_hand(
                coarse_x_rhand,
                rhand_layer,
                coarse_x_obj,
                obj_pc_org,
                data_config.name,
                valid_mask_rhand,
                valid_mask_obj,
                contact_threshold=contact_threshold,
                obj_pc_top_idx=obj_pc_top_idx,
            )
            tail_mask_obj = compute_tail_mask(valid_mask_obj, last_k=last_k)
            has_full_tail_obj = tail_mask_obj.sum(dim=1) == last_k
            batch_size = valid_mask_obj.shape[0]
            sample_contact_mask = torch.zeros_like(valid_mask_obj, dtype=torch.bool)
            texts = _get_batch_text_entries(item, batch_size)
            lhand_joints = get_hand_joints_w_tip(coarse_x_lhand, lhand_layer)
            rhand_joints = get_hand_joints_w_tip(coarse_x_rhand, rhand_layer)
            transf_obj_pc = get_transformed_obj_pc(
                coarse_x_obj, obj_pc_org, data_config.name, obj_pc_top_idx
            )
            transf_obj_pc_normal = _rotate_obj_vectors(
                coarse_x_obj, obj_pc_normal_org, data_config.name
            )
            if obj_meshes is not None:
                left_tail_penetration_depth, left_tail_penetration_hand_points, left_tail_penetration_object_points = _max_tail_joint_penetration_detail_per_hand_from_mesh(
                    lhand_joints,
                    coarse_x_obj,
                    obj_meshes,
                    object_names,
                    tail_mask_obj,
                )
                right_tail_penetration_depth, right_tail_penetration_hand_points, right_tail_penetration_object_points = _max_tail_joint_penetration_detail_per_hand_from_mesh(
                    rhand_joints,
                    coarse_x_obj,
                    obj_meshes,
                    object_names,
                    tail_mask_obj,
                )
            else:
                left_tail_penetration_depth, left_tail_penetration_hand_points, left_tail_penetration_object_points = _max_tail_joint_penetration_detail_per_hand(
                    lhand_joints,
                    transf_obj_pc,
                    transf_obj_pc_normal,
                    tail_mask_obj,
                )
                right_tail_penetration_depth, right_tail_penetration_hand_points, right_tail_penetration_object_points = _max_tail_joint_penetration_detail_per_hand(
                    rhand_joints,
                    transf_obj_pc,
                    transf_obj_pc_normal,
                    tail_mask_obj,
                )
            max_penetration_depth = torch.zeros(
                batch_size,
                last_k,
                device=coarse_x_obj.device,
                dtype=coarse_x_obj.dtype,
            )
            id_max = torch.zeros(
                batch_size,
                device=coarse_x_obj.device,
                dtype=coarse_x_obj.dtype,
            )
            id_max_hand = [None] * batch_size
            id_max_obj = [None] * batch_size
            left_penetration_mm = torch.zeros(
                batch_size,
                device=coarse_x_obj.device,
                dtype=coarse_x_obj.dtype,
            )
            right_penetration_mm = torch.zeros(
                batch_size,
                device=coarse_x_obj.device,
                dtype=coarse_x_obj.dtype,
            )
            left_penetration_hand = [None] * batch_size
            left_penetration_obj = [None] * batch_size
            right_penetration_hand = [None] * batch_size
            right_penetration_obj = [None] * batch_size
            for b in range(batch_size):
                use_left_eval, use_right_eval = _resolve_eval_hands(
                    texts[b], item["is_lhand"][b], item["is_rhand"][b]
                )
                if use_left_eval:
                    sample_contact_mask[b] |= lhand_contact_frame_mask[b]
                if use_right_eval:
                    sample_contact_mask[b] |= rhand_contact_frame_mask[b]
                tail_indices = torch.nonzero(tail_mask_obj[b], as_tuple=False).flatten()
                if tail_indices.numel() > 0:
                    if use_left_eval:
                        left_depth_values = left_tail_penetration_depth[b, tail_indices]
                        if left_depth_values.numel() > 0:
                            left_best_idx = int(torch.argmax(left_depth_values).item())
                            left_best_depth = left_depth_values[left_best_idx]
                            left_penetration_mm[b] = left_best_depth * 1000.0
                            left_penetration_hand[b] = left_tail_penetration_hand_points[b][int(tail_indices[left_best_idx].item())]
                            left_penetration_obj[b] = left_tail_penetration_object_points[b][int(tail_indices[left_best_idx].item())]
                        max_penetration_depth[b, : tail_indices.numel()] = torch.maximum(
                            max_penetration_depth[b, : tail_indices.numel()],
                            left_depth_values,
                        )
                    if use_right_eval:
                        right_depth_values = right_tail_penetration_depth[b, tail_indices]
                        if right_depth_values.numel() > 0:
                            right_best_idx = int(torch.argmax(right_depth_values).item())
                            right_best_depth = right_depth_values[right_best_idx]
                            right_penetration_mm[b] = right_best_depth * 1000.0
                            right_penetration_hand[b] = right_tail_penetration_hand_points[b][int(tail_indices[right_best_idx].item())]
                            right_penetration_obj[b] = right_tail_penetration_object_points[b][int(tail_indices[right_best_idx].item())]
                        max_penetration_depth[b, : tail_indices.numel()] = torch.maximum(
                            max_penetration_depth[b, : tail_indices.numel()],
                            right_depth_values,
                        )
                if use_left_eval and left_penetration_mm[b] >= right_penetration_mm[b]:
                    id_max_hand[b] = left_penetration_hand[b]
                    id_max_obj[b] = left_penetration_obj[b]
                if use_right_eval and right_penetration_mm[b] > left_penetration_mm[b]:
                    id_max_hand[b] = right_penetration_hand[b]
                    id_max_obj[b] = right_penetration_obj[b]
                id_max[b] = max_penetration_depth[b].max() * 1000.0
            end_contact = ((sample_contact_mask | (~tail_mask_obj)).all(dim=1)) & has_full_tail_obj
            end_contact_sum += end_contact.sum().item()

            total_samples += batch_size
            joint_penetration_ok = (
                (max_penetration_depth <= penetration_max_depth).all(dim=1)
                & has_full_tail_obj
            )
            joint_penetration_ok_sum += joint_penetration_ok.sum().item()
            success_mask = end_contact & joint_penetration_ok
            success_sum += success_mask.sum().item()

            if qualitative_save_dir:
                for b in range(batch_size):
                    sample_len = int(valid_mask_obj[b].long().sum().item())
                    sample_len = max(sample_len, 1)
                    object_name = (
                        object_names[b] if object_names and b < len(object_names) else "object"
                    )
                    sample_text = texts[b]
                    use_left_eval, use_right_eval = _resolve_eval_hands(
                        sample_text, item["is_lhand"][b], item["is_rhand"][b]
                    )
                    candidate = {
                        "title": f"{object_name} | {sample_text}",
                        "text": str(sample_text),
                        "object_name": str(object_name),
                        "success": bool(success_mask[b].item()),
                        "canonicalize_object_targets": bool(
                            getattr(config.gaze2hoi.model, "canonicalize_point_targets", True)
                        ),
                        "use_left": bool(use_left_eval),
                        "use_right": bool(use_right_eval),
                        "pred_lhand": coarse_x_lhand[b, :sample_len].detach().cpu(),
                        "pred_rhand": coarse_x_rhand[b, :sample_len].detach().cpu(),
                        "pred_obj": coarse_x_obj[b, :sample_len].detach().cpu(),
                        "gt_lhand": x_lhand[b, :sample_len].detach().cpu(),
                        "gt_rhand": x_rhand[b, :sample_len].detach().cpu(),
                        "gt_obj": x_obj[b, :sample_len].detach().cpu(),
                        "obj_pc": obj_pc_org[b].detach().cpu(),
                    }
                    qualitative_seen = _reservoir_add_sample(
                        qualitative_samples,
                        candidate,
                        qualitative_seen,
                        int(qualitative_num_samples),
                        qualitative_rng,
                    )

            if failure_save_path:
                for b in range(batch_size):
                    use_left_eval, use_right_eval = _resolve_eval_hands(
                        texts[b], item["is_lhand"][b], item["is_rhand"][b]
                    )
                    is_success = bool(success_mask[b].item())
                    object_name = (
                        object_names[b] if object_names and b < len(object_names) else None
                    )
                    sample_text = item.get("text", [None])[b]
                    sample_record = {
                        "text": sample_text,
                        "object_name": object_name,
                        "success": is_success,
                        "contact": bool(end_contact[b].item()),
                        "penetration_ok": bool(joint_penetration_ok[b].item()),
                        "joint_penetration_max_depth_threshold_m": penetration_max_depth,
                        "joint_penetration_max_depth_last_frames": [
                            int(v)
                            for v in (max_penetration_depth[b] * 1000.0).round().long().tolist()
                        ],
                        "penetration_max_mm": float(id_max[b].item()),
                        "id_max_mm": float(id_max[b].item()),
                        "id_max_hand": id_max_hand[b],
                        "id_max_obj": id_max_obj[b],
                        "left_id_max_mm": float(left_penetration_mm[b].item()),
                        "right_id_max_mm": float(right_penetration_mm[b].item()),
                    }
                    image_path = None
                    target_image_dir = success_image_dir if is_success else failure_image_dir
                    if target_image_dir:
                        os.makedirs(target_image_dir, exist_ok=True)
                        sample_index = len(sample_save_list) + 1
                        sample_slug = _sanitize_failure_name(sample_text, default=f"sample_{sample_index:04d}")
                        object_slug = _sanitize_failure_name(object_name, default="object")
                        image_path = osp.join(
                            target_image_dir,
                            f"{sample_index:04d}_{object_slug}_{sample_slug}_penmax_{float(id_max[b].item()):.1f}mm.png",
                        )
                        last_frame_idx = int(valid_mask_obj[b].long().sum().item()) - 1
                        last_frame_idx = max(last_frame_idx, 0)
                        penetration_segments = []
                        if (
                            use_left_eval
                            and left_penetration_hand[b] is not None
                            and left_penetration_obj[b] is not None
                            and float(left_penetration_mm[b].item()) > 0.0
                        ):
                            penetration_segments.append(
                                {
                                    "hand": "left",
                                    "depth_mm": float(left_penetration_mm[b].item()),
                                    "hand_point": left_penetration_hand[b],
                                    "object_point": left_penetration_obj[b],
                                }
                            )
                        if (
                            use_right_eval
                            and right_penetration_hand[b] is not None
                            and right_penetration_obj[b] is not None
                            and float(right_penetration_mm[b].item()) > 0.0
                        ):
                            penetration_segments.append(
                                {
                                    "hand": "right",
                                    "depth_mm": float(right_penetration_mm[b].item()),
                                    "hand_point": right_penetration_hand[b],
                                    "object_point": right_penetration_obj[b],
                                }
                            )
                        _save_failure_visualization(
                            output_path=image_path,
                            sample_title=str(sample_text),
                            object_name=str(object_name),
                            obj_points=transf_obj_pc[b, last_frame_idx].detach().cpu().numpy(),
                            lhand_joints=(
                                lhand_joints[b, last_frame_idx].detach().cpu().numpy()
                                if bool(valid_mask_lhand[b, last_frame_idx].item())
                                else None
                            ),
                            rhand_joints=(
                                rhand_joints[b, last_frame_idx].detach().cpu().numpy()
                                if bool(valid_mask_rhand[b, last_frame_idx].item())
                                else None
                            ),
                            lhand_contact_joint_mask=lhand_contact_joint_mask[
                                b, last_frame_idx
                            ].detach().cpu().numpy(),
                            rhand_contact_joint_mask=rhand_contact_joint_mask[
                                b, last_frame_idx
                            ].detach().cpu().numpy(),
                            use_left_eval=bool(use_left_eval),
                            use_right_eval=bool(use_right_eval),
                            end_contact_ok=bool(end_contact[b].item()),
                            penetration_ok=bool(joint_penetration_ok[b].item()),
                            penetration_mm=float(max_penetration_depth[b].max().item() * 1000.0),
                            penetration_threshold_mm=float(penetration_max_depth * 1000.0),
                            penetration_segments=penetration_segments,
                        )
                    sample_record["image_path"] = image_path
                    sample_save_list.append(sample_record)
                    if not is_success:
                        failed_save_list.append(sample_record)

    success_rate = success_sum / max(total_samples, 1)
    end_contact_rate = end_contact_sum / max(total_samples, 1)
    joint_penetration_ok_rate = joint_penetration_ok_sum / max(total_samples, 1)
    eval_metrics = {
        "success_rate": success_rate,
        "success_samples": int(success_sum),
        "end_contact_rate": end_contact_rate,
        "end_contact_samples": int(end_contact_sum),
        "joint_penetration_ok_rate": joint_penetration_ok_rate,
        "joint_penetration_ok_samples": int(joint_penetration_ok_sum),
        "failed_samples": len(failed_save_list),
        "total_samples": total_samples,
    }

    if failure_save_path:
        import pickle

        failure_dir = osp.dirname(failure_save_path)
        if failure_dir:
            os.makedirs(failure_dir, exist_ok=True)
        with open(failure_save_path, "wb") as f:
            pickle.dump(sample_save_list, f)

    if qualitative_save_dir:
        _save_eval_qualitative_report(
            qualitative_save_dir,
            qualitative_samples,
            lhand_layer,
            rhand_layer,
            data_config.name,
        )

    return eval_metrics
