import numpy as np
import torch

from lib.models.mano import SKELETONS_W_TIP
from lib.utils.rot import rot6d_to_rotmat


def scatter_sparse_values_to_vertices(sparse_values, nearest_idx, npts):
    out = np.zeros((sparse_values.shape[0], npts), dtype=np.float32)
    counts = np.zeros((sparse_values.shape[0], npts), dtype=np.float32)
    for t in range(sparse_values.shape[0]):
        np.add.at(out[t], nearest_idx, sparse_values[t])
        np.add.at(counts[t], nearest_idx, 1.0)
    valid = counts > 0
    out[valid] /= counts[valid]
    vertex_mask = counts.sum(axis=0) > 0
    return out, vertex_mask


def contact_values_for_display(values, contact_mode):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if contact_mode == "distance_raw":
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros_like(values, dtype=np.float32)
        vmax = float(np.percentile(values[finite], 99))
        vmax = max(vmax, 1e-6)
        return 1.0 - np.clip(values / vmax, 0.0, 1.0)
    clipped = np.clip(values, 0.0, 1.0)
    vmin = float(np.min(clipped))
    vmax = float(np.max(clipped))
    if vmax - vmin < 1e-8:
        return clipped
    return (clipped - vmin) / (vmax - vmin)


def gaze_values_for_display(values, scale=1.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    return np.clip((values + 1.0) * 0.5 * scale, 0.0, 1.0)


def contact_colorbar_spec(values, contact_mode):
    if contact_mode == "distance_raw":
        vals = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(vals)
        vmax = float(np.percentile(vals[finite], 99)) if np.any(finite) else 1.0
        vmax = max(vmax, 1e-6)
        return {
            "ticks": [0.0, 0.5, 1.0],
            "ticklabels": [f"{vmax:.3f}", f"{0.5 * vmax:.3f}", "0.000"],
            "label": "distance_raw display (far -> near)",
        }
    if contact_mode == "distance_closeness":
        return {
            "ticks": [0.0, 0.5, 1.0],
            "ticklabels": ["0.0", "0.5", "1.0"],
            "label": "distance_closeness",
        }
    return {
        "ticks": [0.0, 1.0],
        "ticklabels": ["0", "1"],
        "label": "binary_contact",
    }


def gaze_colorbar_spec():
    return {
        "ticks": [0.0, 0.5, 1.0],
        "ticklabels": ["-1.0", "0.0", "1.0"],
        "label": "gaze cosine",
    }


def plot_hand_skeleton(ax, joints, color):
    ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        s=18,
        c=color,
        edgecolors="none",
        alpha=0.95,
    )
    for start_idx, end_idx, _ in SKELETONS_W_TIP:
        p0 = joints[start_idx]
        p1 = joints[end_idx]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=color,
            linewidth=1.8,
            alpha=0.9,
        )


def transform_points_to_canonical(points, x_obj):
    obj_trans = x_obj[..., :3]
    obj_rotmat = rot6d_to_rotmat(x_obj[..., 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], x_obj.shape[1], 3, 3
    )
    return torch.einsum("btji,btnj->btni", obj_rotmat, points - obj_trans.unsqueeze(2))


def transform_points_to_initial_object_canonical(points, x_obj):
    obj_trans = x_obj[:, :1, :3].expand(-1, points.shape[1], -1)
    obj_rotmat = rot6d_to_rotmat(x_obj[:, :1, 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], 1, 3, 3
    )
    obj_rotmat = obj_rotmat.expand(-1, points.shape[1], -1, -1)
    return torch.einsum("btji,btnj->btni", obj_rotmat, points - obj_trans.unsqueeze(2))


def transform_dirs_to_canonical(vectors, x_obj):
    obj_rotmat = rot6d_to_rotmat(x_obj[..., 3:9].reshape(-1, 6)).reshape(
        x_obj.shape[0], x_obj.shape[1], 3, 3
    )
    return torch.einsum("btji,btj->bti", obj_rotmat, vectors)


def plot_gaze_vector(ax, origin, direction, color="#ffd400", length=0.28):
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return
    direction = direction / norm
    origin = np.asarray(origin, dtype=np.float32)
    end = origin + direction * length
    ax.scatter(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        s=52,
        c=color,
        edgecolors="black",
        linewidths=0.6,
        alpha=1.0,
    )
    ax.plot(
        [origin[0], end[0]],
        [origin[1], end[1]],
        [origin[2], end[2]],
        color=color,
        linewidth=3.2,
        alpha=1.0,
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        [end[2]],
        s=28,
        c=color,
        edgecolors="black",
        linewidths=0.4,
        alpha=1.0,
    )


def line_segment_to_object_hit(origin, direction, object_points, eps=1e-8):
    object_points = np.asarray(object_points, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + eps)

    rel = object_points - origin[None, :]
    t_vals = rel @ direction
    forward_mask = t_vals > 0
    if not np.any(forward_mask):
        return None

    rel_fwd = rel[forward_mask]
    t_fwd = t_vals[forward_mask]
    proj = t_fwd[:, None] * direction[None, :]
    perp = rel_fwd - proj
    perp_dist = np.linalg.norm(perp, axis=1)

    obj_extent = np.max(object_points.max(axis=0) - object_points.min(axis=0)) + eps
    hit_band = max(obj_extent * 0.015, 5e-3)
    near_mask = perp_dist <= hit_band
    if np.any(near_mask):
        candidate_t = t_fwd[near_mask]
        hit_t = candidate_t[np.argmin(candidate_t)]
    else:
        score = t_fwd + perp_dist * 4.0
        hit_t = t_fwd[np.argmin(score)]

    line_end = origin + direction * hit_t
    return origin, line_end


def line_segment_through_object(origin, direction, object_points, eps=1e-8):
    object_points = np.asarray(object_points, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + eps)

    rel = object_points - origin[None, :]
    t_vals = rel @ direction
    forward_mask = t_vals > 0
    if not np.any(forward_mask):
        return None

    rel_fwd = rel[forward_mask]
    t_fwd = t_vals[forward_mask]
    proj = t_fwd[:, None] * direction[None, :]
    perp = rel_fwd - proj
    perp_dist = np.linalg.norm(perp, axis=1)

    obj_extent = np.max(object_points.max(axis=0) - object_points.min(axis=0)) + eps
    hit_band = max(obj_extent * 0.015, 5e-3)
    near_mask = perp_dist <= hit_band
    if not np.any(near_mask):
        return None

    t_candidates = t_fwd[near_mask]
    t_min = float(np.min(t_candidates))
    t_max = float(np.max(t_candidates))
    entry = origin + direction * t_min
    exit = origin + direction * t_max
    return entry, exit


def display_rotation_matrix():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def apply_display_rotation(points, rot):
    return np.asarray(points, dtype=np.float32) @ rot.T


def apply_display_rotation_to_dirs(vectors, rot):
    return np.asarray(vectors, dtype=np.float32) @ rot.T


def flip_vertical(points):
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[..., 1] *= -1.0
    return flipped


def flip_xy_plane(points):
    flipped = np.asarray(points, dtype=np.float32).copy()
    flipped[..., 2] *= -1.0
    return flipped


def plot_axes_indicator(ax, points, axis_scale=0.22):
    pts = np.asarray(points, dtype=np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0 + 1e-6
    origin = center - radius * 0.8
    axis_len = radius * axis_scale * 2.0
    axes = [
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), "x", "#d62728"),
        (np.array([0.0, 1.0, 0.0], dtype=np.float32), "y", "#2ca02c"),
        (np.array([0.0, 0.0, 1.0], dtype=np.float32), "z", "#1f77b4"),
    ]
    ax.scatter([origin[0]], [origin[1]], [origin[2]], s=16, c="black", edgecolors="none")
    for direction, label, color in axes:
        end = origin + direction * axis_len
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=2.0,
            alpha=0.95,
        )
        ax.text(end[0], end[1], end[2], label, color=color, fontsize=8)


def set_hand_focused_view(ax, obj_points, lhand_points=None, rhand_points=None):
    scene_parts = [np.asarray(obj_points, dtype=np.float32)]
    hand_parts = []
    if lhand_points is not None and len(lhand_points) > 0:
        lhand_points = np.asarray(lhand_points, dtype=np.float32)
        scene_parts.append(lhand_points)
        hand_parts.append(lhand_points)
    if rhand_points is not None and len(rhand_points) > 0:
        rhand_points = np.asarray(rhand_points, dtype=np.float32)
        scene_parts.append(rhand_points)
        hand_parts.append(rhand_points)
    scene_points = np.concatenate(scene_parts, axis=0)
    obj_center = np.asarray(obj_points, dtype=np.float32).mean(axis=0)
    if hand_parts:
        hand_center = np.concatenate(hand_parts, axis=0).mean(axis=0)
    else:
        hand_center = obj_center
    center = 0.65 * hand_center + 0.35 * obj_center
    mins = scene_points.min(axis=0)
    maxs = scene_points.max(axis=0)
    radius = np.max(maxs - mins) / 2.0 + 1e-6
    radius *= 0.72
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    view_vec = obj_center - hand_center
    horiz = float(np.linalg.norm(view_vec[[0, 2]]))
    elev = float(np.degrees(np.arctan2(view_vec[1], max(horiz, 1e-6))))
    azim = float(np.degrees(np.arctan2(view_vec[2], view_vec[0])) - 90.0)
    ax.view_init(elev=elev * 0.7 + 10.0, azim=azim)
    plot_axes_indicator(ax, scene_points)
