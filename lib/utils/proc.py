import numpy as np
from pathlib import Path

import torch

from lib.models.mano import build_mano_aa
from lib.utils.rot import (
    axis_angle_to_rot6d,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
)
from lib.utils.proc_output import (
    get_hand_joints_w_tip,
    get_hand_layer_out,
    get_transformed_obj_pc,
    get_NN,
    get_interior,
)


def load_bps_basis(
    bps_path,
    device=None,
    dtype=torch.float32,
    basis_key="obj",
    target_max_norm=1.0,
):
    """Load a BPS basis from NumPy or a DiffH2O/GRAB PyTorch payload.

    Text2HOI normalizes object point clouds inside a unit-radius canonical
    space (maximum object radius 0.85).  DiffH2O stores its object basis at a
    metric radius of roughly 0.15, so the loaded basis is rescaled to
    ``target_max_norm`` by default while preserving its angular distribution.
    """
    bps_path = Path(bps_path)
    if bps_path.suffix.lower() in (".pt", ".pth"):
        try:
            payload = torch.load(bps_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 does not expose weights_only.
            payload = torch.load(bps_path, map_location="cpu")
        if isinstance(payload, dict):
            if basis_key not in payload:
                raise ValueError(
                    f"Expected BPS payload key {basis_key!r} in {bps_path}; "
                    f"available keys: {sorted(payload)}"
                )
            payload = payload[basis_key]
        if torch.is_tensor(payload):
            payload = payload.detach().cpu().numpy()
        bps_points = np.asarray(payload, dtype=np.float32)
    else:
        bps_points = np.asarray(np.load(bps_path), dtype=np.float32)

    if bps_points.ndim == 3 and bps_points.shape[0] == 1:
        bps_points = bps_points[0]
    if bps_points.ndim != 2 or bps_points.shape[1] != 3:
        raise ValueError(
            f"Expected BPS basis with shape (K, 3), got {bps_points.shape}"
        )
    if target_max_norm is not None:
        max_norm = float(np.linalg.norm(bps_points, axis=1).max())
        if not np.isfinite(max_norm) or max_norm <= 0.0:
            raise ValueError(f"Invalid BPS basis maximum norm: {max_norm}")
        bps_points = bps_points * (float(target_max_norm) / max_norm)

    bps_points = np.ascontiguousarray(bps_points, dtype=np.float32)
    bps_tensor = torch.from_numpy(bps_points)
    if device is not None:
        bps_tensor = bps_tensor.to(device=device, dtype=dtype)
    elif dtype is not None:
        bps_tensor = bps_tensor.to(dtype=dtype)
    return bps_tensor


def load_bps_distance_stats(stats_path, device=None, dtype=torch.float32):
    stats_np = np.load(stats_path, allow_pickle=True)
    stats = {}
    for key in stats_np.files:
        value = stats_np[key]
        if np.isscalar(value) or value.shape == ():
            stats[key] = value.item()
            continue
        tensor = torch.from_numpy(value)
        if device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        elif dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        stats[key] = tensor
    return stats


def normalize_bps_distance_map(contact_map, stats, use_per_dim=True, eps=1e-8):
    if stats is None:
        return contact_map

    if use_per_dim and "mean_dim" in stats and "std_dim" in stats:
        mean = stats["mean_dim"].to(device=contact_map.device, dtype=contact_map.dtype)
        std = stats["std_dim"].to(device=contact_map.device, dtype=contact_map.dtype)
    else:
        mean_value = stats["mean"]
        std_value = stats["std"]
        mean = torch.as_tensor(mean_value, device=contact_map.device, dtype=contact_map.dtype)
        std = torch.as_tensor(std_value, device=contact_map.device, dtype=contact_map.dtype)

    return (contact_map - mean) / std.clamp_min(eps)


def _get_bps_nearest_data(obj_pc, bps_points):
    if obj_pc.dim() != 3 or obj_pc.shape[-1] != 3:
        raise ValueError(f"Expected obj_pc shape (B, N, 3), got {tuple(obj_pc.shape)}")
    if bps_points.dim() != 2 or bps_points.shape[-1] != 3:
        raise ValueError(
            f"Expected bps_points shape (K, 3), got {tuple(bps_points.shape)}"
        )

    basis = bps_points.to(device=obj_pc.device, dtype=obj_pc.dtype)
    basis_batch = basis.unsqueeze(0).expand(obj_pc.shape[0], -1, -1)
    pairwise_dist = torch.cdist(basis_batch, obj_pc)
    nearest_idx = pairwise_dist.argmin(dim=-1)
    gather_idx = nearest_idx.unsqueeze(-1).expand(-1, -1, obj_pc.shape[-1])
    nearest_points = torch.gather(obj_pc, 1, gather_idx)
    return basis_batch, nearest_points, nearest_idx


def compute_bps_feature(obj_pc, bps_points):
    """
    Match BimArt's BPS feature: nearest object point minus basis point.

    Args:
        obj_pc: (B, N, 3) normalized object point cloud
        bps_points: (K, 3) BPS basis points

    Returns:
        (B, K * 3) flattened BPS feature
    """
    basis_batch, nearest_points, _ = _get_bps_nearest_data(obj_pc, bps_points)
    return (nearest_points - basis_batch).reshape(obj_pc.shape[0], -1)


def compute_bps_distance_feature(obj_pc, bps_points):
    """
    Scalar BPS feature: distance from each basis point to its nearest object point.

    Args:
        obj_pc: (B, N, 3) normalized object point cloud
        bps_points: (K, 3) BPS basis points

    Returns:
        (B, K) BPS distance feature
    """
    basis_batch, nearest_points, _ = _get_bps_nearest_data(obj_pc, bps_points)
    return (nearest_points - basis_batch).norm(dim=-1)


def compute_bps_gaze_alignment_sequence(
    gaze,
    x_obj,
    obj_pc,
    bps_points,
    obj_cent,
    obj_scale,
    valid_mask=None,
    positive_only=False,
    alignment_method="direction",
    eps=1e-8,
):
    """
    Compare gaze direction and BPS-to-object directions for each frame.

    Args:
        gaze: (B, T, 2, 3) gaze arrows [origin, direction] or (B, T, 3) directions
        x_obj: (B, T, >=9) object motion in world space. Only its initial pose
            is used to map gaze into a fixed object frame.
        obj_pc: (B, N, 3) normalized object point cloud
        bps_points: (K, 3) normalized BPS basis
        obj_cent: (B, 3) object centroid before normalization
        obj_scale: (B,) normalization scale
        valid_mask: (B, T) optional valid-frame mask
        alignment_method: "direction" uses gaze direction; "origin" uses the
            previous origin-to-object direction.

    Returns:
        (B, T, K) frame-wise gaze-alignment score per BPS point. By default
        this keeps the full cosine range in [-1, 1].
    """
    if x_obj.dim() != 3 or x_obj.shape[-1] < 9:
        raise ValueError(f"Expected x_obj shape (B,T,>=9), got {tuple(x_obj.shape)}")

    if gaze.dim() == 5 and gaze.shape[-1] == 1:
        gaze = gaze.squeeze(-1)

    device = x_obj.device
    dtype = x_obj.dtype
    B, T = x_obj.shape[:2]
    basis = bps_points.to(device=device, dtype=dtype)
    method = str(alignment_method).lower()
    if method in ("direction_vector", "gaze_direction", "new"):
        method = "direction"
    elif method in ("origin_vector", "gaze_origin", "old"):
        method = "origin"
    if method not in ("direction", "origin"):
        raise ValueError(
            f"Unknown gaze alignment_method={alignment_method!r}; expected 'direction' or 'origin'."
        )

    init_obj_trans = x_obj[:, :1, :3]
    init_obj_rot6d = x_obj[:, :1, 3:9]
    init_obj_rotmat = rot6d_to_rotmat(init_obj_rot6d.reshape(-1, 6)).reshape(
        B, 1, 3, 3
    )

    if gaze.dim() == 4 and gaze.shape[2:] == (2, 3):
        gaze_origin = gaze[:, :, 0, :].to(device=device, dtype=dtype)
        gaze_dir = gaze[:, :, 1, :].to(device=device, dtype=dtype)
    elif gaze.dim() == 3 and gaze.shape[-1] == 3:
        gaze_origin = init_obj_trans.expand(-1, T, -1)
        gaze_dir = gaze.to(device=device, dtype=dtype)
    else:
        raise ValueError(
            f"Expected gaze shape (B,T,2,3) or (B,T,3), got {tuple(gaze.shape)}"
        )

    obj_cent = obj_cent.to(device=device, dtype=dtype)
    obj_scale = obj_scale.to(device=device, dtype=dtype).view(B, 1, 1)
    basis_local = basis.unsqueeze(0) * obj_scale + obj_cent.unsqueeze(1)

    _, nearest_points_norm, _ = _get_bps_nearest_data(obj_pc, basis)
    nearest_points_local = nearest_points_norm.to(dtype=dtype) * obj_scale + obj_cent.unsqueeze(1)

    bps_to_obj = nearest_points_local - basis_local
    bps_to_obj = bps_to_obj / bps_to_obj.norm(dim=-1, keepdim=True).clamp_min(eps)

    if method == "origin":
        gaze_origin_canon = torch.einsum(
            "btji,btj->bti",
            init_obj_rotmat.expand(-1, T, -1, -1),
            gaze_origin - init_obj_trans,
        )
        gaze_to_obj = nearest_points_local.unsqueeze(1) - gaze_origin_canon.unsqueeze(2)
        gaze_to_obj = gaze_to_obj / gaze_to_obj.norm(dim=-1, keepdim=True).clamp_min(eps)
        alignment = torch.einsum("btki,bki->btk", gaze_to_obj, bps_to_obj)
    else:
        gaze_dir_canon = torch.einsum(
            "btji,btj->bti",
            init_obj_rotmat.expand(-1, T, -1, -1),
            gaze_dir,
        )
        gaze_dir_canon = gaze_dir_canon / gaze_dir_canon.norm(dim=-1, keepdim=True).clamp_min(eps)
        alignment = torch.einsum("bti,bki->btk", gaze_dir_canon, bps_to_obj)

    if positive_only:
        alignment = alignment.clamp_min(0.0)

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=device, dtype=dtype)
        alignment = alignment * valid_mask.unsqueeze(-1)
    return alignment


def compute_bps_gaze_alignment_map(
    gaze,
    x_obj,
    obj_pc,
    bps_points,
    obj_cent,
    obj_scale,
    valid_mask=None,
    positive_only=False,
    alignment_method="direction",
    eps=1e-8,
):
    alignment = compute_bps_gaze_alignment_sequence(
        gaze,
        x_obj,
        obj_pc,
        bps_points,
        obj_cent,
        obj_scale,
        valid_mask=valid_mask,
        positive_only=positive_only,
        alignment_method=alignment_method,
        eps=eps,
    )
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=alignment.device, dtype=alignment.dtype)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return alignment.sum(dim=1) / denom
    return alignment.mean(dim=1)


def compute_bps_contact_map(obj_pc, bps_points, lcontact_map, rcontact_map):
    """
    Build BimArt-style sparse contact conditioning on top of BPS indices.

    Args:
        obj_pc: (B, N, 3) normalized object point cloud
        bps_points: (K, 3) BPS basis points
        lcontact_map: (B, N) dense left-hand contact labels
        rcontact_map: (B, N) dense right-hand contact labels

    Returns:
        (B, 2 * K) sparse contact map [left_bps, right_bps]
    """
    _, _, nearest_idx = _get_bps_nearest_data(obj_pc, bps_points)
    left_sparse = torch.gather(
        lcontact_map.to(device=obj_pc.device, dtype=obj_pc.dtype), 1, nearest_idx
    )
    right_sparse = torch.gather(
        rcontact_map.to(device=obj_pc.device, dtype=obj_pc.dtype), 1, nearest_idx
    )
    return torch.cat([left_sparse, right_sparse], dim=1)


def compute_bps_contact_feature_sequence(
    obj_pc,
    bps_points,
    ldist_map,
    rdist_map,
    lvalid_mask=None,
    rvalid_mask=None,
    mode="distance_raw",
    contact_threshold=0.02,
    distance_scale=0.02,
):
    """
    Build a frame-wise sparse BPS contact feature using one of several mappings.

    Modes:
        - distance_raw: raw minimum distance per BPS point
        - distance_closeness: 1 - clamp(d / distance_scale, 0, 1)
        - binary_contact: 1 if d < contact_threshold else 0

    Returns:
        (B, T, 2 * K) sparse frame-wise feature [left_bps, right_bps]
    """
    _, _, nearest_idx = _get_bps_nearest_data(obj_pc, bps_points)
    nearest_idx_t = nearest_idx.unsqueeze(1)

    def _map_dense(dist_map, valid_mask):
        if dist_map.dim() != 4:
            raise ValueError(
                "Expected dist_map shape (B, T, N, J), got "
                f"{tuple(dist_map.shape)}"
            )
        dense = dist_map.amin(dim=-1)
        if mode == "distance_raw":
            mapped = dense
        elif mode == "distance_closeness":
            mapped = 1.0 - torch.clamp(dense / max(distance_scale, 1e-8), 0.0, 1.0)
        elif mode == "binary_contact":
            mapped = (dense < contact_threshold).to(dtype=dense.dtype)
        else:
            raise ValueError(f"Unsupported contact feature mode: {mode}")

        gather_idx = nearest_idx_t.expand(-1, mapped.shape[1], -1)
        sparse = torch.gather(mapped, 2, gather_idx)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=sparse.device, dtype=sparse.dtype)
            sparse = sparse * valid_mask.unsqueeze(-1)
        return sparse

    left_sparse = _map_dense(ldist_map, lvalid_mask)
    right_sparse = _map_dense(rdist_map, rvalid_mask)
    return torch.cat([left_sparse, right_sparse], dim=-1)


def compute_bps_contact_feature_map(
    obj_pc,
    bps_points,
    ldist_map,
    rdist_map,
    lvalid_mask=None,
    rvalid_mask=None,
    mode="distance_raw",
    contact_threshold=0.02,
    distance_scale=0.02,
):
    """
    Reduce frame-wise sparse BPS contact features over time to a sequence-level map.

    Returns:
        (B, 2 * K) sparse feature [left_bps, right_bps]
    """
    sparse_seq = compute_bps_contact_feature_sequence(
        obj_pc,
        bps_points,
        ldist_map,
        rdist_map,
        lvalid_mask=lvalid_mask,
        rvalid_mask=rvalid_mask,
        mode=mode,
        contact_threshold=contact_threshold,
        distance_scale=distance_scale,
    )

    if lvalid_mask is not None or rvalid_mask is not None:
        valid_mask = None
        if lvalid_mask is not None and rvalid_mask is not None:
            valid_mask = lvalid_mask | rvalid_mask
        elif lvalid_mask is not None:
            valid_mask = lvalid_mask
        else:
            valid_mask = rvalid_mask
        valid_mask = valid_mask.to(device=sparse_seq.device, dtype=sparse_seq.dtype)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (sparse_seq * valid_mask.unsqueeze(-1)).sum(dim=1) / denom

    return sparse_seq.mean(dim=1)


def _reduce_sequence_dist_map(dist_map, valid_mask=None):
    if dist_map.dim() != 4:
        raise ValueError(
            f"Expected dist_map shape (B, T, N, J), got {tuple(dist_map.shape)}"
        )

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=dist_map.device, dtype=torch.bool)
        if valid_mask.dim() != 2 or valid_mask.shape != dist_map.shape[:2]:
            raise ValueError(
                "valid_mask must have shape (B, T) matching the first two "
                f"dimensions of dist_map, got {tuple(valid_mask.shape)}"
            )
        inf = torch.full((), torch.finfo(dist_map.dtype).max, device=dist_map.device, dtype=dist_map.dtype)
        masked_dist_map = torch.where(
            valid_mask.unsqueeze(-1).unsqueeze(-1), dist_map, inf
        )
        dense_dist = masked_dist_map.amin(dim=(1, 3))
        has_valid = valid_mask.any(dim=1, keepdim=True)
        return torch.where(has_valid, dense_dist, torch.zeros_like(dense_dist))

    return dist_map.amin(dim=(1, 3))


def compute_bps_distance_map_sequence(
    obj_pc,
    bps_points,
    ldist_map,
    rdist_map,
    lvalid_mask=None,
    rvalid_mask=None,
):
    """
    Build a frame-wise sparse BPS distance feature.

    Args:
        obj_pc: (B, N, 3) normalized object point cloud
        bps_points: (K, 3) BPS basis points
        ldist_map: (B, T, N, J) left-hand object-to-joint distances
        rdist_map: (B, T, N, J) right-hand object-to-joint distances
        lvalid_mask: (B, T) optional valid-frame mask for the left hand
        rvalid_mask: (B, T) optional valid-frame mask for the right hand

    Returns:
        (B, T, 2 * K) sparse frame-wise distance map [left_bps, right_bps]
    """
    _, _, nearest_idx = _get_bps_nearest_data(obj_pc, bps_points)
    nearest_idx_t = nearest_idx.unsqueeze(1)

    def _gather_framewise(dist_map, valid_mask):
        if dist_map.dim() != 4:
            raise ValueError(
                "Expected dist_map shape (B, T, N, J), got "
                f"{tuple(dist_map.shape)}"
            )
        frame_dense = dist_map.amin(dim=-1)
        gather_idx = nearest_idx_t.expand(
            -1,
            frame_dense.shape[1],
            -1,
        )
        sparse = torch.gather(frame_dense, 2, gather_idx)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=sparse.device, dtype=sparse.dtype)
            sparse = sparse * valid_mask.unsqueeze(-1)
        return sparse

    left_sparse = _gather_framewise(ldist_map, lvalid_mask)
    right_sparse = _gather_framewise(rdist_map, rvalid_mask)
    return torch.cat([left_sparse, right_sparse], dim=-1)


def compute_bps_distance_map(
    obj_pc,
    bps_points,
    ldist_map,
    rdist_map,
    lvalid_mask=None,
    rvalid_mask=None,
):
    """
    Build a Text2HOI-style sequence-level sparse distance feature by averaging
    frame-wise BPS distances over time.

    Args:
        obj_pc: (B, N, 3) normalized object point cloud
        bps_points: (K, 3) BPS basis points
        ldist_map: (B, T, N, J) left-hand object-to-joint distances
        rdist_map: (B, T, N, J) right-hand object-to-joint distances
        lvalid_mask: (B, T) optional valid-frame mask for the left hand
        rvalid_mask: (B, T) optional valid-frame mask for the right hand

    Returns:
        (B, 2 * K) sparse distance map [left_bps, right_bps]
    """
    sparse_seq = compute_bps_distance_map_sequence(
        obj_pc,
        bps_points,
        ldist_map,
        rdist_map,
        lvalid_mask,
        rvalid_mask,
    )

    if lvalid_mask is not None or rvalid_mask is not None:
        valid_mask = None
        if lvalid_mask is not None and rvalid_mask is not None:
            valid_mask = lvalid_mask | rvalid_mask
        elif lvalid_mask is not None:
            valid_mask = lvalid_mask
        else:
            valid_mask = rvalid_mask
        valid_mask = valid_mask.to(device=sparse_seq.device, dtype=sparse_seq.dtype)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (sparse_seq * valid_mask.unsqueeze(-1)).sum(dim=1) / denom

    return sparse_seq.mean(dim=1)


def _flatten_obj_feature(obj_feat):
    if obj_feat.dim() == 2:
        return obj_feat
    if obj_feat.dim() == 3:
        if obj_feat.shape[1] == 1:
            return obj_feat[:, 0]
        return obj_feat.reshape(obj_feat.shape[0], -1)
    raise ValueError(f"Unsupported obj_feat shape: {tuple(obj_feat.shape)}")


def _compute_gaze_ray_distance_and_mask(
    gaze,
    nframes,
    x_obj,
    obj_pc,
    obj_cent=None,
    obj_scale=None,
):
    if gaze.dim() != 4 or gaze.shape[2:] != (2, 3):
        raise ValueError(f"Expected gaze shape (B,T,2,3), got {tuple(gaze.shape)}")
    if x_obj.dim() != 3 or x_obj.shape[-1] < 9:
        raise ValueError(f"Expected x_obj shape (B,T,>=9), got {tuple(x_obj.shape)}")
    if obj_pc.dim() != 3 or obj_pc.shape[-1] != 3:
        raise ValueError(f"Expected obj_pc shape (B,N,3), got {tuple(obj_pc.shape)}")

    B, T = gaze.shape[:2]
    device = gaze.device

    if not torch.is_tensor(nframes):
        nframes = torch.as_tensor(nframes, device=device)
    nframes = nframes.to(device=device).long().clamp(min=0, max=T)

    init_obj_trans = x_obj[:, :1, :3]
    init_obj_rot6d = x_obj[:, :1, 3:9]
    init_obj_rotmat = rot6d_to_rotmat(init_obj_rot6d.reshape(-1, 6)).reshape(
        B, 1, 3, 3
    )

    gaze_origin_world = gaze[:, :, 0, :]  # (B,T,3)
    gaze_dir_world = gaze[:, :, 1, :]  # (B,T,3)
    gaze_origin = torch.einsum(
        "btji,btj->bti",
        init_obj_rotmat.expand(-1, T, -1, -1),
        gaze_origin_world - init_obj_trans,
    )
    gaze_dir = torch.einsum(
        "btji,btj->bti",
        init_obj_rotmat.expand(-1, T, -1, -1),
        gaze_dir_world,
    )
    if (obj_cent is None) != (obj_scale is None):
        raise ValueError("obj_cent and obj_scale must be provided together")
    if obj_cent is not None:
        obj_cent = obj_cent.to(device=device, dtype=gaze.dtype).view(B, 1, 3)
        obj_scale = obj_scale.to(device=device, dtype=gaze.dtype).view(B, 1, 1)
        gaze_origin = (gaze_origin - obj_cent) / obj_scale.clamp_min(1e-8)

    gaze_dir_norm = torch.linalg.norm(gaze_dir, dim=-1, keepdim=True)
    valid_dir_mask = gaze_dir_norm.squeeze(-1) > 1e-8
    gaze_dir = gaze_dir / gaze_dir_norm.clamp_min(1e-8)

    frame_mask = torch.arange(T, device=device).unsqueeze(0) < nframes.unsqueeze(1)
    frame_mask = frame_mask & valid_dir_mask

    origin = gaze_origin.unsqueeze(2)  # (B,T,1,3)
    direction = gaze_dir.unsqueeze(2)  # (B,T,1,3)
    object_points = obj_pc.unsqueeze(1)
    vec = object_points - origin  # (B,T,N,3)
    proj_len = (vec * direction).sum(dim=-1).clamp_min(0.0)
    closest = origin + proj_len.unsqueeze(-1) * direction
    dist = torch.linalg.norm(object_points - closest, dim=-1)  # (B,T,N)
    return dist, frame_mask


def build_gaze_map_from_arrow(
    gaze,
    nframes,
    x_obj,
    obj_pc,
    sigma=0.02,
    obj_cent=None,
    obj_scale=None,
):
    """
    Convert gaze arrows (origin, direction) into a point-wise gaze map.

    Args:
        gaze: (B, T, 2, 3) tensor. gaze[..., 0, :] is origin, gaze[..., 1, :] is direction.
        nframes: (B,) tensor-like valid frame counts (padding frames are ignored).
        x_obj: (B, T, 9) object params [trans(3), rot6d(6)]. Only the initial
            pose is used to define a fixed object frame.
        obj_pc: (B, N, 3) object points in that fixed frame.
        sigma: Gaussian width in the coordinate system of ``obj_pc``.
        obj_cent, obj_scale: Optional normalization parameters used to map the
            gaze origin into the normalized object-point coordinate system.

    Returns:
        gaze_map: (B, N) accumulated/averaged gaze score on canonical object points.
    """
    dist, frame_mask = _compute_gaze_ray_distance_and_mask(gaze, nframes, x_obj, obj_pc)
    dtype = gaze.dtype
    device = gaze.device
    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(1e-8)
    frame_score = torch.exp(-(dist ** 2) / (2 * sigma_t ** 2))
    frame_score = frame_score * frame_mask.unsqueeze(-1).to(dtype)

    denom = frame_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
    gaze_map = frame_score.sum(dim=1) / denom
    return gaze_map


def build_gaze_sequence_from_arrow(
    gaze,
    nframes,
    x_obj,
    obj_pc,
    sigma=0.02,
    obj_cent=None,
    obj_scale=None,
):
    """
    Convert gaze arrows (origin, direction) into frame-wise point scores.

    Args:
        gaze: (B, T, 2, 3) tensor. gaze[..., 0, :] is origin, gaze[..., 1, :] is direction.
        nframes: (B,) tensor-like valid frame counts (padding frames are ignored).
        x_obj: (B, T, 9) object params [trans(3), rot6d(6)]. Only the initial
            pose is used to define a fixed object frame.
        obj_pc: (B, N, 3) object points in that fixed frame.
        sigma: Gaussian width in the coordinate system of ``obj_pc``.
        obj_cent, obj_scale: Optional normalization parameters used to map the
            gaze origin into the normalized object-point coordinate system.

    Returns:
        gaze_sequence: (B, T, N) frame-wise gaze closeness score on canonical object points.
    """
    dist, frame_mask = _compute_gaze_ray_distance_and_mask(
        gaze,
        nframes,
        x_obj,
        obj_pc,
        obj_cent=obj_cent,
        obj_scale=obj_scale,
    )
    dtype = gaze.dtype
    device = gaze.device
    sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(1e-8)
    frame_score = torch.exp(-(dist ** 2) / (2 * sigma_t ** 2))
    return frame_score * frame_mask.unsqueeze(-1).to(dtype)


def build_gaze_distance_map_from_arrow(
    gaze,
    nframes,
    x_obj,
    obj_pc,
    return_frame_mask=False,
    obj_cent=None,
    obj_scale=None,
):
    """
    Convert gaze arrows to per-frame point-wise ray distance map.

    Returns:
        gaze_dist_map: (B, T, N) ray-to-point distance map.
        frame_mask: (B, T) returned only when return_frame_mask=True.
    """
    dist, frame_mask = _compute_gaze_ray_distance_and_mask(
        gaze,
        nframes,
        x_obj,
        obj_pc,
        obj_cent=obj_cent,
        obj_scale=obj_scale,
    )
    gaze_dist_map = dist * frame_mask.unsqueeze(-1).to(dist.dtype)
    if return_frame_mask:
        return gaze_dist_map, frame_mask
    return gaze_dist_map


def build_temporal_binned_gaze_map(
    gaze_map_seq,
    nframes,
    gaze_bin_count=4,
):
    """
    Temporal binning for precomputed gaze maps.

    Args:
        gaze_map_seq: (B, T, N)
        nframes: (B,)
        gaze_bin_count: number of temporal bins (K)

    Returns:
        gaze_bin_maps: (B, K, N)
    """
    if gaze_map_seq.dim() != 3:
        raise ValueError(f"Expected gaze_map_seq shape (B,T,N), got {tuple(gaze_map_seq.shape)}")
    if gaze_bin_count <= 0:
        raise ValueError(f"gaze_bin_count must be > 0, got {gaze_bin_count}")

    B, T, _ = gaze_map_seq.shape
    device = gaze_map_seq.device
    dtype = gaze_map_seq.dtype

    if not torch.is_tensor(nframes):
        nframes = torch.as_tensor(nframes, device=device)
    nframes = nframes.to(device=device).long().clamp(min=1, max=T)

    frame_mask = torch.arange(T, device=device).unsqueeze(0) < nframes.unsqueeze(1)
    frame_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    bin_idx = torch.div(
        frame_idx * gaze_bin_count,
        nframes.unsqueeze(1),
        rounding_mode="floor",
    ).clamp(max=gaze_bin_count - 1)

    gaze_bin_maps = []
    for k in range(gaze_bin_count):
        bin_mask = frame_mask & (bin_idx == k)
        denom = bin_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
        bin_map = (gaze_map_seq * bin_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / denom
        gaze_bin_maps.append(bin_map)

    return torch.stack(gaze_bin_maps, dim=1)


def process_dist_map(
    max_nframes, init_frame, cf_idx, cov_idx, chj_idx, dist_value, is_hand
):
    dist_map = np.zeros((max_nframes, 1024, 21), dtype=np.float32)
    if is_hand:
        f_idx_filtered = np.where(
            (init_frame <= cf_idx) & (cf_idx < init_frame + max_nframes)
        )[0]
        cf_idx_selected = cf_idx[f_idx_filtered]
        cf_idx_moved = cf_idx_selected - init_frame
        cov_idx_selected = cov_idx[f_idx_filtered]
        chj_idx_selected = chj_idx[f_idx_filtered]
        dist_value_selected = dist_value[f_idx_filtered]
        dist_map[cf_idx_moved, cov_idx_selected, chj_idx_selected] = dist_value_selected
    return dist_map


def get_contact_frame_start_from_indices(cf_idx, init_frame, max_nframes):
    if len(cf_idx) == 0:
        return max_nframes

    cf_arr = np.asarray(cf_idx, dtype=np.int32)
    valid_mask = (init_frame <= cf_arr) & (cf_arr < init_frame + max_nframes)
    if not valid_mask.any():
        return max_nframes
    local_idx = cf_arr[valid_mask] - init_frame
    return int(local_idx.min())


def get_first_contact_frame(
    lcf_idx, rcf_idx, init_frame, max_nframes, use_lhand=True, use_rhand=True
):
    start_frame = max_nframes
    if use_lhand:
        start_frame = min(
            start_frame,
            get_contact_frame_start_from_indices(lcf_idx, init_frame, max_nframes),
        )
    if use_rhand:
        start_frame = min(
            start_frame,
            get_contact_frame_start_from_indices(rcf_idx, init_frame, max_nframes),
        )
    return start_frame


def process_contact_map(dist_map, contact_frames):
    dist_map = dist_map.copy()[contact_frames]
    contact_obj_map = dist_map.sum(2) != 0
    dist_map[dist_map == 0] = 1
    contact_hand_map = dist_map.min(1) != 1
    return contact_obj_map, contact_hand_map


def process_contact_frame_idx(lcf_idx, rcf_idx):
    if len(lcf_idx) > 0:
        min_lcf_idx = lcf_idx.min()
        max_lcf_idx = lcf_idx.max()
    else:
        min_lcf_idx = 999
        max_lcf_idx = -1

    if len(rcf_idx) > 0:
        min_rcf_idx = rcf_idx.min()
        max_rcf_idx = rcf_idx.max()
    else:
        min_rcf_idx = 999
        max_rcf_idx = -1

    min_cf_idx = min(min_lcf_idx, min_rcf_idx)
    max_cf_idx = max(max_lcf_idx, max_rcf_idx)
    return np.array([min_cf_idx, max_cf_idx])


def get_contact_map(idx_f, idx_v, v_num, is_hand):
    contact_map = np.zeros([60, v_num])
    if is_hand:
        contact_map[idx_f, idx_v] = 1
    return contact_map


def pc_normalize(pc, return_params=False):
    target_max_norm = 0.85
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    effective_scale = scale / target_max_norm
    pc = pc / effective_scale
    if return_params:
        return pc, centroid, effective_scale
    else:
        return pc


def transform_hand_to_xdata(trans, pose):
    trans_torch, pose_torch = proc_torch_frame(trans), proc_torch_frame(pose)
    nframes = pose_torch.shape[0]
    rot6d_torch = axis_angle_to_rot6d(pose_torch.reshape(-1, 3)).reshape(
        nframes, 16 * 6
    )
    xdata = torch.cat([trans_torch, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata


def transform_xdata_to_joints(xdata, hand_layer):
    xdata = proc_torch_cuda(xdata).unsqueeze(0)
    hand_joints = get_hand_joints_w_tip(xdata, hand_layer)
    hand_joints = proc_numpy(hand_joints.squeeze(0))
    return hand_joints


def transform_obj_to_xdata(obj_matrix):
    orl = proc_torch_frame(obj_matrix)  # object rotation list
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]
    nframes = obj_rotmat.shape[0]
    rot6d_torch = rotmat_to_rot6d(obj_rotmat).reshape(nframes, 6)
    xdata = torch.cat([obj_trans, rot6d_torch], dim=1)
    xdata = proc_numpy(xdata)
    return xdata


def get_contact_info(
    lhand_pose_list,
    lhand_beta_list,
    lhand_trans_list,
    rhand_pose_list,
    rhand_beta_list,
    rhand_trans_list,
    object_rotmat_list,
    lhand_layer,
    rhand_layer,
    sampled_obj_verts_org,
    mul_rv=True,
):
    contact_threshold = 0.02

    sampled_obj_verts_org = proc_torch_cuda(sampled_obj_verts_org)
    orl = proc_torch_frame(object_rotmat_list)
    obj_rotmat = orl[:, :3, :3]
    obj_trans = orl[:, :3, 3]

    if mul_rv:
        sampled_obj_verts = torch.einsum(
            "tij,kj->tki", obj_rotmat, sampled_obj_verts_org
        ) + obj_trans.unsqueeze(1)
    else:
        sampled_obj_verts = torch.einsum(
            "tij,ki->tkj", obj_rotmat, sampled_obj_verts_org
        ) + obj_trans.unsqueeze(1)

    if len(lhand_pose_list) > 0:
        lpl, lbl, ltl = (
            proc_torch_frame(lhand_pose_list),
            proc_torch_frame(lhand_beta_list),
            proc_torch_frame(lhand_trans_list),
        )
        out_l = lhand_layer(lbl, lpl[..., :3], lpl[..., 3:])
        lhand_joints = out_l.joints_w_tip + ltl.unsqueeze(1)
        ldist = get_hand_object_dist(
            lhand_joints,
            sampled_obj_verts,
        )
        lcf_idx, lcov_idx, lchj_idx = get_contact_idx(ldist, contact_threshold)
        ldist_value = ldist[lcf_idx, lcov_idx, lchj_idx]

        lhand_vertices = out_l.vertices + ltl.unsqueeze(1)
        ldist = get_hand_object_dist(
            lhand_vertices,
            sampled_obj_verts,
        )
        lcf_ver_idx, lcov_ver_idx, lchj_ver_idx = get_contact_idx(
            ldist, contact_threshold
        )
        ldist_value_vertex = ldist[lcf_idx, lcov_idx, lchj_idx]

    else:
        lcf_idx, lcov_idx, lchj_idx = np.array([]), np.array([]), np.array([])
        lcf_ver_idx, lcov_ver_idx, lchj_ver_idx = (
            np.array([]),
            np.array([]),
            np.array([]),
        )
        ldist_value, rdist_value_vertex = np.array([]), np.array([])

    if len(rhand_pose_list) > 0:
        rpl, rbl, rtl = (
            proc_torch_frame(rhand_pose_list),
            proc_torch_frame(rhand_beta_list),
            proc_torch_frame(rhand_trans_list),
        )
        out_r = rhand_layer(rbl, rpl[..., :3], rpl[..., 3:])

        rhand_joints = out_r.joints_w_tip + rtl.unsqueeze(1)
        rdist = get_hand_object_dist(
            rhand_joints,
            sampled_obj_verts,
        )
        rcf_idx, rcov_idx, rchj_idx = get_contact_idx(rdist, contact_threshold)
        rdist_value = rdist[rcf_idx, rcov_idx, rchj_idx]

        rhand_vertices = out_r.vertices + rtl.unsqueeze(1)
        rdist = get_hand_object_dist(
            rhand_vertices,
            sampled_obj_verts,
        )
        rcf_ver_idx, rcov_ver_idx, rchj_ver_idx = get_contact_idx(
            rdist, contact_threshold
        )
        rdist_value_vertex = rdist[rcf_idx, rcov_idx, rchj_idx]

    else:
        rcf_idx, rcov_idx, rchj_idx = np.array([]), np.array([]), np.array([])
        rcf_ver_idx, rcov_ver_idx, rchj_ver_idx = (
            np.array([]),
            np.array([]),
            np.array([]),
        )
        rdist_value, rdist_value_vertex = np.array([]), np.array([])

    is_lhand, is_rhand = get_which_hands_inter(lcf_idx, rcf_idx)

    lcf_idx = proc_numpy(lcf_idx)
    lcov_idx = proc_numpy(lcov_idx)
    lchj_idx = proc_numpy(lchj_idx)
    ldist_value = proc_numpy(ldist_value)
    lcf_ver_idx = proc_numpy(lcf_ver_idx)
    lcov_ver_idx = proc_numpy(lcov_ver_idx)
    lchj_ver_idx = proc_numpy(lchj_ver_idx)
    ldist_value_vertex = proc_numpy(ldist_value_vertex)

    rcf_idx = proc_numpy(rcf_idx)
    rcov_idx = proc_numpy(rcov_idx)
    rchj_idx = proc_numpy(rchj_idx)
    rdist_value = proc_numpy(rdist_value)
    rcf_ver_idx = proc_numpy(rcf_ver_idx)
    rcov_ver_idx = proc_numpy(rcov_ver_idx)
    rchj_ver_idx = proc_numpy(rchj_ver_idx)
    rdist_value_vertex = proc_numpy(rdist_value_vertex)

    return (
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
    )


def get_hand_object_dist(hand_joints, sampled_obj_verts):
    hand_joints = proc_torch_cuda(hand_joints)
    sampled_obj_verts = proc_torch_cuda(sampled_obj_verts)

    # sampled_obj_verts = obj_verts[:, point_set]
    dist = torch.cdist(sampled_obj_verts, hand_joints)
    return dist


def get_contact_idx(dist, contact_threshold):
    # Contact frame idx, Contact object verts idx, Contact hand joints idx
    cf_idx, cov_idx, chj_idx = torch.where(dist < contact_threshold)
    return cf_idx, cov_idx, chj_idx


def get_which_hands_inter(lcf_idx, rcf_idx):
    # Contact frame idx, Contact object verts idx, Contact hand joints idx
    is_lhand = 0
    is_rhand = 0
    if len(lcf_idx) > 0:
        is_lhand = 1
    if len(rcf_idx) > 0:
        is_rhand = 1
    return is_lhand, is_rhand


def get_hand_org(hand_pose, hand_beta, hand_trans, hand_layer):
    hand_pose = proc_torch_cuda(hand_pose)
    hand_beta = proc_torch_cuda(hand_beta)
    hand_trans = proc_torch_cuda(hand_trans)
    mano_keypoints_3d = hand_layer(
        hand_beta,
        hand_pose[:, :3],
        hand_pose[:, 3:],
    ).joints

    hand_origin = mano_keypoints_3d[:, 0]
    return hand_origin


def proc_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.FloatTensor(d)
    if d.device != "cuda":
        d = d.cuda()
    return d


def proc_long_torch_cuda(d):
    if not isinstance(d, torch.Tensor):
        d = torch.LongTensor(d)
    if d.device != "cuda":
        d = d.cuda()
    return d


def proc_torch_frame(l):
    if isinstance(l, list) or isinstance(l, np.ndarray):
        l = [torch.FloatTensor(_l).unsqueeze(0) for _l in l]
        l = torch.cat(l)
        l = l.cuda()
    return l


def proc_numpy(d):
    if isinstance(d, torch.Tensor):
        if d.requires_grad:
            d = d.detach()
        if d.is_cuda:
            d = d.cpu()
        d = d.numpy()
    return d


def proc_cond_contact_estimator(obj_scale, obj_feat, enc_text, npts, ):
    enc_text_expand = enc_text.unsqueeze(1)
    enc_text_expand = enc_text_expand.expand(-1, npts, -1)
    obj_scale_expand2 = obj_scale.unsqueeze(1).unsqueeze(2)
    obj_scale_expand2 = obj_scale_expand2.expand(-1, npts, -1)
    condition = torch.cat([obj_scale_expand2, obj_feat, enc_text_expand], dim=2)
    
    return condition


def proc_obj_feat_cov(
    contact_estimator,
    obj_scale,
    obj_feat,
    enc_text,
    npts,
):
    obj_feat_global = obj_feat[:, 0, :1024]
    # condition = proc_cond_contact_estimator(
    #     obj_scale, obj_feat, enc_text, npts, 
    # )
    
    # est_contact_map = contact_estimator.decode(condition)
    # est_contact_map_plot = est_contact_map.squeeze(-1).clone()
    # est_contact_map = (est_contact_map.squeeze(-1) > 0.5).long()
    
    # obj_feat_cov = torch.cat([obj_feat_global, est_contact_map], dim=1)
    # obj_feat_cov = torch.cat([obj_feat_global, cov_map[]], dim=1)
    # est_contact_map = None
    # est_contact_map_plot = None

    # return obj_feat_global, est_contact_map, est_contact_map_plot
    return obj_feat_global



def proc_obj_feat_final(
    contact_estimator,
    obj_scale,
    obj_cent,
    obj_feat,
    npts,
    gaze_map,
    enc_text=None,
    use_obj_scale_centroid=True,
    use_contact_feat=True,
    gaze_bin_maps=None,
):
    obj_feat_global = _flatten_obj_feature(obj_feat)
    target_dtype = obj_feat_global.dtype
    target_device = obj_feat_global.device
    if use_contact_feat:
        if gaze_bin_maps is not None:
            gaze_feat = gaze_bin_maps.to(
                device=target_device, dtype=target_dtype
            ).mean(dim=1)
        else:
            gaze_feat = gaze_map.to(device=target_device, dtype=target_dtype)
        obj_feat_cov = torch.cat([obj_feat_global, gaze_feat], dim=1)
    else:
        obj_feat_cov = obj_feat_global

    if use_obj_scale_centroid:
        obj_scale_expand1 = obj_scale.to(
            device=target_device, dtype=target_dtype
        ).unsqueeze(1)
        obj_cent = obj_cent.to(device=target_device, dtype=target_dtype)
        obj_feat_final = torch.cat([obj_feat_cov, obj_scale_expand1, obj_cent], dim=1)
    else:
        obj_feat_final = obj_feat_cov
    
    return obj_feat_final



def proc_obj_feat_final_train(
    gaze_map,
    obj_scale,
    obj_cent,
    obj_feat,
    use_obj_scale=True,
    use_obj_centroid=False,
):
    obj_feat = _flatten_obj_feature(obj_feat)
    target_dtype = obj_feat.dtype
    target_device = obj_feat.device
    gaze_map = gaze_map.to(device=target_device, dtype=target_dtype)
    global_parts = [obj_feat]
    if use_obj_scale:
        obj_scale_expand1 = obj_scale.to(
            device=target_device, dtype=target_dtype
        ).unsqueeze(1)
        global_parts.append(obj_scale_expand1)
    if use_obj_centroid:
        obj_cent = obj_cent.to(device=target_device, dtype=target_dtype)
        global_parts.append(obj_cent)
    if gaze_map.dim() in (3, 4):
        obj_feat_final = torch.cat(global_parts, dim=1)
        return {"global": obj_feat_final, "gaze": gaze_map}
    obj_feat_final = torch.cat([obj_feat, gaze_map] + global_parts[1:], dim=1)

    return obj_feat_final


def get_hand2obj_dist(hand_joints, obj_pc, obj_pc_normal):
    B, T = hand_joints.shape[:2]
    hand_joints = hand_joints.reshape(B * T, -1, 3)
    obj_pc = obj_pc.reshape(B * T, -1, 3)
    obj_pc_normal = obj_pc_normal.reshape(B * T, -1, 3)
    hand_nn_dist, hand_nn_idx = get_NN(hand_joints, obj_pc)
    hand_interior = get_interior(obj_pc_normal, obj_pc, hand_joints, hand_nn_idx)
    hand_nn_dist = hand_nn_dist.sqrt()
    # hand_nn_dist[hand_interior] *= -1
    hand_nn_dist = torch.abs(hand_nn_dist)
    hand_nn_idx_expand = hand_nn_idx.unsqueeze(-1).expand(*hand_nn_idx.shape, 3)
    obj_pc_contact = torch.gather(obj_pc, 1, hand_nn_idx_expand)
    hand_dist_values_xyz = (hand_joints - obj_pc_contact) ** 2
    hand_dist_values_xyz = hand_dist_values_xyz.reshape(B, T, -1, 3)
    hand_nn_dist = hand_nn_dist.reshape(B, T, -1)
    obj_pc_contact = obj_pc_contact.reshape(B, T, -1, 3)
    return hand_dist_values_xyz, hand_nn_dist, obj_pc_contact


def get_contact_frame(
    lhand_dist_values,
    rhand_dist_values,
    valid_mask_lhand=None,
    valid_mask_rhand=None,
    valid_mask_obj=None,
    threshold=0.005,
    num_keypoints=2,
):
    if valid_mask_obj is not None:
        lhand_contact_frame_mask = (
            (lhand_dist_values < threshold).sum(2) >= num_keypoints
        ) * (torch.logical_and(valid_mask_obj, valid_mask_lhand))
        rhand_contact_frame_mask = (
            (rhand_dist_values < threshold).sum(2) >= num_keypoints
        ) * (torch.logical_and(valid_mask_obj, valid_mask_rhand))
    else:
        lhand_contact_frame_mask = (lhand_dist_values < threshold).sum(
            2
        ) >= num_keypoints
        rhand_contact_frame_mask = (rhand_dist_values < threshold).sum(
            2
        ) >= num_keypoints
    contact_frame_mask = torch.logical_or(
        lhand_contact_frame_mask, rhand_contact_frame_mask
    )
    return contact_frame_mask


def proc_refiner_input(
    pred_lhand,
    pred_rhand,
    pred_obj,
    lhand_layer,
    rhand_layer,
    obj_pc_org,
    obj_pc_normal_org,
    valid_mask_lhand,
    valid_mask_rhand,
    valid_mask_obj,
    cov_map,
    dataset_name,
    return_psuedo_gt=False,
    obj_pc_top_idx=None,
):
    threshold = 0.02
    bs, T = pred_obj.shape[:2]
    # Post-Processing object generation
    lhand_joints = get_hand_joints_w_tip(pred_lhand, lhand_layer)
    rhand_joints = get_hand_joints_w_tip(pred_rhand, rhand_layer)
    tf_obj_pc = get_transformed_obj_pc(
        pred_obj, obj_pc_org, dataset_name, obj_pc_top_idx
    )
    tf_obj_pc_normal = get_transformed_obj_pc(
        pred_obj, obj_pc_normal_org, dataset_name, obj_pc_top_idx
    )

    lhand_dist_values_xyz, lhand_dist_values, obj_pc_contact_lhand = get_hand2obj_dist(
        lhand_joints, tf_obj_pc, tf_obj_pc_normal
    )
    rhand_dist_values_xyz, rhand_dist_values, obj_pc_contact_rhand = get_hand2obj_dist(
        rhand_joints, tf_obj_pc, tf_obj_pc_normal
    )

    lhand_attn = torch.exp(-50 * lhand_dist_values_xyz)
    rhand_attn = torch.exp(-50 * rhand_dist_values_xyz)

    cov_map = cov_map.unsqueeze(1)
    cov_map = cov_map.expand(-1, T, -1)
    input_lhand = torch.cat(
        [
            pred_lhand,
            lhand_joints.reshape(bs, T, -1),
            lhand_attn.reshape(bs, T, -1),
            tf_obj_pc.norm(dim=-1),
            cov_map,
        ],
        dim=2,
    )
    input_rhand = torch.cat(
        [
            pred_rhand,
            rhand_joints.reshape(bs, T, -1),
            rhand_attn.reshape(bs, T, -1),
            tf_obj_pc.norm(dim=-1),
            cov_map,
        ],
        dim=2,
    )
    if return_psuedo_gt:
        lhand_contact_joint_mask = (lhand_dist_values < threshold) * (
            torch.logical_and(valid_mask_obj, valid_mask_lhand)
        ).unsqueeze(2)
        rhand_contact_joint_mask = (rhand_dist_values < threshold) * (
            torch.logical_and(valid_mask_obj, valid_mask_rhand)
        ).unsqueeze(2)
        obj_pc_contact_lhand_psuedo = obj_pc_contact_lhand.clone()
        obj_pc_contact_rhand_psuedo = obj_pc_contact_rhand.clone()
        return (
            input_lhand,
            input_rhand,
            pred_obj,
            obj_pc_contact_lhand_psuedo,
            obj_pc_contact_rhand_psuedo,
            lhand_contact_joint_mask,
            rhand_contact_joint_mask,
        )
    else:
        return input_lhand, input_rhand, pred_obj


def filter_obj_params(pred_obj, contact_mask):
    bs, nframes = pred_obj.shape[:2]
    input_obj = pred_obj.clone()
    for B in range(bs):
        start_idx = -1
        end_idx = -1
        for T in range(nframes):
            if start_idx == -1 and not contact_mask[B, T]:
                start_idx = T
            if start_idx != -1 and contact_mask[B, T]:
                end_idx = T
                if start_idx != 0:
                    input_obj[B, start_idx:end_idx] = input_obj[
                        B, start_idx - 1 : start_idx
                    ]
                else:
                    input_obj[B, start_idx:end_idx] = input_obj[
                        B, end_idx : end_idx + 1
                    ]

                start_idx = -1
                end_idx = -1
            if start_idx != -1 and T == nframes - 1:
                end_idx = T
                if start_idx != 0:
                    input_obj[B, start_idx:end_idx] = input_obj[
                        B, start_idx - 1 : start_idx
                    ]
                else:
                    input_obj[B, start_idx:end_idx] = input_obj[B, 0:1]

                start_idx = -1
                end_idx = -1
    return input_obj


def farthest_point_sample(xyz, npoint, random=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if random:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = 0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def select_from_groups(data, num_groups=15):
    N = data.shape[0]
    ave_duration, remainder = divmod(N, num_groups)
    offset1 = np.multiply(np.arange(remainder), (ave_duration + 1)) + np.random.randint(
        ave_duration + 1, size=remainder
    )
    offset2 = (
        np.multiply(np.arange(remainder, num_groups), (ave_duration))
        + remainder
        + np.random.randint(ave_duration, size=num_groups - remainder)
    )
    final_offsets = np.append(offset1, offset2)
    selected_data = data[final_offsets]
    return selected_data


def get_hand_obj_dist_map(
    pred_lhand, pred_rhand, pred_obj, obj_pc, lhand_layer, rhand_layer
):
    bs, nframes = pred_lhand.shape[:2]
    pred_trans = pred_lhand[..., :3]
    out = get_hand_layer_out(pred_lhand, lhand_layer)
    hand_joints_w_tip = out.joints_w_tip.reshape(bs, nframes, 21, 3)
    lhand_joints_w_tip = hand_joints_w_tip + pred_trans.unsqueeze(2)

    bs, nframes = pred_rhand.shape[:2]
    pred_trans = pred_rhand[..., :3]
    out = get_hand_layer_out(pred_rhand, rhand_layer)
    hand_joints_w_tip = out.joints_w_tip.reshape(bs, nframes, 21, 3)
    rhand_joints_w_tip = hand_joints_w_tip + pred_trans.unsqueeze(2)

    bs, nframes = pred_obj.shape[:2]
    obj_trans = pred_obj[..., :3]
    obj_rot6d = pred_obj[..., 3:9]
    obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(bs, nframes, 3, 3)
    obj_pc_rotated = torch.einsum("btij,bkj->btki", obj_rotmat, obj_pc)
    obj_pc_transformed = obj_pc_rotated + obj_trans.unsqueeze(2)

    pred_ldist_map = torch.cdist(
        obj_pc_transformed.reshape(-1, 1024, 3), lhand_joints_w_tip.reshape(-1, 21, 3)
    ).reshape(bs, nframes, 1024, 21)

    pred_rdist_map = torch.cdist(
        obj_pc_transformed.reshape(-1, 1024, 3), rhand_joints_w_tip.reshape(-1, 21, 3)
    ).reshape(bs, nframes, 1024, 21)

    return pred_ldist_map, pred_rdist_map


def temporal_smooth_position_loss(
    lhand_pred,
    pos_idx=slice(96, 99),
    valid_mask=None,
    w_vel=1.0,
    w_acc=0.2,
    robust="l2",
):
    """
    lhand_pred: (B, T, 99)
    pos_idx: 위치 3D의 인덱스 (예: 마지막 3차원이라면 slice(96,99))
    valid_mask: (B, T) True=유효 / None 가능
    robust: 'l2' 또는 'l1' (TV-L1 스타일)
    """
    B, T, _ = lhand_pred.shape
    pos = lhand_pred[..., pos_idx]  # (B, T, 3)

    # 1) velocity: x_{t+1} - x_{t}  → (B, T-1, 3)
    vel = pos[:, 1:, :] - pos[:, :-1, :]

    # 2) acceleration: x_{t+1} - 2*x_{t} + x_{t-1}  → (B, T-2, 3)
    acc = pos[:, 2:, :] - 2 * pos[:, 1:-1, :] + pos[:, :-2, :]

    if robust == "l1":
        vel_term = torch.abs(vel).sum(-1)  # (B, T-1)
        acc_term = torch.abs(acc).sum(-1)  # (B, T-2)
    else:  # 'l2'
        vel_term = (vel**2).sum(-1)  # (B, T-1)
        acc_term = (acc**2).sum(-1)  # (B, T-2)

    # 마스크 적용 (유효 프레임 쌍/트리플만 사용)
    if valid_mask is not None:
        m = valid_mask.bool()
        pair_mask = m[:, 1:] & m[:, :-1]  # (B, T-1)
        tri_mask = m[:, 2:] & m[:, 1:-1] & m[:, :-2]  # (B, T-2)

        vel_term = (vel_term * pair_mask).sum() / pair_mask.sum().clamp(min=1)
        acc_term = (acc_term * tri_mask).sum() / tri_mask.sum().clamp(min=1)
    else:
        vel_term = vel_term.mean()
        acc_term = acc_term.mean()

    return w_vel * vel_term + w_acc * acc_term
