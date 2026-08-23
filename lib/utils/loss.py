import numpy as np

import torch
import torch.nn.functional as F

from lib.utils.frame import sample_with_window_size
from lib.utils.rot import rot6d_to_rotmat
from lib.utils.proc_output import (
    get_pytorch3d_meshes, 
    get_hand_joints_w_tip, 
    get_transformed_obj_pc, 
    get_NN, get_interior
)
from lib.utils.proc import get_hand_obj_dist_map


def build_soft_lift_gate_target(
    object_translation,
    valid_mask=None,
    velocity_threshold=1e-3,
    velocity_temperature=5e-4,
    smoothing_radius=2,
):
    """Build a soft per-frame lift target from ground-truth object translation.

    The target describes whether the object is translating at each frame.  It is
    intentionally independent of hand contact and gaze: those signals will be
    inputs to the learned gate, while this function supplies its supervision.

    Args:
        object_translation: Tensor shaped ``(B, T, 3)`` in dataset units.
        valid_mask: Optional boolean tensor shaped ``(B, T)``.
        velocity_threshold: Translation speed mapped to a gate value of 0.5.
        velocity_temperature: Softness of the sigmoid transition.
        smoothing_radius: Radius of the temporal moving-average filter.

    Returns:
        Tensor shaped ``(B, T, 1)`` with values in ``[0, 1]``.
    """
    if object_translation.dim() != 3 or object_translation.shape[-1] != 3:
        raise ValueError(
            "object_translation must have shape (B, T, 3), got "
            f"{tuple(object_translation.shape)}"
        )
    if velocity_temperature <= 0:
        raise ValueError("velocity_temperature must be positive")
    if smoothing_radius < 0:
        raise ValueError("smoothing_radius must be non-negative")

    batch_size, frame_count, _ = object_translation.shape
    if frame_count == 0:
        return object_translation.new_zeros((batch_size, 0, 1))

    if valid_mask is None:
        valid = torch.ones(
            (batch_size, frame_count),
            device=object_translation.device,
            dtype=torch.bool,
        )
    else:
        if valid_mask.shape != (batch_size, frame_count):
            raise ValueError(
                f"valid_mask must have shape {(batch_size, frame_count)}, got "
                f"{tuple(valid_mask.shape)}"
            )
        valid = valid_mask.to(device=object_translation.device, dtype=torch.bool)

    speed = object_translation.new_zeros((batch_size, frame_count))
    if frame_count > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        displacement = object_translation[:, 1:] - object_translation[:, :-1]
        pair_speed = torch.linalg.vector_norm(displacement, dim=-1)
        pair_speed = pair_speed * pair_valid.to(pair_speed.dtype)
        # Associate a displacement with its destination frame.  The first frame
        # has no preceding displacement and therefore remains non-lift.
        speed[:, 1:] = pair_speed

    gate = torch.sigmoid(
        (speed - float(velocity_threshold)) / float(velocity_temperature)
    )
    gate = gate * valid.to(gate.dtype)

    if smoothing_radius > 0 and frame_count > 1:
        kernel_size = 2 * int(smoothing_radius) + 1
        gate_1d = gate.unsqueeze(1)
        valid_1d = valid.to(gate.dtype).unsqueeze(1)
        kernel = gate.new_ones((1, 1, kernel_size))
        numerator = F.conv1d(gate_1d, kernel, padding=smoothing_radius)
        denominator = F.conv1d(valid_1d, kernel, padding=smoothing_radius)
        gate = (numerator / denominator.clamp_min(1.0)).squeeze(1)
        gate = gate * valid.to(gate.dtype)

    return gate.unsqueeze(-1)


def _motion_position(motion, entity):
    """Extract a representative 3D position from pose or point-token motion."""
    if motion.dim() != 3:
        raise ValueError(f"Expected (B,T,D) motion, got {tuple(motion.shape)}")
    feature_dim = motion.shape[-1]
    if entity == "object" and feature_dim == 9:
        return motion[..., :3]
    if entity == "hand" and feature_dim == 99:
        return motion[..., -3:]
    if entity == "object":
        if feature_dim % 3 != 0:
            raise ValueError(f"Object point feature dim {feature_dim} is not divisible by 3")
        return motion.reshape(*motion.shape[:2], -1, 3).mean(dim=2)
    if entity == "hand":
        if feature_dim % 6 != 0:
            raise ValueError(
                "Point-token hand features must contain equal XYZ and direction "
                f"blocks, got feature dim {feature_dim}"
            )
        point_coord_dim = feature_dim // 2
        return motion[..., :point_coord_dim].reshape(
            *motion.shape[:2], -1, 3
        ).mean(dim=2)
    raise ValueError(f"Unknown entity {entity!r}")


def get_lift_gate_loss(pred_gate, target_gate, valid_mask):
    if pred_gate is None:
        raise ValueError("pred_gate is required when lift-gate loss is enabled")
    valid = valid_mask.to(device=pred_gate.device, dtype=pred_gate.dtype).unsqueeze(-1)
    loss = F.binary_cross_entropy(pred_gate, target_gate, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def get_object_static_loss(pred_obj, target_gate, valid_mask):
    """Suppress predicted object translation before the lift interval."""
    position = _motion_position(pred_obj, "object")
    velocity = position[:, 1:] - position[:, :-1]
    pair_valid = (valid_mask[:, 1:] & valid_mask[:, :-1]).to(velocity.dtype)
    static_weight = (1.0 - target_gate[:, 1:, 0]) * pair_valid
    per_frame = velocity.square().sum(dim=-1)
    return (per_frame * static_weight).sum() / static_weight.sum().clamp_min(1.0)


def get_predicted_contact_static_loss(
    pred_lhand,
    pred_rhand,
    pred_obj,
    obj_points,
    valid_mask_lhand,
    valid_mask_rhand,
    valid_mask_obj,
    dataset_name,
    contact_threshold=0.02,
    contact_temperature=0.005,
    max_obj_points=128,
    distance_chunk_size=512,
):
    """Suppress object translation when neither predicted hand is in contact.

    Contact is a differentiable score computed from the minimum distance between
    predicted hand surface points and the object transformed by the predicted
    pose.  This makes the constraint depend on generated contact rather than on
    the ground-truth lift schedule.
    """
    if contact_temperature <= 0:
        raise ValueError("contact_temperature must be positive")
    if pred_lhand.shape[-1] % 6 != 0 or pred_rhand.shape[-1] % 6 != 0:
        raise ValueError("Predicted-contact loss requires hand point-and-direction tokens")

    point_dim_l = pred_lhand.shape[-1] // 2
    point_dim_r = pred_rhand.shape[-1] // 2
    lpoints = pred_lhand[..., :point_dim_l].reshape(*pred_lhand.shape[:2], -1, 3)
    rpoints = pred_rhand[..., :point_dim_r].reshape(*pred_rhand.shape[:2], -1, 3)
    transformed_obj = get_transformed_obj_pc(pred_obj, obj_points, dataset_name)
    if transformed_obj.shape[2] > int(max_obj_points):
        sample_idx = torch.linspace(
            0,
            transformed_obj.shape[2] - 1,
            steps=int(max_obj_points),
            device=transformed_obj.device,
        ).long()
        transformed_obj = transformed_obj.index_select(2, sample_idx)

    batch_size, frame_count = pred_obj.shape[:2]
    flat_obj = transformed_obj.reshape(-1, transformed_obj.shape[2], 3)
    flat_l = lpoints.reshape(-1, lpoints.shape[2], 3)
    flat_r = rpoints.reshape(-1, rpoints.shape[2], 3)
    min_l, min_r = [], []
    chunk_size = max(int(distance_chunk_size), 1)
    for start in range(0, flat_obj.shape[0], chunk_size):
        end = min(start + chunk_size, flat_obj.shape[0])
        obj_chunk = flat_obj[start:end]
        min_l.append(torch.cdist(flat_l[start:end], obj_chunk).amin(dim=(1, 2)))
        min_r.append(torch.cdist(flat_r[start:end], obj_chunk).amin(dim=(1, 2)))
    min_l = torch.cat(min_l).reshape(batch_size, frame_count)
    min_r = torch.cat(min_r).reshape(batch_size, frame_count)

    temperature = float(contact_temperature)
    threshold = float(contact_threshold)
    lcontact = torch.sigmoid((threshold - min_l) / temperature)
    rcontact = torch.sigmoid((threshold - min_r) / temperature)
    lcontact = lcontact * valid_mask_lhand.to(lcontact.dtype)
    rcontact = rcontact * valid_mask_rhand.to(rcontact.dtype)
    # Soft union: probability that at least one hand is in contact.
    contact = 1.0 - (1.0 - lcontact) * (1.0 - rcontact)

    obj_position = _motion_position(pred_obj, "object")
    obj_velocity = obj_position[:, 1:] - obj_position[:, :-1]
    pair_valid = (valid_mask_obj[:, 1:] & valid_mask_obj[:, :-1]).to(obj_velocity.dtype)
    # Require contact at both ends of a displacement before object motion is free.
    pair_contact = contact[:, 1:] * contact[:, :-1]
    static_weight = (1.0 - pair_contact) * pair_valid
    per_frame = obj_velocity.square().sum(dim=-1)
    return (per_frame * static_weight).sum() / static_weight.sum().clamp_min(1.0)


def get_object_pose_losses(pred_obj, target_obj, valid_mask):
    """Physical translation MSE and SO(3) geodesic loss for 9D object poses."""
    if pred_obj.shape[-1] != 9 or target_obj.shape[-1] != 9:
        zero = pred_obj.new_zeros(())
        return zero, zero
    valid = valid_mask.to(device=pred_obj.device, dtype=pred_obj.dtype)
    translation_error = (pred_obj[..., :3] - target_obj[..., :3]).square().sum(dim=-1)
    translation_loss = (translation_error * valid).sum() / valid.sum().clamp_min(1.0)

    pred_rot = rot6d_to_rotmat(pred_obj[..., 3:9].reshape(-1, 6)).reshape(
        *pred_obj.shape[:2], 3, 3
    )
    target_rot = rot6d_to_rotmat(target_obj[..., 3:9].reshape(-1, 6)).reshape(
        *target_obj.shape[:2], 3, 3
    )
    relative = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    angle = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    rotation_loss = (angle * valid).sum() / valid.sum().clamp_min(1.0)
    return translation_loss, rotation_loss


def get_hand_object_coupling_loss(
    pred_lhand,
    pred_rhand,
    pred_obj,
    target_lhand,
    target_rhand,
    target_obj,
    target_gate,
    valid_mask_lhand,
    valid_mask_rhand,
    valid_mask_obj,
    hand_selection_temperature=0.05,
):
    """Match object velocity to the nearer hand during ground-truth lift."""
    pred_lpos = _motion_position(pred_lhand, "hand")
    pred_rpos = _motion_position(pred_rhand, "hand")
    pred_opos = _motion_position(pred_obj, "object")
    target_lpos = _motion_position(target_lhand, "hand")
    target_rpos = _motion_position(target_rhand, "hand")
    target_opos = _motion_position(target_obj, "object")

    temperature = max(float(hand_selection_temperature), 1e-6)
    hand_distances = torch.stack(
        (
            torch.linalg.vector_norm(target_lpos - target_opos, dim=-1),
            torch.linalg.vector_norm(target_rpos - target_opos, dim=-1),
        ),
        dim=-1,
    )
    hand_weights = torch.softmax(-hand_distances / temperature, dim=-1)

    pred_lvel = pred_lpos[:, 1:] - pred_lpos[:, :-1]
    pred_rvel = pred_rpos[:, 1:] - pred_rpos[:, :-1]
    pred_ovel = pred_opos[:, 1:] - pred_opos[:, :-1]
    weights = hand_weights[:, 1:]
    selected_hand_velocity = (
        weights[..., 0:1] * pred_lvel + weights[..., 1:2] * pred_rvel
    )

    pair_valid_obj = valid_mask_obj[:, 1:] & valid_mask_obj[:, :-1]
    pair_valid_l = valid_mask_lhand[:, 1:] & valid_mask_lhand[:, :-1]
    pair_valid_r = valid_mask_rhand[:, 1:] & valid_mask_rhand[:, :-1]
    weighted_hand_valid = (
        weights[..., 0] * pair_valid_l.to(weights.dtype)
        + weights[..., 1] * pair_valid_r.to(weights.dtype)
    )
    lift_weight = (
        target_gate[:, 1:, 0]
        * pair_valid_obj.to(weights.dtype)
        * weighted_hand_valid
    )
    per_frame = (pred_ovel - selected_hand_velocity).square().sum(dim=-1)
    return (per_frame * lift_weight).sum() / lift_weight.sum().clamp_min(1.0)

def l2_loss_unit(pred, targ, mask=None, weight=None):
    l2_loss = F.mse_loss(pred, targ, reduction='none')
    if mask is not None:
        filtered_loss = get_filtered_loss_valid_mask(l2_loss, mask, weight)
    else:
        filtered_loss = l2_loss.mean()
    return filtered_loss

def get_l2_loss(
        pred_lhand=None, pred_rhand=None, pred_obj=None, 
        targ_lhand=None, targ_rhand=None, targ_obj=None, 
        mask_lhand=None, mask_rhand=None, mask_obj=None, 
        weight=None, 
    ):
    total_loss = 0
    if targ_lhand is not None:
        lhand_loss = l2_loss_unit(pred_lhand, targ_lhand, mask_lhand, weight)
        total_loss += lhand_loss
    if targ_rhand is not None:
        rhand_loss = l2_loss_unit(pred_rhand, targ_rhand, mask_rhand, weight)
        total_loss += rhand_loss
    if targ_obj is not None:
        if pred_obj.dim() == 4:
            pred_obj = pred_obj.reshape(-1, pred_obj.shape[-2], pred_obj.shape[-1])
            targ_obj = targ_obj.reshape(-1, targ_obj.shape[-2], targ_obj.shape[-1])
            if mask_obj is not None:
                mask_obj = mask_obj.reshape(-1, mask_obj.shape[-1])
        obj_loss = l2_loss_unit(pred_obj, targ_obj, mask_obj, weight)
        total_loss += obj_loss
    return total_loss

def get_joint_contact_loss(
    pred_lhand, pred_rhand, 
    lhand_obj_cont_v, rhand_obj_cont_v,
    lhand_layer, rhand_layer, 
    mask_lhand=None, mask_rhand=None, 
    weight=None, loss_type="l2"
):
    
    lhand_joints = get_hand_joints_w_tip(pred_lhand, lhand_layer)
    rhand_joints = get_hand_joints_w_tip(pred_rhand, rhand_layer)
    cont_lhand_loss = joint_contact_loss_unit(lhand_joints, lhand_obj_cont_v, mask_lhand, weight=weight, loss_type=loss_type)
    cont_rhand_loss = joint_contact_loss_unit(rhand_joints, rhand_obj_cont_v, mask_rhand, weight=weight, loss_type=loss_type)
    return cont_lhand_loss + cont_rhand_loss
    
def joint_contact_loss_unit(pred, targ, mask=None, weight=None, loss_type="l2"):
    if loss_type == "l2":
        loss = F.mse_loss(pred, targ, reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(pred, targ, reduction='none')
    if mask is not None:
        filtered_loss = get_filtered_joint_loss_valid_mask(loss, mask, weight)
    else:
        filtered_loss = loss.mean()
    return filtered_loss

def get_smth_loss(
        model, window_size, window_step, 
        pred_lhand, pred_rhand, pred_obj, 
        targ_lhand, targ_rhand, targ_obj, 
        mask_lhand, mask_rhand, mask_obj, 
        weight, 
    ):
    pred_X0_lhand_sampled, target_lhand_sampled = sample_with_window_size(pred_lhand, targ_lhand, mask_lhand, window_size, window_step)
    pred_X0_rhand_sampled, target_rhand_sampled = sample_with_window_size(pred_rhand, targ_rhand, mask_rhand, window_size, window_step)
    pred_X0_obj_sampled, target_obj_sampled = sample_with_window_size(pred_obj, targ_obj, mask_obj, window_size, window_step)

    pred_X0_lhand_smoothed = model.hand_smoother(pred_X0_lhand_sampled.detach())
    pred_X0_rhand_smoothed = model.hand_smoother(pred_X0_rhand_sampled.detach())
    pred_X0_obj_smoothed = model.obj_smoother(pred_X0_obj_sampled.detach())

    # Smth pos loss
    smth_pos_loss_lhand = smth_pos_loss_unit(pred_X0_lhand_smoothed, target_lhand_sampled, weight)
    smth_pos_loss_rhand = smth_pos_loss_unit(pred_X0_rhand_smoothed, target_rhand_sampled, weight)
    smth_pos_loss_obj = smth_pos_loss_unit(pred_X0_obj_smoothed, target_obj_sampled, weight)
    smth_pos_loss = smth_pos_loss_lhand+smth_pos_loss_rhand+smth_pos_loss_obj

    # Smth accel loss
    smth_accel_loss_lhand = smth_accel_loss_unit(pred_X0_lhand_smoothed, target_lhand_sampled, weight)
    smth_accel_loss_rhand = smth_accel_loss_unit(pred_X0_rhand_smoothed, target_rhand_sampled, weight)
    smth_accel_loss_obj = smth_accel_loss_unit(pred_X0_obj_smoothed, target_obj_sampled, weight)
    smth_accel_loss = smth_accel_loss_lhand+smth_accel_loss_rhand+smth_accel_loss_obj
    return smth_pos_loss, smth_accel_loss

def smth_pos_loss_unit(pred, targ, weight):
    smth_pos_loss = weight*F.l1_loss(pred, targ, reduction='none').mean([1, 2])
    smth_pos_loss = smth_pos_loss.mean()
    return smth_pos_loss

def smth_accel_loss_unit(pred, targ, weight):
    pred_accel = pred[:, 2:]-2*pred[:, 1:-1]+pred[:, :-2]
    targ_accel = targ[:, 2:]-2*targ[:, 1:-1]+targ[:, :-2]
    smth_acc_loss = weight*F.l1_loss(pred_accel, targ_accel, reduction='none').mean([1, 2])
    smth_acc_loss = smth_acc_loss.mean()
    return smth_acc_loss

def get_distance_map_loss(
        pred_ldist, pred_rdist, 
        targ_ldist, targ_rdist, 
        weight=None, 
    ):
    
    ldist_loss = F.mse_loss(pred_ldist, targ_ldist, reduction="none")
    rdist_loss = F.mse_loss(pred_rdist, targ_rdist, reduction="none")
    valid_map_ldist = targ_ldist > 0
    valid_map_rdist = targ_rdist > 0
    filtered_ldist_loss = get_filtered_loss_valid_map(ldist_loss, valid_map_ldist, weight)
    filtered_rdist_loss = get_filtered_loss_valid_map(rdist_loss, valid_map_rdist, weight)
    return filtered_ldist_loss+filtered_rdist_loss

def get_relative_orientation_loss(
        pred_lhand, pred_rhand, pred_obj, 
        targ_lhand, targ_rhand, targ_obj, 
        mask_lhand, mask_rhand, 
        weight=None
    ):
    pred_ro_lhand = get_ro(pred_lhand, pred_obj, mask_lhand)
    pred_ro_rhand = get_ro(pred_rhand, pred_obj, mask_rhand)
    targ_ro_lhand = get_ro(targ_lhand, targ_obj, mask_lhand)
    targ_ro_rhand = get_ro(targ_rhand, targ_obj, mask_rhand)
    if weight is not None:
        nframes = targ_obj.shape[1]
        weight = weight.unsqueeze(1).expand(-1, nframes)
        weight_lhand = weight[mask_lhand]
        weight_rhand = weight[mask_rhand]
        ro_lhand_loss = F.mse_loss(pred_ro_lhand, targ_ro_lhand, reduction="none")
        ro_rhand_loss = F.mse_loss(pred_ro_rhand, targ_ro_rhand, reduction="none")
        ro_lhand_loss = ro_lhand_loss.mean([1, 2])*weight_lhand
        ro_rhand_loss = ro_rhand_loss.mean([1, 2])*weight_rhand
        ro_lhand_loss = ro_lhand_loss.mean()
        ro_rhand_loss = ro_rhand_loss.mean()
    else:
        if mask_lhand.sum().item() != 0:
            ro_lhand_loss = F.mse_loss(pred_ro_lhand, targ_ro_lhand)
        else:
            ro_lhand_loss = torch.tensor(0)
        if mask_rhand.sum().item() != 0:
            ro_rhand_loss = F.mse_loss(pred_ro_rhand, targ_ro_rhand)
        else:
            ro_rhand_loss = torch.tensor(0)
    return ro_lhand_loss + ro_rhand_loss

# ro: relative orientation
def get_ro(hand, obj, valid_mask):
    hand_orient = hand[valid_mask][..., 3:9]
    obj_orient = obj[valid_mask][..., 3:9]
    hand_orient_rotmat = rot6d_to_rotmat(hand_orient)
    obj_orient_rotmat = rot6d_to_rotmat(obj_orient)
    ro_hand_obj = relative_rotation_matrix(hand_orient_rotmat, obj_orient_rotmat)
    return ro_hand_obj

def get_filtered_loss_valid_mask(loss, valid_mask, loss_weight=None):
    if len(loss.shape)==3:
        loss_mean = loss.mean([2])
    elif len(loss.shape)==4:
        loss_mean = loss.mean([2, 3])
    filtered_loss = torch.where(valid_mask, loss_mean, torch.zeros_like(loss_mean))
    filtered_loss_summed = filtered_loss.sum(1)
    valid_mask_summed = valid_mask.sum(1)
    valid_mask_summed = torch.where(valid_mask_summed!=0, valid_mask_summed, torch.tensor(1).to(valid_mask_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed/valid_mask_summed
    if loss_weight is not None:
        filtered_loss_bm = filtered_loss_bm*loss_weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def get_filtered_joint_loss_valid_mask(loss, valid_mask, loss_weight=None):
    loss_mean = loss.mean([3])
    filtered_loss = torch.where(valid_mask, loss_mean, torch.zeros_like(loss_mean))
    filtered_loss_summed = filtered_loss.sum(2)
    valid_mask_summed = valid_mask.sum(2)
    valid_mask_summed = torch.where(valid_mask_summed!=0, valid_mask_summed, torch.tensor(1).to(valid_mask_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed/valid_mask_summed
    if loss_weight is not None:
        filtered_loss_bm = filtered_loss_bm*loss_weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def get_filtered_loss_valid_map(loss, valid_map, weight=None):
    filtered_loss = torch.where(valid_map, loss, torch.zeros_like(loss))
    filtered_loss_summed = filtered_loss.sum([1, 2, 3]) # batch, nframes, 1024, 21
    valid_map_summed = valid_map.sum([1, 2, 3])
    valid_map_summed = torch.where(valid_map_summed!=0, valid_map_summed, torch.tensor(1).to(valid_map_summed.device))
    # batch_mean
    filtered_loss_bm = filtered_loss_summed/valid_map_summed
    if weight is not None:
        filtered_loss_bm = filtered_loss_bm*weight
    filtered_loss = filtered_loss_bm.mean()
    return filtered_loss

def relative_rotation_matrix(R1, R2):
    R1_inv = torch.inverse(R1)
    relative_matrix = torch.matmul(R2, R1_inv)
    return relative_matrix

def get_penetration_loss(
    pred_X0_lhand, pred_X0_rhand, pred_X0_obj, 
    lhand_layer, rhand_layer, obj_pc_org, 
    valid_mask_lhand, valid_mask_rhand, 
    dataset_name, obj_pc_top_idx=None
):
    lhand_mesh, lhand_verts = get_pytorch3d_meshes(pred_X0_lhand, lhand_layer)
    rhand_mesh, rhand_verts = get_pytorch3d_meshes(pred_X0_rhand, rhand_layer)
    lhand_normal = lhand_mesh.verts_normals_packed().view(-1, 778, 3)
    rhand_normal = rhand_mesh.verts_normals_packed().view(-1, 778, 3)
    batch_size, npts = obj_pc_org.shape[:2]
    transf_obj_pc = get_transformed_obj_pc(pred_X0_obj, obj_pc_org, dataset_name, obj_pc_top_idx)
    transf_obj_pc = transf_obj_pc.reshape(-1, npts, 3)
    
    valid_mask_lhand = valid_mask_lhand.reshape(-1)
    valid_mask_rhand = valid_mask_rhand.reshape(-1)
    
    penet_loss_lhand = get_penet_hand_obj_loss(
        lhand_verts, lhand_normal, 
        transf_obj_pc, valid_mask_lhand
    )
    
    penet_loss_rhand = get_penet_hand_obj_loss(
        rhand_verts, rhand_normal, 
        transf_obj_pc, valid_mask_rhand
    )
    return penet_loss_lhand + penet_loss_rhand

def get_penet_hand_obj_loss(hand_verts, hand_normal, obj_pc, valid_mask_hand):
    nn_dist, nn_idx = get_NN(obj_pc, hand_verts)
    interior = get_interior(hand_normal, hand_verts, obj_pc, nn_idx)
    nn_dist = nn_dist.sqrt()
    nn_dist = nn_dist[valid_mask_hand]
    interior = interior[valid_mask_hand]
    if interior.sum() > 0:
        penet_loss = nn_dist[interior].mean()
    else:
        penet_loss = torch.FloatTensor(1).fill_(0).cuda()
    return penet_loss

import torch
import torch.nn.functional as F

def temporal_smoothness_loss(motion, valid_mask):
    """
    motion: (B, T, D)
    valid_mask: (B, T) - optional
    """
    vel = motion[:, 1:] - motion[:, :-1]  # (B, T-1, D)
    loss = torch.norm(vel, dim=-1)        # (B, T-1)

    valid_vel_mask = valid_mask[:, 1:] & valid_mask[:, :-1]  # (B, T-1)
    loss = loss * valid_vel_mask.float()

    denom = valid_vel_mask.sum() if valid_mask is not None else loss.numel()
    
    return loss.sum() / (denom + 1e-8)
    
    
def calculate_loss(batch, modules, weights):
    """
    batch: 텐서 묶음(dict) – pred_contact_map, l_hand_cm, r_hand_cm, ...
    modules: 외부 모듈 묶음(dict) – lhand_layer, rhand_layer 등
    weights: 손실 가중치/옵션(dict) – hand_rec, commit, dist, orient, contact 등
    """
    pred_contact_map = batch["pred_contact_map"]
    l_hand_cm = batch["l_hand_cm"]
    r_hand_cm = batch["r_hand_cm"]
    l_mask = batch["l_mask"]          # (B, T)
    r_mask = batch["r_mask"]          # (B, T)
    pred_pose = batch["pred_pose"]
    gt_pose   = batch["gt_pose"]
    valid_mask_lhand = batch["valid_mask_lhand"]
    valid_mask_rhand = batch["valid_mask_rhand"]
    gt_obj    = batch["gt_obj"]
    pred_obj = batch["pred_obj"]
    obj_verts_org = batch["obj_verts_org"]
    loss_commit = batch["loss_commit"]

    lhand_layer = modules["lhand_layer"]
    rhand_layer = modules["rhand_layer"]

    # pos_weight은 같은 device에 올리기
    pos_weight = torch.tensor(5.0, device=pred_contact_map.device)
    ce = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    l1 = F.mse_loss

    # (1) contact-map loss
    loss_contact_map = (
        torch.mean(ce(pred_contact_map[..., :778], l_hand_cm) * l_mask.unsqueeze(-1)) +
        torch.mean(ce(pred_contact_map[..., 778:],  r_hand_cm) * r_mask.unsqueeze(-1))
    )

    # (2) explicit L2
    lhand_pred, rhand_pred = pred_pose[..., :99], pred_pose[..., 99:]
    lhand_gt,   rhand_gt   = gt_pose[...,  :99], gt_pose[...,   99:]
    loss_explicit = l1(pred_pose, gt_pose, reduction='none')
    loss_lhand = get_filtered_loss_valid_mask(loss_explicit[..., :99], valid_mask_lhand[..., :30])
    loss_rhand = get_filtered_loss_valid_mask(loss_explicit[..., 99:],  valid_mask_rhand[..., :30])
    loss_obj = l1(pred_obj, gt_obj, reduction='none')

    # (3) distance-map loss (주의: ldist_map/rdist_map은 배치에서 같이 받아와야 함)
    pred_ldist, pred_rdist = get_hand_obj_dist_map(
        lhand_pred, rhand_pred, gt_obj, obj_verts_org, lhand_layer, rhand_layer
    )
    ldist_map, rdist_map = batch["ldist_map"][:, :30], batch["rdist_map"][:, :30]
    loss_dist = get_distance_map_loss(pred_ldist, pred_rdist, ldist_map, rdist_map)

    # (4) relative orientation loss
    loss_orient = get_relative_orientation_loss(
        lhand_pred, rhand_pred, pred_obj,
        lhand_gt,   rhand_gt,   gt_obj,
        valid_mask_lhand[..., :30], valid_mask_rhand[..., :30]
    )
    
    # (5)
    loss_lpos=temporal_smoothness_loss(lhand_pred[..., :3], valid_mask_lhand[..., :30])
    loss_rpos=temporal_smoothness_loss(rhand_pred[..., :3], valid_mask_rhand[..., :30])
    
    # loss_lrot=temporal_smoothness_loss(lhand_pred[..., 3:])
    # loss_rrot=temporal_smoothness_loss(rhand_pred[..., 3:])
                

    # 가중치
    w = weights  # e.g., w = config.codebook.loss_hyper
    loss = (
        w["hand_rec"] * (loss_lhand + loss_rhand)
        + w["commit"]   * loss_commit
        + w["dist"]     * loss_dist
        + w["orient"]   * loss_orient
        + w["contact"]  * loss_contact_map
        + w["pos"] * (loss_lpos + loss_rpos)
        # + w["rot"] * (loss_lrot + loss_rrot)
    )

    return {
        "loss": loss,
        "loss_lhand": loss_lhand,
        "loss_rhand": loss_rhand,
        "loss_obj": loss_obj, 
        "loss_commit": loss_commit,
        "loss_dist": loss_dist,
        "loss_orient": loss_orient,
        "loss_contact_map": loss_contact_map,
        "loss_pos": (loss_lpos + loss_rpos),
        # "loss_rot": (loss_lrot + loss_rrot),
    }
