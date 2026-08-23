import os.path as osp

import numpy as np

import torch
import torch.nn.functional as F
from gaze2hoi.test_contact import rotate_pc_y
from lib.utils.proc import proc_cond_contact_estimator
from preprocess.preprocessing_hot3d import process_obj_result
from lib.networks.clip import encoded_text
from lib.utils.rot import (
    rot6d_to_axis_angle,
)

def process_hand_result(hand_layer, hand_params):
    hand_pose = hand_params[:, 3:]
    hand_pose = rot6d_to_axis_angle(hand_pose).reshape(-1, 48)
    hand_trans = hand_params[:, :3]
    duration = hand_trans.shape[0]
    out = hand_layer(
        global_orient=hand_pose[:, :3],
        hand_pose=hand_pose[:, 3:48],
        betas=torch.zeros((duration, 10)).to(hand_pose.device)
    )
    hand_trans = hand_trans.unsqueeze(1)
    hand_vertices = out.vertices + hand_trans
    hand_faces = hand_layer.faces.copy().astype(np.int16)
    hand_faces = torch.LongTensor(hand_faces).to(hand_pose.device)
    return hand_vertices, hand_faces

# def process_obj_result(obj_verts, obj_params, dataset_name=None, obj_top_idx=None):
#     nframes = obj_params.shape[0]
#     obj_trans = obj_params[:, :3]
#     obj_rot6d = obj_params[:, 3:9]
#     obj_rotmat = rot6d_to_rotmat(obj_rot6d).reshape(-1, 3, 3)
#     if dataset_name == "arctic" and obj_params.shape[-1] == 10 and obj_top_idx is not None:
#         obj_top_idx = obj_top_idx.bool()
#         obj_angle = obj_params[..., 9:10]
#         quat_arti = axis_angle_to_quaternion(torch.FloatTensor([0, 0, -1]).to(obj_params.device).view(1, 3)*obj_angle)
#         obj_verts = obj_verts.unsqueeze(0).expand(nframes, -1, -1)
#         obj_verts2 = obj_verts.clone()
#         obj_top_idx = obj_top_idx.unsqueeze(0).expand(nframes, -1)
#         obj_verts2[obj_top_idx] = quaternion_apply(quat_arti[:, None], obj_verts)[obj_top_idx]
#         obj_pc_rotated = torch.einsum("tij,tkj->tki", obj_rotmat, obj_verts2)
#     elif dataset_name == "h2o":
#         obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
#     elif dataset_name == "grab":
#         obj_pc_rotated = torch.einsum("tij,ki->tkj", obj_rotmat, obj_verts)
#     else:
#         obj_pc_rotated = torch.einsum("tij,kj->tki", obj_rotmat, obj_verts)
#     obj_verts_transformed = obj_pc_rotated+obj_trans.unsqueeze(1)
#     return obj_verts_transformed

def processe_params(hand_data_path, obj_data_path):
    hand_data = np.load(hand_data_path, allow_pickle=True)
    object_data = np.load(obj_data_path, allow_pickle=True)
    left_hand_data = hand_data[()]["left"]
    left_global_rot = left_hand_data["rot"]
    left_pose = left_hand_data["pose"]
    left_pose = np.concatenate([left_global_rot, left_pose], axis=1)
    frame_cnt = left_pose.shape[0]
    left_shape = left_hand_data["shape"]
    matrix_size = (frame_cnt, ) + left_shape.shape
    left_shape = np.full(matrix_size, left_shape)
    left_trans = left_hand_data["trans"]

    right_hand_data = hand_data[()]["right"]
    right_global_rot = right_hand_data["rot"]
    right_pose = right_hand_data["pose"]
    right_pose = np.concatenate([right_global_rot, right_pose], axis=1)
    frame_cnt = right_pose.shape[0]
    right_shape = right_hand_data["shape"]
    matrix_size = (frame_cnt, ) + right_shape.shape
    right_shape = np.full(matrix_size, right_shape)
    right_trans = right_hand_data["trans"]

    hand_data = {
        "left.pose": left_pose, 
        "left.shape": left_shape, 
        "left.trans": left_trans, 
        "right.pose": right_pose, 
        "right.shape": right_shape, 
        "right.trans": right_trans, 
    }
    
    object_angle = object_data[:, :1]
    object_global_rot = object_data[:, 1:4]
    object_trans = object_data[:, 4:]
    object_data = {
        "object.angle": object_angle,
        "object.global_rot": object_global_rot,
        "object.trans": object_trans,
    }
    return hand_data, object_data


def run_epoch(item, config, pointnet, clip_model, contact_estimator, val=False):
    normalized_obj_pc = item["normalized_obj_pc"].cuda()
    obj_scale = item["obj_scale"].cuda()
    text = item["text"]
    cov_map = item["cov_map"].cuda()
    cov_map = cov_map.unsqueeze(2)
    gaze_map = item["gaze_map"][:, -1].unsqueeze(-1).cuda()
    bs, npts = normalized_obj_pc.shape[:2]

########################################################################################################################
    # feature_map = [cov_map, gaze_map] if config.use_gaze else [cov_map]
    feature_map = [cov_map]

    if config.contact.rot_obj:  ## only applied to rotate, not translate
        if config.contact.aug_rot and not val:
            rotated_pc = rotate_pc_y(
                    normalized_obj_pc.cuda(),
                    (torch.rand(bs) * 360).unsqueeze(-1),
                )
            normalized_obj_pc = rotated_pc.clone()  # if aug
        _, normalized_obj_pc_rotated = process_obj_result(
            normalized_obj_pc.cuda(), item["x_obj"].cuda()
        )
        normalized_obj_pc_rotated = normalized_obj_pc_rotated[:, 0]
    else:
        normalized_obj_pc_rotated = normalized_obj_pc.cuda()
########################################################################################################################

    obj_feat = pointnet(normalized_obj_pc_rotated)  # (B, 1024, 1088)
    enc_text = encoded_text(clip_model, text)  # (B, 512)

    encoder_input = torch.cat(
        [normalized_obj_pc_rotated, *feature_map], dim=2
    )  # (B, 1024, 3+1)
    condition = proc_cond_contact_estimator(  # (B, 1024, 1601) obj_feat -> B, 1024, 1088 (64 global + 1024 local)
        obj_scale, obj_feat, enc_text, npts, config.contact.use_scale
    )
    
########################################################################################################################
    if config.use_gaze: 
        condition = torch.cat([condition, gaze_map], dim = 2)

    # condition = condition * (1 + gaze_map) #  reinforcement condition with gaze
########################################################################################################################
    
    contact_map, mu, logvar = contact_estimator(
        encoder_input,
        condition,

    )
    recon_loss = F.binary_cross_entropy(
        contact_map, cov_map, reduction="sum"
    )
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    dice_loss = 1 - 2 * (
        torch.sum(contact_map * cov_map)
        / (torch.sum(contact_map) + torch.sum(cov_map))
    )
    losses = recon_loss + kl_div + dice_loss

    if val:
        return losses, bs, recon_loss, kl_div, dice_loss
    else:
        return losses, bs   
