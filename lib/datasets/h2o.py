import os
import pickle
import time
import numpy as np
import json

import pandas as pd
from torch.utils.data import Dataset
import trimesh

from collections import Counter
from lib.models.object import build_object_model
from lib.utils.frame import get_valid_mask
from lib.utils.augm import (
    augmentation,
    augmentation_joints,
    get_augm_rot,
    get_augm_scale,
)
from lib.utils.proc_h2o import process_text
from lib.utils.proc import (
    get_contact_map,
    pc_normalize,
    process_dist_map,
    select_from_groups,
    process_contact_map,
)
from lib.utils.proc_output import (
    get_hand_verts,
    get_transformed_obj_pc,
)
from lib.models.mano import build_mano_aa
import torch


def _compute_afford_map(
    obj_pc,
    x_obj,
    x_lhand,
    x_rhand,
    lhand_layer,
    rhand_layer,
    is_lhand,
    is_rhand,
    init_frame,
    nframes,
    lcf_idx,
    rcf_idx,
    dataset_name,
):
    if (not is_lhand) and (not is_rhand):
        return np.zeros((1024,), dtype=np.float32)

    contact_frames = []
    if is_lhand and len(lcf_idx) > 0:
        lcf_idx = np.asarray(lcf_idx, dtype=np.int64)
        valid_l = (init_frame <= lcf_idx) & (lcf_idx < init_frame + nframes)
        if valid_l.any():
            contact_frames.append(lcf_idx[valid_l] - init_frame)
    if is_rhand and len(rcf_idx) > 0:
        rcf_idx = np.asarray(rcf_idx, dtype=np.int64)
        valid_r = (init_frame <= rcf_idx) & (rcf_idx < init_frame + nframes)
        if valid_r.any():
            contact_frames.append(rcf_idx[valid_r] - init_frame)

    if not contact_frames:
        return np.zeros((1024,), dtype=np.float32)

    contact_frames = np.unique(np.concatenate(contact_frames)).astype(np.int64)

    with torch.no_grad():
        obj_params = torch.from_numpy(x_obj[:nframes]).float().unsqueeze(0)
        obj_pc_t = torch.from_numpy(obj_pc).float().unsqueeze(0)
        obj_pc_transformed = get_transformed_obj_pc(
            obj_params, obj_pc_t, dataset_name
        )
        obj_pc_transformed = obj_pc_transformed.squeeze(0)

        hand_verts_list = []
        if is_lhand:
            lhand_params = torch.from_numpy(
                x_lhand[:nframes]).float().unsqueeze(0)
            lhand_verts = get_hand_verts(lhand_params, lhand_layer).squeeze(0)
            hand_verts_list.append(lhand_verts)
        if is_rhand:
            rhand_params = torch.from_numpy(
                x_rhand[:nframes]).float().unsqueeze(0)
            rhand_verts = get_hand_verts(rhand_params, rhand_layer).squeeze(0)
            hand_verts_list.append(rhand_verts)

        if not hand_verts_list:
            return np.zeros((1024,), dtype=np.float32)

        hand_verts = torch.cat(hand_verts_list, dim=1)
        frame_idx = torch.from_numpy(contact_frames)
        obj_sel = obj_pc_transformed[frame_idx]
        hand_sel = hand_verts[frame_idx]

        dist = torch.cdist(obj_sel, hand_sel)
        di = dist.min(dim=2).values
        dmin = di.min(dim=0).values

        pc = torch.from_numpy(obj_pc).float()
        dist_pp = torch.cdist(pc.unsqueeze(0), pc.unsqueeze(0)).squeeze(0)
        dist_mask = dist_pp + torch.eye(dist_pp.shape[0]) * 1e6
        nn_dist = dist_mask.min(dim=1).values
        sigma = nn_dist.mean().clamp(min=1e-6)
        weights = torch.exp(-(dist_pp ** 2) / (2 * sigma ** 2))
        weights = weights / weights.sum(dim=1, keepdim=True)
        afford = (weights * dmin.unsqueeze(0)).sum(dim=1)

    return afford.cpu().numpy().astype(np.float32)


class SequenceH2O(Dataset):  # point encoder
    def __init__(
        self,
        data_path,
        data_obj_pc_path,
        text_json,
        max_nframes,
        data_ratio=1.0,
        augm=False,
        **kwargs,
    ):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data h2o!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
            self.nframes = data["nframes"]

        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data h2o!!!", f"{time.time()-start_time:.2f}s")
        print(f"length of data: {self.__len__()}")

    def __len__(self):
        return int(len(self.action_name) * self.data_ratio)

    def __getitem__(self, index):
        item = {}

        nframes = self.nframes[index]
        if nframes > self.max_nframes:
            nframes = self.max_nframes
        seq_time = np.array([nframes / 150], dtype=np.float32)
        if self.augm:
            augm_scale = 1 - (2 * np.random.rand() * 0.05 - 0.05)
            seq_time *= augm_scale
            if seq_time > 1:
                seq_time = np.array([1.0], dtype=np.float32)
        item["seq_time"] = seq_time

        action_name = self.action_name[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]

        text = process_text(
            action_name,
            is_lhand,
            is_rhand,
            self.text_description,
        )
        item["text"] = text
        return item


class ContactH2O(Dataset):  # point encoder
    def __init__(
        self,
        data_path,
        data_obj_pc_path,
        text_json,
        max_nframes,
        obj_name,
        data_ratio=1.0,
        augm=False,
        **kwargs,
    ):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.object_name = obj_name
        self.data_ratio = data_ratio
        self.augm = augm
        self.use_afford_map = kwargs.get("use_afford_map", True)
        flat_hand = kwargs.get("flat_hand", False)
        self.lhand_layer = build_mano_aa(is_rhand=False, flat_hand=flat_hand)
        self.rhand_layer = build_mano_aa(is_rhand=True, flat_hand=flat_hand)

        start_time = time.time()
        print("Start to read data h2o!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.object_idx = data["object_idx"]
            self.lcov_idx = data["lcov_idx"]  # left contact object verts idx
            self.rcov_idx = data["rcov_idx"]  # right contact object verts idx
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data h2o!!!", f"{time.time()-start_time:.2f}s")

    def __len__(self):
        return int(len(self.action_name) * self.data_ratio)

    def __getitem__(self, index):
        item = {}

        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]
        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand

        action_name = self.action_name[index]

        text = process_text(
            action_name,
            is_lhand,
            is_rhand,
            self.text_description,
        )
        item["text"] = text

        object_idx = self.object_idx[index]
        object_name = self.object_name[object_idx]
        item["action_name"] = action_name
        # obj_pc, obj_pc_normal  = self.object_model(object_name)
        _, obj_pc, _, _ = self.object_model(object_name)

        if self.augm:
            aug_scale = get_augm_scale(0.2).numpy()
            obj_pc = obj_pc * aug_scale
            aug_rotmat = get_augm_rot(15, 15, 15).numpy()
            obj_pc = np.einsum("ij,kj->ki", aug_rotmat, obj_pc)
        normalized_obj_pc, _, obj_norm_scale = pc_normalize(
            obj_pc, return_params=True)
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_scale"] = obj_norm_scale

        lcov_idx = self.lcov_idx[index]
        rcov_idx = self.rcov_idx[index]
        lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
        rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
        cov_map = (lcov_map + rcov_map) > 0
        item["cov_map"] = cov_map.astype(np.float32)
        if self.use_afford_map:
            afford_map = _compute_afford_map(
                obj_pc,
                x_obj,
                x_lhand,
                x_rhand,
                self.lhand_layer,
                self.rhand_layer,
                is_lhand,
                is_rhand,
                init_frame,
                nframes,
                lcf_idx,
                rcf_idx,
                "h2o",
            )
            item["afford_map"] = afford_map
        return item


class MotionH2O(Dataset):
    def __init__(
        self,
        data_path,
        data_obj_pc_path,
        text_json,
        max_nframes,
        obj_name,
        data_ratio=1.0,
        augm=False,
        **kwargs,
    ):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.data_path = data_path
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.object_name = obj_name
        self.data_ratio = data_ratio
        self.augm = augm

        start_time = time.time()
        print("Start to read data h2o!!!")
        with np.load(data_path, allow_pickle=True) as data:
            self.object_idx = data["object_idx"]
            self.x_lhand = data["x_lhand"]
            self.x_rhand = data["x_rhand"]
            self.x_obj = data["x_obj"]
            self.lhand_org = data["lhand_org"]
            self.rhand_org = data["rhand_org"]
            self.lcf_idx = data["lcf_idx"]  # left hand contact frame idx
            self.lcov_idx = data["lcov_idx"]  # left contact object verts idx
            self.lchj_idx = data["lchj_idx"]  # left contact hand joints idx
            self.ldist_value = data["ldist_value"]
            self.rcf_idx = data["rcf_idx"]  # right hand contact frame idx
            self.rcov_idx = data["rcov_idx"]  # right contact object verts idx
            self.rchj_idx = data["rchj_idx"]  # right contact hand joints idx
            self.rdist_value = data["rdist_value"]
            self.is_lhand = data["is_lhand"]
            self.is_rhand = data["is_rhand"]
            self.action_name = data["action_name"]
            self.nframes = data["nframes"]

        with open(text_json, "r") as f:
            self.text_description = json.load(f)

        self.object_model = build_object_model(data_obj_pc_path)

        print("Finish to read data h2o!!!", f"{time.time()-start_time:.2f}s")

    def __len__(self):
        return int(len(self.action_name) * self.data_ratio)

    def __getitem__(self, index):
        item = {}

        nframes = self.nframes[index]
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]

        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand
        item["object_idx"] = self.object_idx[index]

        if nframes > self.max_nframes:
            init_frame = np.random.randint(0, nframes - self.max_nframes)
            nframes = self.max_nframes
        else:
            init_frame = 0

        x_obj = self.x_obj[index][init_frame: init_frame + self.max_nframes]
        if self.augm:
            x_obj[:nframes], aug_rotmat, aug_trans = augmentation(
                x_obj[:nframes])
        item["x_obj"] = x_obj

        if is_lhand:
            x_lhand = self.x_lhand[index][init_frame: init_frame +
                                          self.max_nframes]
            if self.augm:
                lhand_org = self.lhand_org[index][init_frame: init_frame + nframes]
                x_lhand[:nframes], _, _ = augmentation(
                    x_lhand[:nframes],
                    hand_org=lhand_org,
                    aug_rotmat=aug_rotmat,
                    aug_trans=aug_trans,
                )
        else:
            x_lhand = np.zeros((150, 99), dtype=np.float32)

        item["x_lhand"] = x_lhand

        if is_rhand:
            x_rhand = self.x_rhand[index][init_frame: init_frame +
                                          self.max_nframes]
            if self.augm:
                rhand_org = self.rhand_org[index][init_frame: init_frame + nframes]
                x_rhand[:nframes], _, _ = augmentation(
                    x_rhand[:nframes],
                    hand_org=rhand_org,
                    aug_rotmat=aug_rotmat,
                    aug_trans=aug_trans,
                )
        else:
            x_rhand = np.zeros((150, 99), dtype=np.float32)
        item["x_rhand"] = x_rhand

        action_name = self.action_name[index]
        max_nframes = self.max_nframes

        valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask(
            is_lhand, is_rhand, max_nframes, nframes
        )  # max_nframes: 2x frames
        item["valid_mask_lhand"] = valid_mask_lhand
        item["valid_mask_rhand"] = valid_mask_rhand
        item["valid_mask_obj"] = valid_mask_obj

        text = process_text(
            action_name,
            is_lhand,
            is_rhand,
            self.text_description,
        )
        item["text"] = text

        object_idx = self.object_idx[index]
        object_name = self.object_name[object_idx]
        item["object_name"] = object_name
        _, obj_pc, obj_pc_normal, _ = self.object_model(object_name)

        normalized_obj_pc, obj_norm_cent, obj_norm_scale = pc_normalize(
            obj_pc, return_params=True
        )
        item["obj_pc"] = obj_pc
        item["normalized_obj_pc"] = normalized_obj_pc
        item["obj_pc_normal"] = obj_pc_normal
        item["obj_cent"] = obj_norm_cent
        item["obj_scale"] = obj_norm_scale

        lcf_idx = self.lcf_idx[index]
        lcov_idx = self.lcov_idx[index]
        lchj_idx = self.lchj_idx[index]
        ldist_value = self.ldist_value[index]

        ldist_map = process_dist_map(
            self.max_nframes,
            init_frame,
            lcf_idx,
            lcov_idx,
            lchj_idx,
            ldist_value,
            is_lhand,
        )
        item["ldist_map"] = ldist_map

        rcf_idx = self.rcf_idx[index]
        rcov_idx = self.rcov_idx[index]
        rchj_idx = self.rchj_idx[index]
        rdist_value = self.rdist_value[index]

        rdist_map = process_dist_map(
            self.max_nframes,
            init_frame,
            rcf_idx,
            rcov_idx,
            rchj_idx,
            rdist_value,
            is_rhand,
        )
        item["rdist_map"] = rdist_map

        lcov_map = get_contact_map(lcov_idx, 1024, is_lhand)
        rcov_map = get_contact_map(rcov_idx, 1024, is_rhand)
        cov_map = (lcov_map + rcov_map) > 0
        item["cov_map"] = cov_map.astype(np.float32)
        return item


class MotionHOT3D(Dataset):
    def __init__(
        self,
        data_name,
        data_root,
        data_obj_pc_path,
        max_nframes,
        test_flag=False,
        data_ratio=1.0,
        augm=False,
        **kwargs,
    ):
        super().__init__()

        self.data_path = os.path.join(data_root, f"{data_name}.pkl")
        self.data_obj_pc_path = data_obj_pc_path
        self.max_nframes = max_nframes
        self.data_ratio = data_ratio
        self.data_name = data_name
        self.augm = augm
        self.test_flag = test_flag
        self.exclude_obj = kwargs.get("exclude_obj", None)
        self.use_afford_map = kwargs.get("use_afford_map", False)
        flat_hand = kwargs.get("flat_hand", False)
        self.lhand_layer = build_mano_aa(is_rhand=False, flat_hand=flat_hand)
        self.rhand_layer = build_mano_aa(is_rhand=True, flat_hand=flat_hand)

        with open(
            "data/hot3d/previous/instance.json", "r"
        ) as f:
            self.obj_json = json.load(f)

        with open("data/hot3d/text.json", "r") as f:
            self.text_description = json.load(f)

        self.obj_list = []
        for key, value in self.obj_json.items():
            self.obj_list.append([key, value["instance_name"]])

        data_paths = self.data_path

        with open(data_paths, "rb") as f:
            dataset_data = pickle.load(f)
        dataset_data = dict(dataset_data)

        (
            self.obj_pc_list,
            self.norm_obj_pc_list,
            self.obj_scale_list,
            self.ori_obj,
            self.obj_cent,
            self.obj_normal_list,
        ) = (dict(), dict(), dict(), dict(), dict(), dict())
        object_model = build_object_model(data_obj_pc_path)

        object_name = "<unknown>"
        if "object_name" in dataset_data:
            name_value = dataset_data["object_name"]
            if torch.is_tensor(name_value):
                name_value = name_value.detach().cpu().numpy()
            if isinstance(name_value, (list, tuple, np.ndarray)) and len(name_value) > 0:
                object_name = str(name_value[0])
            elif name_value is not None:
                object_name = str(name_value)
        elif "object_idx" in dataset_data:
            idx_value = dataset_data["object_idx"]
            if torch.is_tensor(idx_value):
                idx_value = idx_value.detach().cpu().numpy()
            if isinstance(idx_value, (list, tuple, np.ndarray)) and len(idx_value) > 0:
                idx_value = idx_value[0]
            try:
                object_name = self.obj_json[str(
                    int(idx_value))]["instance_name"]
            except Exception:
                object_name = "<unknown>"

        field_map = {
            "object_idx": "object_idx",
            "x_lhand": "x_lhand",
            "x_rhand": "x_rhand",
            "x_obj": "x_obj",
            "lhand_org": "lhand_org",
            "rhand_org": "rhand_org",
            "lcf_idx": "lcf_idx",
            "lcov_idx": "lcov_idx",
            "lchj_idx": "lchj_idx",
            "lcf_ver_idx": "lcf_ver_idx",
            "lcov_ver_idx": "lcov_ver_idx",
            "lchj_ver_idx": "lchj_ver_idx",
            "ldist_value": "ldist_value",
            "rcf_idx": "rcf_idx",
            "rcov_idx": "rcov_idx",
            "rchj_idx": "rchj_idx",
            "rcf_ver_idx": "rcf_ver_idx",
            "rcov_ver_idx": "rcov_ver_idx",
            "rchj_ver_idx": "rchj_ver_idx",
            "rdist_value": "rdist_value",
            "is_lhand": "is_lhand",
            "is_rhand": "is_rhand",
            "action": "action",
            "nframes": "nframes",
            "gaze": "gaze",
            "gaze_map": "gaze_map",
            "act_id": "act_id",
            "afford_map": "afford_map",
        }
        if "cam_pose" in dataset_data:
            field_map["cam_pose"] = "cam_pose"
        combined_fields = {dst: [] for dst in set(field_map.values())}

        for src_key, dst_key in field_map.items():
            if src_key in dataset_data:
                source_value = dataset_data[src_key]
            elif src_key == "rcov_idx" and "rcf_idx" in dataset_data:
                source_value = dataset_data["rcf_idx"]
            else:
                raise KeyError(
                    f"Missing expected key '{src_key}' in HOT3D dataset for object '{object_name}'."
                )

            if torch.is_tensor(source_value):
                source_value = source_value.detach().cpu().numpy()
            elif isinstance(source_value, (list, tuple)):
                processed = [
                    elem.detach().cpu().numpy() if torch.is_tensor(elem) else elem
                    for elem in source_value
                ]
                source_value = np.empty(len(processed), dtype=object)
                for idx, elem in enumerate(processed):
                    source_value[idx] = elem
            elif not isinstance(source_value, np.ndarray):
                if torch.is_tensor(source_value):
                    source_value = source_value.detach().cpu().numpy()
                else:
                    wrapped = np.empty(1, dtype=object)
                    wrapped[0] = source_value
                    source_value = wrapped

            combined_fields[dst_key].append(source_value)

        for attr, parts in combined_fields.items():
            if not parts:
                continue
            setattr(self, attr, np.concatenate(parts, axis=0))

        is_lhand = np.asarray(self.is_lhand)
        is_rhand = np.asarray(self.is_rhand)

        obj_idx = np.asarray(self.object_idx)
        self.obj_names = np.array(
            [self.obj_json[str(i)]["instance_name"] for i in obj_idx]
        )
        if self.exclude_obj:
            excluded_obj = np.atleast_1d(self.exclude_obj)
            target_obj = np.setdiff1d(np.unique(self.obj_names), excluded_obj)
        else:
            excluded_obj = np.array([], dtype=self.obj_names.dtype)
            target_obj = np.unique(self.obj_names)

        keep_mask = np.isin(self.obj_names, target_obj) & ~(
            (is_lhand == 0) & (is_rhand == 0)
        )  # remove no hand interaction samples

        field_map["object_names"] = "object_names"
        for key in field_map.values():
            if hasattr(self, key):
                arr = getattr(self, key)
                if isinstance(arr, np.ndarray) and arr.shape[0] == keep_mask.shape[0]:
                    setattr(self, key, arr[keep_mask])

        for obj_number, object_name in self.obj_list:
            obj_mesh = trimesh.load(
                f"data/hot3d/pre-process/{obj_number}.ply",
                maintain_order=True,
            )

            _, obj_pc, obj_normal, _ = object_model(object_name)
            normalized_obj_pc, obj_cent, obj_norm_scale = pc_normalize(
                obj_pc, return_params=True
            )

            self.obj_pc_list[object_name] = obj_pc
            self.norm_obj_pc_list[object_name] = normalized_obj_pc
            self.obj_scale_list[object_name] = obj_norm_scale
            self.ori_obj[object_name] = [
                torch.tensor(obj_mesh.vertices, dtype=torch.float32),
                torch.tensor(obj_mesh.faces, dtype=torch.float32),
            ]
            self.obj_cent[object_name] = obj_cent
            self.obj_normal_list[object_name] = obj_normal

        text_keys = []
        for i in range(len(self.action)):
            text_key = self.action[i].capitalize()
            text_keys.append(text_key)

        text_counter = Counter(text_keys)
        self.balance_weights = np.array(
            [1.0 / text_counter[k] for k in text_keys], dtype=np.float32
        )

        num_samples = len(self.act_id)
        unique_objs = len(set(self.object_idx.tolist()))
        print(
            f"[MotionHOT3D] samples={num_samples} \n"
            f"unique_objs={unique_objs} \n"
            f"data_name={self.data_name} excluded_obj={excluded_obj.tolist()} obj_name={target_obj} \n"
        )

    def __len__(self):
        return int(len(self.gaze_map))

    def __getitem__(self, index):
        item = {}

        target_length = 100  # 최대 frame 99
        nframes = int(self.nframes[index])

        item["max_nframes"] = target_length
        item["nframes"] = self.nframes[index]

        effective_frames = min(nframes, target_length)
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]

        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand
        item["act_id"] = self.act_id[index]

        object_idx = self.object_idx[index]
        object_name = self.obj_json[str(object_idx)]["instance_name"]

        item["obj_name"] = object_name

        raw_x_obj = np.asarray(self.x_obj[index], dtype=np.float32)
        if raw_x_obj.ndim == 1:
            raw_x_obj = raw_x_obj[np.newaxis, :]
        obj_frames = min(raw_x_obj.shape[0], target_length)
        effective_frames = min(effective_frames, obj_frames)
        x_obj = np.zeros(
            (target_length, raw_x_obj.shape[-1]), dtype=raw_x_obj.dtype)
        x_obj[:obj_frames] = raw_x_obj[:obj_frames]
        if obj_frames > 0 and obj_frames < target_length:
            x_obj[obj_frames:] = raw_x_obj[obj_frames - 1]
        item["x_obj"] = x_obj.astype(np.float32, copy=False)

        if is_lhand:
            raw_x_lhand = np.asarray(self.x_lhand[index], dtype=np.float32)
            if raw_x_lhand.ndim == 1:
                raw_x_lhand = raw_x_lhand[np.newaxis, :]
            lhand_frames = min(raw_x_lhand.shape[0], target_length)
            effective_frames = min(effective_frames, lhand_frames)
            x_lhand = np.zeros(
                (target_length, raw_x_lhand.shape[-1]), dtype=raw_x_lhand.dtype
            )
            x_lhand[:lhand_frames] = raw_x_lhand[:lhand_frames]
            if lhand_frames > 0 and lhand_frames < target_length:
                x_lhand[lhand_frames:] = raw_x_lhand[lhand_frames - 1]
            x_lhand = x_lhand.astype(np.float32, copy=False)
        else:
            x_lhand = np.zeros((target_length, 99), dtype=np.float32)

        item["x_lhand"] = x_lhand

        if is_rhand:
            raw_x_rhand = np.asarray(self.x_rhand[index], dtype=np.float32)
            if raw_x_rhand.ndim == 1:
                raw_x_rhand = raw_x_rhand[np.newaxis, :]
            rhand_frames = min(raw_x_rhand.shape[0], target_length)
            effective_frames = min(effective_frames, rhand_frames)
            x_rhand = np.zeros(
                (target_length, raw_x_rhand.shape[-1]), dtype=raw_x_rhand.dtype
            )
            x_rhand[:rhand_frames] = raw_x_rhand[:rhand_frames]
            if rhand_frames > 0 and rhand_frames < target_length:
                x_rhand[rhand_frames:] = raw_x_rhand[rhand_frames - 1]
            x_rhand = x_rhand.astype(np.float32, copy=False)
        else:
            x_rhand = np.zeros((target_length, 99), dtype=np.float32)
        item["x_rhand"] = x_rhand

        if effective_frames == 0:
            x_obj[:] = 0
            x_lhand[:] = 0
            x_rhand[:] = 0

        valid_mask_lhand, valid_mask_rhand, valid_mask_obj = get_valid_mask(
            is_lhand, is_rhand, target_length, effective_frames
        )
        item[
            "valid_mask_lhand"
        ] = valid_mask_lhand  # If some videos contain missing frames, zero-padding is automatically applied to fill those gaps
        item["valid_mask_rhand"] = valid_mask_rhand
        item["valid_mask_obj"] = valid_mask_obj

        if not self.test_flag:
            text = process_text(
                self.action[index].lower(),
                is_lhand,
                is_rhand,
                self.text_description,
            )
        else:
            text = self.action[index]

        item["text"] = text
        item["obj_pc"] = self.obj_pc_list[object_name]
        
        raw_gaze_seq = torch.as_tensor(self.gaze[index], dtype=torch.float32)
        gaze_frames_seq = min(raw_gaze_seq.shape[0], target_length)
        gaze_seq = torch.zeros((target_length,) + tuple(raw_gaze_seq.shape[1:]), dtype=raw_gaze_seq.dtype)
        gaze_seq[:gaze_frames_seq] = raw_gaze_seq[:gaze_frames_seq]
        if gaze_frames_seq < target_length:
            gaze_seq[gaze_frames_seq:] = raw_gaze_seq[gaze_frames_seq - 1]
            
        item["gaze"] = gaze_seq
        if hasattr(self, "cam_pose"):
            raw_cam_pose_value = self.cam_pose[index]
            if torch.is_tensor(raw_cam_pose_value):
                raw_cam_pose = raw_cam_pose_value.detach().to(dtype=torch.float32)
            else:
                raw_cam_pose = torch.as_tensor(
                    np.asarray(raw_cam_pose_value, dtype=np.float32),
                    dtype=torch.float32,
                )
            if raw_cam_pose.ndim == 2:
                raw_cam_pose = raw_cam_pose.unsqueeze(0)
            cam_pose_frames = min(raw_cam_pose.shape[0], target_length)
            cam_pose = torch.zeros(
                (target_length,) + tuple(raw_cam_pose.shape[1:]),
                dtype=raw_cam_pose.dtype,
            )
            cam_pose[:cam_pose_frames] = raw_cam_pose[:cam_pose_frames]
            if cam_pose_frames > 0 and cam_pose_frames < target_length:
                cam_pose[cam_pose_frames:] = raw_cam_pose[cam_pose_frames - 1]
            item["cam_pose"] = cam_pose
        raw_gaze = self.gaze_map[index]
        gaze_map = torch.zeros([target_length, 1024], dtype=torch.float32)
        gaze_frames = min(len(raw_gaze), target_length, effective_frames)

        for idx in range(gaze_frames):
            gaze_map[idx][raw_gaze[idx]] = 1
            
        if gaze_map[idx].sum() == 0:
            raise ValueError(
                f"Gaze map at frame {idx} is empty for sample index {index} (object: {object_name})."
            )
        if gaze_frames > 0 and gaze_frames < target_length:
            gaze_map[gaze_frames:] = gaze_map[gaze_frames - 1]

        item["gaze_map"] = gaze_map
        item["normalized_obj_pc"] = self.norm_obj_pc_list[object_name]
        item["obj_pc_normal"] = self.norm_obj_pc_list[object_name]
        item["obj_scale"] = self.obj_scale_list[object_name]
        item["obj_cent"] = self.obj_cent[object_name]

        lcf_idx = self.lcf_idx[index]
        lcov_idx = self.lcov_idx[index]
        lchj_idx = self.lchj_idx[index]
        ldist_value = self.ldist_value[index]
        lcf_ver_idx = self.lcf_ver_idx[index]
        lcov_ver_idx = self.lcov_ver_idx[index]
        lchj_ver_idx = self.lchj_ver_idx[index]

        ldist_map = process_dist_map(
            target_length, 0, lcf_idx, lcov_idx, lchj_idx, ldist_value, is_lhand
        )
        item["ldist_map"] = ldist_map

        rcf_idx = self.rcf_idx[index]
        rcov_idx = self.rcov_idx[index]
        rchj_idx = self.rchj_idx[index]
        rcf_ver_idx = self.rcf_ver_idx[index]
        rcov_ver_idx = self.rcov_ver_idx[index]
        rchj_ver_idx = self.rchj_ver_idx[index]
        rdist_value = self.rdist_value[index]

        rdist_map = process_dist_map(
            target_length, 0, rcf_idx, rcov_idx, rchj_idx, rdist_value, is_rhand
        )
        item["rdist_map"] = rdist_map

        lcf_ver_idx = np.asarray(lcf_ver_idx)
        lcov_ver_idx = np.asarray(lcov_ver_idx)
        lchj_ver_idx = np.asarray(lchj_ver_idx)
        rcf_ver_idx = np.asarray(rcf_ver_idx)
        rcov_ver_idx = np.asarray(rcov_ver_idx)
        rchj_ver_idx = np.asarray(rchj_ver_idx)

        if lcf_ver_idx.size > 0:
            valid_left_mask = lcf_ver_idx < target_length
            lcf_ver_idx = lcf_ver_idx[valid_left_mask]
            lcov_ver_idx = lcov_ver_idx[valid_left_mask]
            lchj_ver_idx = lchj_ver_idx[valid_left_mask]
        if rcf_ver_idx.size > 0:
            valid_right_mask = rcf_ver_idx < target_length
            rcf_ver_idx = rcf_ver_idx[valid_right_mask]
            rcov_ver_idx = rcov_ver_idx[valid_right_mask]
            rchj_ver_idx = rchj_ver_idx[valid_right_mask]

        l_map = np.zeros((target_length, 778), dtype=np.float32)
        if is_lhand and lcf_ver_idx.size > 0:
            l_map[lcf_ver_idx.astype(
                np.int64), lchj_ver_idx.astype(np.int64)] = 1
        r_map = np.zeros((target_length, 778), dtype=np.float32)
        if is_rhand and rcf_ver_idx.size > 0:
            r_map[rcf_ver_idx.astype(
                np.int64), rchj_ver_idx.astype(np.int64)] = 1

        item["lhand_map"] = l_map
        item["rhand_map"] = r_map

        lcov_map = np.zeros([target_length, 1024])
        if lcf_ver_idx.size > 0:
            lcov_map[lcf_ver_idx.astype(
                np.int64), lcov_ver_idx.astype(np.int64)] = 1

        rcov_map = np.zeros([target_length, 1024])
        if rcf_ver_idx.size > 0:
            rcov_map[rcf_ver_idx.astype(
                np.int64), rcov_ver_idx.astype(np.int64)] = 1

        capped_nframes = min(nframes, target_length)
        cov_map_idx = max(0, capped_nframes - 1)
        item["cov_map"] = ((lcov_map + rcov_map) >
                           0).astype(np.float32)[cov_map_idx]
        item["afford_map"] = self.afford_map[index]

        return item


class ContactHOT3D(MotionHOT3D):  # point encoder
    def __init__(
        self,
        data_name,
        data_root,
        data_obj_pc_path,
        max_nframes,
        test_flag=False,
        data_ratio=1.0,
        augm=False,
        **kwargs,
    ):
        super().__init__(
            data_name,
            data_root,
            data_obj_pc_path,
            max_nframes,
            test_flag=test_flag,
            **kwargs,
        )

    def __len__(self):
        return len(self.act_id) - 1

    def __getitem__(self, index):
        item = {}

        target_length = 100  # 최대 frame 99
        nframes = int(self.nframes[index])

        item["max_nframes"] = target_length
        item["nframes"] = self.nframes[index]

        effective_frames = min(nframes, target_length)
        is_lhand = self.is_lhand[index]
        is_rhand = self.is_rhand[index]

        item["is_lhand"] = is_lhand
        item["is_rhand"] = is_rhand
        item["act_id"] = self.act_id[index]

        object_idx = self.object_idx[index]
        object_name = self.obj_json[str(object_idx)]["instance_name"]

        item["obj_name"] = object_name

        raw_x_obj = np.asarray(self.x_obj[index], dtype=np.float32)
        if raw_x_obj.ndim == 1:
            raw_x_obj = raw_x_obj[np.newaxis, :]
        obj_frames = min(raw_x_obj.shape[0], target_length)
        effective_frames = min(effective_frames, obj_frames)
        x_obj = np.zeros(
            (target_length, raw_x_obj.shape[-1]), dtype=raw_x_obj.dtype)
        x_obj[:obj_frames] = raw_x_obj[:obj_frames]
        if obj_frames > 0 and obj_frames < target_length:
            x_obj[obj_frames:] = raw_x_obj[obj_frames - 1]
        item["x_obj"] = x_obj.astype(np.float32, copy=False)

        if not self.test_flag and self.data_name == "grab_ori":
            text = process_text(
                self.action[index].lower(),
                is_lhand,
                is_rhand,
                self.text_description,
            )
        else:
            text = self.action[index]

        item["text"] = text

        item["obj_pc"] = self.obj_pc_list[object_name]

        item["normalized_obj_pc"] = self.norm_obj_pc_list[object_name]
        item["obj_scale"] = self.obj_scale_list[object_name]
        item["obj_cent"] = self.obj_cent[object_name]

        lcf_idx = self.lcf_idx[index]
        lcov_idx = self.lcov_idx[index]
        lchj_idx = self.lchj_idx[index]
        ldist_value = self.ldist_value[index]
        lcf_ver_idx = self.lcf_ver_idx[index]
        lcov_ver_idx = self.lcov_ver_idx[index]
        lchj_ver_idx = self.lchj_ver_idx[index]

        ldist_map = process_dist_map(
            target_length, 0, lcf_idx, lcov_idx, lchj_idx, ldist_value, is_lhand
        )
        item["ldist_map"] = ldist_map

        rcf_idx = self.rcf_idx[index]
        rcov_idx = self.rcov_idx[index]
        rchj_idx = self.rchj_idx[index]
        rcf_ver_idx = self.rcf_ver_idx[index]
        rcov_ver_idx = self.rcov_ver_idx[index]
        rchj_ver_idx = self.rchj_ver_idx[index]
        rdist_value = self.rdist_value[index]

        rdist_map = process_dist_map(
            target_length, 0, rcf_idx, rcov_idx, rchj_idx, rdist_value, is_rhand
        )
        item["rdist_map"] = rdist_map

        lcf_ver_idx = np.asarray(lcf_ver_idx)
        lcov_ver_idx = np.asarray(lcov_ver_idx)
        lchj_ver_idx = np.asarray(lchj_ver_idx)
        rcf_ver_idx = np.asarray(rcf_ver_idx)
        rcov_ver_idx = np.asarray(rcov_ver_idx)
        rchj_ver_idx = np.asarray(rchj_ver_idx)

        if lcf_ver_idx.size > 0:
            valid_left_mask = lcf_ver_idx < target_length
            lcf_ver_idx = lcf_ver_idx[valid_left_mask]
            lcov_ver_idx = lcov_ver_idx[valid_left_mask]
            lchj_ver_idx = lchj_ver_idx[valid_left_mask]

        if rcf_ver_idx.size > 0:
            valid_right_mask = rcf_ver_idx < target_length
            rcf_ver_idx = rcf_ver_idx[valid_right_mask]
            rcov_ver_idx = rcov_ver_idx[valid_right_mask]
            rchj_ver_idx = rchj_ver_idx[valid_right_mask]

        l_map = np.zeros((target_length, 778), dtype=np.float32)
        if is_lhand and lcf_ver_idx.size > 0:
            l_map[lcf_ver_idx.astype(
                np.int64), lchj_ver_idx.astype(np.int64)] = 1
        r_map = np.zeros((target_length, 778), dtype=np.float32)
        if is_rhand and rcf_ver_idx.size > 0:
            r_map[rcf_ver_idx.astype(
                np.int64), rchj_ver_idx.astype(np.int64)] = 1

        item["lhand_map"] = l_map
        item["rhand_map"] = r_map

        lcov_map = np.zeros([target_length, 1024])
        if lcf_ver_idx.size > 0:
            lcov_map[lcf_ver_idx.astype(
                np.int64), lcov_ver_idx.astype(np.int64)] = 1

        rcov_map = np.zeros([target_length, 1024])
        if rcf_ver_idx.size > 0:
            rcov_map[rcf_ver_idx.astype(
                np.int64), rcov_ver_idx.astype(np.int64)] = 1

        capped_nframes = min(nframes, target_length)
        cov_map_idx = max(0, capped_nframes - 1)
        item["cov_map"] = ((lcov_map + rcov_map) >
                           0).astype(np.float32)[cov_map_idx]

        return item
