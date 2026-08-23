import os.path as osp
import pickle
import sys

import torch


sys.path.append(osp.dirname(osp.abspath(osp.dirname(__file__))))
from lib.utils.proc import proc_cond_contact_estimator

from lib.utils.augm import rotate_pc_y
from preprocess.preprocessing_hot3d import process_obj_result
import tqdm
import numpy as np
import hydra
from easydict import EasyDict as edict
from omegaconf import OmegaConf, open_dict, MISSING


from lib.networks.clip import load_and_freeze_clip, encoded_text
from lib.datasets.datasets import get_dataloader
from lib.utils.model_utils import (
    build_pointnetfeat,
    build_contact_estimator,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    # Fill missing mandatory values so test_contact can run without extra overrides.
    default_overrides = {
        "contact.rot_obj": False,
        "contact.aug_rot": False,
        "gaze2hoi.use_obj_init": False,
    }
    with open_dict(config):
        for key, value in default_overrides.items():
            if OmegaConf.select(config, key, default=MISSING) is MISSING:
                OmegaConf.update(config, key, value, merge=False)

    config = OmegaConf.to_object(config)
    config = edict(config)
    data_config = config.dataset
    dataset_name = data_config.name

    config.shuffle = False
    config.drop_last = False
    config.test = True
    data_config.augm = False
    
    if config.use_gaze: 
        config.contact.cond_dim = 1602

    dataset, data_loader = get_dataloader(
        "Motion" + dataset_name, config, data_config, test=True
    )

    clip_model = load_and_freeze_clip(config.clip.clip_version)
    clip_model = clip_model.cuda()

    pointnet = build_pointnetfeat(config, test=True)
    contact_estimator = build_contact_estimator(config, test=True)

    pointnet.eval()
    save_file = []
    contact_estimator.eval()

    # obj_list = ["mug_white"]
    # part_list = ["handle"]
    # side_list = ["right"]
    
    obj_list = ["mug_patterned", "mug_white", "flask"]
    part_list = ["handle", "body", "rim"]
    side_list = ["left", "right"]
    
    for item in data_loader:
        text = item['text']
        x_obj = item['x_obj'].cuda()
        obj_scale = item['obj_scale'].cuda()
        normalized_obj_pc = item['normalized_obj_pc'].cuda()
        _, rotated_pc = process_obj_result(
            normalized_obj_pc, x_obj
        )                    
        gaze_map = item["gaze_map"][:, -1].unsqueeze(-1).cuda()
        
        # text = f"Grab {part_name} of {obj_name} with {side} hand."
        # if text not in item["text"]:
        #     continue
        # idx = item["text"].index(text)  # 없으면 ValueError
        # x_obj = item["x_obj"][idx].cuda()
        # normalized_obj_pc = item["normalized_obj_pc"][idx].cuda()

        # _, rotated_pc = process_obj_result(
        #     normalized_obj_pc, item["x_obj"].cuda()
        # )

        # for deg in range(0, 360, 30):
        #             # for deg in [0]:
        #     rotated_pc = rotate_pc_y(normalized_obj_pc.cuda(), deg).cuda()
        #     _, rotated_pc = process_obj_result(
        #         rotated_pc, x_obj
        #     )
            # rotated_pc = rotated_pc[0].cuda()
            
        # obj_feat = pointnet(rotated_pc.unsqueeze(0)).repeat(32, 1, 1)
        # enc_text = encoded_text(clip_model, text).repeat(32, 1)

        # obj_scale = (
        #     torch.tensor(dataset.obj_scale_list[obj_name])
        #     .reshape(1)
        #     .repeat(32)
        #     .cuda()
        # )
        
        rotated_pc = rotated_pc[:, 0]
        obj_feat = pointnet(rotated_pc)
        enc_text = encoded_text(clip_model, text)

        condition = proc_cond_contact_estimator(
            obj_scale, obj_feat, enc_text, 1024, config.contact.use_scale
        )
        condition = torch.cat([condition, gaze_map], dim = 2)

        contact_map = contact_estimator.decode(condition)
        contact_map = (contact_map.squeeze(-1) > 0.5).long()

        save_file.append(
            [
                rotated_pc.detach().cpu(),
                contact_map.detach().cpu(),
                text
                # [f"{text} [rot_y={deg}]" for _ in range(32)],
            ]
        )

    # bs = 32
    # with torch.no_grad():
    #     for obj_name in obj_list:
    #         for part_name in part_list:
    #             for side in side_list:
    #                 text = f"Grab {part_name} of {obj_name} with {side} hand."
    #                 normalized_obj_pc = (
    #                     torch.tensor(dataset.norm_obj_pc_list[obj_name])
    #                     .unsqueeze(0)
    #                     .repeat(bs, 1, 1)
    #                     .cuda()
    #                 )
    #                 obj_scale = (
    #                     torch.tensor(dataset.obj_scale_list[obj_name])
    #                     .reshape(1)
    #                     .repeat(bs)
    #                     .cuda()
    #                 )
    #                 _, npts = normalized_obj_pc.shape[:2]
                    
    #                 obj_feat = pointnet(normalized_obj_pc)
    #                 enc_text = encoded_text(clip_model, text).repeat(bs, 1)

    #                 condition = proc_cond_contact_estimator(
    #                     obj_scale, obj_feat, enc_text, npts, config.contact.use_scale
    #                 )

    #                 contact_map = contact_estimator.decode(condition)
    #                 contact_map = (contact_map.squeeze(-1) > 0.5).long()
                    
    #                 save_file.append(
    #                     [
    #                         normalized_obj_pc.detach().cpu(),
    #                         contact_map.detach().cpu(),
    #                         [f"{text}" for _ in range(bs)],
    #                     ]
    #                 )

    save_name = (
        config.gaze2hoi.save_name
        if config.gaze2hoi.save_name
        else config.gaze2hoi.name
    )
    with open(f"{save_name}.pkl", "wb") as f:
        pickle.dump(save_file, f)

    print(f"Saved at {save_name}.pkl")


if __name__ == "__main__":
    main()
