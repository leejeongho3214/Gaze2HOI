import os
import os.path as osp

from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn

from lib.networks.cvae import SeqCVAE, CTCVAE
from lib.networks.gaze2hoi import Gaze2HOI
from lib.networks.refiner import Refiner
from lib.networks.diffusion import Diffusion
from lib.networks.pointnet import PointNetfeat

def _print_separator():
    print("-" * 60)

def init_weights_to_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def build_mpnet(args):
    print(f"build mpnet {args.mpnet.version}")
    mpnet = SentenceTransformer(args.mpnet.version)
    mpnet = mpnet.cuda()
    return mpnet

def build_refiner(args, test=False):
    args_refiner = args.refiner
    refiner = Refiner(**args_refiner)
    refiner = refiner.cuda()
    print("build refiner")

    if args_refiner.zero_initialized:
        print("initialize refiner with zero")
        refiner.apply(init_weights_to_zero)

    if test:
        weight_path = args_refiner.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load refiner, {weight_path}")
        checkpoints = torch.load(weight_path)
        refiner.load_state_dict(checkpoints["model"])
        if "epoch" in checkpoints and "loss" in checkpoints:
            print(f"epoch : {checkpoints['epoch']}, loss : {checkpoints['loss']:.5f}")
        _print_separator()
        refiner.eval()
        for p in refiner.parameters():
            p.requires_grad = False
    return refiner

def build_model_and_diffusion(args, lhand_layer, rhand_layer, test=False):

    args_diffusion = args.diffusion

    gaze2hoi_args = dict(args.gaze2hoi.model)
    gaze2hoi_args.pop("use_gaze_alignment", None)
    gaze2hoi_args.pop("gaze_condition_mode", None)
    gaze2hoi_args.pop("null_gaze_condition", None)
    gaze2hoi_args.pop("gaze_alignment_method", None)
    gaze2hoi_args.pop("gaze_alignment_temporal_scope", None)
    gaze2hoi_args.pop("gaze_ray_distance_sigma", None)
    gaze2hoi_args.pop("gaze_ray_distance_sigma_method1", None)
    gaze2hoi_args.pop("gaze_ray_distance_sigma_method2", None)
    gaze2hoi_args.pop("gaze_map_dim", None)
    gaze2hoi_args.pop("contact_map_dim", None)
    gaze2hoi_args.pop("relative_gaze_feat_dim", None)
    gaze2hoi_args.pop("relative_gaze_sequence_length", None)
    gaze2hoi_args.pop("relative_gaze_hidden_dim", None)
    gaze2hoi_args.pop("relative_gaze_dropout", None)
    gaze2hoi_args.pop("output_representation", None)
    gaze2hoi_args.pop("use_point_token_output", None)
    gaze2hoi_args.pop("point_token_count", None)
    gaze2hoi_args.pop("predict_object_pose", None)
    gaze2hoi_args.pop("canonicalize_point_targets", None)
    gaze2hoi_args.pop("include_hand_object_dirvec", None)
    gaze2hoi_args.pop("canonicalize_hand_point_targets", None)
    gaze2hoi_args.pop("canonicalize_object_point_targets", None)
    gaze2hoi_args.pop("object_feature_type", None)
    gaze2hoi_args.pop("object_bps_feature_mode", None)
    gaze2hoi_args.pop("pointnet_obj_dim", None)
    gaze2hoi_args.pop("motion_normalization_min_std", None)
    gaze2hoi_args.pop("use_lift_gate", None)

    checkpoints = None
    if test:
        weight_path = args.gaze2hoi.exp.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        checkpoints = torch.load(weight_path)
        use_ema_for_eval = bool(
            getattr(args.gaze2hoi.exp, "use_ema_for_eval", True)
        )
        ema_state = checkpoints.get("ema")
        checkpoint_model_state = (
            ema_state.get("shadow")
            if use_ema_for_eval
            and isinstance(ema_state, dict)
            and ema_state.get("shadow") is not None
            else checkpoints["model"]
        )
        checkpoint_fusion = checkpoints.get("gaze_token_fusion")
        if checkpoint_fusion is None and "gaze_film.weight" in checkpoint_model_state:
            checkpoint_fusion = "film"
        if checkpoint_fusion is not None:
            configured_fusion = str(gaze2hoi_args.get("gaze_token_fusion", "add")).lower()
            checkpoint_fusion = str(checkpoint_fusion).lower()
            if configured_fusion != checkpoint_fusion:
                print(
                    "Override gaze2hoi.model.gaze_token_fusion from "
                    f"{configured_fusion} to {checkpoint_fusion} for checkpoint compatibility."
                )
                gaze2hoi_args["gaze_token_fusion"] = checkpoint_fusion
                args.gaze2hoi.model.gaze_token_fusion = checkpoint_fusion
        checkpoint_use_scale = checkpoints.get("use_obj_scale")
        checkpoint_use_centroid = checkpoints.get("use_obj_centroid")
        legacy_lift_keys = [
            key
            for key in checkpoint_model_state
            if key.startswith("lift_gate_mlp.")
        ]
        if legacy_lift_keys:
            checkpoint_model_state = checkpoint_model_state.copy()
            for key in legacy_lift_keys:
                del checkpoint_model_state[key]
            print(
                "Ignoring the unused legacy lift-gate head in this checkpoint."
            )
        embed_obj_weight = checkpoint_model_state.get("embed_obj.weight")
        if embed_obj_weight is not None and (
            checkpoint_use_scale is None or checkpoint_use_centroid is None
        ):
            obj_dim = int(gaze2hoi_args.get("obj_dim", 1024))
            metadata_dim = int(embed_obj_weight.shape[1]) - obj_dim
            if metadata_dim not in (0, 1, 3, 4):
                raise ValueError(
                    "Cannot infer object metadata condition from checkpoint: "
                    f"embed_obj input dim={embed_obj_weight.shape[1]}, obj_dim={obj_dim}."
                )
            checkpoint_use_scale = metadata_dim in (1, 4)
            checkpoint_use_centroid = metadata_dim in (3, 4)

        # Do not let the deprecated combined flag override the independent flags.
        gaze2hoi_args.pop("use_obj_scale_centroid", None)
        if hasattr(args.gaze2hoi.model, "use_obj_scale_centroid"):
            del args.gaze2hoi.model["use_obj_scale_centroid"]

        for key, checkpoint_value, default in (
            ("use_obj_scale", checkpoint_use_scale, True),
            ("use_obj_centroid", checkpoint_use_centroid, False),
        ):
            if checkpoint_value is None:
                continue
            checkpoint_value = bool(checkpoint_value)
            configured_value = bool(gaze2hoi_args.get(key, default))
            if configured_value != checkpoint_value:
                print(
                    f"Override gaze2hoi.model.{key} from {configured_value} "
                    f"to {checkpoint_value} for checkpoint compatibility."
                )
            gaze2hoi_args[key] = checkpoint_value
            args.gaze2hoi.model[key] = checkpoint_value

    gaze2hoi = Gaze2HOI(**gaze2hoi_args)
    diffusion = Diffusion(
        lhand_layer=lhand_layer, 
        rhand_layer=rhand_layer, 
        **args_diffusion
    )
    gaze2hoi = gaze2hoi.cuda()
    diffusion = diffusion.cuda()
    print("build gaze2hoi, diffusion")
    
    if test:
        weight_path = args.gaze2hoi.exp.weight_path
        print(f"load gaze2hoi, {weight_path}")
        gaze2hoi.load_state_dict(checkpoint_model_state)
        state_name = (
            "EMA"
            if checkpoints.get("model_is_ema")
            or (
                use_ema_for_eval
            and isinstance(checkpoints.get("ema"), dict)
            and checkpoints["ema"].get("shadow") is not None
            )
            else "raw"
        )
        progress_text = (
            f"iteration : {checkpoints['global_step']}"
            if checkpoints.get("global_step") is not None
            else f"legacy epoch : {checkpoints.get('epoch', 'unknown')}"
        )
        print(
            f"{progress_text}, "
            f"weights : {state_name}, "
            f"loss : {float(checkpoints.get('loss', float('nan'))):.5f}"
        )
        _print_separator()
        gaze2hoi.eval()
        diffusion.eval()
        for p in gaze2hoi.parameters():
            p.requires_grad = False
            
    return gaze2hoi, diffusion

def build_seq_cvae(args, test=False):
    args_cvae = args.seq_cvae
    seq_cvae = SeqCVAE(**args_cvae)
    seq_cvae = seq_cvae.cuda()
    print("build seq cvae")
    
    if test:
        weight_path = args_cvae.weight_path
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"    
        print(f"load seq cvae, {weight_path}")
        checkpoints = torch.load(weight_path)
        seq_cvae.load_state_dict(checkpoints["model"])
        if "epoch" in checkpoints and "loss" in checkpoints:
            print(f"epoch : {checkpoints['epoch']}, loss : {checkpoints['loss']:.5f}")
        _print_separator()
        seq_cvae.eval()
        for p in seq_cvae.parameters():
            p.requires_grad = False
    return seq_cvae

def build_pointnetfeat(args, test=False):
    args_pointfeat = args.pointfeat
    point_encoder = PointNetfeat(**args_pointfeat)
    point_encoder = point_encoder.cuda()
    print("build point encoder")
    
    if test:
        weight_path = args.contact.weight_path if args.contact.epoch is None else osp.join(args.contact.save_root, args.contact.name, f"best_model.pth")
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load point encoder, {weight_path}")

        checkpoints = torch.load(weight_path)
        state_dict = checkpoints.get("model") or checkpoints.get("pointnet_model")
        assert state_dict is not None, "point encoder checkpoint needs either `model` or `pointnet_model` state dict"
        point_encoder.load_state_dict(state_dict)
        if "epoch" in checkpoints:
            if "best_loss" in checkpoints and "best_epoch" in checkpoints:
                print(
                    f"epoch : {checkpoints['epoch']}, "
                    f"best loss : {checkpoints['best_loss']:.0f} ({checkpoints['best_epoch']})"
                )
            elif "loss" in checkpoints:
                print(f"epoch : {checkpoints['epoch']}, loss : {checkpoints['loss']:.5f}")
        _print_separator()
        point_encoder.eval()
        for p in point_encoder.parameters():
            p.requires_grad = False
            
    return point_encoder

def build_contact_estimator(args, test=False):
    args_contact = args.contact
    contact_estimator = CTCVAE(**args_contact)
    contact_estimator = contact_estimator.cuda()
    print("build contact estimator")
    
    if test:
        weight_path = args.contact.weight_path if args.contact.epoch is None else osp.join(args.contact.save_root, args.contact.name, f"best_model.pth")
        assert osp.exists(weight_path), f"{weight_path} deosn't exist!"
        print(f"load contact estimator, {weight_path}")

        checkpoints = torch.load(weight_path)
        state_dict = checkpoints.get("model") or checkpoints.get("contact_model")
        assert state_dict is not None, "contact estimator checkpoint needs either `model` or `contact_model` state dict"
        contact_estimator.load_state_dict(state_dict)
        if "epoch" in checkpoints:
            if "best_loss" in checkpoints and "best_epoch" in checkpoints:
                msg = (
                    f"epoch : {checkpoints['epoch']}, "
                    f"best loss : {checkpoints['best_loss']:.0f} ({checkpoints['best_epoch']})"
                )
                if "valid_best_loss" in checkpoints and "valid_best_epoch" in checkpoints:
                    msg += (
                        f", valid best loss : {checkpoints['valid_best_loss']:.0f} "
                        f"({checkpoints['valid_best_epoch']})"
                    )
                print(msg)
            elif "loss" in checkpoints:
                print(f"epoch : {checkpoints['epoch']}, loss : {checkpoints['loss']:.5f}")
        _print_separator()
        contact_estimator.eval()
        for p in contact_estimator.parameters():
            p.requires_grad = False
    return contact_estimator
