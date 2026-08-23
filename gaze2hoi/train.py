import os.path as osp
import random
import shutil
import sys

# Keep this bootstrap before project imports (formatters may run isort separately).
PROJECT_ROOT = osp.dirname(osp.abspath(osp.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# isort: split

from lib.utils.train_gaze2hoi_utils import (
    TeeIO,
    get_current_git_branch,
    save_code_snapshot,
)
from lib.utils.proc import (
    load_bps_basis,
    proc_obj_feat_final_train,
)
from lib.utils.seed import seed_everything
from lib.utils.file import (
    make_model_result_folder,
    wandb_login,
)
from lib.utils.metric import AverageMeter
from lib.utils.gaze2hoi_config import (
    get_gaze2hoi_paths,
    move_batch_to_cuda,
)
from lib.utils.model_utils import (
    build_model_and_diffusion,
    build_pointnetfeat,
)
from lib.datasets.datasets import get_train_validation_dataloaders
from lib.models.mano import build_mano_aa
from lib.utils.gaze2hoi_train_helpers import (
    _is_relative_gaze_mlp_mode,
    _is_temporal_gaze_token_mode,
    build_gaze_alignment_temporal_mask,
    apply_null_gaze_condition,
    build_relative_gaze_mlp_for_gaze2hoi,
    build_bps_correspondence_cache,
    build_gaze_condition_feature_for_gaze2hoi,
    build_point_token_motion_targets_for_gaze2hoi,
    build_hybrid_motion_targets_for_gaze2hoi,
    compute_bps_feature_from_mesh_cache_for_gaze2hoi,
    configure_gaze_token_fusion_for_mode,
    get_bps_correspondence_source,
    get_gaze2hoi_gaze_condition_dim,
    gaze_condition_requires_bps,
    load_hand_sample_indices_for_gaze2hoi,
    load_object_mesh_bps_cache,
    repeat_initial_object_pose_for_gaze_condition,
    resolve_partwise_bps_context,
)
from lib.utils.motion_normalizer import (
    MaskedFeatureStats,
    MotionNormalizer,
    finalize_motion_normalizer,
)
from lib.utils.training import ExponentialMovingAverage, warmup_cosine_factor
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from datetime import datetime
import hydra
import numpy as np
import tqdm
import os


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


@torch.no_grad()
def _compute_hybrid_motion_normalizer(
    dataloader,
    lhand_layer,
    rhand_layer,
    hand_sample_indices,
    dataset_name,
    include_hand_object_dirvec=True,
    canonicalize_hand_targets=True,
    min_std=1e-4,
):
    point_count = int(hand_sample_indices.numel())
    hand_dim = point_count * (6 if include_hand_object_dirvec else 3)
    left_stats = MaskedFeatureStats(hand_dim)
    right_stats = MaskedFeatureStats(hand_dim)
    object_stats = MaskedFeatureStats(9)
    stat_keys = [
        "x_lhand",
        "x_rhand",
        "x_obj",
        "obj_pc",
        "valid_mask_lhand",
        "valid_mask_rhand",
        "valid_mask_obj",
    ]
    for item in tqdm.tqdm(dataloader, desc="COMPUTING MOTION NORMALIZATION"):
        batch = move_batch_to_cuda(item, stat_keys)
        target_lhand, target_rhand, target_obj = build_hybrid_motion_targets_for_gaze2hoi(
            batch["x_lhand"],
            batch["x_rhand"],
            batch["x_obj"],
            batch["obj_pc"],
            lhand_layer,
            rhand_layer,
            dataset_name,
            hand_sample_indices,
            include_hand_object_dirvec=include_hand_object_dirvec,
            canonicalize_hand_targets=canonicalize_hand_targets,
        )
        left_stats.update(target_lhand, batch["valid_mask_lhand"])
        right_stats.update(target_rhand, batch["valid_mask_rhand"])
        object_stats.update(target_obj, batch["valid_mask_obj"])
    return finalize_motion_normalizer(
        left_stats, right_stats, object_stats, min_std=min_std
    )


def _load_or_compute_motion_normalizer(
    stats_path,
    checkpoint_path,
    compute_fn,
):
    if osp.exists(stats_path):
        state = torch.load(stats_path, map_location="cpu")
        print(f"Load motion normalization statistics from {stats_path}")
        return MotionNormalizer.from_state_dict(state)
    if checkpoint_path and osp.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("motion_normalization")
        if state is not None:
            print(f"Load motion normalization statistics from {checkpoint_path}")
            normalizer = MotionNormalizer.from_state_dict(state)
            torch.save(normalizer.state_dict(), stats_path)
            return normalizer
    normalizer = compute_fn()
    torch.save(normalizer.state_dict(), stats_path)
    print(f"Saved motion normalization statistics to {stats_path}")
    return normalizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    config = OmegaConf.to_object(config)
    config = edict(config)
    train_seed = seed_everything(
        getattr(config.gaze2hoi.exp, "seed", 0),
        deterministic=bool(getattr(config.gaze2hoi.exp, "deterministic", False)),
    )
    print(
        f"Using seed={train_seed} "
        f"deterministic={bool(getattr(config.gaze2hoi.exp, 'deterministic', False))}"
    )
    print(
        "Null gaze condition: "
        f"{'ON' if bool(getattr(config.gaze2hoi.model, 'null_gaze_condition', False)) else 'OFF'}"
    )

    train_data_name = getattr(config.gaze2hoi.exp, "train_data_name", None)
    if train_data_name:
        config.dataset.data_name = train_data_name

    wandb = wandb_login(config, config.gaze2hoi.exp, relogin=False)
    (
        model_name,
        save_root,
        data_config,
        lambda_simple,
        max_iterations,
    ) = get_gaze2hoi_paths(config)

    model_folder = make_model_result_folder(save_root)
    log_path = osp.join(model_folder, "log.txt")
    reset = bool(getattr(config, "reset", False))

    # training state init
    cur_loss = float("nan")
    start_epoch = 0
    resume_checkpoint = None

    if reset and osp.isdir(model_folder):
        for filename in os.listdir(model_folder):
            path = osp.join(model_folder, filename)
            if osp.isfile(path) or osp.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        print("Reset gaze2hoi: cleared existing checkpoints/logs.")

    os.makedirs(model_folder, exist_ok=True)
    git_branch = get_current_git_branch(PROJECT_ROOT)
    if (not osp.exists(log_path)) or osp.getsize(log_path) == 0:
        cmd_line = "python " + " ".join(sys.argv)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "w") as f:
            f.write(f"[{timestamp}] cmd: {cmd_line}\n")
            f.write(f"[{timestamp}] git_branch: {git_branch}\n")
    log_stream = open(log_path, "a", buffering=1)
    sys.stdout = TeeIO(sys.stdout, log_stream)

    lhand_layer = build_mano_aa(is_rhand=False, flat_hand=False).cuda()
    rhand_layer = build_mano_aa(is_rhand=True, flat_hand=False).cuda()

    (
        train_dataset,
        dataloader,
        validation_dataset,
        _validation_loader,
        train_indices,
        validation_indices,
    ) = get_train_validation_dataloaders(
        "Motion" + data_config.name,
        config,
        data_config,
        validation_ratio=float(
            getattr(config.gaze2hoi.exp, "validation_ratio", 0.1)
        ),
    )
    normalization_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )
    print(
        "Train/validation split: "
        f"train={len(train_dataset)} (test_flag=False, "
        f"{'weighted sampler' if config.balance_weights else 'shuffle'}), "
        f"validation={len(validation_dataset)} "
        "(test_flag=True, deterministic order)."
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
        print(
            f"Using GRAB BPS basis from {bps_path} "
            f"with shape {tuple(bps_basis.shape)}"
        )
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
    use_contact_condition = gaze_condition_mode in (
        "contact_map",
        "cov_map",
        "gt_contact_map",
        "bps_contact_map",
        "bps_cov_map",
        "raw_contact_map",
        "raw_cov_map",
    )
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
    use_obj_scale = bool(getattr(config.gaze2hoi.model, "use_obj_scale", True))
    use_obj_centroid = bool(
        getattr(config.gaze2hoi.model, "use_obj_centroid", False)
    )
    config.gaze2hoi.model.use_obj_scale = use_obj_scale
    config.gaze2hoi.model.use_obj_centroid = use_obj_centroid
    canonicalize_point_targets = bool(
        getattr(config.gaze2hoi.model, "canonicalize_point_targets", True)
    )
    canonicalize_hand_point_targets = canonicalize_point_targets
    canonicalize_object_point_targets = canonicalize_point_targets
    config.gaze2hoi.model.canonicalize_hand_point_targets = canonicalize_point_targets
    config.gaze2hoi.model.canonicalize_object_point_targets = canonicalize_point_targets
    if use_point_token_output:
        point_coord_dim = int(hand_sample_indices.numel()) * 3
        config.gaze2hoi.model.hand_nfeats = point_coord_dim * 2
        config.gaze2hoi.model.obj_nfeats = 9 if predict_object_pose else point_coord_dim
    else:
        config.gaze2hoi.model.hand_nfeats = 99
        config.gaze2hoi.model.obj_nfeats = 9
    if use_point_token_output:
        print(
            "Using point-token Gaze2HOI targets: "
            f"{hand_sample_indices.numel()} points per entity, "
            f"hand feature dim {config.gaze2hoi.model.hand_nfeats}, "
            f"object representation {'relative 9D pose' if predict_object_pose else 'point tokens'} "
            f"with dim {config.gaze2hoi.model.obj_nfeats}. "
            f"Hand targets are {'canonicalized' if canonicalize_hand_point_targets else 'world-space'}. "
            f"Object targets are {'canonicalized' if canonicalize_object_point_targets else 'world-space'}. "
            f"Object scale condition is {'enabled' if use_obj_scale else 'disabled'}; "
            f"object centroid condition is {'enabled' if use_obj_centroid else 'disabled'}. "
            f"Legacy relative-orientation loss is disabled; hybrid object pose uses "
            f"translation/geodesic rotation losses. Hand points and direction vectors "
            f"are supervised by simple loss only."
        )
    else:
        print(
            "Using MANO-parameter Gaze2HOI targets: "
            f"hand feature dim {config.gaze2hoi.model.hand_nfeats}, "
            f"object pose dim {config.gaze2hoi.model.obj_nfeats}."
        )
    motion_normalizer = None
    if use_point_token_output and predict_object_pose:
        stats_path = osp.join(model_folder, "motion_normalization_stats.pt")
        configured_resume_path = getattr(config.gaze2hoi.exp, "resume_path", None)
        default_resume_path = osp.join(model_folder, "latest_model.pth")
        stats_checkpoint_path = (
            configured_resume_path
            if configured_resume_path and osp.exists(configured_resume_path)
            else default_resume_path
        )
        motion_normalizer = _load_or_compute_motion_normalizer(
            stats_path,
            stats_checkpoint_path,
            lambda: _compute_hybrid_motion_normalizer(
                normalization_loader,
                lhand_layer,
                rhand_layer,
                hand_sample_indices,
                data_config.name,
                include_hand_object_dirvec=include_hand_object_dirvec,
                canonicalize_hand_targets=canonicalize_hand_point_targets,
                min_std=float(
                    getattr(config.gaze2hoi.model, "motion_normalization_min_std", 1e-4)
                ),
            ),
        )
        print(
            "Hybrid motion normalization enabled: "
            f"left std [{motion_normalizer.left_std.min():.4g}, {motion_normalizer.left_std.max():.4g}], "
            f"right std [{motion_normalizer.right_std.min():.4g}, {motion_normalizer.right_std.max():.4g}], "
            f"object std [{motion_normalizer.object_std.min():.4g}, {motion_normalizer.object_std.max():.4g}]."
        )
    gaze2hoi, diffusion = build_model_and_diffusion(
        config, lhand_layer, rhand_layer)
    relative_gaze_mlp = (
        build_relative_gaze_mlp_for_gaze2hoi(config, output_dim=gaze_condition_dim).cuda()
        if use_relative_gaze_mlp
        else None
    )
    if relative_gaze_mlp is not None:
        print(
            "Using object-relative raw gaze MLP condition: "
            f"sequence length={getattr(config.gaze2hoi.model, 'relative_gaze_sequence_length', 100)}, "
            f"feature dim={gaze_condition_dim}."
        )
    pointnet = None
    if use_pointnet_object_feature:
        config.pointfeat.global_feat = True
        pointnet = build_pointnetfeat(config, test=False)
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

    resume_path_cfg = getattr(config.gaze2hoi.exp, "resume_path", None)
    latest_model_path = osp.join(model_folder, "latest_model.pth")
    resume_path = resume_path_cfg or latest_model_path
    if resume_path_cfg and not osp.exists(resume_path_cfg):
        raise FileNotFoundError(
            f"Explicit gaze2hoi.exp.resume_path does not exist: {resume_path_cfg}"
        )
    if not resume_path_cfg and osp.exists(latest_model_path):
        print(f"Found latest training checkpoint: {latest_model_path}")

    if osp.exists(resume_path):
        print(f"Resume gaze2hoi from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cuda")
        resume_checkpoint = ckpt
        try:
            gaze2hoi.load_state_dict(ckpt["model"])
            if use_pointnet_object_feature and pointnet is not None:
                pointnet_state = ckpt.get("pointnet_model")
                if pointnet_state is None:
                    raise RuntimeError(
                        "Checkpoint does not contain `pointnet_model` state."
                    )
                pointnet.load_state_dict(pointnet_state)
            if relative_gaze_mlp is not None:
                relative_gaze_state = ckpt.get("relative_gaze_mlp")
                if relative_gaze_state is None:
                    raise RuntimeError(
                        "Checkpoint does not contain `relative_gaze_mlp` state, "
                        "but gaze_condition_mode uses relative_gaze_mlp."
                    )
                relative_gaze_mlp.load_state_dict(relative_gaze_state)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint. This usually means the checkpoint was trained "
                "with a different object feature layout, such as BPS vs PointNet "
                "or old 1-part dimensions. Use a fresh exp.name/weight_path."
            ) from exc
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step")
        if global_step is None:
            global_step = start_epoch * len(dataloader)
            print(
                "Legacy checkpoint has no global_step; approximating it as "
                f"completed_epochs * steps_per_epoch = {global_step}."
            )
    else:
        print("No pretrained gaze2hoi found, training from scratch")
        global_step = 0

    trainable_params = list(gaze2hoi.parameters())
    if pointnet is not None:
        trainable_params += list(pointnet.parameters())
    if relative_gaze_mlp is not None:
        trainable_params += list(relative_gaze_mlp.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.gaze2hoi.exp.lr)
    warmup_iterations = int(
        getattr(config.gaze2hoi.exp, "warmup_iterations", 2000)
    )
    min_lr_ratio = float(getattr(config.gaze2hoi.exp, "min_lr_ratio", 0.0))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_factor(
            step,
            warmup_steps=warmup_iterations,
            total_steps=max_iterations,
            min_lr_ratio=min_lr_ratio,
        ),
    )
    use_amp = bool(getattr(config.gaze2hoi.exp, "use_amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ExponentialMovingAverage(
        gaze2hoi,
        decay=float(getattr(config.gaze2hoi.exp, "ema_decay", 0.9999)),
    )
    pointnet_ema = (
        ExponentialMovingAverage(pointnet, decay=ema.decay)
        if pointnet is not None
        else None
    )
    relative_gaze_ema = (
        ExponentialMovingAverage(relative_gaze_mlp, decay=ema.decay)
        if relative_gaze_mlp is not None
        else None
    )
    if resume_checkpoint is not None:
        optimizer_state = resume_checkpoint.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print("Restored optimizer state from checkpoint")
        else:
            print(
                "Checkpoint has no optimizer state; continuing with a newly "
                "initialized optimizer."
            )
        scheduler_state = resume_checkpoint.get("scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            print("Restored LR scheduler state from checkpoint")
        else:
            scheduler.last_epoch = global_step
            factor = warmup_cosine_factor(
                global_step,
                warmup_steps=warmup_iterations,
                total_steps=max_iterations,
                min_lr_ratio=min_lr_ratio,
            )
            for param_group, base_lr in zip(
                optimizer.param_groups, scheduler.base_lrs
            ):
                param_group["lr"] = base_lr * factor
            scheduler._last_lr = [
                param_group["lr"] for param_group in optimizer.param_groups
            ]
            print("Initialized LR scheduler position from global_step")
        scaler_state = resume_checkpoint.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
            print("Restored AMP GradScaler state from checkpoint")
        ema_state = resume_checkpoint.get("ema")
        if ema_state is not None:
            ema.load_state_dict(ema_state)
            print("Restored EMA state from checkpoint")
        elif "ema_model" in resume_checkpoint:
            ema.load_state_dict(
                {
                    "decay": ema.decay,
                    "num_updates": global_step,
                    "shadow": resume_checkpoint["ema_model"],
                }
            )
            print("Restored legacy EMA model state from checkpoint")
        else:
            print("Checkpoint has no EMA state; initialized EMA from raw model")
        if pointnet_ema is not None and resume_checkpoint.get("pointnet_ema"):
            pointnet_ema.load_state_dict(resume_checkpoint["pointnet_ema"])
            print("Restored PointNet EMA state from checkpoint")
        if (
            relative_gaze_ema is not None
            and resume_checkpoint.get("relative_gaze_ema")
        ):
            relative_gaze_ema.load_state_dict(
                resume_checkpoint["relative_gaze_ema"]
            )
            print("Restored relative-gaze MLP EMA state from checkpoint")
        rng_state = resume_checkpoint.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in rng_state["cuda"]]
            )
            print("Restored Python/NumPy/PyTorch RNG state from checkpoint")

    lambda_object_translation = float(
        getattr(config.gaze2hoi.exp, "lambda_obj_trans", 1.0)
    )
    lambda_object_rotation = float(
        getattr(config.gaze2hoi.exp, "lambda_obj_rot", 0.1)
    )
    print(
        "Using paper training objective only: "
        f"L = {lambda_simple:.4f} * L_rec + "
        f"{lambda_object_translation:.4f} * L_trans + "
        f"{lambda_object_rotation:.4f} * L_rot."
    )

    log_iteration_freq = int(
        getattr(config.gaze2hoi.exp, "log_iteration_freq", 100)
    )
    latest_checkpoint_freq = int(
        getattr(config.gaze2hoi.exp, "latest_checkpoint_freq", 1000)
    )
    checkpoint_freq = int(
        getattr(config.gaze2hoi.exp, "checkpoint_freq", 10000)
    )
    grad_clip_norm = float(
        getattr(config.gaze2hoi.exp, "grad_clip_norm", 1.0)
    )
    print(
        "Iteration-based training: "
        f"start={global_step}, target={max_iterations}, "
        f"steps_per_epoch={len(dataloader)}, AMP={use_amp}, "
        f"EMA={ema.decay}, warmup={warmup_iterations}, "
        f"latest_every={latest_checkpoint_freq}, "
        f"milestone_every={checkpoint_freq}."
    )
    cuda_keys = [
        "x_lhand",
        "x_rhand",
        "x_obj",
        "obj_pc",
        "normalized_obj_pc",
        "obj_scale",
        "obj_cent",
        "ldist_map",
        "rdist_map",
        "valid_mask_lhand",
        "valid_mask_rhand",
        "valid_mask_obj",
        "gaze",
    ]
    if gaze_condition_mode in ("gaze_map", "raw_gaze_map", "dataset_gaze_map"):
        cuda_keys.append("gaze_map")
    if use_contact_condition:
        cuda_keys.append("cov_map")
    if global_step >= max_iterations:
        print(
            f"Checkpoint iteration {global_step} already reaches target "
            f"{max_iterations}, nothing to train."
        )
        return

    snapshot_dir = osp.join(model_folder, "code_snapshot")
    if reset or not osp.isdir(snapshot_dir):
        save_code_snapshot(PROJECT_ROOT, snapshot_dir)

    def save_checkpoint(epoch, save_path, iteration_loss, resumable):
        checkpoint = {
            "model": (
                gaze2hoi.state_dict()
                if resumable
                else ema.model_state_dict()
            ),
            "model_is_ema": not resumable,
            "resumable": bool(resumable),
            "epoch": epoch,
            "global_step": global_step,
            "iteration": global_step,
            # Keep `loss` for compatibility with older loading code.
            "loss": iteration_loss,
            "iteration_loss": iteration_loss,
            "use_contact_map": use_contact_condition,
            "gaze_condition_mode": gaze_condition_mode,
            "null_gaze_condition": bool(
                getattr(config.gaze2hoi.model, "null_gaze_condition", False)
            ),
            "gaze_token_fusion": str(
                getattr(config.gaze2hoi.model, "gaze_token_fusion", "add")
            ),
            "cross_attn_order": str(
                getattr(config.gaze2hoi.model, "cross_attn_order", "hand_object")
            ),
            "gaze_token_target": str(
                getattr(config.gaze2hoi.model, "gaze_token_target", "object")
            ),
            "use_obj_scale": bool(
                getattr(config.gaze2hoi.model, "use_obj_scale", True)
            ),
            "use_obj_centroid": bool(
                getattr(config.gaze2hoi.model, "use_obj_centroid", False)
            ),
            "lambda_reconstruction": float(lambda_simple),
            "lambda_object_translation": lambda_object_translation,
            "lambda_object_rotation": lambda_object_rotation,
            "seed": int(getattr(config.gaze2hoi.exp, "seed", 0)),
            "deterministic": bool(getattr(config.gaze2hoi.exp, "deterministic", False)),
            "object_bps_feature_mode": object_bps_feature_mode,
            "gaze_alignment_method": str(
                getattr(config.gaze2hoi.model, "gaze_alignment_method", "direction")
            ),
            "gaze_alignment_temporal_scope": str(
                getattr(config.gaze2hoi.model, "gaze_alignment_temporal_scope", "all")
            ),
            "gaze_ray_distance_sigma": float(
                getattr(config.gaze2hoi.model, "gaze_ray_distance_sigma", 0.05)
            ),
            "gaze_ray_distance_sigma_method1": float(
                getattr(config.gaze2hoi.model, "gaze_ray_distance_sigma_method1", 0.05)
            ),
            "gaze_ray_distance_sigma_method2": float(
                getattr(config.gaze2hoi.model, "gaze_ray_distance_sigma_method2", 0.16)
            ),
            "object_feature_type": object_feature_type,
            "output_representation": output_representation,
            "use_point_token_output": use_point_token_output,
            "predict_object_pose": predict_object_pose,
            "object_pose_is_relative": bool(
                use_point_token_output and predict_object_pose
            ),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "max_iterations": max_iterations,
            "amp_enabled": use_amp,
            "grad_clip_norm": grad_clip_norm,
            "warmup_iterations": warmup_iterations,
            "min_lr_ratio": min_lr_ratio,
        }
        if resumable:
            checkpoint.update(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict(),
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                    },
                }
            )
        if motion_normalizer is not None:
            checkpoint["motion_normalization"] = motion_normalizer.state_dict()
        if pointnet is not None:
            checkpoint["pointnet_model"] = (
                pointnet.state_dict()
                if resumable
                else pointnet_ema.model_state_dict()
            )
            if resumable:
                checkpoint["pointnet_ema"] = pointnet_ema.state_dict()
        if relative_gaze_mlp is not None:
            checkpoint["relative_gaze_mlp"] = (
                relative_gaze_mlp.state_dict()
                if resumable
                else relative_gaze_ema.model_state_dict()
            )
            if resumable:
                checkpoint["relative_gaze_ema"] = (
                    relative_gaze_ema.state_dict()
                )
        temporary_path = f"{save_path}.tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, save_path)

    def prune_iteration_checkpoints():
        keep_count = int(
            getattr(config.gaze2hoi.exp, "keep_last_checkpoints", 3)
        )
        if keep_count <= 0:
            return
        iteration_paths = sorted(
            (
                osp.join(model_folder, filename)
                for filename in os.listdir(model_folder)
                if filename.startswith("iteration_") and filename.endswith(".pth")
            ),
            key=lambda path: int(osp.basename(path)[len("iteration_") : -len(".pth")]),
        )
        for stale_path in iteration_paths[:-keep_count]:
            os.remove(stale_path)

    epoch = start_epoch
    last_saved_iteration = -1
    with tqdm.tqdm(
        total=max_iterations,
        initial=global_step,
        unit="iter",
        dynamic_ncols=True,
    ) as pbar:
        while global_step < max_iterations:
            gaze2hoi.train()
            if pointnet is not None:
                pointnet.train()
            if relative_gaze_mlp is not None:
                relative_gaze_mlp.train()
            loss_meter = AverageMeter()
            loss_simple_meter = AverageMeter()
            loss_obj_trans_meter = AverageMeter()
            loss_obj_rot_meter = AverageMeter()

            for item in dataloader:
                if global_step >= max_iterations:
                    break
                raw_object_names = item.get("object_name", item.get("obj_name"))
                if raw_object_names is None:
                    object_names = None
                elif isinstance(raw_object_names, (list, tuple)):
                    object_names = [
                        str(name) if name is not None else None
                        for name in raw_object_names
                    ]
                else:
                    object_names = [str(raw_object_names)] * item["x_obj"].shape[0]
                obj_pc_top_idx = None
                batch_cuda = move_batch_to_cuda(item, cuda_keys)
                x_lhand = batch_cuda["x_lhand"]
                x_rhand = batch_cuda["x_rhand"]
                x_obj = batch_cuda["x_obj"]
                obj_pc_org = batch_cuda["obj_pc"]
                normalized_obj_pc = batch_cuda["normalized_obj_pc"]
    
                obj_scale = batch_cuda["obj_scale"]
                obj_cent = batch_cuda["obj_cent"]
                valid_mask_lhand = batch_cuda["valid_mask_lhand"]
                valid_mask_rhand = batch_cuda["valid_mask_rhand"]
                valid_mask_obj = batch_cuda["valid_mask_obj"]
                ldist_map = batch_cuda["ldist_map"]
                rdist_map = batch_cuda["rdist_map"]
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

                bs = x_obj.shape[0]
                if use_pointnet_object_feature:
                    obj_feat = pointnet(normalized_obj_pc)
                else:
                    with torch.no_grad():
                        obj_feat = compute_bps_feature_from_mesh_cache_for_gaze2hoi(
                            normalized_obj_pc,
                            bps_basis,
                            object_names=object_names,
                            mesh_cache=mesh_bps_cache,
                            mesh_bps_correspondence_cache=mesh_bps_correspondence_cache,
                            part_label_map=part_label_map,
                            bbox_margin=float(getattr(config.dataset, "mesh_part_bbox_margin", 0.03)),
                            feature_mode=object_bps_feature_mode,
                        )
                gaze_condition_x_obj = (
                    repeat_initial_object_pose_for_gaze_condition(x_obj)
                )
                if relative_gaze_mlp is None:
                    with torch.no_grad():
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
                        contact_feat = gaze_score
                else:
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
                        relative_gaze_mlp=relative_gaze_mlp,
                    )
                    gaze_score = apply_null_gaze_condition(config, gaze_score)
                    contact_feat = gaze_score

                obj_feat_final = proc_obj_feat_final_train(
                    contact_feat,
                    obj_scale,
                    obj_cent,
                    obj_feat,
                    use_obj_scale=use_obj_scale,
                    use_obj_centroid=use_obj_centroid,
                )
                if use_point_token_output:
                    if predict_object_pose:
                        target_lhand, target_rhand, target_obj = (
                            build_hybrid_motion_targets_for_gaze2hoi(
                                x_lhand,
                                x_rhand,
                                x_obj,
                                obj_pc_org,
                                lhand_layer,
                                rhand_layer,
                                data_config.name,
                                hand_sample_indices,
                                obj_pc_top_idx=obj_pc_top_idx,
                                include_hand_object_dirvec=include_hand_object_dirvec,
                                canonicalize_hand_targets=canonicalize_hand_point_targets,
                            )
                        )
                    else:
                        target_lhand, target_rhand, target_obj = build_point_token_motion_targets_for_gaze2hoi(
                            x_lhand,
                            x_rhand,
                            x_obj,
                            obj_pc_org,
                            lhand_layer,
                            rhand_layer,
                            data_config.name,
                            hand_sample_indices,
                            obj_pc_top_idx=obj_pc_top_idx,
                            include_hand_object_dirvec=include_hand_object_dirvec,
                            canonicalize_hand_targets=canonicalize_hand_point_targets,
                            canonicalize_object_targets=canonicalize_object_point_targets,
                        )
                else:
                    target_lhand = x_lhand
                    target_rhand = x_rhand
                    target_obj = x_obj
                if motion_normalizer is not None:
                    target_lhand, target_rhand, target_obj = motion_normalizer.normalize(
                        target_lhand, target_rhand, target_obj
                    )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_lhand, pred_rhand, pred_obj, losses_dict = diffusion(
                        gaze2hoi,
                        target_lhand,
                        target_rhand,
                        target_obj,
                        obj_feat_final,
                        get_losses=True,
                        valid_mask_lhand=valid_mask_lhand,
                        valid_mask_rhand=valid_mask_rhand,
                        valid_mask_obj=valid_mask_obj,
                        motion_normalizer=motion_normalizer,
                    )
                    simple_loss = losses_dict["simple_loss"]
                    object_translation_loss = losses_dict["object_translation_loss"]
                    object_rotation_loss = losses_dict["object_rotation_loss"]

                    losses = (
                        lambda_simple * simple_loss
                        + lambda_object_translation * object_translation_loss
                        + lambda_object_rotation * object_rotation_loss
                    )

                optimizer.zero_grad()
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=grad_clip_norm,
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = (
                    not use_amp or scaler.get_scale() >= scale_before_step
                )
                if not optimizer_updated:
                    pbar.set_description(
                        f"{model_name} | AMP overflow; optimizer update skipped"
                    )
                    continue
                scheduler.step()
                ema.update(gaze2hoi)
                if pointnet_ema is not None:
                    pointnet_ema.update(pointnet)
                if relative_gaze_ema is not None:
                    relative_gaze_ema.update(relative_gaze_mlp)
                global_step += 1
                pbar.update(1)
                loss_meter.update(losses.item(), bs)
                loss_simple_meter.update(simple_loss.item(), bs)
                loss_obj_trans_meter.update(object_translation_loss.item(), bs)
                loss_obj_rot_meter.update(object_rotation_loss.item(), bs)
                cur_loss = float(losses.item())
                current_lr = float(optimizer.param_groups[0]["lr"])
                pbar.set_description(
                    f"{model_name} | iter {global_step}/{max_iterations} "
                    f"| loss {cur_loss:.4f} | lr {current_lr:.2e}"
                )

                should_log = (
                    global_step % log_iteration_freq == 0
                    or global_step == max_iterations
                )
                if should_log:
                    log_payload = {
                        "iteration": global_step,
                        "loss": cur_loss,
                        "simple_loss": float(simple_loss.item()),
                        "object_translation_loss": float(
                            object_translation_loss.item()
                        ),
                        "object_rotation_loss": float(
                            object_rotation_loss.item()
                        ),
                        "learning_rate": current_lr,
                        "grad_norm": float(grad_norm.item()),
                    }
                    wandb.log(log_payload, step=global_step)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_line_with_time = (
                        f"[{timestamp}] [{git_branch}] "
                        f"Iteration {global_step}/{max_iterations} | "
                        f"Train loss: {cur_loss:.4f} | "
                        f"LR: {current_lr:.8f} | "
                        f"Grad norm: {float(grad_norm.item()):.4f}"
                    )
                    with open(log_path, "a") as f:
                        f.write(log_line_with_time + "\n")

                should_save_latest = (
                    latest_checkpoint_freq > 0
                    and global_step % latest_checkpoint_freq == 0
                )
                should_save_milestone = (
                    checkpoint_freq > 0
                    and global_step % checkpoint_freq == 0
                )
                if should_save_latest or global_step == max_iterations:
                    save_checkpoint(
                        epoch,
                        latest_model_path,
                        cur_loss,
                        resumable=True,
                    )
                    last_saved_iteration = global_step
                if should_save_milestone or global_step == max_iterations:
                    periodic_path = osp.join(
                        model_folder,
                        f"iteration_{global_step:07d}.pth",
                    )
                    save_checkpoint(
                        epoch,
                        periodic_path,
                        cur_loss,
                        resumable=False,
                    )
                    prune_iteration_checkpoints()
                    last_saved_iteration = global_step

            epoch += 1

    if last_saved_iteration != global_step:
        save_checkpoint(
            epoch - 1,
            latest_model_path,
            cur_loss,
            resumable=True,
        )
    wandb.finish()


if __name__ == "__main__":
    main()
