import numpy as np

import torch
import torch.nn as nn

class Gaze2HOI(nn.Module):
    def __init__(self, hand_nfeats=99, obj_nfeats=9, latent_dim=512, 
                ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                activation="gelu", clip_dim=512, obj_dim=1024,
                cond_mask_prob=0.1, use_cond_fc=True, 
                use_obj_scale=True,
                use_obj_centroid=False,
                use_obj_scale_centroid=None,
                use_frame_pos=True, 
                use_gaze_feat = False,
                use_inst_pos=True, 
                obj_global_dim=1024,
                gaze_token_dim=0,
                gaze_token_fusion="add",
                gaze_token_target="object",
                cross_attn_order="hand_object",
                ):
        super().__init__()

        self.cond_mask_prob = cond_mask_prob
        self.use_gaze_feat = use_gaze_feat
        self.nfeats = hand_nfeats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.clip_dim = clip_dim
        self.obj_dim = obj_dim

        self.input_feats_hand = hand_nfeats
        self.input_feats_obj = obj_nfeats

        self.use_cond_fc = use_cond_fc
        # `use_obj_scale_centroid` is retained only for loading legacy configs.
        # New experiments control scale and centroid independently.
        if use_obj_scale_centroid is not None:
            use_obj_scale = bool(use_obj_scale_centroid)
            use_obj_centroid = bool(use_obj_scale_centroid)
        self.use_obj_scale = bool(use_obj_scale)
        self.use_obj_centroid = bool(use_obj_centroid)
        self.use_frame_pos = use_frame_pos
        self.use_inst_pos = use_inst_pos
        self.obj_global_dim = obj_global_dim
        self.gaze_token_dim = int(gaze_token_dim)
        self.gaze_token_fusion = str(gaze_token_fusion).lower()
        if self.gaze_token_fusion not in (
            "add",
            "additive",
            "film",
            "token",
            "separate",
            "separate_token",
            "cross",
            "cross_attn",
            "cross_attention",
            "append",
            "append_tokens",
            "sequence_append",
            "global_tokens",
        ):
            raise ValueError(
                f"Unknown gaze_token_fusion={gaze_token_fusion!r}; "
                "expected 'add', 'film', 'token', 'cross_attn', or 'append'."
            )
        self.gaze_token_target = str(gaze_token_target).lower()
        self.cross_attn_order = str(cross_attn_order).lower()
        if self.gaze_token_target not in ("object", "obj", "all"):
            raise ValueError(
                f"Unknown gaze_token_target={gaze_token_target!r}; expected 'object' or 'all'."
            )
        if self.cross_attn_order not in ("hand_object", "object_hand", "parallel"):
            raise ValueError(
                f"Unknown cross_attn_order={cross_attn_order!r}; expected "
                "'hand_object', 'object_hand', or 'parallel'."
            )
        
        ### Architecture
        self.init_fc_lhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_rhand = InitFC(self.input_feats_hand, self.latent_dim)
        self.init_fc_obj = InitFC(self.input_feats_obj, self.latent_dim)
        self.init_fc_gaze = (
            InitFC(self.gaze_token_dim, self.latent_dim)
            if self.gaze_token_dim > 0
            else None
        )
        self.gaze_film = (
            nn.Linear(self.gaze_token_dim, self.latent_dim * 2)
            if self.gaze_token_dim > 0 and self.gaze_token_fusion == "film"
            else None
        )
        if self.gaze_token_dim > 0 and self._uses_cross_attention():
            self.gaze_cross_attn = nn.MultiheadAttention(
                self.latent_dim,
                self.num_heads,
                dropout=self.dropout,
            )
            self.gaze_cross_attn_dropout = nn.Dropout(self.dropout)
            self.gaze_cross_attn_norm = nn.LayerNorm(self.latent_dim)
            self.hand_object_cross_attn = nn.MultiheadAttention(
                self.latent_dim,
                self.num_heads,
                dropout=self.dropout,
            )
            self.hand_object_cross_attn_dropout = nn.Dropout(self.dropout)
            self.hand_object_cross_attn_norm = nn.LayerNorm(self.latent_dim)
        else:
            self.gaze_cross_attn = None
            self.gaze_cross_attn_dropout = None
            self.gaze_cross_attn_norm = None
            self.hand_object_cross_attn = None
            self.hand_object_cross_attn_dropout = None
            self.hand_object_cross_attn_norm = None

        if not self.use_frame_pos:
            self.sequence_pos_encoder = OrgPositionalEncoding(self.latent_dim, self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        self.sequence_hand_encoder = HandObjectEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer,
            num_layers=self.num_layers
        )
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        obj_metadata_dim = int(self.use_obj_scale) + 3 * int(self.use_obj_centroid)
        self.embed_obj = nn.Linear(
            self.obj_dim + obj_metadata_dim,
            self.latent_dim,
        )

        if use_cond_fc:
            self.out_fc_lhand = CondOutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = CondOutFC(self.input_feats_hand, self.latent_dim)
        else:
            self.out_fc_lhand = OutFC(self.input_feats_hand, self.latent_dim)
            self.out_fc_rhand = OutFC(self.input_feats_hand, self.latent_dim)
        self.out_fc_obj = OutFC(self.input_feats_obj, self.latent_dim)

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask_shape = [bs] + [1] * (cond.dim() - 1)
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(*mask_shape)  # 1-> use null_cond, 0-> use real cond        모두 0.1로 채우고 10% 확률로 안에 값이 0, 1로 채워짐. 
            return cond * (1. - mask)
        else:
            return cond

    def _split_obj_condition(self, obj_feat):
        if isinstance(obj_feat, dict):
            return obj_feat.get("global"), obj_feat.get("gaze")
        if isinstance(obj_feat, (tuple, list)):
            if len(obj_feat) == 2:
                return obj_feat[0], obj_feat[1]
            raise ValueError(f"Expected obj_feat tuple/list of length 2, got {len(obj_feat)}")
        return obj_feat, None

    def _uses_separate_gaze_token(self):
        return self.gaze_token_fusion in ("token", "separate", "separate_token")

    def _uses_cross_attention(self):
        return self.gaze_token_fusion in ("cross", "cross_attn", "cross_attention")

    def _uses_appended_gaze_tokens(self):
        return self.gaze_token_fusion in (
            "append",
            "append_tokens",
            "sequence_append",
            "global_tokens",
        )

    def _apply_gaze_condition(self, token, gaze_feat):
        if self.gaze_token_fusion in ("add", "additive"):
            return token + self.init_fc_gaze(gaze_feat)
        scale, shift = self.gaze_film(gaze_feat).chunk(2, dim=-1)
        scale = scale.permute(1, 0, 2)
        shift = shift.permute(1, 0, 2)
        return token * (1.0 + scale) + shift

    def _embed_gaze_memory(self, gaze_feat):
        if gaze_feat.dim() == 3:
            return self.init_fc_gaze(gaze_feat), 1
        if gaze_feat.dim() == 4:
            batch_size, frame_count, token_count, feat_dim = gaze_feat.shape
            gaze_feat = gaze_feat.reshape(batch_size, frame_count * token_count, feat_dim)
            return self.init_fc_gaze(gaze_feat), token_count
        raise ValueError(
            f"Expected gaze_feat shape (B,T,D) or (B,T,M,D), got {tuple(gaze_feat.shape)}."
        )

    def forward(
        self, x_lhand, x_rhand, 
        x_obj, obj_feat,
        timesteps, 
        force_mask=False,
        valid_mask_lhand=None, 
        valid_mask_rhand=None, 
        valid_mask_obj=None
    ):
        # obj_feat -> [gaze_map, feature w. pointnet, obj s, obj c]
        bs = timesteps.shape[0] 
        obj_feat_global, gaze_feat = self._split_obj_condition(obj_feat)
        emb = self.embed_timestep(timesteps)
        emb += self.embed_obj(self.mask_cond(obj_feat_global, force_mask=force_mask))
        
        x_init_lhand = self.init_fc_lhand(x_lhand)
        x_init_rhand = self.init_fc_rhand(x_rhand)
        x_init_obj = self.init_fc_obj(x_obj)
        x_init_gaze = None
        gaze_memory = None
        appended_gaze_tokens = None

        if gaze_feat is not None:
            if self.init_fc_gaze is None:
                raise ValueError("Received temporal gaze features but gaze_token_dim is 0.")
            if self._uses_appended_gaze_tokens():
                if gaze_feat.dim() != 3 or gaze_feat.shape[0] != x_obj.shape[0]:
                    raise ValueError(
                        "Appended gaze tokens require gaze_feat shape (B,M,D), got "
                        f"{tuple(gaze_feat.shape)}."
                )
                gaze_feat = self.mask_cond(gaze_feat, force_mask=force_mask)
                appended_gaze_tokens = self.init_fc_gaze(gaze_feat)
            elif gaze_feat.shape[:2] != x_obj.shape[:2]:
                raise ValueError(
                    f"Expected gaze_feat shape (B,T,D) or (B,T,M,D) matching x_obj, got "
                    f"{tuple(gaze_feat.shape)} and x_obj {tuple(x_obj.shape)}."
                )
            else:
                gaze_feat = self.mask_cond(gaze_feat, force_mask=force_mask)
                if gaze_feat.dim() == 4 and not self._uses_cross_attention():
                    if self._uses_separate_gaze_token():
                        # The non-CGIA token-fusion ablation keeps gaze
                        # information but has one gaze token per frame.
                        # Collapse the per-frame BPS token axis before the
                        # regular (B,T,D) gaze-token projection.
                        gaze_feat = gaze_feat.mean(dim=2)
                    else:
                        raise ValueError(
                            "4D gaze features (B,T,M,D) require either "
                            "gaze_token_fusion='cross_attn' or the separate "
                            "gaze-token fusion path."
                        )
                if self._uses_separate_gaze_token():
                    x_init_gaze = self.init_fc_gaze(gaze_feat)
                elif self._uses_cross_attention():
                    gaze_memory, gaze_tokens_per_frame = self._embed_gaze_memory(gaze_feat)
                elif self.gaze_token_target == "all":
                    x_init_lhand = self._apply_gaze_condition(x_init_lhand, gaze_feat)
                    x_init_rhand = self._apply_gaze_condition(x_init_rhand, gaze_feat)
                    x_init_obj = self._apply_gaze_condition(x_init_obj, gaze_feat)
                else:
                    x_init_obj = self._apply_gaze_condition(x_init_obj, gaze_feat)

        if x_init_gaze is None:
            x_init = torch.stack((x_init_lhand, x_init_rhand, x_init_obj), dim=1)
            tokens_per_frame = 3
        else:
            x_init = torch.stack(
                (x_init_lhand, x_init_rhand, x_init_obj, x_init_gaze), dim=1
            )
            tokens_per_frame = 4
        x_init = x_init.reshape(-1, bs, self.latent_dim)

        xseq = torch.cat((emb, x_init), dim=0)
        xseq = self.sequence_pos_encoder(xseq, tokens_per_frame=tokens_per_frame)
        xseq = self.sequence_hand_encoder(xseq, tokens_per_frame=tokens_per_frame)
        if appended_gaze_tokens is not None:
            gaze_token_count = appended_gaze_tokens.shape[0]
            gaze_positions = self.sequence_pos_encoder.pe[
                x_obj.shape[1] + 1 : x_obj.shape[1] + 1 + gaze_token_count
            ]
            gaze_role = self.sequence_hand_encoder.pe[
                len(self.sequence_hand_encoder.pe) * 3 // 4 :
                len(self.sequence_hand_encoder.pe) * 3 // 4 + 1
            ]
            appended_gaze_tokens = appended_gaze_tokens + gaze_positions + gaze_role
            appended_gaze_tokens = self.sequence_pos_encoder.dropout(
                appended_gaze_tokens
            )
            appended_gaze_tokens = self.sequence_hand_encoder.dropout(
                appended_gaze_tokens
            )
            xseq = torch.cat((xseq, appended_gaze_tokens), dim=0)
        if gaze_memory is not None:
            if self.gaze_cross_attn is None:
                raise ValueError(
                    "gaze_token_fusion uses cross-attention but cross-attention "
                    "modules were not initialized."
                )
            gaze_frame_count = x_obj.shape[1]
            if gaze_tokens_per_frame == 1:
                gaze_pos = self.sequence_pos_encoder.pe[1 : gaze_frame_count + 1]
            else:
                gaze_pos = self.sequence_pos_encoder.pe[1 : gaze_frame_count + 1]
                gaze_pos = (
                    gaze_pos.reshape(gaze_frame_count, 1, 1, self.latent_dim)
                    .expand(-1, gaze_tokens_per_frame, -1, -1)
                    .reshape(gaze_frame_count * gaze_tokens_per_frame, 1, self.latent_dim)
                )
            gaze_memory = gaze_memory + gaze_pos
            gaze_key_padding_mask = (
                ~valid_mask_obj.to(device=x_obj.device, dtype=torch.bool)
                if valid_mask_obj is not None
                else None
            )
            if gaze_key_padding_mask is not None and gaze_tokens_per_frame > 1:
                gaze_key_padding_mask = (
                    gaze_key_padding_mask.unsqueeze(-1)
                    .expand(-1, -1, gaze_tokens_per_frame)
                    .reshape(gaze_key_padding_mask.shape[0], -1)
                )
            motion_tokens = xseq[1:]
            left_idx = torch.arange(
                0, motion_tokens.shape[0], tokens_per_frame, device=motion_tokens.device
            )
            right_idx = torch.arange(
                1, motion_tokens.shape[0], tokens_per_frame, device=motion_tokens.device
            )
            hand_idx = torch.stack((left_idx, right_idx), dim=1).reshape(-1)
            object_idx = torch.arange(
                2, motion_tokens.shape[0], tokens_per_frame, device=motion_tokens.device
            )
            motion_tokens = motion_tokens.clone()

            if self.cross_attn_order == "hand_object":
                hand_query = motion_tokens.index_select(0, hand_idx)
                gaze_out, _ = self.gaze_cross_attn(
                    hand_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    need_weights=False,
                )
                updated_hands = self.gaze_cross_attn_norm(
                    hand_query + self.gaze_cross_attn_dropout(gaze_out)
                ).to(dtype=motion_tokens.dtype)
                motion_tokens.index_copy_(0, hand_idx, updated_hands)

                object_query = motion_tokens.index_select(0, object_idx)
                hand_padding_mask = None
                if valid_mask_lhand is not None and valid_mask_rhand is not None:
                    hand_padding_mask = ~torch.stack(
                        (
                            valid_mask_lhand.to(dtype=torch.bool),
                            valid_mask_rhand.to(dtype=torch.bool),
                        ),
                        dim=2,
                    ).reshape(bs, -1)
                object_out, _ = self.hand_object_cross_attn(
                    object_query,
                    updated_hands,
                    updated_hands,
                    key_padding_mask=hand_padding_mask,
                    need_weights=False,
                )
                updated_objects = self.hand_object_cross_attn_norm(
                    object_query + self.hand_object_cross_attn_dropout(object_out)
                ).to(dtype=motion_tokens.dtype)
                motion_tokens.index_copy_(0, object_idx, updated_objects)
            elif self.cross_attn_order == "object_hand":
                object_query = motion_tokens.index_select(0, object_idx)
                gaze_out, _ = self.gaze_cross_attn(
                    object_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    need_weights=False,
                )
                updated_objects = self.gaze_cross_attn_norm(
                    object_query + self.gaze_cross_attn_dropout(gaze_out)
                ).to(dtype=motion_tokens.dtype)
                motion_tokens.index_copy_(0, object_idx, updated_objects)

                hand_query = motion_tokens.index_select(0, hand_idx)
                object_padding_mask = (
                    ~valid_mask_obj.to(device=x_obj.device, dtype=torch.bool)
                    if valid_mask_obj is not None
                    else None
                )
                hand_out, _ = self.hand_object_cross_attn(
                    hand_query,
                    updated_objects,
                    updated_objects,
                    key_padding_mask=object_padding_mask,
                    need_weights=False,
                )
                updated_hands = self.hand_object_cross_attn_norm(
                    hand_query + self.hand_object_cross_attn_dropout(hand_out)
                ).to(dtype=motion_tokens.dtype)
                motion_tokens.index_copy_(0, hand_idx, updated_hands)
            else:
                # Both branches read the same gaze memory and neither depends on
                # the other branch's update: Gaze -> {Hand, Object}.
                hand_query = motion_tokens.index_select(0, hand_idx)
                object_query = motion_tokens.index_select(0, object_idx)
                hand_out, _ = self.gaze_cross_attn(
                    hand_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    need_weights=False,
                )
                object_out, _ = self.hand_object_cross_attn(
                    object_query,
                    gaze_memory,
                    gaze_memory,
                    key_padding_mask=gaze_key_padding_mask,
                    need_weights=False,
                )
                updated_hands = self.gaze_cross_attn_norm(
                    hand_query + self.gaze_cross_attn_dropout(hand_out)
                ).to(dtype=motion_tokens.dtype)
                updated_objects = self.hand_object_cross_attn_norm(
                    object_query + self.hand_object_cross_attn_dropout(object_out)
                ).to(dtype=motion_tokens.dtype)
                motion_tokens.index_copy_(0, hand_idx, updated_hands)
                motion_tokens.index_copy_(0, object_idx, updated_objects)
            xseq = torch.cat((xseq[:1], motion_tokens), dim=0)

        if valid_mask_obj is not None:
            bs = x_obj.shape[0]
            emb_mask = torch.ones((bs, 1), device=x_obj.device, dtype=bool)
            frame_masks = [valid_mask_lhand, valid_mask_rhand, valid_mask_obj]
            if tokens_per_frame == 4:
                frame_masks.append(valid_mask_obj)
            aug_mask_no_emb = torch.stack(frame_masks, dim=2)
            aug_mask_no_emb = aug_mask_no_emb.reshape(bs, -1)
            aug_mask = torch.cat([emb_mask, aug_mask_no_emb], dim=1)
            if appended_gaze_tokens is not None:
                gaze_valid = valid_mask_obj.to(dtype=torch.bool).any(
                    dim=1, keepdim=True
                )
                gaze_valid = gaze_valid.expand(-1, appended_gaze_tokens.shape[0])
                aug_mask = torch.cat((aug_mask, gaze_valid), dim=1)
            x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[1:]
        else:
            x_enc = self.seqTransEncoder(xseq, src_key_padding_mask=None)[1:]

        motion_token_count = x_obj.shape[1] * tokens_per_frame
        x_enc_motion = x_enc[:motion_token_count]
        x_enc_lhand = x_enc_motion[::tokens_per_frame]
        x_enc_rhand = x_enc_motion[1::tokens_per_frame]
        x_enc_obj = x_enc_motion[2::tokens_per_frame]

        pred_lhand = self.out_fc_lhand(x_enc_lhand, x_enc_obj)
        pred_rhand = self.out_fc_rhand(x_enc_rhand, x_enc_obj)
        pred_obj = self.out_fc_obj(x_enc_obj)

        return pred_lhand, pred_rhand, pred_obj


class InitFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        return x


class OutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return x
    

class CondOutFC(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.fc = nn.Linear(self.latent_dim*2, self.input_feats)
        
    def forward(self, x, cond):
        x = self.fc(torch.cat([x, cond], dim=2))
        x = x.permute(1, 0, 2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class OrgPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(OrgPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, tokens_per_frame=None):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, tokens_per_frame=None):
        if tokens_per_frame is None:
            tokens_per_frame = 3
        frame_count = (x.shape[0] - 1) // tokens_per_frame
        x[0] = x[0] + self.pe[0]
        frame_pe = self.pe[1 : frame_count + 1]
        for token_offset in range(tokens_per_frame):
            x[1 + token_offset :: tokens_per_frame] = (
                x[1 + token_offset :: tokens_per_frame] + frame_pe
            )
        return self.dropout(x)


class HandObjectEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(HandObjectEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, tokens_per_frame=None):
        if tokens_per_frame is None:
            tokens_per_frame = 3
        if tokens_per_frame == 4:
            role_positions = [
                1,
                len(self.pe) // 4,
                len(self.pe) // 2,
                len(self.pe) * 3 // 4,
            ]
        else:
            role_positions = [
                1,
                len(self.pe) // 3,
                len(self.pe) * 2 // 3,
            ]
        for token_offset, role_position in enumerate(role_positions):
            x[1 + token_offset :: tokens_per_frame] = (
                x[1 + token_offset :: tokens_per_frame]
                + self.pe[role_position : role_position + 1]
            )
        return self.dropout(x)
