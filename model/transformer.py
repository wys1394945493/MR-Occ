import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmengine.utils import deprecated_api_warning
from mmcv.cnn import Scale
from mmengine.model import bias_init_with_prob

from mmdet3d.registry import MODELS


from ops import MSMV_CUDA
from .utils.misc import DUMP, safe_sigmoid
from .utils.checkpoint import checkpoint as cp
from .core import encode_points, decode_points
from .utils.sampling import sampling_4d
from .mixture_of_experts import SparseMoEBlock


@MODELS.register_module()
class MROccTransformer(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_frames=8,
        num_views=6,
        num_points=4,
        num_layers=6,
        num_levels=4,
        num_classes=10,
        num_groups=4,
        num_refines=[1, 2, 4, 8, 16, 32],
        calibration_stage=False,
        use_ego=True,
        pc_range=[],
        init_cfg=None,
    ):
        assert init_cfg is None, (
            "To prevent abnormal initialization "
            "behavior, init_cfg is not allowed to be set"
        )
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines

        self.decoder = MROccDecoder(
            embed_dims,
            num_frames,
            num_views,
            num_points,
            num_layers,
            num_levels,
            num_classes,
            num_refines,
            num_groups,
            use_ego,
            calibration_stage=calibration_stage,
            pc_range=pc_range,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(
        self, query_feat, query_points, mlvl_feats, data, img_metas, dn_attn_mask=None
    ):
        """
        Args:
            query_feat: (B, N_query, C)
            query_points: (B, N_query, 1, 3)
            temp_memory: (B, Mem, C)
            temp_reference_points: (B, Mem, 3)
            mlvl_feats: List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
        Returns:
            cls_scores: List[(B, N_query, n_refine_1, n_cls), (B, N_query, n_refine_2, n_cls), ...]
            refine_pts: List[(B, N_query, n_refine_1, 3), (B, N_query, n_refine_2, 3), ...]
        """
        query_feats, cls_scores, refine_pts = self.decoder(
            query_feat, query_points, mlvl_feats, data, img_metas, dn_attn_mask
        )

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return query_feats, cls_scores, refine_pts


class MROccDecoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_frames=1,
        num_views=6,
        num_points=4,
        num_layers=6,
        num_levels=4,
        num_classes=10,
        num_refines=16,
        num_groups=4,
        use_ego=True,
        calibration_stage=False,
        pc_range=[],
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.use_ego = use_ego

        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                MROccDecoderLayer(
                    embed_dims,
                    num_frames,
                    num_views,
                    num_points,
                    num_levels,
                    num_classes,
                    num_groups,
                    num_refines[i],
                    last_refines[i],
                    pc_range=pc_range,
                    ffn_type="moe",
                    calibration_stage=calibration_stage,
                )
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(
        self, query_feat, query_points, mlvl_feats, data, img_metas, dn_attn_mask=None
    ):
        """
        Args:
            query_feat: (B, N_query, C)
            query_points: (B, N_query, n_refine, 3)
            temp_memory: (B, Mem, C)
            temp_reference_points: (B, Mem, 3)
            mlvl_feats: List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
        Returns:
            cls_scores: List[(B, N_query, n_refine_1, n_cls), (B, N_query, n_refine_2, n_cls), ...]
            refine_pts: List[(B, N_query, n_refine_1, 3), (B, N_query, n_refine_2, 3), ...]
        """
        cls_scores_list, refine_sqs_list = [], []
        query_feat_list = []

        # organize projections matrix and copy to CUDA
        occ2img = data["ego2img"] if self.use_ego else data["lidar2img"]
        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            (
                B,
                TN,
                GC,
                H,
                W,
            ) = feat.shape
            N, T, G, C = (
                self.num_views,
                self.num_frames,
                self.num_groups,
                GC // self.num_groups,
            )  # 6 1 4 64
            assert T * N == TN
            # (B, N, C=256, H2, W2) --> (B, T, N_view, G, C, fH, fW)
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:
                # (B, T, N_view, G, C, fH, fW) --> (B, T, G, N_view, fH, fW, C)
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
                # (B*T*G, N_view, fH, fW, C)
                feat = feat.reshape(B * T * G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                # (B, T, N_view, G, C, fH, fW) --> (B, T, G, C, N_view, fH, fW)
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
                # (B*T*G, C, N_view, fH, fW)
                feat = feat.reshape(B * T * G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()

            query_feat, cls_score, refine_sqs = decoder_layer(
                query_feat,
                query_points,
                mlvl_feats,
                occ2img,
                img_metas,
                dn_attn_mask,
            )

            query_points = refine_sqs[..., :3]
            cls_scores_list.append(cls_score)
            refine_sqs_list.append(refine_sqs)
            query_feat_list.append(query_feat)

        return query_feat_list, cls_scores_list, refine_sqs_list


class MROccDecoderLayer(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_frames=1,
        num_views=6,
        num_points=4,
        num_levels=4,
        num_classes=10,
        num_groups=4,
        num_refines=16,
        last_refines=16,
        num_cls_fcs=2,
        num_reg_fcs=2,
        pc_range=[],
        ffn_type="ffn",
        calibration_stage=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = MROccSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range
        )
        self.sampling = MROccSampling(
            embed_dims,
            num_frames=num_frames,
            num_views=num_views,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range,
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames,
        )

        if ffn_type == "ffn":
            self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)
        elif ffn_type == "moe":
            self.ffn = SparseMoEBlock(
                embed_dims,
                mlp_ratio=8,
                num_experts=4,
                num_experts_per_tok=2,
                pretraining_tp=1,
                n_shared_experts=None,
            )
        else:
            raise ValueError(f"Unknown ffn type: {ffn_type}")

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        # Parameters used by the second-stage classifier rebalancing.
        base_classifier = self.cls_branch[-1]
        self.cls_scale = nn.Parameter(
            torch.ones(self.num_classes), requires_grad=calibration_stage
        )
        self.cls_delta_weight = nn.Parameter(
            torch.zeros_like(base_classifier.weight), requires_grad=calibration_stage
        )
        self.cls_delta_bias = nn.Parameter(
            torch.zeros_like(base_classifier.bias), requires_grad=calibration_stage
        )

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 13 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.output_dim = 13
        self.scale = Scale([1.0] * self.output_dim)
        self.register_buffer(
            "unit_xyz", torch.tensor([3.0, 3.0, 3.0], dtype=torch.float)
        )

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_sqs(self, points_proposal, points_delta):
        """
        Args:
            points_proposal: (B, N_query, N_refine, 3)
            points_delta: (B, N_query, N_refine', 13)
        Returns:
            refine_sqs: (B, N_query, N_refine‘, 13)
        """
        B, Q = points_delta.shape[:2]
        # (B, N_query, N_refine‘ * 3) --> (B, N_query, N_refine’, 13)
        points_delta = points_delta.reshape(B, Q, self.num_refines, 13)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        delta_xyz = (2 * safe_sigmoid(points_delta[..., :3]) - 1.0) * self.unit_xyz[
            None, None, None
        ]
        new_points = points_proposal + delta_xyz

        xyz = encode_points(new_points, self.pc_range)
        xyz = torch.clamp(xyz, min=1e-6, max=1 - 1e-6)

        scale = points_delta[..., 3:6]
        rot = torch.nn.functional.normalize(points_delta[..., 6:10], p=2, dim=-1)
        feat = points_delta[..., 10:]
        refine_sqs = torch.cat([xyz, scale, rot, feat], dim=-1)
        return refine_sqs

    def forward(
        self,
        query_feat,
        query_points,
        mlvl_feats,
        occ2img,
        img_metas,
        dn_attn_mask=None,
    ):
        """
        Args:
            query_feat: (B, N_query, C)
            query_points: (B, N_query, N_refine, 3)
            temp_memory: (B, Mem, C)
            temp_reference_points: (B, Mem, 3)
            mlvl_feats: List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
            occ2img: (B, N=T*N_view, 4, 4)
        Returns:
            query_feat: (B, N_query, C)
            cls_score:  (B, N_query, N_refine, n_cls)
            refine_pt:  (B, N_query, N_refine, 3)
        """
        # (B, N_query, N_refine*3) --> (B, N_query, C)
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        sampled_feat = self.sampling(
            query_feat,
            query_pos,
            query_points,
            mlvl_feats,
            occ2img,
            img_metas,
        )  # (B, N_query, n_group, T*n_points, C)
        query_feat = self.norm1(
            self.mixing(sampled_feat, query_feat, query_pos)
        )  # (B, N_query, C)
        query_feat = self.norm2(
            self.self_attn(query_feat, query_pos, query_points, dn_attn_mask)
        )  # (B, N_query, C)
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_feat = self.cls_branch[:-1](query_feat)
        base_classifier = self.cls_branch[-1]
        cls_score = F.linear(
            cls_feat,
            base_classifier.weight + self.cls_delta_weight,
            base_classifier.bias + self.cls_delta_bias,
        )
        cls_score = cls_score * self.cls_scale
        reg_offset = self.reg_branch(query_feat)
        reg_offset = reg_offset.reshape(B, Q, self.num_refines, -1)
        reg_offset = self.scale(reg_offset)
        cls_score = cls_score.unsqueeze(dim=2).repeat(1, 1, self.num_refines, 1)
        refine_sqs = self.refine_sqs(query_points, reg_offset)

        if DUMP.enabled:
            pass

        return query_feat, cls_score, refine_sqs


class MROccSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""

    def __init__(
        self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None
    ):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.num_heads = num_heads
        self.attention = MultiheadAttention(
            embed_dims, num_heads, dropout, batch_first=True
        )
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_feat, query_pos, reference_points, dn_attn_mask=None):
        """
        Args:
            query_feat: (B, Q, C)
            query_pos:  (B, Q, C)
            reference_points: (B, Q, n_refine, 3)

        Return:
            query_feat: (B, Q, C)
        """
        reference_points = reference_points.mean(dim=2)
        temp_key = temp_value = query_feat
        temp_reference_points = reference_points
        temp_pos = query_pos

        dist = self.calc_points_dists(reference_points, temp_reference_points)
        tau = self.gen_tau(query_feat)
        tau = tau.permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)
        if dn_attn_mask is not None:
            dn_bias = dn_attn_mask.to(attn_mask.device).float()
            dn_bias = dn_bias.masked_fill(dn_attn_mask, float("-inf"))
            dn_bias = dn_bias.unsqueeze(0).expand_as(attn_mask)
            attn_mask = attn_mask + dn_bias

        return self.attention(
            query_feat,
            temp_key,
            temp_value,
            identity=None,
            query_pos=query_pos,
            key_pos=temp_pos,
            attn_mask=attn_mask,
            key_padding_mask=None,
        )

    def forward(self, query_feat, query_pos, query_points, dn_attn_mask=None):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward,
                query_feat,
                query_pos,
                query_points,
                dn_attn_mask,
                use_reentrant=False,
            )
        else:
            return self.inner_forward(query_feat, query_pos, query_points, dn_attn_mask)

    @torch.no_grad()
    def calc_points_dists(self, points, temp_points):
        """
        Args:
            points: (B, Q, 3)
            temp_points: (B, K, 3)
        Returns:
            -dist: (B, Q, K)
        """
        points = decode_points(points, self.pc_range)  # (B, Q, 3)
        temp_points = decode_points(temp_points, self.pc_range)  # (B, K, 3)
        # (B, Q, 1, 3) - (B, 1, K, 3)  --> (B, Q, K)
        dist = torch.norm(points.unsqueeze(-2) - temp_points.unsqueeze(-3), dim=-1)
        return -dist


class MROccSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""

    def __init__(
        self,
        embed_dims=256,
        num_frames=4,
        num_views=6,
        num_groups=4,
        num_points=8,
        num_levels=4,
        pc_range=[],
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(
        self, query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas
    ):
        """
        Args:
            query_feat: (B, N_query, C)
            query_pos: (B, N_query, C)
            query_points: (B, N_query, N_refine, 3)
            mlvl_feats: List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
            occ2img: (B, N=T*N_view, 4, 4)
        Returns:
            sampled_feats: (B, N_query, n_group, T*n_points, C)
        """
        query_feat = query_feat + query_pos
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]["img_shape"][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        sampling_points = query_center + sampling_offset
        sampling_points = sampling_points.view(
            B, Q, self.num_groups, self.num_points, 3
        )
        sampling_points = sampling_points.reshape(
            B, Q, 1, self.num_groups, self.num_points, 3
        )
        sampling_points = sampling_points.expand(
            B, Q, self.num_frames, self.num_groups, self.num_points, 3
        )
        # scale weights
        scale_weights = self.scale_weights(query_feat).view(
            B, Q, self.num_groups, 1, self.num_points, self.num_levels
        )
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(
            B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels
        )

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h,
            image_w,  # 256 704
            self.num_views,  # 6
        )

        return sampled_feats

    def forward(
        self, query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas
    ):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward,
                query_feat,
                query_pos,
                query_points,
                mlvl_feats,
                occ2img,
                img_metas,
                use_reentrant=False,
            )
        else:
            return self.inner_forward(
                query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas
            )


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""

    def __init__(
        self,
        in_dim,
        in_points,
        n_groups=1,
        query_dim=None,
        out_dim=None,
        out_points=None,
    ):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(
            self.query_dim, self.n_groups * self.total_parameters
        )
        self.out_proj = nn.Linear(
            self.eff_out_dim * self.out_points * self.n_groups, self.query_dim
        )
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query, query_pos):
        """
        Args:
            x: (B, Q, G, P=N_frames*N_points, C)
            query: (B, Q, C)
            query_pos: (B, Q, C)
        Returns:
            out: (B, Q, C)
        """
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        """generate mixing parameters"""
        # (B, N_query, C)  --> (B, Q, G*(64*64+32*32))
        params = self.parameter_generator(query + query_pos)
        params = params.reshape(B * Q, G, -1)
        out = x.reshape(B * Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B * Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B * Q, G, self.out_points, self.in_points)

        """adaptive channel mixing"""
        # (B*Q, G, P, C_in) @ (B*Q, G, C_in, C_out) --> (B*Q, G, P, C_out)
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        """adaptive point mixing"""
        # (B*Q, G, out_P, in_P) @ (B*Q, G, in_P, C_out) --> (B*Q, G, out_P, C_out)
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        """linear transfomation to query dim"""
        out = out.reshape(B, Q, -1)  # (B*Q, G, out_P, C_out) --> (B, Q, G*out_P*C_out)
        out = self.out_proj(out)  # (B, Q, G*out_P*C_out) --> (B, Q, C)
        out = query + out

        return out

    def forward(self, x, query, query_pos):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, query_pos, use_reentrant=False)
        else:
            return self.inner_forward(x, query, query_pos)


class CustomMultiheadAttention(MultiheadAttention):
    @deprecated_api_warning({"residual": "identity"}, cls_name="MultiheadAttention")
    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is"
                        f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))
