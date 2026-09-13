import torch
from pathlib import Path
from einops import rearrange
from mmengine.model import BaseModel
from mmdet3d.registry import MODELS
from .utils.augmentation import GpuPhotoMetricDistortion
from .dpt_head import DPTHead
from .mixture_of_experts import SparseMoEBlock


@MODELS.register_module()
class MROcc(BaseModel):
    """Multi-path representation learning for semantic occupancy prediction.

    The image path uses four selected DINOv3 layers, one shared expert bank,
    four layer-specific routers, layer-specific pre-norms and DPTHead. The
    router and experts consume normalized tokens while the residual preserves
    raw DINO features. Dual-path queries, query-specialized expert routing,
    density-aware geometry losses and optional classifier calibration are
    implemented in the head and transformer.
    """

    def __init__(
        self,
        data_aug=None,
        model_url="",
        num_levels=4,
        patch_embed="dinov3_vitl16",
        dim_in=1024,
        out_dim=256,
        backbone_repo="dinov3",
        calibration_stage=False,
        pts_bbox_head=None,
        **kwargs
    ):
        super().__init__()

        backbone_repo = Path(backbone_repo)
        if not backbone_repo.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            backbone_repo = repo_root / backbone_repo
        self.img_backbone = torch.hub.load(
            str(backbone_repo), patch_embed, source="local", weights=model_url
        )
        self.img_backbone.requires_grad_(False)
        self.img_backbone.is_init = True
        self.patch_size = self.img_backbone.patch_size
        self.n_blocks = self.img_backbone.n_blocks

        self.patch_start_idx = self.img_backbone.n_storage_tokens + 1
        self.num_levels = num_levels

        self.intermediate_layer_idx = {
            "dinov3_vits16": [2, 5, 8, 11],
            "dinov3_vitb16": [2, 5, 8, 11],
            "dinov3_vitl16": [4, 11, 17, 23],
            "dinov3_vitg16": [9, 19, 29, 39],
        }[patch_embed]

        self.moe_encoder = SparseMoEBlock(
            dim_in,
            mlp_ratio=2,
            num_experts=4,
            num_experts_per_tok=2,
            pretraining_tp=1,
            n_shared_experts=None,
            num_routers=self.num_levels,
            use_pre_norm=True,
        )
        self.dpt_head = DPTHead(
            dim_in=dim_in,
            patch_size=self.patch_size,
            features=out_dim,
            out_channels=[256, 512, 1024, 1024],
            intermediate_layer_idx=[0, 1, 2, 3],
            feature_only=True,
        )

        self.pts_bbox_head = MODELS.build(pts_bbox_head)

        if calibration_stage:
            self.requires_grad_(False)
            for layer in self.pts_bbox_head.transformer.decoder.decoder_layers:
                layer.cls_scale.requires_grad_(True)
                layer.cls_delta_weight.requires_grad_(True)
                layer.cls_delta_bias.requires_grad_(True)

        self.data_aug = data_aug
        self.color_aug = GpuPhotoMetricDistortion()

    def extract_feat(self, img, img_metas):
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()
        if self.data_aug is not None:
            if (
                "img_color_aug" in self.data_aug
                and self.data_aug["img_color_aug"]
                and self.training
            ):
                img = self.color_aug(img)
            if "img_norm_cfg" in self.data_aug:
                img_norm_cfg = self.data_aug["img_norm_cfg"]
                norm_mean = torch.tensor(img_norm_cfg["mean"], device=img.device)
                norm_std = torch.tensor(img_norm_cfg["std"], device=img.device)
                if img_norm_cfg["to_rgb"]:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB
                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)
            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]["img_shape"] = [img_shape for _ in range(N)]
                img_metas[b]["ori_shape"] = [img_shape for _ in range(N)]

        self.img_backbone.eval()
        with torch.no_grad():
            aggregated_tokens_list = self.img_backbone.get_intermediate_layers(
                img,
                n=self.intermediate_layer_idx,
                reshape=False,
                return_class_token=False,
                return_extra_tokens=False,
                norm=False,
            )

        image_tokens_list = []
        for level_idx, aggregated_tokens in enumerate(aggregated_tokens_list):
            image_tokens = self.moe_encoder(
                aggregated_tokens.contiguous(), router_idx=level_idx
            )
            image_tokens = rearrange(image_tokens, "(b n) l c -> b n l c", b=B, n=N)
            image_tokens_list.append(image_tokens)

        images = img.reshape(B, N, C, H, W)
        mlvl_feats = self.dpt_head(image_tokens_list, images, patch_start_idx=0)
        return mlvl_feats

    def forward(self, mode="loss", **data):
        img_metas = data["img_metas"]
        img = data["img"]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        data["img_feats"] = img_feats
        outs = self.pts_bbox_head(**data)

        voxel_semantics = data["voxel_semantics"]
        mask_camera = data["mask_camera"]
        if mode == "predict":
            occ_pred, occ_logits = self.pts_bbox_head.get_occ(outs)
            outputs = [
                {
                    "occ_pred": occ_pred,
                    "occ_logits": occ_logits,
                    "occ_gt": voxel_semantics,
                    "occ_mask": mask_camera,
                    "refine_sqs": outs["all_refine_sqs"][-1],
                    "cls_scores": outs["all_cls_scores"][-1],
                }
            ]
            return outputs

        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses
