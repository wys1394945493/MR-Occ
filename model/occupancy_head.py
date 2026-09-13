import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.ops import knn
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from ops import LocalAggregator
from .core import decode_points
from .utils.misc import safe_sigmoid, get_rotation_matrix
from loss import lovasz_softmax


@MODELS.register_module()
class MROccHead(BaseModule):
    def __init__(
        self,
        num_classes,
        in_channels,
        num_query,
        transformer=None,
        empty_label=17,
        ignore_label=255,
        pc_range=[],
        voxel_size=[],
        scale_range=[0.01, 3.2],
        u_range=[0.1, 2],
        v_range=[0.1, 2],
        nusc_class_frequencies=[],
        manual_class_weight=None,
        noise_scale=0.3,
        alpha=1.0,
        calibration_stage=False,
        score_thres=None,
        loss_occ=None,
        loss_pts=None,
        train_cfg=dict(),
        test_cfg=dict(),
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.empty_label = empty_label
        self.transformer = MODELS.build(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.score_thres = score_thres

        self.scale_range = scale_range
        self.u_range = u_range
        self.v_range = v_range
        self.ignore_label = ignore_label

        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()

        self.aggregator = LocalAggregator(
            scale_multiplier=3,
            H=voxel_num[0],
            W=voxel_num[1],
            D=voxel_num[2],
            pc_min=pc_range[:3],
            grid_size=voxel_size[0],
        )
        self.register_buffer("pc_range", pc_range)
        self.register_buffer("scene_size", scene_size)
        self.register_buffer("voxel_size", voxel_size)
        self.register_buffer("voxel_num", voxel_num)
        xyz = self.get_meshgrid(pc_range, voxel_num, voxel_size)
        self.register_buffer("gt_xyz", torch.tensor(xyz))

        self._init_layers()

        if manual_class_weight is not None:
            self.class_weights = torch.tensor(manual_class_weight, dtype=torch.float)
            self.cls_weights = (num_classes + 1) * F.normalize(
                self.class_weights, 1, -1
            )
        else:
            class_freqs = nusc_class_frequencies
            self.cls_weights = torch.from_numpy(
                1 / np.log(np.array(class_freqs[: num_classes + 1]) + 0.001)
            )

        loss_occ["class_weight"] = self.cls_weights
        loss_occ["ignore_label"] = self.ignore_label
        self.loss_occ = MODELS.build(loss_occ)
        self.loss_pts = MODELS.build(loss_pts)

        self.noise_scale = noise_scale
        self.alpha = alpha
        self.calibration_stage = calibration_stage

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    def get_meshgrid(self, ranges, grid, reso):
        xxx = (
            torch.arange(grid[0], dtype=torch.float) * reso[0]
            + 0.5 * reso[0]
            + ranges[0]
        )
        yyy = (
            torch.arange(grid[1], dtype=torch.float) * reso[1]
            + 0.5 * reso[1]
            + ranges[1]
        )
        zzz = (
            torch.arange(grid[2], dtype=torch.float) * reso[2]
            + 0.5 * reso[2]
            + ranges[2]
        )

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1).numpy()
        return xyz

    def _build_queries(self, voxel_semantics, batch_size):
        num_queries = self.num_query
        init_points = self.init_points.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        use_noisy_queries = self.training and not self.calibration_stage

        if not use_noisy_queries:
            query_feat = init_points.new_zeros(batch_size, num_queries, self.embed_dims)
            return init_points, query_feat, init_points, None

        denoising_points = []
        for semantics in voxel_semantics:
            occupied_coords = (semantics != self.empty_label).nonzero()
            gt_points = (
                self.pc_range[:3] + (occupied_coords.float() + 0.5) * self.voxel_size
            ).contiguous()
            selected_indices = torch.randperm(
                gt_points.shape[0], device=gt_points.device
            )[:num_queries]
            selected_points = gt_points[selected_indices]
            noise = torch.empty_like(selected_points).uniform_(-1, 1) * self.noise_scale
            noisy_points = torch.clamp(
                selected_points + noise,
                min=self.pc_range[:3],
                max=self.pc_range[3:],
            )
            denoising_points.append(noisy_points)

        denoising_points = torch.stack(denoising_points)
        denoising_points = (denoising_points - self.pc_range[:3]) / self.scene_size
        reference_points = torch.cat([denoising_points, init_points], dim=1)
        query_feat = init_points.new_zeros(batch_size, 2 * num_queries, self.embed_dims)
        dn_attn_mask = torch.zeros(
            2 * num_queries,
            2 * num_queries,
            dtype=torch.bool,
            device=init_points.device,
        )
        # Standard queries cannot attend to ground-truth-derived noisy queries.
        dn_attn_mask[num_queries:, :num_queries] = True
        return init_points, query_feat, reference_points, dn_attn_mask

    def forward(self, img_metas, **data):
        mlvl_feats = data["img_feats"]
        batch_size = mlvl_feats[0].shape[0]
        init_points, query_feat, reference_points, dn_attn_mask = self._build_queries(
            data["voxel_semantics"], batch_size
        )
        use_noisy_queries = self.training and not self.calibration_stage

        _, cls_scores, refine_sqs = self.transformer(
            query_feat,
            reference_points.unsqueeze(dim=2),
            mlvl_feats,
            data,
            img_metas=img_metas,
            dn_attn_mask=dn_attn_mask,
        )

        pred_occ_list = []
        dn_pred_occ_list = []
        for i, (refine_sq, cls_score) in enumerate(zip(refine_sqs, cls_scores)):
            if not self.training and i < len(refine_sqs) - 1:
                continue

            sq_mean = decode_points(refine_sq[..., 0:3], self.pc_range)
            sq_scales = safe_sigmoid(refine_sq[..., 3:6])
            sq_scales = (
                self.scale_range[0]
                + (self.scale_range[1] - self.scale_range[0]) * sq_scales
            )
            rot = refine_sq[..., 6:10]
            opa = safe_sigmoid(refine_sq[..., 10:11])
            uv = safe_sigmoid(refine_sq[..., 11:13])
            u = self.u_range[0] + (self.u_range[1] - self.u_range[0]) * uv[..., :1]
            v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * uv[..., 1:]
            sqs = torch.cat([sq_mean, sq_scales, rot, opa, u, v], dim=-1)
            if use_noisy_queries:
                dn_cls_score, norm_cls_score = cls_score.split(self.num_query, dim=1)
                dn_sqs, norm_sqs = sqs.split(self.num_query, dim=1)

                occ_pred = self.sq2occ(norm_cls_score, norm_sqs)

                dn_occ_pred = self.sq2occ(dn_cls_score, dn_sqs)
                dn_pred_occ_list.append(dn_occ_pred)

            else:
                occ_pred = self.sq2occ(cls_score, sqs)

            pred_occ_list.append(occ_pred)

        return dict(
            init_points=init_points,
            all_cls_scores=cls_scores,
            all_refine_sqs=refine_sqs,
            all_pred_occ_list=pred_occ_list,
            all_dn_pred_occ_list=dn_pred_occ_list,
        )

    def sq2occ(self, cls_scores, refine_sqs):
        """
        :param cls_scores: (B, N_query, n_refine, n_cls)
        :param refine_sqs: (B, N_query, n_refine, 13)
        :return:
        """
        num_imgs = cls_scores.size(0)
        cls_scores = cls_scores.flatten(1, 2)
        refine_sqs = refine_sqs.flatten(1, 2)

        gs_mean = refine_sqs[..., :3]
        scales = refine_sqs[..., 3:6]
        rot = refine_sqs[..., 6:10]
        origi_opa = refine_sqs[..., 10:11]
        u = refine_sqs[..., 11:12]
        v = refine_sqs[..., 12:13]

        rots = get_rotation_matrix(rot)
        origi_opa = origi_opa.flatten(1, 2)
        u = u.flatten(1, 2)
        v = v.flatten(1, 2)

        opacities = cls_scores.softmax(dim=-1)

        opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        gt_xyz = self.gt_xyz[None, ...].repeat([num_imgs, 1, 1, 1, 1])
        sampled_xyz = gt_xyz.flatten(1, 3).float()

        semantics = []
        for i in range(num_imgs):

            semantic = self.aggregator(
                sampled_xyz[i : (i + 1)],
                gs_mean[i : (i + 1)],
                origi_opa[i : (i + 1)],
                u[i : (i + 1)],
                v[i : (i + 1)],
                opacities[i : (i + 1)],
                scales[i : (i + 1)],
                rots[i : (i + 1)],
            )

            sem = semantic[0][:, :-1] * semantic[1].unsqueeze(-1)
            geo = 1 - semantic[1].unsqueeze(-1)
            geosem = torch.cat([sem, geo], dim=-1)
            geosem = geosem.reshape(
                self.voxel_num[0], self.voxel_num[1], self.voxel_num[2], -1
            )

            semantics.append(geosem)
        occ_pred = torch.stack(semantics, dim=0)
        return occ_pred

    def loss_single(
        self,
        pred_occ,
        refine_sqs,
        voxel_semantics,
        gt_points_list,
        gt_labels_list,
    ):
        """
        Args:
            cls_scores: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx=200, Dy=200, Dz=16)
        """
        voxel_semantics = voxel_semantics.long()
        preds = pred_occ.permute(0, 4, 1, 2, 3).contiguous()
        preds = torch.clamp(preds, 1e-6, 1.0 - 1e-6)  # clamp: 1e-6 1.-1e-6

        num_total_samples = 0
        for i in range(self.num_classes + 1):
            if i == self.ignore_label:
                continue
            num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]

        # CE Loss
        loss_occ = self.loss_occ(  # CE_loss
            preds,
            voxel_semantics,
            avg_factor=num_total_samples,
        )

        # Lovasz-softmax Loss

        loss_voxel_lovasz = lovasz_softmax(
            preds, voxel_semantics, ignore=[self.ignore_label, self.empty_label]
        )

        # CD Loss
        num_imgs = refine_sqs.shape[0]
        refine_pts = refine_sqs[..., :3].reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        loss_pts = self.loss_pts_single(refine_pts, gt_points_list)[0]

        return loss_occ, loss_voxel_lovasz, loss_pts

    # chamfer distance
    def loss_chamfer_distance(self, reference_points_list, gt_points_list):

        (
            gt_paired_pts,
            pred_paired_pts,
            gt_pts_weights,
            gt_paired_idx,
            pred_paired_idx,
        ) = multi_apply(self._get_paired_pts, reference_points_list, gt_points_list)

        gt_pts = torch.cat(gt_points_list)  # (N_gt=N_occ0+N_occ1+..., 3)
        gt_paired_pts = torch.cat(gt_paired_pts)  # (N_gt=N_occ0+N_occ1+..., )
        pred_pts = torch.cat(reference_points_list)  # (N_pred=B*Q, 3)
        pred_paired_pts = torch.cat(pred_paired_pts)  # (N_pred=B*Q, 3)

        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts, gt_paired_pts, avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(
            pred_pts, pred_paired_pts, avg_factor=pred_pts.shape[0]
        )
        return loss_pts

    # density aware chamfer distance
    def loss_density_aware_chamfer_distance(
        self, reference_points_list, gt_points_list
    ):

        (
            gt_paired_pts,
            pred_paired_pts,
            gt_pts_weights,
            gt_paired_idx,
            pred_paired_idx,
        ) = multi_apply(self._get_paired_pts, reference_points_list, gt_points_list)

        num_imgs = len(reference_points_list)

        loss_gt_list = []
        loss_pred_list = []

        for i in range(num_imgs):

            gt_pts = gt_points_list[i]
            pred_pts = reference_points_list[i]

            gt_nn_pts = gt_paired_pts[i]
            pred_nn_pts = pred_paired_pts[i]

            gt_idx = gt_paired_idx[i]  # (39368)
            pred_idx = pred_paired_idx[i]  # (2400)

            # GT -> Pred

            dist_gt = ((gt_pts - gt_nn_pts) ** 2).sum(-1)
            pred_count = torch.bincount(gt_idx, minlength=pred_pts.shape[0])

            weight_gt = (pred_count[gt_idx].float() + 1e-6).pow(-1)
            frac_gt = gt_pts.shape[0] / pred_pts.shape[0]

            weight_gt = weight_gt * frac_gt
            loss_gt = (1.0 - weight_gt * torch.exp(-self.alpha * dist_gt)).mean()

            # Pred -> GT

            dist_pred = ((pred_pts - pred_nn_pts) ** 2).sum(-1)
            gt_count = torch.bincount(pred_idx, minlength=gt_pts.shape[0])
            weight_pred = (gt_count[pred_idx].float() + 1e-6).pow(-1)

            frac_pred = pred_pts.shape[0] / gt_pts.shape[0]
            weight_pred = weight_pred * frac_pred

            loss_pred = (1.0 - weight_pred * torch.exp(-self.alpha * dist_pred)).mean()
            loss_gt_list.append(loss_gt)
            loss_pred_list.append(loss_pred)

        loss_gt = torch.stack(loss_gt_list).mean()
        loss_pred = torch.stack(loss_pred_list).mean()

        loss_pts = 0.5 * (loss_gt + loss_pred)
        return loss_pts

    def loss_pts_single(
        self,
        reference_points,
        gt_points_list,
    ):
        """
        Args:
            reference_points: (B, N_q, 3)
            gt_points_list: List[(N_occ0, 3), (N_occ1, 3), ...]
        """
        num_imgs = reference_points.size(0)
        reference_points = reference_points.reshape(num_imgs, -1, 3).contiguous()
        reference_points_list = [
            reference_points[i] for i in range(num_imgs)
        ]  # List[(Q, 3), (Q, 3), ...]
        loss_pts = self.loss_density_aware_chamfer_distance(
            reference_points_list, gt_points_list
        )

        return (loss_pts,)

    def loss_pts_consis(self, refine_sqs, dn_refine_sqs):
        num_imgs = refine_sqs.shape[0]
        refine_pts = refine_sqs[..., :3].reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)

        dn_refine_pts = dn_refine_sqs[..., :3].reshape(num_imgs, -1, 3)
        dn_refine_pts = decode_points(dn_refine_pts, self.pc_range)

        reference_points_list = [refine_pts[i] for i in range(num_imgs)]
        gt_points_list = [dn_refine_pts[i] for i in range(num_imgs)]

        loss_pts = self.loss_density_aware_chamfer_distance(
            reference_points_list, gt_points_list
        )
        return (loss_pts,)

    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        """
        Args:
            voxel_semantics: (B, Dx=200, Dy=200, Dz=16)
            mask_camera:
            preds_dicts: dict{
                'init_points': (B, N_query, 1, 3),
                'cls_scores': List[(B, N_query, n_cls), (B, N_query, n_cls), ...]
                'refine_sqs': List[(B, N_query, n_refine_1, 13), (B, N_query, n_refine_2, 13), ...]
                'all_pred_occ_list': List[(B, Dx, Dy, Dz, 18), (B, Dx, Dy, Dz, 18), ...]
            }
        """

        if self.calibration_stage:
            losses_occ = []
            for pred_occ in preds_dicts["all_pred_occ_list"]:
                preds = pred_occ.permute(0, 4, 1, 2, 3).contiguous()
                preds = torch.clamp(preds, 1e-6, 1.0 - 1e-6)
                num_total_samples = sum(
                    (voxel_semantics == i).sum() * self.cls_weights[i]
                    for i in range(self.num_classes + 1)
                )
                losses_occ.append(
                    self.loss_occ(
                        preds, voxel_semantics.long(), avg_factor=num_total_samples
                    )
                )
            loss_dict = {"loss_occ": losses_occ[-1]}
            for i, loss_occ in enumerate(losses_occ[:-1]):
                loss_dict[f"d{i}.loss_occ"] = loss_occ
            return loss_dict

        all_pred_occ_list = preds_dicts["all_pred_occ_list"]

        all_dn_pred_occ_list = preds_dicts["all_dn_pred_occ_list"]

        all_dn_refine_sqs, all_refine_sqs = map(
            list,
            zip(
                *[
                    refine_sqs.split(self.num_query, dim=1)
                    for refine_sqs in preds_dicts["all_refine_sqs"]
                ]
            ),
        )

        gt_points_list, gt_labels_list = self.get_sparse_voxels(voxel_semantics)

        num_dec_layers = len(all_pred_occ_list)  # 6
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        voxel_semantics_list = [voxel_semantics for _ in range(num_dec_layers)]

        losses_occ, losses_voxel_lovasz, losses_pts = multi_apply(
            self.loss_single,
            all_pred_occ_list,  # List[(B, Dx, Dy, Dz, 18), (B, Dx, Dy, Dz, 18), ...]
            all_refine_sqs,
            voxel_semantics_list,  # List[(B, Dx=200, Dy=200, Dz=16), (B, Dx=200, Dy=200, Dz=16), ...]
            all_gt_points_list,
            all_gt_labels_list,
        )

        losses_occ_dn, losses_voxel_lovasz_dn, losses_pts_dn = multi_apply(
            self.loss_single,
            all_dn_pred_occ_list,  # List[(B, Dx, Dy, Dz, 18), (B, Dx, Dy, Dz, 18), ...]
            all_dn_refine_sqs,
            voxel_semantics_list,  # List[(B, Dx=200, Dy=200, Dz=16), (B, Dx=200, Dy=200, Dz=16), ...]
            all_gt_points_list,
            all_gt_labels_list,
        )

        losses_pts_consis = multi_apply(
            self.loss_pts_consis, all_refine_sqs, all_dn_refine_sqs
        )[0]

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict["loss_occ"] = losses_occ[-1]
        loss_dict["loss_voxel_lovasz"] = losses_voxel_lovasz[-1]
        loss_dict["loss_pts"] = losses_pts[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_occ_i, loss_voxel_lovasz_i, loss_pts_i in zip(
            losses_occ[:-1], losses_voxel_lovasz[:-1], losses_pts[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_occ"] = loss_occ_i
            loss_dict[f"d{num_dec_layer}.loss_voxel_lovasz"] = loss_voxel_lovasz_i
            loss_dict[f"d{num_dec_layer}.loss_pts"] = loss_pts_i
            num_dec_layer += 1

        # DN and Consis loss from the last decoder layer
        loss_dict["dn_loss_occ"] = losses_occ_dn[-1]
        loss_dict["dn_loss_voxel_lovasz"] = losses_voxel_lovasz_dn[-1]
        loss_dict["dn_loss_pts"] = losses_pts_dn[-1]
        loss_dict["dn_loss_pts_consis"] = losses_pts_consis[-1]
        # DN loss from other decoder layers
        num_dec_layer = 0
        for loss_occ_i, loss_voxel_lovasz_i, loss_pts_i, loss_pts_consis_i in zip(
            losses_occ_dn[:-1],
            losses_voxel_lovasz_dn[:-1],
            losses_pts_dn[:-1],
            losses_pts_consis[:-1],
        ):
            loss_dict[f"dn_d{num_dec_layer}.loss_occ"] = loss_occ_i
            loss_dict[f"dn_d{num_dec_layer}.loss_voxel_lovasz"] = loss_voxel_lovasz_i
            loss_dict[f"dn_d{num_dec_layer}.loss_pts"] = loss_pts_i
            loss_dict[f"dn_d{num_dec_layer}.loss_pts_consis"] = loss_pts_consis_i
            num_dec_layer += 1

        # (4) loss of init_points (CD Loss)
        gt_points_list, gt_labels_list = self.get_sparse_voxels(voxel_semantics)
        init_points = preds_dicts["init_points"]
        init_points = decode_points(init_points, self.pc_range)
        init_loss_pts = self.loss_pts_single(init_points, gt_points_list)[
            0
        ]  # init_loss_pts
        loss_dict["init_loss_pts"] = init_loss_pts
        return loss_dict

    def _get_paired_pts(self, pts, gt_points):
        """

        :param pts: (Q, 3)
        :param gt_points: (N_gt, 3)
        :return:
        """
        gt_paired_idx = knn(1, pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()  # (N_gt, )
        pred_paired_idx = knn(1, gt_points[None, ...], pts[None, ...])
        pred_paired_idx = (
            pred_paired_idx.permute(0, 2, 1).squeeze().long()
        )  # (N_pred, )
        gt_paired_pts = pts[gt_paired_idx]  # (N_gt, 3)
        pred_paired_pts = gt_points[pred_paired_idx]  # (N_pred, 3)

        empty_dist_thr = 0.2
        empty_weights = 5.0
        gt_pts_weights = pts.new_ones(gt_paired_pts.shape[0])  # (N_gt, )
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)  # (N_gt, )
        mask = dist > empty_dist_thr
        gt_pts_weights[mask] = empty_weights

        return (
            gt_paired_pts,
            pred_paired_pts,
            gt_pts_weights,
            gt_paired_idx,
            pred_paired_idx,
        )

    def get_sparse_voxels(self, voxel_semantics):
        """
        Args:
            voxel_semantics: (B, Dx, Dy, Dz)
        Returns:
            gt_points: List[(N_occ0, 3), (N_occ1, 3), ...]
            gt_labels: List[(N_occ0, ), (N_occ1, ), ...]
        """
        coors = self.gt_xyz
        voxel_semantics = voxel_semantics.long()

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(len(voxel_semantics)):
            mask = (voxel_semantics[i] != self.empty_label) & (
                voxel_semantics[i] != self.ignore_label
            )
            gt_points.append(coors[mask])  # (N_occ, 3)
            gt_labels.append(voxel_semantics[i][mask])  # (N_occ, )

        return gt_points, gt_labels

    def get_occ(self, pred_dicts):
        occ_logits = pred_dicts["all_pred_occ_list"][-1]

        if self.score_thres is None:
            occ_res = occ_logits.argmax(-1).int()  # (B, Dx, Dy, Dz)
        else:
            fg_prob = occ_logits[..., :-1]  # (B, Dx, Dy, Dz, 17)
            score, occ_res = fg_prob.max(dim=-1)  # (B, Dx, Dy, Dz)
            score_mask = score < self.score_thres
            occ_res = occ_res.int()
            occ_res[score_mask] = 17

        return occ_res, occ_logits


class MLN(nn.Module):
    """
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    """

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
