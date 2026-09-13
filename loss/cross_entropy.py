# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss


@MODELS.register_module()
class CELoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        class_weight=None,
        activated=True,
        ignore_label=255,
        reduction="mean",
        label_smoothing=0.0,
        **kwargs,
    ):
        super(CELoss, self).__init__()

        self.activated = activated
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing

    def forward(self, ce_input, ce_label, weight=None, avg_factor=None):
        ce_input = ce_input.float()
        ce_label = ce_label.long()

        if self.label_smoothing <= 0:
            if not self.activated:
                ce_loss = F.cross_entropy(
                    ce_input,
                    ce_label,
                    weight=self.class_weight.to(ce_input),
                    ignore_index=self.ignore_label,
                    reduction="none",
                )
            else:

                ce_loss = F.nll_loss(
                    torch.log(ce_input),
                    ce_label,
                    weight=self.class_weight.to(ce_input),
                    ignore_index=self.ignore_label,
                    reduction="none",
                )
        else:
            # label smoothing
            if not self.activated:
                ce_loss = F.cross_entropy(
                    ce_input,
                    ce_label,
                    weight=self.class_weight.to(ce_input),
                    ignore_index=self.ignore_label,
                    reduction="none",
                    label_smoothing=self.label_smoothing,
                )

            else:
                num_classes = ce_input.shape[1]  # 18
                log_prob = torch.log(ce_input.clamp(min=1e-8))
                target = F.one_hot(
                    ce_label.clamp(min=0), num_classes=num_classes
                ).float()
                target = target.permute(0, 4, 1, 2, 3)

                target = target * (1.0 - self.label_smoothing) + (
                    1.0 - target
                ) * self.label_smoothing / (num_classes - 1)
                ce_loss = -(target * log_prob).sum(dim=1)
                ignore_mask = ce_label == self.ignore_label
                ce_loss = ce_loss.masked_fill(ignore_mask, 0)

                if self.class_weight is not None:
                    cls_weight = self.class_weight.to(ce_input)  # (18)
                    gt_weight = cls_weight[ce_label.clamp(0, num_classes - 1)]
                    gt_weight = gt_weight.masked_fill(ignore_mask, 0)
                    ce_loss = ce_loss * gt_weight

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        ce_loss = (
            weight_reduce_loss(
                ce_loss, weight=weight, reduction=self.reduction, avg_factor=avg_factor
            )
            * self.loss_weight
        )
        return ce_loss


@MODELS.register_module()
class LabelAwareCELoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        class_weight=None,
        activated=True,
        ignore_label=255,
        reduction="mean",
        label_smoothing=0.0,
        label_aware_smoothing=False,
        class_counts=None,
        alpha_head=0.2,
        alpha_tail=0.0,
        las_form="linear",
        empty_label=17,
        smooth_empty=False,
        **kwargs,
    ):
        super().__init__()

        self.activated = activated
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.class_weight = class_weight
        self.label_smoothing = label_smoothing
        self.label_aware_smoothing = label_aware_smoothing
        self.empty_label = empty_label
        self.smooth_empty = smooth_empty

        if label_aware_smoothing:
            assert class_counts is not None
            class_counts = torch.as_tensor(class_counts, dtype=torch.float64)  # (17)
            self.register_buffer(
                "class_smoothing",
                self.build_class_smoothing(
                    class_counts, alpha_head, alpha_tail, las_form, empty_label
                ),
            )
        else:
            self.class_smoothing = None

    @staticmethod
    def build_class_smoothing(
        class_counts, alpha_head, alpha_tail, las_form, empty_label
    ):
        """Compute per-class smoothing coefficients from voxel counts."""
        num_classes = len(class_counts)

        if num_classes == empty_label:
            class_counts = torch.cat([class_counts, class_counts.new_zeros(1)])

        semantic_counts = class_counts[:empty_label]
        n_max = semantic_counts.max()  # 2.84e8
        n_min = semantic_counts.min()  # 301900
        ratio = (semantic_counts - n_min) / (n_max - n_min).clamp_min(1.0)

        if las_form == "linear":
            alpha = alpha_tail + (alpha_head - alpha_tail) * ratio
        elif las_form == "concave":
            alpha = alpha_tail + (alpha_head - alpha_tail) * torch.sin(
                torch.pi * ratio / 2
            )
        elif las_form == "convex":
            alpha = alpha_head + (alpha_head - alpha_tail) * torch.sin(
                3 * torch.pi / 2 + torch.pi * ratio / 2
            )
        else:
            raise ValueError(f"Unsupported LAS form: {las_form}")

        alpha = torch.cat([alpha, alpha.new_zeros(1)])
        return alpha.float()

    def forward(self, ce_input, ce_label, weight=None, avg_factor=None):
        ce_input = ce_input.float()  # (B, C, X, Y, Z)
        ce_label = ce_label.long()  # (B, X, Y, Z)
        num_classes = ce_input.shape[1]
        cls_weight = (
            self.class_weight.to(ce_input) if self.class_weight is not None else None
        )

        if not self.label_aware_smoothing and self.label_smoothing <= 0:
            if not self.activated:
                ce_loss = F.cross_entropy(
                    ce_input,
                    ce_label,
                    weight=cls_weight,
                    ignore_index=self.ignore_label,
                    reduction="none",
                )
            else:
                ce_loss = F.nll_loss(
                    torch.log(ce_input.clamp_min(1e-8)),
                    ce_label,
                    weight=cls_weight,
                    ignore_index=self.ignore_label,
                    reduction="none",
                )

        else:
            log_prob = (
                torch.log(ce_input.clamp_min(1e-8))
                if self.activated
                else F.log_softmax(ce_input, dim=1)
            )

            ignore_mask = ce_label == self.ignore_label
            valid_label = ce_label.clamp(0, num_classes - 1)

            target = F.one_hot(valid_label, num_classes=num_classes).float()
            target = target.permute(0, 4, 1, 2, 3)

            if self.label_aware_smoothing:

                eps = self.class_smoothing.to(ce_input)[valid_label].unsqueeze(1)
            else:
                eps = ce_input.new_full(
                    (ce_label.shape[0], 1, *ce_label.shape[1:]), self.label_smoothing
                )

            if self.smooth_empty:

                target = target * (1.0 - eps) + (1.0 - target) * eps / (num_classes - 1)

            else:

                semantic_mask = ce_input.new_ones(num_classes)  # (18)
                semantic_mask[self.empty_label] = 0
                other_mask = semantic_mask.view(1, num_classes, 1, 1, 1) * (
                    1.0 - target
                )
                num_other = other_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

                target = target * (1.0 - eps) + other_mask * eps / num_other

                empty_gt = (valid_label == self.empty_label).unsqueeze(1)
                empty_target = F.one_hot(valid_label, num_classes=num_classes).float()
                empty_target = empty_target.permute(0, 4, 1, 2, 3)
                target = torch.where(empty_gt, empty_target, target)

            ce_loss = -(target * log_prob).sum(dim=1)
            ce_loss = ce_loss.masked_fill(ignore_mask, 0)

            if cls_weight is not None:
                gt_weight = cls_weight[valid_label].masked_fill(ignore_mask, 0)
                ce_loss = ce_loss * gt_weight

        if weight is not None:
            weight = weight.float()

        return (
            weight_reduce_loss(
                ce_loss, weight=weight, reduction=self.reduction, avg_factor=avg_factor
            )
            * self.loss_weight
        )


@MODELS.register_module()
class FocalCELoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        class_weight=None,
        activated=True,
        ignore_label=255,
        gamma=2.0,
        reduction="mean",
        **kwargs,
    ):
        super(FocalCELoss, self).__init__()

        self.activated = activated
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.class_weight = class_weight
        self.gamma = gamma

    def forward(self, ce_input, ce_label, weight=None, avg_factor=None):
        ce_input = ce_input.float()
        ce_label = ce_label.long()

        if not self.activated:
            ce_loss = F.cross_entropy(
                ce_input, ce_label, ignore_index=self.ignore_label, reduction="none"
            )
        else:
            ce_loss = F.nll_loss(
                torch.log(ce_input),
                ce_label,
                ignore_index=self.ignore_label,
                reduction="none",
            )

        pt = torch.exp(-ce_loss)
        if self.class_weight is not None:
            class_weight = self.class_weight.to(ce_input)  # (18)
            alpha_t = class_weight[ce_label]
        else:
            alpha_t = 1.0
        loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = (
            weight_reduce_loss(
                loss, weight=weight, reduction=self.reduction, avg_factor=avg_factor
            )
            * self.loss_weight
        )

        return loss
