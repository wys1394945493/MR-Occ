import os
import json
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable
from mmdet3d.registry import METRICS


@METRICS.register_module()
class ECEMetricDual(BaseMetric):
    def __init__(
        self,
        class_indices,
        empty_label,
        label_str,
        dataset_empty_label=17,
        filter_minmax=True,
        num_ece_bins=10,
        use_softmax_for_ece=False,
        calibration_save_dir="./calibration_results",
        collect_device="cpu",
        prefix=None,
        pklfile_prefix=None,
        submission_prefix=None,
        collect_dir=None,
        **kwargs,
    ):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super().__init__(
            prefix=prefix, collect_device=collect_device, collect_dir=collect_dir
        )

        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.filter_minmax = filter_minmax
        self.num_ece_bins = num_ece_bins
        self.use_softmax_for_ece = use_softmax_for_ece
        self.calibration_save_dir = calibration_save_dir

    def _update_bins(self, confidence, preds, gt_occ, mask):
        cur_confidence = confidence[mask]  # (32503)
        cur_correct = (preds[mask] == gt_occ[mask]).float()  # (32503)
        bin_ids = torch.clamp(
            (cur_confidence * self.num_ece_bins).long(), 0, self.num_ece_bins - 1
        )  # (32503)

        bin_count = torch.bincount(bin_ids, minlength=self.num_ece_bins)
        bin_correct = torch.bincount(
            bin_ids, weights=cur_correct, minlength=self.num_ece_bins
        )  # (10)
        bin_confidence = torch.bincount(
            bin_ids, weights=cur_confidence, minlength=self.num_ece_bins
        )  # (10)
        return (
            bin_count.cpu().double(),
            bin_correct.cpu().double(),
            bin_confidence.cpu().double(),
        )  # (10)

    def process(self, data_batch, data_samples):
        preds = data_samples[0]["occ_pred"]  # (B, 200, 200, 16)
        logits = data_samples[0]["occ_logits"]  # (B, 200, 200, 16, 18)
        gt_occ = data_samples[0]["occ_gt"]  # (B, 200, 200, 16)
        occ_mask = data_samples[0]["occ_mask"].bool()  # (B, 200, 200, 16)

        if self.use_softmax_for_ece:
            confidence = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
        else:
            confidence = logits.float().max(dim=-1).values

        b_seen = torch.zeros(self.num_classes + 1, dtype=torch.float64)
        b_correct = torch.zeros(self.num_classes + 1, dtype=torch.float64)
        b_positive = torch.zeros(self.num_classes + 1, dtype=torch.float64)

        b_bin_count_nonempty = torch.zeros(self.num_ece_bins, dtype=torch.float64)
        b_bin_correct_nonempty = torch.zeros(self.num_ece_bins, dtype=torch.float64)
        b_bin_confidence_nonempty = torch.zeros(self.num_ece_bins, dtype=torch.float64)

        b_bin_count_all = torch.zeros(self.num_ece_bins, dtype=torch.float64)
        b_bin_correct_all = torch.zeros(self.num_ece_bins, dtype=torch.float64)
        b_bin_confidence_all = torch.zeros(self.num_ece_bins, dtype=torch.float64)

        for idx in range(preds.size(0)):
            mask = occ_mask[idx]
            outputs = preds[idx][mask]
            targets = gt_occ[idx][mask]
            # mIoU
            for i, c in enumerate(self.class_indices):
                b_seen[i] += (targets == c).sum().item()
                b_correct[i] += ((targets == c) & (outputs == c)).sum().item()
                b_positive[i] += (outputs == c).sum().item()
            # IoU
            b_seen[-1] += (targets != self.empty_label).sum().item()
            b_correct[-1] += (
                ((targets != self.empty_label) & (outputs != self.empty_label))
                .sum()
                .item()
            )
            b_positive[-1] += (outputs != self.empty_label).sum().item()
            # ECE(nonempty)
            nonempty_mask = mask & (gt_occ[idx] != self.empty_label)
            count, correct, conf = self._update_bins(
                confidence[idx], preds[idx], gt_occ[idx], nonempty_mask
            )
            b_bin_count_nonempty += count
            b_bin_correct_nonempty += correct
            b_bin_confidence_nonempty += conf
            # ECE(all)
            count, correct, conf = self._update_bins(
                confidence[idx], preds[idx], gt_occ[idx], mask
            )
            b_bin_count_all += count
            b_bin_correct_all += correct
            b_bin_confidence_all += conf

        self.results.append(
            {
                "seen": b_seen,
                "correct": b_correct,
                "positive": b_positive,
                "bin_count_nonempty": b_bin_count_nonempty,
                "bin_correct_nonempty": b_bin_correct_nonempty,
                "bin_confidence_nonempty": b_bin_confidence_nonempty,
                "bin_count_all": b_bin_count_all,
                "bin_correct_all": b_bin_correct_all,
                "bin_confidence_all": b_bin_confidence_all,
            }
        )

    def compute_metrics(self, results: list) -> dict:
        if self.submission_prefix:
            if hasattr(self, "format_results"):
                self.format_results(results)
            return {}

        total_seen = sum(res["seen"] for res in results)
        total_correct = sum(res["correct"] for res in results)
        total_positive = sum(res["positive"] for res in results)

        print(f"Computing metrics over {len(results)} frames")

        ret_dict = {}
        ious = []
        header = ["classes"] + list(self.label_str) + ["miou", "iou"]
        table_columns = [["results"]]

        for i in range(self.num_classes):
            denom = total_seen[i] + total_positive[i] - total_correct[i]
            cur_iou = (
                np.nan
                if total_seen[i] == 0 or denom == 0
                else (total_correct[i] / denom).item()
            )
            ious.append(cur_iou)
            table_columns.append([f"{cur_iou:.4f}"])
            ret_dict[self.label_str[i]] = cur_iou * 100

        miou = np.nanmean(ious)
        bin_denom = total_seen[-1] + total_positive[-1] - total_correct[-1]
        iou_bin = (total_correct[-1] / bin_denom).item() if bin_denom > 0 else 0.0
        table_columns.append([f"{miou:.4f}"])
        table_columns.append([f"{iou_bin:.4f}"])

        table = AsciiTable([header] + list(zip(*table_columns)))
        table.inner_footing_row_border = True
        print("\n" + table.table)

        ret_dict["miou"] = miou * 100
        ret_dict["iou"] = iou_bin * 100

        ece_nonempty = self._evaluate_ece(
            total_bin_count=sum(res["bin_count_nonempty"] for res in results),
            total_bin_correct=sum(res["bin_correct_nonempty"] for res in results),
            total_bin_confidence=sum(res["bin_confidence_nonempty"] for res in results),
            scope="nonempty",
            title="Non-empty Voxels",
        )

        ece_all = self._evaluate_ece(
            total_bin_count=sum(res["bin_count_all"] for res in results),
            total_bin_correct=sum(res["bin_correct_all"] for res in results),
            total_bin_confidence=sum(res["bin_confidence_all"] for res in results),
            scope="all",
            title="All Voxels",
        )

        ret_dict["ece_nonempty"] = ece_nonempty * 100
        ret_dict["ece_all"] = ece_all * 100
        return ret_dict

    def _evaluate_ece(
        self, total_bin_count, total_bin_correct, total_bin_confidence, scope, title
    ):
        valid_bins = total_bin_count > 0
        bin_accuracy = torch.zeros_like(total_bin_count)
        bin_avg_confidence = torch.zeros_like(total_bin_count)

        bin_accuracy[valid_bins] = (
            total_bin_correct[valid_bins] / total_bin_count[valid_bins]
        )
        bin_avg_confidence[valid_bins] = (
            total_bin_confidence[valid_bins] / total_bin_count[valid_bins]
        )

        bin_gap = (bin_accuracy - bin_avg_confidence).abs()
        total_samples = total_bin_count.sum()
        bin_weight = total_bin_count / total_samples
        bin_ece = bin_weight * bin_gap
        ece = bin_ece.sum().item()

        calibration_table = [
            [
                "Bin",
                "Range",
                "Count",
                "Accuracy",
                "Confidence",
                "Gap",
                "ECE Contribution",
            ]
        ]
        calibration_results = {
            "scope": scope,
            "includes_empty": scope == "all",
            "num_bins": self.num_ece_bins,
            "total_samples": int(total_samples.item()),
            "ece": ece,
            "bins": [],
        }

        for i in range(self.num_ece_bins):
            lower = i / self.num_ece_bins
            upper = (i + 1) / self.num_ece_bins
            range_str = (
                f"[{lower:.1f}, {upper:.1f}]"
                if i == self.num_ece_bins - 1
                else f"[{lower:.1f}, {upper:.1f})"
            )

            count = int(total_bin_count[i].item())
            accuracy = bin_accuracy[i].item()
            avg_confidence = bin_avg_confidence[i].item()
            gap = bin_gap[i].item()
            contribution = bin_ece[i].item()

            calibration_table.append(
                [
                    str(i),
                    range_str,
                    f"{count:,}",
                    f"{accuracy:.4f}",
                    f"{avg_confidence:.4f}",
                    f"{gap:.4f}",
                    f"{contribution:.6f}",
                ]
            )
            calibration_results["bins"].append(
                {
                    "bin_id": i,
                    "range": range_str,
                    "count": count,
                    "accuracy": accuracy,
                    "confidence": avg_confidence,
                    "gap": gap,
                    "ece_contribution": contribution,
                }
            )

        print(f"\n===== ECE: {title} =====")
        print(AsciiTable(calibration_table).table)
        print(f"ECE ({scope}): {ece:.6f} ({ece * 100:.4f}%)")

        if self.calibration_save_dir is not None:
            os.makedirs(self.calibration_save_dir, exist_ok=True)
            json_path = os.path.join(
                self.calibration_save_dir, f"calibration_statistics_{scope}.json"
            )
            with open(json_path, "w") as f:
                json.dump(calibration_results, f, indent=4)
            print(f"Calibration statistics saved to: {json_path}")
            self.plot_reliability_diagram(
                bin_accuracy.numpy(),
                bin_avg_confidence.numpy(),
                total_bin_count.numpy(),
                ece,
                scope,
                title,
            )
        return ece

    def plot_reliability_diagram(
        self, bin_accuracy, bin_confidence, bin_count, ece, scope, title
    ):
        valid = bin_count > 0
        bin_edges = np.linspace(0, 1, self.num_ece_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / self.num_ece_bins

        accuracy = np.zeros(self.num_ece_bins)
        confidence = np.zeros(self.num_ece_bins)
        accuracy[valid] = bin_accuracy[valid]
        confidence[valid] = bin_confidence[valid]

        over_gap = np.clip(confidence - accuracy, 0, None)
        under_gap = np.clip(accuracy - confidence, 0, None)
        total_correct = np.sum(bin_accuracy[valid] * bin_count[valid])
        total_count = np.sum(bin_count[valid])
        overall_acc = total_correct / total_count if total_count > 0 else 0.0

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.bar(
            bin_centers,
            accuracy,
            width=bin_width * 0.92,
            color="#8ecae6",
            edgecolor="white",
            linewidth=1.0,
            label="Accuracy",
            zorder=2,
        )
        ax.bar(
            bin_centers,
            over_gap,
            bottom=accuracy,
            width=bin_width * 0.92,
            color="#2f80c1",
            edgecolor="white",
            linewidth=1.0,
            label="Over-confidence Gap",
            zorder=2,
        )
        ax.bar(
            bin_centers,
            under_gap,
            bottom=confidence,
            width=bin_width * 0.92,
            color="#f4a261",
            edgecolor="white",
            linewidth=1.0,
            label="Under-confidence Gap",
            zorder=2,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            "--",
            color="gray",
            linewidth=2,
            label="Perfect Calibration",
            zorder=3,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Confidence", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.35, zorder=0)
        ax.legend(loc="upper left", fontsize=10, frameon=True)

        ax.text(
            0.68,
            0.08,
            f"ACC={overall_acc * 100:.1f}%\nECE={ece * 100:.1f}%",
            transform=ax.transAxes,
            fontsize=13,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        )

        plt.tight_layout()
        save_path = os.path.join(
            self.calibration_save_dir, f"reliability_diagram_{scope}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Reliability diagram saved to: {save_path}")
