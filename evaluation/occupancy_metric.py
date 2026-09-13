import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable
from mmdet3d.registry import METRICS


@METRICS.register_module()
class OccMetric(BaseMetric):
    def __init__(
        self,
        class_indices,
        empty_label,
        label_str,
        dataset_empty_label=17,
        filter_minmax=True,
        collect_device="cpu",
        prefix=None,
        pklfile_prefix=None,
        submission_prefix=None,
        collect_dir=None,
        **kwargs,
    ):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.results = []
        super().__init__(
            prefix=prefix, collect_device=collect_device, collect_dir=collect_dir
        )

        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.filter_minmax = filter_minmax

    def process(self, data_batch, data_samples):

        preds = data_samples[0]["occ_pred"]
        gt_occ = data_samples[0]["occ_gt"]
        occ_mask = data_samples[0]["occ_mask"].bool()

        b_seen = torch.zeros(self.num_classes + 1)
        b_correct = torch.zeros(self.num_classes + 1)
        b_positive = torch.zeros(self.num_classes + 1)

        for idx in range(preds.size(0)):
            mask = occ_mask[idx]
            outputs = preds[idx][mask]
            targets = gt_occ[idx][mask]
            for i, c in enumerate(self.class_indices):
                b_seen[i] += torch.sum(targets == c).item()  # TP + FN
                b_correct[i] += torch.sum((targets == c) & (outputs == c)).item()  # TP
                b_positive[i] += torch.sum(outputs == c).item()  # TP + FP

            b_seen[-1] += torch.sum(targets != self.empty_label).item()
            b_correct[-1] += torch.sum(
                (targets != self.empty_label) & (outputs != self.empty_label)
            ).item()
            b_positive[-1] += torch.sum(outputs != self.empty_label).item()

        self.results.append(
            {"seen": b_seen, "correct": b_correct, "positive": b_positive}
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

        ret_dict = dict()
        ious = []

        header = ["classes"] + list(self.label_str) + ["miou", "iou"]
        table_columns = [["results"]]

        for i in range(self.num_classes):
            denom = (
                total_seen[i] + total_positive[i] - total_correct[i]
            )  # (TP+FN) + (TP+FP) - TP = TP + FN + FP
            if total_seen[i] == 0:
                cur_iou = np.nan
            else:
                cur_iou = (total_correct[i] / denom).item()  # TP / (TP + FP + FN)

            ious.append(cur_iou)
            table_columns.append([f"{cur_iou:.4f}"])
            ret_dict[self.label_str[i]] = cur_iou * 100

        miou = np.nanmean(ious)
        bin_denom = total_seen[-1] + total_positive[-1] - total_correct[-1]
        iou_bin = (total_correct[-1] / bin_denom).item() if total_seen[-1] > 0 else 0

        table_columns.append([f"{miou:.4f}"])
        table_columns.append([f"{iou_bin:.4f}"])

        table_data = [header]
        table_rows = list(zip(*table_columns))
        table_data += table_rows
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True

        print("\n" + table.table)

        ret_dict["miou"] = miou * 100
        ret_dict["iou"] = iou_bin * 100

        return ret_dict
