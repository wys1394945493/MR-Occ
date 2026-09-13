import os
import numpy as np
import torch
import mmcv
from mmdet3d.registry import TRANSFORMS
from .formating import to_tensor


@TRANSFORMS.register_module(name="CustomLoadMultiViewImageFromFiles")
class LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):

        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):  # SurroundOcc
    def __init__(self, num_classes=17, gt_instance=False):
        self.num_classes = num_classes
        self.gt_instance = gt_instance

    def __call__(self, results):
        occ_gt_path = results["occ_gt_path"]
        occ_gt_path = os.path.join(
            occ_gt_path, results["pts_filename"].split("/")[-1] + ".npy"
        )  # surroundocc gt

        label = np.load(occ_gt_path)
        new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
        new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

        new_label = torch.from_numpy(new_label)
        mask_camera = new_label != 0

        if results.get("flip_dx", False):
            new_label = torch.flip(new_label, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get("flip_dy", False):
            new_label = torch.flip(new_label, [1])
            mask_camera = torch.flip(mask_camera, [1])

        if self.gt_instance:
            instance_count = 0
            final_instance_class_ids = []
            final_instances = torch.full_like(
                new_label, 255
            )  # empty space has instance id "255"
            for class_id in range(self.num_classes):
                if (new_label == class_id).sum() == 0:
                    continue
                # treat as semantics
                final_instances[new_label == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)

            results["voxel_instances"] = final_instances
            results["instance_class_ids"] = to_tensor(final_instance_class_ids)

        results["voxel_semantics"] = new_label
        results["mask_camera"] = mask_camera

        return results


@TRANSFORMS.register_module()
class LoadMultiScaleOccupancySurroundOcc(object):  # SurroundOcc
    def __init__(
        self,
        num_classes=17,
        gt_instance=False,
        base_occ_size=(200, 200, 16),
        multi_scale_occ_sizes=((100, 100, 8), (50, 50, 4)),
        empty_label=17,
        class_weights=None,
    ):
        self.num_classes = num_classes
        self.gt_instance = gt_instance
        self.base_occ_size = tuple(base_occ_size)
        self.multi_scale_occ_sizes = [tuple(size) for size in multi_scale_occ_sizes]
        self.empty_label = empty_label

        num_labels = max(self.num_classes, self.empty_label) + 1
        if class_weights is None:
            class_weights = [1.0] * num_labels
        assert (
            len(class_weights) >= num_labels
        ), "class_weights should cover labels 0 ~ empty_label"
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def _get_scale_name(self, out_size):
        assert all(
            base_dim % out_dim == 0
            for base_dim, out_dim in zip(self.base_occ_size, out_size)
        ), f"Cannot name occupancy scale from {self.base_occ_size} to {out_size}"
        scale = [
            base_dim // out_dim
            for base_dim, out_dim in zip(self.base_occ_size, out_size)
        ]
        assert (
            scale[0] == scale[1] == scale[2]
        ), f"Only isotropic downsample scale names are supported, got {scale}"
        return f"{scale[0]}x"

    def _downsample_by_weighted_vote(self, voxel_semantics, out_size):
        in_size = voxel_semantics.shape
        assert len(in_size) == len(out_size) == 3
        assert all(
            in_dim % out_dim == 0 for in_dim, out_dim in zip(in_size, out_size)
        ), f"Cannot merge occupancy from {tuple(in_size)} to {out_size}"

        merge_size = [in_dim // out_dim for in_dim, out_dim in zip(in_size, out_size)]
        num_labels = max(
            int(voxel_semantics.max().item()) + 1, self.class_weights.numel()
        )
        class_weights = self.class_weights.to(voxel_semantics.device)
        if class_weights.numel() < num_labels:
            pad = class_weights.new_ones(num_labels - class_weights.numel())
            class_weights = torch.cat([class_weights, pad], dim=0)

        blocks = voxel_semantics.reshape(
            out_size[0],
            merge_size[0],
            out_size[1],
            merge_size[1],
            out_size[2],
            merge_size[2],
        )
        blocks = blocks.permute(0, 2, 4, 1, 3, 5).reshape(-1, np.prod(merge_size))

        votes = torch.nn.functional.one_hot(blocks, num_classes=num_labels).float()
        non_empty_mask = blocks != self.empty_label
        votes = votes * non_empty_mask.unsqueeze(-1)
        votes = votes * class_weights.view(1, 1, -1)
        merged = votes.sum(dim=1).argmax(dim=1)
        all_empty_mask = ~non_empty_mask.any(dim=1)
        merged[all_empty_mask] = self.empty_label
        return merged.reshape(out_size).to(voxel_semantics.dtype)

    def __call__(self, results):
        occ_gt_path = results["occ_gt_path"]
        occ_gt_path = os.path.join(
            occ_gt_path, results["pts_filename"].split("/")[-1] + ".npy"
        )  # surroundocc gt

        label = np.load(occ_gt_path)
        new_label = np.ones(self.base_occ_size, dtype=np.int64) * self.empty_label
        new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

        new_label = torch.from_numpy(new_label)
        mask_camera = new_label != 0

        if results.get("flip_dx", False):
            new_label = torch.flip(new_label, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if results.get("flip_dy", False):
            new_label = torch.flip(new_label, [1])
            mask_camera = torch.flip(mask_camera, [1])

        if self.gt_instance:
            instance_count = 0
            final_instance_class_ids = []
            final_instances = torch.full_like(
                new_label, 255
            )  # empty space has instance id "255"
            for class_id in range(self.num_classes):
                if (new_label == class_id).sum() == 0:
                    continue
                # treat as semantics
                final_instances[new_label == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)

            results["voxel_instances"] = final_instances
            results["instance_class_ids"] = to_tensor(final_instance_class_ids)

        results["voxel_semantics"] = new_label
        results["mask_camera"] = mask_camera
        for occ_size in self.multi_scale_occ_sizes:
            voxel_semantics = self._downsample_by_weighted_vote(new_label, occ_size)
            scale_name = self._get_scale_name(occ_size)
            results[f"voxel_semantics_{scale_name}"] = voxel_semantics
            results[f"mask_camera_{scale_name}"] = voxel_semantics != 0

        return results


@TRANSFORMS.register_module()
class LoadLidarPoints(object):
    def __init__(self, data_root=""):
        self.data_root = data_root

    def __call__(self, results):
        lidar_path = results["pts_filename"].replace("./data/nuscenes/", self.data_root)
        raw_points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        points = torch.from_numpy(raw_points[:, :3])
        results["points"] = points
        return results
