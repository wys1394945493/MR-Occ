import torch
import numpy as np
from PIL import Image
import warnings
from copy import deepcopy
import mmengine
from mmengine.dataset import Compose
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomTransformImage:
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training

    def __call__(self, results):
        ida_mats = []
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        if len(results["lidar2img"]) == len(results["img"]):
            for i in range(len(results["img"])):  # offline: 48
                img = Image.fromarray(
                    np.uint8(results["img"][i])
                )
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,  # test: 0.44
                    resize_dims=resize_dims,  # 704 396
                    crop=crop,
                    flip=flip,  # False
                    rotate=rotate,  # 0
                )
                results["img"][i] = np.array(img).astype(np.uint8)
                results["lidar2img"][i] = ida_mat @ results["lidar2img"][i]
                ida_mats.append(ida_mat)

        elif len(results["img"]) == 6:
            for i in range(len(results["img"])):
                img = Image.fromarray(np.uint8(results["img"][i]))
                # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results["img"][i] = np.array(img).astype(np.uint8)
                results["lidar2img"][i] = ida_mat @ results["lidar2img"][i]
                ida_mats.append(ida_mat)

        else:
            raise ValueError()

        results["ego2img"] = []
        for i in range(len(results["lidar2img"])):
            results["ego2img"].append(results["lidar2img"][i] @ results["ego2lidar"])

        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img_shape"] = [img.shape for img in results["img"]]
        results["pad_shape"] = [img.shape for img in results["img"]]

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L48
        """

        def get_rot(h):
            return torch.Tensor(
                [
                    [np.cos(h), np.sin(h)],
                    [-np.sin(h), np.cos(h)],
                ]
            )

        ida_rot = torch.eye(2)  # [[1 0],[0 1]]
        ida_tran = torch.zeros(2)  # [0 0]

        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize  # resize:0.473 -> ida_rot: [[0.4733 0] [0 0.4733]]
        ida_tran -= torch.Tensor(crop[:2])  # [-31 -169]

        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])  # [704 0]
            ida_rot = A.matmul(ida_rot)  # [[-0.4733 0] [0 0.4733]]
            ida_tran = A.matmul(ida_tran) + b  # [31 -169] + [704 0] = [735 -169]

        A = get_rot(rotate / 180 * np.pi)  # [[1 0],[-0 1]]
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2  # [352 128]
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        """
        https://github.com/Megvii-BaseDetection/BEVStereo/blob/master/dataset/nusc_mv_det_dataset.py#L247
        """
        H, W = self.ida_aug_conf["H"], self.ida_aug_conf["W"]
        fH, fW = self.ida_aug_conf["final_dim"]

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.ida_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module(name="CustomMultiScaleFlipAug3D")
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool, optional): Whether apply flip augmentation.
            Defaults to False.
        flip_direction (str | list[str], optional): Flip augmentation
            directions for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool, optional): Whether apply horizontal
            flip augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool, optional): Whether apply vertical flip
            augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(
        self,
        transforms,
        img_scale,
        pts_scale_ratio,
        flip=False,
        flip_direction="horizontal",
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
    ):

        self.transforms = Compose(transforms)

        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = (
            pts_scale_ratio
            if isinstance(pts_scale_ratio, list)
            else [float(pts_scale_ratio)]
        )

        assert mmengine.is_list_of(self.img_scale, tuple)
        assert mmengine.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = (
            flip_direction if isinstance(flip_direction, list) else [flip_direction]
        )
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ["horizontal"]:
            warnings.warn("flip_direction has no effect when flip is set to False")
        if self.flip and not any(
            [
                (t["type"] == "RandomFlip3D" or t["type"] == "RandomFlip")
                for t in transforms
            ]
        ):
            warnings.warn("flip has no effect when RandomFlip is not in transforms")

    def __call__(self, results):
        """Call function to augment common fields in results.
        Args:
            results (dict): Result dict contains the data to augment.
        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = (
            [False, True] if self.flip and self.pcd_horizontal_flip else [False]
        )
        pcd_vertical_flip_aug = (
            [False, True] if self.flip and self.pcd_vertical_flip else [False]
        )
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results["scale"] = scale
                                _results["flip"] = flip
                                _results["pcd_scale_factor"] = pts_scale_ratio
                                _results["flip_direction"] = direction
                                _results["pcd_horizontal_flip"] = pcd_horizontal_flip
                                _results["pcd_vertical_flip"] = pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)

        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(transforms={self.transforms}, "
        repr_str += f"img_scale={self.img_scale}, flip={self.flip}, "
        repr_str += f"pts_scale_ratio={self.pts_scale_ratio}, "
        repr_str += f"flip_direction={self.flip_direction})"
        return repr_str
