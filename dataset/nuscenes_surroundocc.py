import numpy as np
import torch
import os
from torch.utils.data import Dataset

from nuscenes.eval.common.utils import Quaternion

import mmengine
from mmdet3d.registry import DATASETS
from mmdet3d.registry import TRANSFORMS


@DATASETS.register_module()
class NuScenesDatasetSurroundOcc(Dataset):
    def __init__(
        self,
        data_root,
        ann_file,
        occ_gt,
        pipeline,
        modality,
        test_mode,
        load_interval=1,
        **kwargs
    ):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.occ_gt = occ_gt
        self.modality = modality
        self.test_mode = test_mode
        self.load_interval = load_interval

        self.data_infos = self.load_annotations(self.ann_file)

        self.pipeline = []
        for t in pipeline:
            self.pipeline.append(TRANSFORMS.build(t))

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):

        input_dict = self.get_data_info(index)
        for t in self.pipeline:
            input_dict = t(input_dict)
        return input_dict

    def load_annotations(self, ann_file):
        data = mmengine.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]

        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]

        e2g_rotation = Quaternion(info["ego2global_rotation"]).rotation_matrix
        e2g_translation = info["ego2global_translation"]
        l2e_rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        l2e_translation = info["lidar2ego_translation"]
        ego2global = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        lidar2ego = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        lidar2global = ego2global @ lidar2ego  # lidar2global

        ego2lidar = invert_matrix_egopose_numpy(lidar2ego)
        global2ego = invert_matrix_egopose_numpy(ego2global)
        global2lidar = invert_matrix_egopose_numpy(lidar2global)
        input_dict = dict(
            token=info["token"],
            pts_filename=info["lidar_path"],
            scene_name=info["scene_name"],
            scene_token=info["scene_token"],
            timestamp=info["timestamp"] / 1e6,
            ego_pose=lidar2global,
            ego_pose_inv=global2lidar,
            lidar2global=lidar2global,
            global2lidar=global2lidar,
            ego2global=ego2global,
            global2ego=global2ego,
            ego2lidar=ego2lidar,
            gt_boxes=info["gt_boxes"],
            gt_names=info["gt_names"],
        )
        input_dict["occ_gt_path"] = os.path.join(self.occ_gt, "surroundocc", "samples")
        if self.modality["use_camera"]:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            sensor2egos = []
            ego2globals = []
            cam_names = []

            for _, cam_info in info["cams"].items():
                cam_names.append(cam_info["type"])
                img_paths.append(os.path.relpath(cam_info["data_path"]))

                img_timestamps.append(cam_info["timestamp"] / 1e6)

                cam2lidar_r = cam_info["sensor2lidar_rotation"]
                cam2lidar_t = cam_info["sensor2lidar_translation"]
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt

                sensor2ego = lidar2ego @ cam2lidar_rt

                intrinsics.append(intrinsic)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)

            sensor2egos = torch.from_numpy(np.stack(sensor2egos, axis=0)).float()
            ego2globals = torch.from_numpy(np.stack(ego2globals, axis=0)).float()
            intrins = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
            input_dict.update(
                dict(
                    cam_names=cam_names,
                    img_filename=img_paths,
                    img_timestamp=img_timestamps,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    lss_inputs=(sensor2egos, ego2globals, intrins),
                    cam_sweeps={"prev": cam_sweeps_prev, "next": cam_sweeps_next},
                )
            )

        return input_dict

    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]["cam_sweeps"]
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]["cams"])
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]["cam_sweeps"]
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]["cams"])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next


def invert_matrix_egopose_numpy(egopose):
    """Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix
