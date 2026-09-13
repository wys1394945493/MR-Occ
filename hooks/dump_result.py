import os
import pickle
import shutil
import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS

from model.core import decode_points
from model.utils.misc import safe_sigmoid


class SQSDecoder(nn.Module):
    def __init__(
        self,
        pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
        voxel_size=[0.5, 0.5, 0.5],
        scale_range=[0.01, 3.2],
        u_range=[0.1, 2],
        v_range=[0.1, 2],
    ):
        super().__init__()

        self.u_range = u_range
        self.v_range = v_range
        self.scale_range = scale_range

        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer("pc_range", pc_range)
        self.register_buffer("scene_size", scene_size)
        self.register_buffer("voxel_size", voxel_size)
        self.register_buffer("voxel_num", voxel_num)

    def forward(self, refine_sqs):
        sq_mean = decode_points(refine_sqs[..., 0:3], self.pc_range)
        sq_scales = safe_sigmoid(refine_sqs[..., 3:6])
        sq_scales = (
            self.scale_range[0]
            + (self.scale_range[1] - self.scale_range[0]) * sq_scales
        )
        # rotations: normalized
        rot = refine_sqs[..., 6:10]

        opa = safe_sigmoid(refine_sqs[..., 10:11])
        uv = safe_sigmoid(refine_sqs[..., 11:13])
        u = self.u_range[0] + (self.u_range[1] - self.u_range[0]) * uv[..., :1]
        v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * uv[..., 1:]
        sqs = torch.cat([sq_mean, sq_scales, rot, opa, u, v], dim=-1)
        return sqs


@HOOKS.register_module()
class DumpResultHook(Hook):
    def __init__(
        self,
        save_dir="output/vis",
        save_img=False,
    ):
        super().__init__()
        self.sqs_decoder = SQSDecoder()
        self.save_img = save_img
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.occ_path = os.path.join(save_dir, "occ_pred")
        os.makedirs(self.occ_path, exist_ok=True)
        print(f"Dump results to: {self.save_dir}")

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        results = outputs[0]
        bs = results["occ_pred"].shape[0]

        if "refine_sqs" in results:
            refine_sqs = results["refine_sqs"]
            cls_scores = results["cls_scores"]
            probs = cls_scores.softmax(dim=-1)
            probs = torch.cat([probs, torch.zeros_like(probs[..., :1])], dim=-1)
            pred = probs.argmax(-1).flatten(1, 2).unsqueeze(-1)
            sqs = self.sqs_decoder(refine_sqs).flatten(1, 2)
            sqs = torch.cat([sqs, pred], dim=-1)

        for i in range(bs):
            occ_pred = results["occ_pred"][i].cpu().numpy()
            occ_gt = results["occ_gt"][i].cpu().numpy()
            output = dict(
                occ_pred=occ_pred,
                occ_gt=occ_gt,
            )

            if "refine_sqs" in results:
                output["refine_sqs"] = sqs[i].cpu().numpy()

            scene_token = data_batch["img_metas"][i]["scene_token"]
            token = data_batch["img_metas"][i]["token"]

            scene_dir = os.path.join(self.occ_path, scene_token)
            os.makedirs(scene_dir, exist_ok=True)

            token_dir = os.path.join(scene_dir, token)
            os.makedirs(token_dir, exist_ok=True)

            save_path = os.path.join(token_dir, f"{token}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(output, f)

            if self.save_img:

                img_dir = os.path.join(token_dir, "imgs")
                os.makedirs(img_dir, exist_ok=True)

                filenames = data_batch["img_metas"][i]["filename"]  # list, len=6

                for img_path in filenames:
                    img_name = os.path.basename(img_path)
                    dst_path = os.path.join(img_dir, img_name)
                    shutil.copy2(img_path, dst_path)
