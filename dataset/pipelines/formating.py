import numpy as np
from mmdet3d.registry import TRANSFORMS
from .formatting import to_tensor


@TRANSFORMS.register_module(force=True)
class CustomFormatBundle3D(object):
    def __init__(self, class_names, collect_keys, with_gt=True, with_label=True):
        super(CustomFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label
        self.collect_keys = collect_keys

    def __call__(self, results):
        for key in self.collect_keys + ["voxel_semantics", "mask_camera"]:
            if key not in results:
                continue
            if key in ["timestamp", "img_timestamp"]:
                results[key] = to_tensor(np.array(results[key], dtype=np.float64))
            else:
                results[key] = to_tensor(np.array(results[key], dtype=np.float32))

        if "img" in results:
            if isinstance(results["img"], list):
                imgs = [img.transpose(2, 0, 1) for img in results["img"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results["img"] = to_tensor(imgs)
            else:
                img = np.ascontiguousarray(results["img"].transpose(2, 0, 1))
                results["img"] = to_tensor(img)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_names={self.class_names}, "
        repr_str += f"collect_keys={self.collect_keys}, with_gt={self.with_gt}, with_label={self.with_label})"
        return repr_str


@TRANSFORMS.register_module()
class Collect3D(object):
    def __init__(self, keys, meta_keys):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data["img_metas"] = img_metas
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )
