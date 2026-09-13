import torch
import numpy as np
from mmengine.registry import FUNCTIONS


@FUNCTIONS.register_module()
def custom_collate_fn(instances):
    return_dict = {}
    for k, v in instances[0].items():
        if isinstance(v, np.ndarray):
            return_dict[k] = torch.stack(
                [torch.from_numpy(instance[k]) for instance in instances]
            )
        elif isinstance(v, torch.Tensor):
            vals = [instance[k] for instance in instances]
            shapes = [x.shape for x in vals]
            if all(s == shapes[0] for s in shapes):
                return_dict[k] = torch.stack(vals)
            else:
                return_dict[k] = vals
        elif isinstance(v, (dict, str)):
            return_dict[k] = [instance[k] for instance in instances]
        elif v is None:
            return_dict[k] = [None] * len(instances)
        else:
            raise NotImplementedError
    return return_dict
