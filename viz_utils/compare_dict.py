import torch
import numpy as np


def equal_dicts(dict1, dict2):
    if dict1 is None or dict2 is None:
        return False

    for key in dict1.keys():
        if isinstance(dict1[key], torch.Tensor):
            if not torch.equal(dict1[key], dict2[key]):
                return False
        elif isinstance(dict1[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        else:
            if key not in dict2.keys():
                return False
            if dict1[key] != dict2[key]:
                return False
    return True
