import numpy as np
import torch

from splatviz_utils.cam_utils import LookAtPoseSampler, fov_to_intrinsics


def view_conditioning(cam_params, fov, mapping_conditioning, device="cuda"):
    intrinsics = fov_to_intrinsics(fov, device=device)[None, :]
    gan_camera_params = torch.concat([cam_params.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    if mapping_conditioning == "zero":
        mapping_camera_params = torch.zeros_like(gan_camera_params)
    elif mapping_conditioning == "current":
        mapping_camera_params = gan_camera_params
    elif mapping_conditioning == "frontal":
        cam = LookAtPoseSampler.sample(horizontal_mean=-np.pi / 2, vertical_mean=np.pi / 2, up_vector=torch.tensor([0, 1, 0.]), radius=2.7,
                                       lookat_position=torch.tensor([0, 0, 0.2]))
        cam = cam.to("cuda")
        mapping_camera_params = torch.concat([cam.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    else:
        raise NotImplementedError
    return gan_camera_params, mapping_camera_params