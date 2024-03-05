# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""
import math
import torch


class LookAtPoseSampler:
    @staticmethod
    def sample(
            horizontal_mean,
            vertical_mean,
            lookat_position,
            radius,
            up_vector,
            device=torch.device("cuda")
    ):
        camera_origins = get_origin(horizontal_mean, vertical_mean, radius, lookat_position)
        forward_vectors = get_forward_vector(lookat_position, horizontal_mean, vertical_mean, radius, camera_origins=camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins, up_vector=up_vector).to(device)


def get_origin(horizontal_mean, vertical_mean, radius, lookat_position, device=torch.device("cuda")):
    h = torch.tensor(horizontal_mean)
    v = torch.tensor(vertical_mean)
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    camera_origins = torch.zeros(3, device=device)
    camera_origins[0] = radius * torch.sin(v) * torch.cos(math.pi - h)
    camera_origins[2] = radius * torch.sin(v) * torch.sin(math.pi - h)
    camera_origins[1] = radius * torch.cos(v)
    return camera_origins + lookat_position


def get_forward_vector(lookat_position, horizontal_mean, vertical_mean, radius, camera_origins=None):
    if camera_origins is None:
        camera_origins = get_origin(horizontal_mean, vertical_mean, radius, lookat_position)
    return normalize_vecs(lookat_position.to(camera_origins.device) - camera_origins)


def create_cam2world_matrix(forward_vector, origin, up_vector):
    forward_vector = normalize_vecs(forward_vector)
    up_vector = up_vector.float().to(origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)
    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert cam2world.shape[1:] == (4, 4)
    return cam2world


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


