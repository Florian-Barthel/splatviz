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
        forward_vector=None,
    ):
        camera_origins = get_origin(horizontal_mean, vertical_mean, radius, lookat_position, up_vector)
        if forward_vector is None:
            forward_vector = get_forward_vector(
                lookat_position, horizontal_mean, vertical_mean, radius, up_vector, camera_origins
            )
        return create_cam2world_matrix(forward_vector, camera_origins, up_vector)


def get_origin(horizontal_mean, vertical_mean, radius, lookat_position, up_vector):
    h = torch.tensor(horizontal_mean)
    v = torch.tensor(vertical_mean)
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    camera_origins = torch.zeros(3)
    camera_origins[0] = radius * torch.sin(v) * torch.cos(math.pi - h)
    camera_origins[2] = radius * torch.sin(v) * torch.sin(math.pi - h)
    camera_origins[1] = radius * torch.cos(v)

    camera_origins = rotate_coordinates(camera_origins, up_vector)

    return camera_origins + lookat_position


def rotate_coordinates(coordinates, vector):
    if torch.equal(vector, torch.zeros_like(vector)):
        return coordinates
    unit_vector = normalize_vecs(vector)

    base_vector = torch.tensor([0.0, -1.0, 0.0], device=coordinates.device)
    theta = torch.arccos(torch.dot(unit_vector, base_vector))  # Angle of rotation
    if theta == 0:
        rotation_matrix = torch.eye(3, device=coordinates.device)
    elif theta == torch.pi:
        rotation_matrix = -1 * torch.eye(3, device=coordinates.device)  #
        rotation_matrix[0, 0] = 1
    else:
        k = torch.cross(base_vector, unit_vector, dim=0)
        k /= torch.linalg.norm(k)
        K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=coordinates.device)
        rotation_matrix = (
            torch.eye(3, device=coordinates.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)
        )

    rotated_coordinates = torch.matmul(rotation_matrix, coordinates[..., None])[..., 0]
    return rotated_coordinates


def get_forward_vector(lookat_position, horizontal_mean, vertical_mean, radius, up_vector, camera_origins=None):
    if camera_origins is None:
        camera_origins = get_origin(horizontal_mean, vertical_mean, radius, lookat_position, up_vector)
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


def fov_to_intrinsics(fov_degrees, imsize=1, device="cpu"):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    fov_rad = fov_degrees * 2 * 3.14159 / 360
    focal_length = float(imsize / (2 * math.tan(fov_rad / 2)))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1.0]], device=device)
    return intrinsics
