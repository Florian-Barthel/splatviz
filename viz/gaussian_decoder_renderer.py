# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import copy
import os
import pickle
import re
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from viz.base_renderer import Renderer
from viz_utils.camera_utils import fov_to_intrinsics
from viz_utils.dict import EasyDict


class GaussianDecoderRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.decoder = None
        self.position_prediction = None
        self.last_z = torch.zeros([1, 512], device=self._device)
        self.last_command = ""
        self.latent_map = torch.randn([1, 512, 10, 10], device=self._device, dtype=torch.float)
        self.reload_model = True
        self._current_ply_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        current_ply_names,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        latent_x=0.0,
        latent_y=0.0,
        render_gan_image=False,
        save_ply_path=None,
        fast_render_mode=False,
        **slider,
    ):
        if not fast_render_mode:
            slider = EasyDict(slider)
            self.load_decoder(ply_file_paths[0])

        # create videos
        if len(video_cams) > 0:
            self.render_video("./_videos", video_cams)

        # create camera
        width = resolution
        height = resolution

        intrinsics = fov_to_intrinsics(fov, device=self._device)[None, :]
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(width, height, fovy=fov_rad, fovx=fov_rad, znear=0.01, zfar=10, extr=cam_params)
        gan_camera_params = torch.concat([cam_params.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        # generate latent vector todo optimize
        if not fast_render_mode or self.last_z is None:
            z = self.create_z(latent_x, latent_y)
        else:
            z = self.last_z

        if not torch.equal(self.last_z, z) or self.reload_model or render_gan_image:
            result = self.position_prediction.get_data(z=z, camera_params=gan_camera_params)
            gaussian_attr = self.decoder(z, gan_camera_params, result.vertices, truncation_psi=1.0)
            self.gaussian_model._xyz = gaussian_attr.xyz
            self.gaussian_model._scaling = gaussian_attr.scale
            self.gaussian_model._rotation = gaussian_attr.rotation
            self.gaussian_model._opacity = gaussian_attr.opacity
            self.gaussian_model._features_dc = gaussian_attr.color.unsqueeze(1)
            self.last_z = z
            self.reload_model = False

        if not fast_render_mode:
            command = re.sub(";+", ";", edit_text.replace("\n", ";"))
            gaussian = copy.deepcopy(self.gaussian_model)  # for editing todo optimize
            exec(command)
        else:
            gaussian = self.gaussian_model

        if save_ply_path is not None:
            os.makedirs(save_ply_path, exist_ok=True)
            save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
            print("Model saved in", save_path)
            gaussian.save_ply(save_path)

        render = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)

        img = render["render"]
        if not fast_render_mode:
            res.stats = torch.stack(
                [
                    img.mean(),
                    img.mean(),
                    img.std(),
                    img.std(),
                    img.norm(float("inf")),
                    img.norm(float("inf")),
                ]
            )
            if render_alpha:
                img = render["alpha"]
            if render_depth:
                img = render["depth"] / render["depth"].max()
            res.mean_xyz = torch.mean(gaussian.get_xyz, dim=0)
            res.std_xyz = torch.std(gaussian.get_xyz)

        if render_gan_image:
            gan_image = torch.nn.functional.interpolate(result.img, size=[img.shape[1], img.shape[2]])[0]
            img = torch.concat([img, gan_image], dim=2)
        # Scale and convert to uint8.
        if img_normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img

        if not fast_render_mode:
            if len(eval_text) > 0:
                res.eval = eval(eval_text)

    def create_z(self, latent_x, latent_y):
        latent_x = torch.tensor(latent_x, device="cuda", dtype=torch.float)
        latent_y = torch.tensor(latent_y, device="cuda", dtype=torch.float)
        position = torch.stack([latent_x, latent_y]).reshape(1, 1, 1, 2)
        # todo: interpolate in w
        z = torch.nn.functional.grid_sample(self.latent_map, position, padding_mode="reflection")
        return z.reshape(1, 512)

    def render_video(self, save_path, video_cams):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=self.gaussian_model, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")

    def load_decoder(self, ply_file_path):
        if ply_file_path != self._current_ply_file_path:
            if ply_file_path.endswith(".pkl"):
                with open(ply_file_path, "rb") as input_file:
                    save_file = pickle.load(input_file)
                    self.decoder = save_file["decoder"]
                    self.position_prediction = save_file["dataloader"]
                    self.reload_model = True
                    self._current_ply_file_path = ply_file_path
