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
import re
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from pathlib import Path
from compression.compression_exp import run_single_decompression
from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from viz.base_renderer import Renderer
from viz_utils.dict import EasyDict


class GaussianRenderer(Renderer):
    def __init__(self):
        super().__init__()

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        size,
        ply_file_paths,
        cam_params,
        current_ply_names,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        **slider,
    ):
        slider = EasyDict(slider)
        width = size
        height = size
        images = []

        for scene_index, ply_file_path in enumerate(ply_file_paths):
            if ply_file_path != self._current_ply_file_paths[scene_index]:
                if scene_index + 1 > len(self.gaussian_models):
                    self.gaussian_models.append(GaussianModel(sh_degree=0, disable_xyz_log_activation=True))
                if ply_file_path.endswith(".ply"):
                    self.gaussian_models[scene_index].load_ply(ply_file_path)
                elif ply_file_path.endswith("compression_config.yml"):
                    self.gaussian_models[scene_index] = run_single_decompression(Path(ply_file_path).parent.absolute())
                self._current_ply_file_paths[scene_index] = ply_file_path

            gaussian = copy.deepcopy(self.gaussian_models[scene_index])
            command = re.sub(";+", ";", edit_text.replace("\n", ";"))
            exec(command)

            if len(video_cams) > 0:
                self.render_video("./_videos", video_cams, gaussian)

            fov_rad = fov / 360 * 2 * np.pi
            render_cam = CustomCam(width, height, fovy=fov_rad, fovx=fov_rad, znear=0.01, zfar=10, extr=cam_params)
            render = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                images.append( render["depth"] / render["depth"].max())
            else:
                images.append(render["render"])

        img = torch.concatenate(images, dim=2)
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

        # Scale and convert to uint8.
        if img_normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img
        res.mean_xyz = torch.mean(gaussian.get_xyz, dim=0)
        res.std_xyz = torch.std(gaussian.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def render_video(self, save_path, video_cams, gaussian):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")
