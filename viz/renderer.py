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
from viz_utils.camera_utils import LookAtPoseSampler
from viz_utils.dict import EasyDict
from viz.render_utils import CapturedException


class Renderer:
    def __init__(self):
        self._current_ply_file_path = None
        self._device = torch.device("cuda")
        self._pinned_bufs = dict()  # {(shape, dtype): torch.Tensor, ...}
        self._is_timing = False
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._net_layers = dict()
        self._last_model_input = None
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32).to("cuda")

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = EasyDict()
        try:
            with torch.no_grad():
                self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        self._end_event.record(torch.cuda.current_stream(self._device))
        if "image" in res:
            res.image = self.to_cpu(res.image).detach().numpy()
        if "stats" in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if "error" in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _render_impl(
        self,
        res,
        yaw,
        pitch,
        lookat_point,
        fov,
        edit_text,
        eval_text,
        size,
        ply_file_path,
        up_vector,
        current_ply_name,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        z_near=0.01,
        z_far=10,
        radius=2.7,
        **slider
    ):
        slider = EasyDict(slider)
        if ply_file_path != self._current_ply_file_path:
            if ply_file_path.endswith(".ply"):
                self.gaussian_model.load_ply(ply_file_path)
            elif ply_file_path.endswith("compression_config.yml"):
                self.gaussian_model = run_single_decompression(Path(ply_file_path).parent.absolute())
            self._current_ply_file_path = ply_file_path
        cam = EasyDict(radius=radius, z_near=z_near, z_far=z_far, fov=fov, pitch=pitch, yaw=yaw, lookat_point=lookat_point)
        width = size
        height = size
        gaussian = copy.deepcopy(self.gaussian_model)
        command = re.sub(';+', ';', edit_text.replace("\n", ";"))
        exec(command)

        if len(video_cams) > 0:
            self.render_video(f"./videos/{current_ply_name}", video_cams)

        extrinsic = LookAtPoseSampler.sample(3.14 / 2 + cam.yaw, 3.14 / 2 + cam.pitch, cam.lookat_point, radius=cam.radius, up_vector=up_vector)[0]
        fov_rad = cam.fov / 360 * 2 * np.pi
        render_cam = CustomCam(width, height, fovy=fov_rad, fovx=fov_rad, znear=cam.z_near, zfar=cam.z_far, extr=extrinsic)
        render = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)
        img = render["render"]
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
        # Scale and convert to uint8.
        if img_normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img
        res.mean_xyz = torch.mean(gaussian.get_xyz, dim=0)
        res.std_xyz = torch.std(gaussian.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def render_video(self, save_path, video_cams):
        os.makedirs(save_path, exist_ok=True)
        filename = f'{save_path}/rotate_{len(os.listdir(save_path))}.mp4'
        video = imageio.get_writer(filename, mode='I', fps=30, codec='libx264', bitrate='16M', quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=self.gaussian_model, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")
