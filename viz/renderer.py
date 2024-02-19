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
import re

import numpy as np
import torch
import torch.fft
import torch.nn

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
        self._pkl_data = dict()  # {pkl: dict | CapturedException, ...}
        self._networks = dict()  # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs = dict()  # {(shape, dtype): torch.Tensor, ...}
        self._cmaps = dict()  # {name: torch.Tensor, ...}
        self._is_timing = False
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._net_layers = dict()
        self._last_model_input = None
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)

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
        x,
        size,
        ply_file_path,
        up_vector,
        img_normalize=False,
        z_near=0.01,
        z_far=10,
        radius=2.7,
    ):
        if ply_file_path != self._current_ply_file_path:
            self.gaussian_model.load_ply(ply_file_path)
            self._current_ply_file_path = ply_file_path
        cam = EasyDict(radius=radius, z_near=z_near, z_far=z_far, fov=fov, pitch=pitch, yaw=yaw, lookat_point=lookat_point)
        width = size
        height = size
        bg_color = [0, 0, 0]
        gaussian = copy.deepcopy(self.gaussian_model)
        command = re.sub(';+', ';', edit_text.replace("\n", ";"))
        exec(command)

        extrinsic = LookAtPoseSampler.sample(3.14 / 2 + cam.yaw, 3.14 / 2 + cam.pitch, cam.lookat_point, radius=cam.radius, up_vector=up_vector)[0]
        fov_rad = cam.fov / 360 * 2 * np.pi
        render_cam = CustomCam(width, height, fovy=fov_rad, fovx=fov_rad, znear=cam.z_near, zfar=cam.z_far, extr=extrinsic)
        bg_color = torch.tensor(bg_color, dtype=torch.float32).to("cuda")
        render = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=bg_color)
        img = render["render"]
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

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
