# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import torch
import torch.nn
from viz.render_utils import CapturedException
from viz_utils.dict import EasyDict


class Renderer:
    def __init__(self):
        self._current_ply_file_paths = [None] * 16
        self._device = torch.device("cuda")
        self._pinned_bufs = dict()  # {(shape, dtype): torch.Tensor, ...}
        self._is_timing = False
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._net_layers = dict()
        self._last_model_input = None
        self.gaussian_models = []
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")

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
        fov,
        edit_text,
        eval_text,
        size,
        ply_file_path,
        cam_params,
        current_ply_names,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        **slider,
    ):
        raise NotImplementedError
