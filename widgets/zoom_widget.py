# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import imgui
import torch
import numpy as np

from gui_utils import imgui_utils
from viz_utils.camera_utils import get_forward_vector


# ----------------------------------------------------------------------------


class ZoomWidget:
    def __init__(self, viz):
        self.viz = viz
        self.fov = 45
        self.size = 512
        self.radius = 3
        self.up_vector = torch.tensor([0, -1, 0])

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            self.radius += wheel / 2

        if show:
            imgui.text("Up Vector")
            imgui.same_line()
            _changed, up_vector_tuple = imgui.input_float3("##fov", *self.up_vector, format="%.1f")
            self.up_vector = torch.tensor(up_vector_tuple)
            imgui.same_line()
            if imgui.button("Set current direction"):
                self.up_vector = -get_forward_vector(lookat_position=viz.args.lookat_point, horizontal_mean=viz.args.yaw + np.pi / 2, vertical_mean=viz.args.pitch + np.pi / 2, radius=self.radius)
                self.viz.args.yaw = 0
                self.viz.args.pitch = 0
                self.viz.pose_widget.yaw = 0
                self.viz.pose_widget.pitch = 0

            imgui.text("FOV")
            imgui.same_line()
            _changed, self.fov = imgui.slider_float("##fov", self.fov, 1, 180, format="%.1f °")

            imgui.text("Radius")
            imgui.same_line()
            _changed, self.radius = imgui.slider_float("##radius", self.radius, 0, 10, format="%.1f")

            imgui.text("Size")
            imgui.same_line()
            _changed, self.size = imgui.input_int("##size", self.size, 1, 64)

        viz.args.fov = self.fov
        viz.args.radius = self.radius
        viz.args.size = self.size
        viz.args.up_vector = self.up_vector



# ----------------------------------------------------------------------------
