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
from viz_utils.dict import EasyDict


# ----------------------------------------------------------------------------


class CamWidget:
    def __init__(self, viz):
        self.viz = viz
        self.fov = 45
        self.size = 512
        self.radius = 3
        self.up_vector = torch.tensor([0, -1, 0], device="cuda")
        self.pose = EasyDict(yaw=0, pitch=0)
        self.lookat_point = torch.tensor((0.0, 0.0, 0.0), device="cuda")
        self.invert_x = False
        self.invert_y = False

    def drag(self, dx, dy):
        viz = self.viz
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1
        self.pose.yaw += x_dir * dx / viz.font_size * 3e-2
        self.pose.pitch = np.clip(self.pose.pitch + y_dir * dy / viz.font_size * 3e-2, -np.pi / 2, np.pi / 2)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            self.radius += wheel / 2

        if show:
            imgui.push_item_width(200)
            imgui.text("Up Vector")
            imgui.same_line()
            _changed, up_vector_tuple = imgui.input_float3("##up_vector", *self.up_vector, format="%.1f")
            if _changed:
                self.up_vector = torch.tensor(up_vector_tuple, device="cuda")
            imgui.same_line()
            if imgui.button("Set current direction"):
                self.up_vector = get_forward_vector(
                    lookat_position=self.lookat_point,
                    horizontal_mean=self.pose.yaw + np.pi / 2,
                    vertical_mean=self.pose.pitch + np.pi / 2,
                    radius=self.radius,
                )
                self.pose.yaw = 0
                self.pose.pitch = 0

            imgui.text("Look at point")
            imgui.same_line()
            _changed, look_at_point_tuple = imgui.input_float3("##lookat", *self.lookat_point, format="%.1f")
            self.lookat_point = torch.tensor(look_at_point_tuple, device=torch.device("cuda"))
            imgui.same_line()
            if imgui.button("Set to xyz mean") and "mean_xyz" in viz.result.keys():
                self.lookat_point = viz.result.mean_xyz
            imgui.pop_item_width()

            imgui.text("FOV")
            imgui.same_line()
            _changed, self.fov = imgui.slider_float("##fov", self.fov, 1, 180, format="%.1f °")

            imgui.text("Radius")
            imgui.same_line()
            _changed, self.radius = imgui.slider_float("##radius", self.radius, 0, 10, format="%.1f")
            imgui.same_line()
            if imgui.button("Set to xyz stddev") and "std_xyz" in viz.result.keys():
                self.radius = viz.result.std_xyz.item()

            imgui.text("Size")
            imgui.same_line()
            _changed, self.size = imgui.input_int("##size", self.size, 1, 64)

            imgui.text("Pose")
            imgui.same_line()
            yaw = self.pose.yaw
            pitch = self.pose.pitch
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_yaw, new_pitch) = imgui.input_float2(
                    "##pose", yaw, pitch, format="%+.2f", flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
                )
                if changed:
                    self.pose.yaw = new_yaw
                    self.pose.pitch = new_pitch
            imgui.same_line()
            _clicked, dragging, dx, dy = imgui_utils.drag_button("Drag", width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            _, self.invert_x = imgui.checkbox("invert x", self.invert_x)
            _, self.invert_y = imgui.checkbox("invert y", self.invert_y)

        viz.args.yaw = self.pose.yaw
        viz.args.pitch = self.pose.pitch
        viz.args.lookat_point = self.lookat_point
        viz.args.fov = self.fov
        viz.args.radius = self.radius
        viz.args.size = self.size
        viz.args.up_vector = self.up_vector
