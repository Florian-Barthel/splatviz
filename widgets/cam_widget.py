# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from imgui_bundle import imgui
import torch
import numpy as np

from gui_utils import imgui_utils
from viz_utils.camera_utils import get_forward_vector, create_cam2world_matrix, get_origin, normalize_vecs
from viz_utils.dict import EasyDict


# ----------------------------------------------------------------------------


class CamWidget:
    def __init__(self, viz):
        self.viz = viz
        self.fov = 45
        self.radius = 3
        self.lookat_point = torch.tensor((0.0, 0.0, 0.0), device="cuda")
        self.cam_pos = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        self.up_vector = torch.tensor([0.0, -1.0, 0.0], device="cuda")
        self.forward = torch.tensor([0.0, 0.0, -1.0], device="cuda")

        # controls
        self.pose = EasyDict(yaw=0, pitch=0)
        self.invert_x = False
        self.invert_y = False
        self.move_speed = 0.02
        self.control_modes = ["Orbit", "WASD"]
        self.current_control_mode = 0

    def drag(self, dx, dy):
        viz = self.viz
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1
        self.pose.yaw += x_dir * dx / viz.font_size * 3e-2
        self.pose.pitch = np.clip(self.pose.pitch + y_dir * dy / viz.font_size * 3e-2, -np.pi / 2, np.pi / 2)

    def handle_wasd(self):
        if self.control_modes[self.current_control_mode] == "WASD":
            self.forward = get_forward_vector(
                lookat_position=self.cam_pos,
                horizontal_mean=self.pose.yaw + np.pi / 2,
                vertical_mean=self.pose.pitch + np.pi / 2,
                radius=0.01,
                up_vector=self.up_vector
            )
            self.sideways = torch.cross(self.forward, self.up_vector)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.cam_pos += self.forward * self.move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.cam_pos -= self.sideways * self.move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.cam_pos -= self.forward * self.move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.cam_pos += self.sideways * self.move_speed

        elif self.control_modes[self.current_control_mode] == "Orbit":
            self.cam_pos = get_origin(self.pose.yaw + np.pi / 2, self.pose.pitch + np.pi / 2, self.radius, self.lookat_point, device=torch.device("cuda"), up_vector=self.up_vector)
            self.forward = normalize_vecs(self.lookat_point - self.cam_pos)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.pose.pitch += self.move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.pose.yaw += self.move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.pose.pitch -= self.move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.pose.yaw -= self.move_speed

    def handle_mouse(self):
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= self.viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            if self.control_modes[self.current_control_mode] == "WASD":
                self.cam_pos += self.forward * self.move_speed * wheel
            elif self.control_modes[self.current_control_mode] == "Orbit":
                self.radius -= wheel / 10

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self.handle_mouse()
        self.handle_wasd()

        if show:
            imgui.text("Camera Mode")
            imgui.same_line(viz.label_w)
            _clicked, self.current_control_mode = imgui.combo(
                "##Camera Modes", self.current_control_mode, self.control_modes
            )

            imgui.push_item_width(200)
            imgui.text("Up Vector")
            imgui.same_line(viz.label_w)
            _changed, up_vector_tuple = imgui.input_float3("##up_vector", v=self.up_vector.tolist(), format="%.1f")
            if _changed:
                self.up_vector = torch.tensor(up_vector_tuple, device="cuda")
            imgui.same_line()
            if imgui.button("Set current direction"):
                self.up_vector = self.forward
                self.pose.yaw = 0
                self.pose.pitch = 0

            imgui.text("FOV")
            imgui.same_line(viz.label_w)
            _changed, self.fov = imgui.slider_float("##fov", self.fov, 1, 180, format="%.1f °")

            if self.control_modes[self.current_control_mode] == "Orbit":
                imgui.text("Radius")
                imgui.same_line(viz.label_w)
                _changed, self.radius = imgui.slider_float("##radius", self.radius, 0, 10, format="%.1f")
                imgui.same_line()
                if imgui.button("Set to xyz stddev") and "std_xyz" in viz.result.keys():
                    self.radius = viz.result.std_xyz.item()
                imgui.text("Look at point")
                imgui.same_line(viz.label_w)
                _changed, look_at_point_tuple = imgui.input_float3("##lookat", self.lookat_point.tolist(), format="%.1f")
                self.lookat_point = torch.tensor(look_at_point_tuple, device=torch.device("cuda"))
                imgui.same_line()
                if imgui.button("Set to xyz mean") and "mean_xyz" in viz.result.keys():
                    self.lookat_point = viz.result.mean_xyz
            imgui.pop_item_width()

            imgui.text("Invert X")
            imgui.same_line(viz.label_w)
            _, self.invert_x = imgui.checkbox("##invert_x", self.invert_x)

            imgui.text("Invert Y")
            imgui.same_line(viz.label_w)
            _, self.invert_y = imgui.checkbox("##invert_y", self.invert_y)

        self.cam_params = create_cam2world_matrix(self.forward, self.cam_pos, self.up_vector).to("cuda")[0]

        viz.args.yaw = self.pose.yaw
        viz.args.pitch = self.pose.pitch
        viz.args.fov = self.fov
        viz.args.cam_params = self.cam_params
