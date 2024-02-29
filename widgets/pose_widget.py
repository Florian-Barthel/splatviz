# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import imgui
import torch

from gui_utils import imgui_utils
from viz_utils.dict import EasyDict


# ----------------------------------------------------------------------------


class PoseWidget:
    def __init__(self, viz):
        self.viz = viz
        self.pose = EasyDict(yaw=0, pitch=0, anim=False, speed=0.25)
        self.pose_def = EasyDict(self.pose)

        self.lookat_point_choice = 0
        self.lookat_point_option = ["auto", "ffhq", "shapenet", "afhq", "manual"]
        self.lookat_point_labels = ["Auto Detect", "FFHQ Default", "Shapenet Default", "AFHQ Default", "Manual"]
        self.lookat_point = (0.0, 0.0, 0.0)

    def drag(self, dx, dy):
        viz = self.viz
        self.pose.yaw -= dx / viz.font_size * 3e-2
        self.pose.pitch = np.clip(self.pose.pitch - dy / viz.font_size * 3e-2, -np.pi/2, np.pi/2)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
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

        viz.args.yaw = self.pose.yaw
        viz.args.pitch = self.pose.pitch
        viz.args.lookat_point = torch.tensor(self.lookat_point, device="cuda")


# ----------------------------------------------------------------------------
