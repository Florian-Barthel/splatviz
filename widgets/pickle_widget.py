# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import imgui
from gui_utils import imgui_utils


class PickleWidget:
    def __init__(self, viz, root):
        self.viz = viz
        self.root = root
        self.filter = "iteration_30000"
        self.items = self.list_runs_and_pkls()
        self.ply = self.items[0]

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            _changed, self.filter = imgui.input_text("Filter", self.filter)
            if imgui_utils.button('Browse', width=viz.button_w, enabled=True):
                imgui.open_popup('browse_pkls_popup')

            if imgui.begin_popup('browse_pkls_popup'):
                for item in self.list_runs_and_pkls():
                    clicked, _state = imgui.menu_item(item)
                    if clicked:
                        self.ply = item
                imgui.end_popup()

            imgui.same_line()
            imgui.text(self.ply)
        viz.args.ply_file_path = os.path.join(self.root, self.ply)
        viz.args.current_ply_name = self.ply.replace('/', "_").replace('\\', "_")

    def list_runs_and_pkls(self):
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith("cloud.ply"):
                    current_path = root[len(self.root)+1:] + "/" + file
                    if self.filter in current_path:
                        self.items.append(current_path)
        return self.items
