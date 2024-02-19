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
from gui_utils import imgui_utils


# ----------------------------------------------------------------------------


class EditWidget:
    def __init__(self, viz):
        self.viz = viz

        # Add a button to create slider that can be used in the code
        self.text = """start = 0
end = -1
gaussian._xyz = gaussian._xyz[start:end, ...]
gaussian._rotation = gaussian._rotation[start:end, ...]
gaussian._scaling = gaussian._scaling[start:end, ...]
gaussian._opacity = gaussian._opacity[start:end, ...]
gaussian._features_dc = gaussian._features_dc[start:end, ...]
gaussian._features_rest = gaussian._features_rest[start:end, ...]
bg_color = [1, 1, 1]
"""

        self.x = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            # changed, self.text = imgui.input_text("Variable", self.text, 2000, width=viz.pane_w, height=10 + viz.font_size * (self.text.count("\n") + 2))
            _changed, self.x = imgui.slider_float("x", self.x, -10, 10)

            _changed, self.text = imgui.input_text_multiline("", self.text, 2000, width=viz.pane_w, height=10 + viz.font_size * (self.text.count("\n") + 2))
        viz.args.edit_text = self.text
        viz.args.x = self.x
