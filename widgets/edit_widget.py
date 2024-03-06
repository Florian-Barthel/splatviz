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
from viz_utils.dict import EasyDict


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
self.bg_color[:] = 0
"""

        self.slider_values = EasyDict()
        self.slider_ranges = EasyDict()
        self.var_names = "xyzijklmnuvwabcdefghopqrst"
        self.var_name_index = 1
        self._cur_min_slider = -10
        self._cur_max_slider = 10
        self._cur_val_slider = 0
        self._cur_name_slider = "x"

        self.render_alpha = False
        self.render_depth = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            alpha_changed, self.render_alpha = imgui.checkbox("Render alpha", self.render_alpha)
            depth_changed, self.render_depth = imgui.checkbox("Render depth", self.render_depth)
            if self.render_alpha and alpha_changed:
                self.render_depth = False
            if self.render_depth and depth_changed:
                self.render_alpha = False

            self.render_sliders()
            imgui.new_line()

            dynamic_height = 10 + viz.font_size * (self.text.count("\n") + 2)
            _changed, self.text = imgui.input_text_multiline(
                "##input_text", self.text, width=viz.pane_w, height=dynamic_height
            )
        viz.args.edit_text = self.text
        viz.args.render_alpha = self.render_alpha
        viz.args.render_depth = self.render_depth
        viz.args.update(self.slider_values)

    def render_sliders(self):
        delete_keys = []
        for slider_key in self.slider_values.keys():
            _changed, self.slider_values[slider_key] = imgui.slider_float(
                slider_key,
                self.slider_values[slider_key],
                self.slider_ranges[slider_key][0],
                self.slider_ranges[slider_key][1],
            )
            imgui.same_line()
            if imgui.button("Remove " + slider_key):
                delete_keys.append(slider_key)

        for key in delete_keys:
            del self.slider_values[key]
            del self.slider_ranges[key]

        imgui.push_item_width(70)
        imgui.text("Var name")
        imgui.same_line()
        _changed, self._cur_name_slider = imgui.input_text("##input_name", self._cur_name_slider)

        imgui.same_line()
        imgui.text("min")
        imgui.same_line()
        _changed, self._cur_min_slider = imgui.input_int("##input_min", self._cur_min_slider, 0)

        imgui.same_line()
        imgui.text("val")
        imgui.same_line()
        _changed, self._cur_val_slider = imgui.input_int("##input_val", self._cur_val_slider, 0)

        imgui.same_line()
        imgui.text("max")
        imgui.same_line()
        _changed, self._cur_max_slider = imgui.input_int("##input_max", self._cur_max_slider, 0)
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.button("Add Slider"):
            self.slider_values[self._cur_name_slider] = self._cur_val_slider
            self.slider_ranges[self._cur_name_slider] = [self._cur_min_slider, self._cur_max_slider]
            self._cur_name_slider = self.var_names[self.var_name_index % len(self.var_names)]
            self.var_name_index += 1
