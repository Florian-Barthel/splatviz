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
from imgui_bundle import imgui, implot
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from . import glfw_window
from . import text_utils
from splatviz_utils.gui_utils import style


# ----------------------------------------------------------------------------


class ImguiWindow(glfw_window.GlfwWindow):
    def __init__(self, *, title="ImguiWindow", font=None, code_font=None, font_sizes=range(16, 64), **glfw_kwargs):
        if font is None:
            font = text_utils.get_default_font()
            code_font = text_utils.get_default_font()
        font_sizes = {int(size) for size in font_sizes}
        super().__init__(title=title, **glfw_kwargs)

        # Init fields.
        self._imgui_context = None
        self._implot_context = None
        self._imgui_renderer = None
        self._imgui_fonts = None
        self._imgui_fonts_code = None
        self._cur_font_size = max(font_sizes)

        # Delete leftover imgui.ini to avoid unexpected behavior.
        if os.path.isfile("imgui.ini"):
            os.remove("imgui.ini")

        # Init ImGui.
        self._imgui_context = imgui.create_context()
        self._implot_context = implot.create_context()
        self._imgui_renderer = _GlfwRenderer(self._glfw_window)
        self._attach_glfw_callbacks()
        imgui.get_io().ini_saving_rate = 0  # Disable creating imgui.ini at runtime.
        imgui.get_io().mouse_drag_threshold = 0  # Improve behavior with imgui_utils.drag_custom().
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(font, size + 3) for size in font_sizes}
        self._imgui_fonts_code = {size: imgui.get_io().fonts.add_font_from_file_ttf(code_font, size) for size in font_sizes}
        self._imgui_renderer.refresh_font_texture()

    def close(self):
        self.make_context_current()
        self._imgui_fonts = None
        self._imgui_fonts_code = None
        if self._imgui_renderer is not None:
            self._imgui_renderer.shutdown()
            self._imgui_renderer = None
        if self._imgui_context is not None:
            # imgui.destroy_context(self._imgui_context) # Commented out to avoid creating imgui.ini at the end.
            self._imgui_context = None
        if self._implot_context is not None:
            implot.destroy_context(self._implot_context)
            self._implot_context = None
        super().close()

    def _glfw_key_callback(self, *args):
        super()._glfw_key_callback(*args)
        self._imgui_renderer.keyboard_callback(*args)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.4)

    def set_font_size(self, target):  # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def begin_frame(self):
        # Begin glfw frame.
        super().begin_frame()

        # Process imgui events.
        self._imgui_renderer.mouse_wheel_multiplier = self._cur_font_size / 10
        if self.content_width > 0 and self.content_height > 0:
            self._imgui_renderer.process_inputs()

        # Begin imgui frame.
        imgui.new_frame()
        imgui.push_font(self._imgui_fonts[self._cur_font_size])
        style.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size + 4)

    def end_frame(self):
        imgui.pop_font()
        imgui.render()
        imgui.end_frame()
        self._imgui_renderer.render(imgui.get_draw_data())
        super().end_frame()


# ----------------------------------------------------------------------------
# Wrapper class for GlfwRenderer to fix a mouse wheel bug on Linux.


class _GlfwRenderer(GlfwRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_wheel_multiplier = 1

    def scroll_callback(self, window, x_offset, y_offset):
        self.io.mouse_wheel += y_offset * self.mouse_wheel_multiplier


# ----------------------------------------------------------------------------
