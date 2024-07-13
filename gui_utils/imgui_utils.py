# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib
from imgui_bundle import imgui
from gui_utils.constants import *


# ----------------------------------------------------------------------------


def set_default_style(color_scheme="dark", spacing=5, indent=20, scrollbar=10):
    s = imgui.get_style()
    s.window_padding = [spacing, spacing]
    s.item_spacing = [spacing, spacing]
    s.item_inner_spacing = [spacing, spacing]
    s.columns_min_spacing = spacing
    s.indent_spacing = indent
    s.scrollbar_size = scrollbar
    s.frame_padding = [3, 3]
    # s.window_border_size    = 1
    # s.child_border_size     = 1
    # s.popup_border_size     = 1
    # s.frame_border_size     = 1
    # s.window_rounding       = 0
    # s.child_rounding        = 0
    # s.popup_rounding        = 1
    s.frame_rounding = 2
    # s.scrollbar_rounding    = 1
    # s.grab_rounding         = 1

    getattr(imgui, f"style_colors_{color_scheme}")(s)

    s.set_color_(COLOR_TEXT,                        imgui.ImVec4(1.00, 1.00, 1.00, 1.00))
    s.set_color_(COLOR_TEXT_DISABLED,               imgui.ImVec4(0.50, 0.50, 0.50, 1.00))
    s.set_color_(COLOR_WINDOW_BACKGROUND,           imgui.ImVec4(0.30, 0.30, 0.30, 1.00))
    s.set_color_(COLOR_CHILD_BACKGROUND,            imgui.ImVec4(0.20, 0.20, 0.20, 1.00))
    s.set_color_(COLOR_POPUP_BACKGROUND,            imgui.ImVec4(0.19, 0.19, 0.19, 0.92))
    s.set_color_(COLOR_BORDER,                      imgui.ImVec4(0.19, 0.19, 0.19, 0.29))
    s.set_color_(COLOR_BORDER_SHADOW,               imgui.ImVec4(0.00, 0.00, 0.00, 0.0))
    s.set_color_(COLOR_FRAME_BACKGROUND,            imgui.ImVec4(0.05, 0.05, 0.05, 0.54))
    s.set_color_(COLOR_FRAME_BACKGROUND_HOVERED,    imgui.ImVec4(0.19, 0.19, 0.19, 0.54))
    s.set_color_(COLOR_FRAME_BACKGROUND_ACTIVE,     imgui.ImVec4(0.20, 0.22, 0.23, 1.00))
    s.set_color_(COLOR_TITLE_BACKGROUND,            imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    s.set_color_(COLOR_TITLE_BACKGROUND_ACTIVE,     imgui.ImVec4(0.06, 0.06, 0.06, 1.00))
    s.set_color_(COLOR_TITLE_BACKGROUND_COLLAPSED,  imgui.ImVec4(0.00, 0.00, 0.00, 1.00))
    s.set_color_(COLOR_MENUBAR_BACKGROUND,          imgui.ImVec4(0.14, 0.14, 0.14, 1.00))
    s.set_color_(COLOR_SCROLLBAR_BACKGROUND,        imgui.ImVec4(0.05, 0.05, 0.05, 0.54))
    s.set_color_(COLOR_SCROLLBAR_GRAB,              imgui.ImVec4(0.34, 0.34, 0.34, 0.54))
    s.set_color_(COLOR_SCROLLBAR_GRAB_HOVERED,      imgui.ImVec4(0.40, 0.40, 0.40, 0.54))
    s.set_color_(COLOR_SCROLLBAR_GRAB_ACTIVE,       imgui.ImVec4(0.56, 0.56, 0.56, 0.54))
    s.set_color_(COLOR_CHECK_MARK,                  imgui.ImVec4(0.33, 0.67, 0.86, 1.00))
    s.set_color_(COLOR_SLIDER_GRAB,                 imgui.ImVec4(0.90, 0.70, 0.00, 0.75))
    s.set_color_(COLOR_SLIDER_GRAB_ACTIVE,          imgui.ImVec4(0.90, 0.70, 0.00, 0.90))
    s.set_color_(COLOR_BUTTON,                      imgui.ImVec4(0.90, 0.70, 0.00, 0.75))
    s.set_color_(COLOR_BUTTON_HOVERED,              imgui.ImVec4(0.90, 0.70, 0.00, 0.90))
    s.set_color_(COLOR_BUTTON_ACTIVE,               imgui.ImVec4(0.90, 0.70, 0.00, 1.00))
    s.set_color_(COLOR_HEADER,                      imgui.ImVec4(0.00, 0.00, 0.00, 0.52))
    s.set_color_(COLOR_HEADER_HOVERED,              imgui.ImVec4(0.00, 0.00, 0.00, 0.36))
    s.set_color_(COLOR_HEADER_ACTIVE,               imgui.ImVec4(0.20, 0.22, 0.23, 0.33))
    s.set_color_(COLOR_SEPARATOR,                   imgui.ImVec4(0.28, 0.28, 0.28, 0.29))
    s.set_color_(COLOR_SEPARATOR_HOVERED,           imgui.ImVec4(0.44, 0.44, 0.44, 0.29))
    s.set_color_(COLOR_SEPARATOR_ACTIVE,            imgui.ImVec4(0.40, 0.44, 0.47, 1.00))
    s.set_color_(COLOR_RESIZE_GRIP,                 imgui.ImVec4(0.28, 0.28, 0.28, 0.29))
    s.set_color_(COLOR_RESIZE_GRIP_HOVERED,         imgui.ImVec4(0.44, 0.44, 0.44, 0.29))
    s.set_color_(COLOR_RESIZE_GRIP_ACTIVE,          imgui.ImVec4(0.40, 0.44, 0.47, 1.00))
    s.set_color_(COLOR_TAB,                         imgui.ImVec4(0.10, 0.00, 0.00, 0.52))
    s.set_color_(COLOR_TAB_HOVERED,                 imgui.ImVec4(0.14, 0.14, 0.14, 1.00))
    s.set_color_(COLOR_TAB_ACTIVE,                  imgui.ImVec4(0.20, 0.20, 0.20, 0.36))
    s.set_color_(COLOR_TAB_UNFOCUSED,               imgui.ImVec4(0.00, 0.00, 0.00, 0.52))
    s.set_color_(COLOR_TAB_UNFOCUSED_ACTIVE,        imgui.ImVec4(0.14, 0.14, 0.14, 1.00))
    s.set_color_(COLOR_PLOT_LINES,                  imgui.ImVec4(1.00, 0.80, 0.00, 0.90))
    s.set_color_(COLOR_PLOT_LINES_HOVERED,          imgui.ImVec4(1.00, 0.80, 0.00, 1.00))
    s.set_color_(COLOR_PLOT_HISTOGRAM,              imgui.ImVec4(1.00, 0.80, 0.00, 0.90))
    s.set_color_(COLOR_PLOT_HISTOGRAM_HOVERED,      imgui.ImVec4(1.00, 0.80, 0.00, 1.00))
    s.set_color_(COLOR_TEXT_SELECTED_BACKGROUND,    imgui.ImVec4(0.20, 0.22, 0.23, 1.00))
    s.set_color_(COLOR_NAV_HIGHLIGHT,               imgui.ImVec4(1.00, 0.00, 0.00, 1.00))
    s.set_color_(COLOR_NAV_WINDOWING_HIGHLIGHT,     imgui.ImVec4(1.00, 0.00, 0.00, 0.70))
    s.set_color_(COLOR_NAV_WINDOWING_HIGHLIGHT,     imgui.ImVec4(1.00, 0.00, 0.00, 0.20))
    s.set_color_(COLOR_NAV_WINDOWING_DIM_BACKGROUND,imgui.ImVec4(1.00, 0.00, 0.00, 0.35))



@contextlib.contextmanager
def eval_color():
    imgui.push_style_color(COLOR_HEADER,            imgui.ImVec4(0.19, 0.22, 0.25, 1.00))
    imgui.push_style_color(COLOR_HEADER_HOVERED,    imgui.ImVec4(0.21, 0.24, 0.27, 1.00))
    imgui.push_style_color(COLOR_HEADER_ACTIVE,     imgui.ImVec4(0.21, 0.24, 0.27, 1.00))
    yield
    imgui.pop_style_color(3)


@contextlib.contextmanager
def change_font(font):
    imgui.push_font(font)
    yield
    imgui.pop_font()

# ----------------------------------------------------------------------------


@contextlib.contextmanager
def item_width(width=None):
    if width is not None:
        imgui.push_item_width(width)
        yield
        imgui.pop_item_width()
    else:
        yield


# ----------------------------------------------------------------------------


def scoped_by_object_id(method):
    def decorator(self, *args, **kwargs):
        imgui.push_id(str(id(self)))
        res = method(self, *args, **kwargs)
        imgui.pop_id()
        return res

    return decorator


# ----------------------------------------------------------------------------


def button(label, width=0, enabled=True):
    clicked = imgui.button(label, size=imgui.ImVec2(width, 0))
    clicked = clicked and enabled
    return clicked


# ----------------------------------------------------------------------------


def collapsing_header(text, visible=None, flags=0, default=False, enabled=True, show=True):
    expanded = False
    if show:
        if default:
            flags |= TREE_NODE_DEFAULT_OPEN
        if not enabled:
            flags |= TREE_NODE_LEAF
        expanded = imgui.collapsing_header(text, flags=flags)
        expanded = expanded and enabled
    return expanded, visible


# ----------------------------------------------------------------------------


def popup_button(label, width=0, enabled=True):
    if button(label, width, enabled):
        imgui.open_popup(label)
    opened = imgui.begin_popup(label)
    return opened


# ----------------------------------------------------------------------------


def drag_previous_control(enabled=True):
    dragging = False
    dx = 0
    dy = 0
    if imgui.begin_drag_drop_source(imgui.DragDropFlags_.source_no_preview_tooltip.value):
        if enabled:
            dragging = True
            dx, dy = imgui.get_mouse_drag_delta()
            imgui.reset_mouse_drag_delta()
        imgui.end_drag_drop_source()
    return dragging, dx, dy


# ----------------------------------------------------------------------------


def drag_button(label, width=0, enabled=True):
    clicked = button(label, width=width, enabled=enabled)
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    return clicked, dragging, dx, dy


# ----------------------------------------------------------------------------


def drag_hidden_window(label, x, y, width, height, enabled=True):
    imgui.push_style_color(COLOR_WINDOW_BACKGROUND, 0)
    imgui.push_style_color(COLOR_BORDER, 0)
    imgui.set_next_window_pos(imgui.ImVec2(x, y))
    imgui.set_next_window_size(imgui.ImVec2(width, height))
    imgui.begin(label, p_open=False, flags=(WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE))
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    imgui.end()
    imgui.pop_style_color(2)
    return dragging, dx, dy


# ----------------------------------------------------------------------------
