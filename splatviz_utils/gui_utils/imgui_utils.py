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
from splatviz_utils.gui_utils.constants import *


@contextlib.contextmanager
def change_font(font):
    imgui.push_font(font)
    yield
    imgui.pop_font()


@contextlib.contextmanager
def item_width(width=None):
    if width is not None:
        imgui.push_item_width(width)
        yield
        imgui.pop_item_width()
    else:
        yield


@contextlib.contextmanager
def indent(width=0.0):
    imgui.indent(width)
    yield
    imgui.unindent(width)

def scoped_by_object_id(method):
    def decorator(self, *args, **kwargs):
        imgui.push_id(str(id(self)))
        res = method(self, *args, **kwargs)
        imgui.pop_id()
        return res
    return decorator


def button(label, width=0, enabled=True):
    clicked = imgui.button(label, size=imgui.ImVec2(width, 0))
    clicked = clicked and enabled
    return clicked


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


def popup_button(label, width=0, enabled=True):
    if button(label, width, enabled):
        imgui.open_popup(label)
    opened = imgui.begin_popup(label)
    return opened


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


def drag_button(label, width=0, enabled=True):
    clicked = button(label, width=width, enabled=enabled)
    dragging, dx, dy = drag_previous_control(enabled=enabled)
    return clicked, dragging, dx, dy


def did_drag_start_in_window(x, y, width, height, drag_delta):
    mouse_pos_at_drag_start = imgui.get_mouse_pos() - drag_delta
    return (x <= mouse_pos_at_drag_start.x <= x + width) and (y <= mouse_pos_at_drag_start.y <= y + height)
