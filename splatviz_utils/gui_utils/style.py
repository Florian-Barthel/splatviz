import contextlib
from imgui_bundle import imgui
from splatviz_utils.gui_utils.constants import *


def set_default_style(color_scheme="dark", spacing=5, indent=20, scrollbar=10):
    s = imgui.get_style()
    s.window_padding = [0, 0]
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
    s.set_color_(COLOR_BORDER_SHADOW,               imgui.ImVec4(0.00, 0.00, 0.00, 0.00))

    # background of all text areas and checkboxes
    s.set_color_(COLOR_FRAME_BACKGROUND,            imgui.ImVec4(0.15, 0.15, 0.15, 0.50))
    s.set_color_(COLOR_FRAME_BACKGROUND_HOVERED,    imgui.ImVec4(0.25, 0.25, 0.25, 0.50))
    s.set_color_(COLOR_FRAME_BACKGROUND_ACTIVE,     imgui.ImVec4(0.25, 0.25, 0.25, 1.00))

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

    # color of selected text
    s.set_color_(COLOR_NAV_WINDOWING_HIGHLIGHT,     imgui.ImVec4(0.90, 0.60, 0.00, 0.70))
    s.set_color_(COLOR_NAV_WINDOWING_HIGHLIGHT,     imgui.ImVec4(0.90, 0.60, 0.00, 0.20))
    s.set_color_(COLOR_NAV_WINDOWING_DIM_BACKGROUND,imgui.ImVec4(0.90, 0.60, 0.00, 0.35))


@contextlib.contextmanager
def eval_color():
    imgui.push_style_color(COLOR_HEADER,            imgui.ImVec4(0.19, 0.22, 0.25, 1.00))
    imgui.push_style_color(COLOR_HEADER_HOVERED,    imgui.ImVec4(0.21, 0.24, 0.27, 1.00))
    imgui.push_style_color(COLOR_HEADER_ACTIVE,     imgui.ImVec4(0.21, 0.24, 0.27, 1.00))
    yield
    imgui.pop_style_color(3)
