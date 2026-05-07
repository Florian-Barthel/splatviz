import glfw
from imgui_bundle import imgui

from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.gui_utils import gl_utils
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import text_utils
from splatviz_utils.gui_utils.constants import *


class WindowHelper:
    def init_window_helper(self):
        self._last_error_print = None
        self._tex_img = None
        self._tex_obj = None
        self.pane_w = None
        self.splitter_w = 30
        self.render_x = 0
        self.render_y = 0
        self.render_w = 1
        self.render_h = 1
        self.is_dragging_splitter = False
        self._cursor_arrow = glfw.create_standard_cursor(glfw.ARROW_CURSOR)
        self._cursor_resize_ew = glfw.create_standard_cursor(glfw.HRESIZE_CURSOR)
        self._is_using_resize_cursor = False

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def set_sizes(self):
        if self.pane_w is None:
            self.pane_w = max(self.content_width - self.content_height, 500)
        self.splitter_w = max(round(self.font_size * 0.75), 12)
        available_w = max(self.content_width - self.splitter_w, 1)
        min_widget_w = min(360, available_w)
        min_render_w = min(360, max(available_w - min_widget_w, 1))
        max_widget_w = max(min_widget_w, available_w - min_render_w)
        self.pane_w = min(max(self.pane_w, min_widget_w), max_widget_w)
        self.render_x = self.pane_w + self.splitter_w
        self.render_y = 0
        self.render_w = max(self.content_width - self.render_x, 1)
        self.render_h = self.content_height
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100
        self.label_w_large = round(self.font_size * 5.5) + 150

    def draw_widgets_pane(self):
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.pane_w, self.content_height))
        flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE | WINDOW_NO_SAVED_SETTINGS
        imgui.begin("##widgets_pane", p_open=True, flags=flags)
        for widget in self.widgets:
            expanded, _visible = imgui_utils.collapsing_header(widget.name, default=widget.name == "Load")
            imgui.indent()
            widget(expanded)
            imgui.unindent()
        imgui.end()

    def draw_splitter(self):
        imgui.set_next_window_pos(imgui.ImVec2(self.pane_w, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.splitter_w, self.content_height))
        flags = (
            WINDOW_NO_DECORATION
            | WINDOW_NO_SCROLLBAR
            | WINDOW_NO_SCROLL_WITH_MOUSE
            | WINDOW_NO_SAVED_SETTINGS
            | WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
            | WINDOW_NO_NAV
        )
        imgui.push_style_color(COLOR_WINDOW_BACKGROUND, imgui.ImVec4(0.5, 0.5, 0.5, 1))
        imgui.begin("##widgets_render_splitter", p_open=True, flags=flags)
        imgui.invisible_button("##resize", imgui.ImVec2(self.splitter_w, self.content_height))
        if imgui.is_item_active():
            self.is_dragging_splitter = True
        if self.is_dragging_splitter and not imgui.is_mouse_down(0):
            self.is_dragging_splitter = False
        use_resize_cursor = imgui.is_item_hovered() or imgui.is_item_active() or self.is_dragging_splitter
        if use_resize_cursor:
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        self.set_resize_cursor(use_resize_cursor)
        if imgui.is_item_active():
            self.pane_w += imgui.get_io().mouse_delta.x
            self.set_sizes()
        imgui.end()
        imgui.pop_style_color()

    def set_resize_cursor(self, enabled):
        if enabled == self._is_using_resize_cursor:
            return
        glfw.set_cursor(self._glfw_window, self._cursor_resize_ew if enabled else self._cursor_arrow)
        self._is_using_resize_cursor = enabled

    def draw_rendering_pane(self):
        imgui.set_next_window_pos(imgui.ImVec2(self.render_x, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.render_w, self.content_height))
        flags = WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE | WINDOW_NO_SAVED_SETTINGS
        imgui.push_style_color(COLOR_WINDOW_BACKGROUND, imgui.ImVec4(0.1, 0.1, 0.1, 1))
        imgui.begin("##rendering_pane", p_open=True, flags=flags)

        avail = imgui.get_content_region_avail()
        cursor = imgui.get_cursor_screen_pos()
        max_w = max(avail.x, 1)
        max_h = max(avail.y, 1)
        self.render_x = cursor.x
        self.render_y = cursor.y
        self.render_w = max_w
        self.render_h = max_h

        if "image" in self.result:
            self._draw_result_image(max_w, max_h)

        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            self._draw_result_message(max_w, max_h)

        imgui.pop_style_color()
        imgui.end()

    def update_render_args(self, args: EasyDict):
        args.render_width = max(int(round(self.render_w)), 1)
        args.render_height = max(int(round(self.render_h)), 1)

    def _draw_result_image(self, max_w, max_h):
        if self._tex_img is not self.result.image:
            self._tex_img = self.result.image
            if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
            else:
                self._tex_obj.update(self._tex_img)
        self._draw_centered_texture(self._tex_obj, max_w, max_h)

    def _draw_result_message(self, max_w, max_h):
        padding = max(self.font_size, 16)
        padded_w = max(max_w - padding * 2, 1)
        padded_h = max(max_h - padding * 2, 1)
        tex = text_utils.get_texture(
            self.result.message,
            size=self.font_size,
            max_width=padded_w,
            max_height=padded_h,
            outline=0,
        )
        cursor = imgui.get_cursor_screen_pos()
        imgui.set_cursor_screen_pos(imgui.ImVec2(cursor.x + padding, cursor.y + padding))
        self._draw_centered_texture(tex, padded_w, padded_h)
        imgui.set_cursor_screen_pos(cursor)
        imgui.dummy(imgui.ImVec2(max_w, max_h))

    def _draw_centered_texture(self, tex, max_w, max_h):
        if tex.width <= 0 or tex.height <= 0 or max_w <= 0 or max_h <= 0:
            return
        zoom = min(max_w / tex.width, max_h / tex.height)
        draw_w = max(tex.width * zoom, 1)
        draw_h = max(tex.height * zoom, 1)
        cursor = imgui.get_cursor_screen_pos()
        imgui.set_cursor_screen_pos(
            imgui.ImVec2(cursor.x + (max_w - draw_w) * 0.5, cursor.y + (max_h - draw_h) * 0.5)
        )
        imgui.image(tex.gl_id, imgui.ImVec2(draw_w, draw_h), imgui.ImVec2(0, 0), imgui.ImVec2(1, 1))
        imgui.set_cursor_screen_pos(cursor)
