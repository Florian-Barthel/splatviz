from imgui_bundle import imgui
import numpy as np
import torch
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)
from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.attach_renderer import AttachRenderer
from splatviz_utils.gui_utils import imgui_window
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import gl_utils
from splatviz_utils.gui_utils import text_utils
from splatviz_utils.gui_utils.constants import *
from splatviz_utils.dict_utils import EasyDict
from widgets import (
    edit,
    eval,
    performance,
    load_ply,
    camera,
    save,
    render,
    training,
)


class Splatviz(imgui_window.ImguiWindow):
    def __init__(self, mode, host, port):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(
            title="splatviz",
            window_width=1920,
            window_height=1080,
            font=self.regular_font_path,
            code_font=self.code_font_path,
        )

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self._imgui_renderer.refresh_font_texture()

        # Internals.
        self._last_error_print = None

        self.widgets = []
        update_all_the_time = False
        if mode == "default":
            self.widgets = [
                load_ply.LoadWidget(self),
                camera.CamWidget(self),
                performance.PerformanceWidget(self),
                save.CaptureWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                eval.EvalWidget(self),
            ]
            renderer = GaussianRenderer()
        elif mode == "attach":
            self.widgets = [
                camera.CamWidget(self),
                performance.PerformanceWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                training.TrainingWidget(self),
            ]
            renderer = AttachRenderer(host=host, port=port)
            update_all_the_time = True
        else:
            raise NotImplementedError(f"Mode '{mode}' not recognized.")

        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self._tex_img = None
        self._tex_obj = None
        self.pane_w = None
        self.splitter_w = 30
        self.render_x = 0
        self.render_y = 0
        self.render_w = 1
        self.render_h = 1

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()
        self.eval_result = ""

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()
        self.preprocessed_images = []

    def close(self):
        for widget in self.widgets:
            widget.close()
        super().close()

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print(f"\n{error}\n")
            self._last_error_print = error

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()

    def _set_sizes(self):
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

    def _draw_widgets_pane(self):
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

    def _draw_splitter(self):
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
        if imgui.is_item_hovered() or imgui.is_item_active():
            imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
        if imgui.is_item_active():
            self.pane_w += imgui.get_io().mouse_delta.x
            self._set_sizes()
        imgui.end()
        imgui.pop_style_color()

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

    def _draw_rendering_pane(self):
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
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            self._draw_centered_texture(self._tex_obj, max_w, max_h)

        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message,
                size=self.font_size,
                max_width=max_w,
                max_height=max_h,
                outline=0,
            )
            self._draw_centered_texture(tex, max_w, max_h)

            imgui.dummy(imgui.ImVec2(max_w, max_h))

        imgui.pop_style_color()
        imgui.end()

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        self._set_sizes()

        self._draw_widgets_pane()
        self._draw_splitter()
        self.args.render_width = max(int(round(self.render_w)), 1)
        self.args.render_height = max(int(round(self.render_h)), 1)

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result

        self._draw_rendering_pane()

        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        if "preprocessed_images" in self.result:
            self.preprocessed_images = self.result.preprocessed_images

        # End frame.
        self._adjust_font_size()
        self.end_frame()
