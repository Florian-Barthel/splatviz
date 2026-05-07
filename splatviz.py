import numpy as np
import torch
from imgui_bundle import imgui

torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)
from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.attach_renderer import AttachRenderer
from splatviz_utils.gui_utils import imgui_window
from splatviz_utils.gui_utils.detached_render_window import DetachedRenderWindow
from splatviz_utils.gui_utils.window_helper import WindowHelper
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


class Splatviz(WindowHelper, imgui_window.ImguiWindow):
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
        self.init_window_helper()

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()
        self.eval_result = ""
        self.detached_render_window = None

        # Initialize window.
        self.set_position(0, 0)
        self.adjust_font_size()
        self.skip_frame()
        self.preprocessed_images = []

    def close(self):
        self.close_detached_render_window()
        for widget in self.widgets:
            widget.close()
        super().close()

    def toggle_detached_render_window(self):
        if self.detached_render_window is None:
            self.detached_render_window = DetachedRenderWindow()
            self.make_context_current()
        else:
            self.close_detached_render_window()

    def close_detached_render_window(self):
        if self.detached_render_window is not None:
            self.detached_render_window.close()
            self.detached_render_window = None
            self.make_context_current()

    def draw_frame(self):
        # main loop
        self.begin_frame()
        self.args = EasyDict()
        if self.detached_render_window is not None and self.detached_render_window.should_close():
            self.close_detached_render_window()
        self.set_sizes()
        is_render_detached = self.detached_render_window is not None
        if is_render_detached:
            self.pane_w = self.content_width
            self.set_resize_cursor(False)

        self.draw_widgets_pane()
        if not is_render_detached:
            self.draw_splitter()
        self.update_render_args(self.args)
        if self.detached_render_window is not None:
            if self.detached_render_window.should_close():
                self.close_detached_render_window()
            else:
                self.detached_render_window.update_render_args(self.args)

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result

        if self.detached_render_window is None:
            self.draw_rendering_pane()

        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        if "preprocessed_images" in self.result:
            self.preprocessed_images = self.result.preprocessed_images

        # End frame.
        self.adjust_font_size()
        self.end_frame()

        if self.detached_render_window is not None:
            if self.detached_render_window.should_close():
                self.close_detached_render_window()
            else:
                self.detached_render_window.draw_frame(self.result)
