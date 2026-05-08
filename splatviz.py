import numpy as np
import torch
from imgui_bundle import imgui

torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)
from renderer.renderer_wrapper import RendererWrapper
from renderer.gaussian_renderer import GaussianRenderer
from renderer.attach_renderer import AttachRenderer
from splatviz_utils.gui_utils import imgui_window
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
    video,
)


class Splatviz(WindowHelper, imgui_window.ImguiWindow):
    def __init__(self, mode, host, port):
        self.code_font_path = "resources/fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "resources/fonts/source_sans_pro/SourceSansPro-Regular.otf"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                video.VideoWidget(self),
                performance.PerformanceWidget(self),
                save.CaptureWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                eval.EvalWidget(self),
            ]
            renderer = GaussianRenderer(device=self.device)
        elif mode == "attach":
            self.widgets = [
                camera.CamWidget(self),
                video.VideoWidget(self),
                performance.PerformanceWidget(self),
                render.RenderWidget(self),
                edit.EditWidget(self),
                training.TrainingWidget(self),
            ]
            renderer = AttachRenderer(host=host, port=port, device=self.device)
            update_all_the_time = True
        else:
            raise NotImplementedError(f"Mode '{mode}' not recognized.")

        self.renderer = RendererWrapper(renderer, update_all_the_time)
        self.init_window_helper()

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()
        self.eval_result = ""

        # Initialize window.
        self.set_position(0, 0)
        self.adjust_font_size()
        self.skip_frame()
        self.preprocessed_images = []

    def close(self):
        for widget in self.widgets:
            widget.close()
        super().close()

    def draw_frame(self):
        # main loop
        self.begin_frame()
        self.args = EasyDict()
        self.set_sizes()

        self.draw_widgets_pane()
        self.draw_splitter()
        self.update_render_args(self.args)

        # Render
        if self.is_skipping_frames():
            pass
        else:
            self.renderer.set_args(**self.args)
            result = self.renderer.result
            if result is not None:
                self.result = result

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
