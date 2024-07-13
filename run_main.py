import click
from imgui_bundle import imgui
import numpy as np
import torch
import sys

sys.path.append("./gaussian-splatting")
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from gui_utils.constants import *
from viz_utils.dict import EasyDict
from widgets import (
    edit_widget,
    eval_widget,
    performance_widget,
    load_widget_pkl,
    load_widget_ply,
    video_widget,
    cam_widget,
    capture_widget,
    latent_widget,
    render_widget,
)
from viz.async_renderer import AsyncRenderer
from viz.gaussian_renderer import GaussianRenderer
from viz.gaussian_decoder_renderer import GaussianDecoderRenderer


class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, data_path=None, use_gan_decoder=False):
        self.code_font_path = "fonts/jetbrainsmono/JetBrainsMono-Regular.ttf"
        self.regular_font_path = "fonts/source_sans_pro/SourceSansPro-Regular.otf"

        super().__init__(title="splatviz", window_width=1920, window_height=1080, font=self.regular_font_path, code_font=self.code_font_path)

        self.code_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self.regular_font = imgui.get_io().fonts.add_font_from_file_ttf(self.code_font_path, 14)
        self._imgui_renderer.refresh_font_texture()

        # Internals.
        self._last_error_print = None
        self.use_gan_decoder = use_gan_decoder
        renderer = GaussianDecoderRenderer() if use_gan_decoder else GaussianRenderer()
        self._async_renderer = AsyncRenderer(renderer)
        self._defer_rendering = 0
        self._tex_img = None
        self._tex_obj = None
        self.eval_result = ""

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()

        # Widgets.
        if self.use_gan_decoder:
            self.load_widget = load_widget_pkl.LoadWidget(self, data_path)
        else:
            self.load_widget = load_widget_ply.LoadWidget(self, data_path)
        self.cam_widget = cam_widget.CamWidget(self)
        self.latent_widget = latent_widget.LatentWidget(self)
        self.edit_widget = edit_widget.EditWidget(self)
        self.eval_widget = eval_widget.EvalWidget(self)
        self.perf_widget = performance_widget.PerformanceWidget(self)
        self.video_widget = video_widget.VideoWidget(self)
        self.capture_widget = capture_widget.CaptureWidget(self)
        self.rener_widget = render_widget.RenderWidget(self)

        # Initialize window.
        self.set_position(0, 0)
        self._adjust_font_size()
        self.skip_frame()  # Layout may change after first frame.

    def close(self):
        super().close()
        if self._async_renderer is not None:
            self._async_renderer.close()
            self._async_renderer = None

    def print_error(self, error):
        error = str(error)
        if error != self._last_error_print:
            print("\n" + error + "\n")
            self._last_error_print = error

    def defer_rendering(self, num_frames=1):
        self._defer_rendering = max(self._defer_rendering, num_frames)

    def clear_result(self):
        self._async_renderer.clear_result()

    def _adjust_font_size(self):
        old = self.font_size
        self.set_font_size(min(self.content_width / 120, self.content_height / 60))
        if self.font_size != old:
            self.skip_frame()  # Layout changed.

    def draw_frame(self):
        self.begin_frame()
        self.args = EasyDict()
        self.pane_w = max(self.content_width - self.content_height, 500)
        self.button_w = self.font_size * 5
        self.button_large_w = self.font_size * 10
        self.label_w = round(self.font_size * 5.5) + 100

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window(
            "##result_area", x=self.pane_w, y=0, width=self.content_width - self.pane_w, height=self.content_height
        )
        if dragging:
            self.cam_widget.drag(dx, dy)

        # Begin control pane.
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(self.pane_w, self.content_height))
        imgui.begin(
            "##control_pane",
            p_open=True,
            flags=(WINDOW_NO_TITLE_BAR | WINDOW_NO_RESIZE | WINDOW_NO_MOVE),
        )

        # Widgets.

        expanded, _visible = imgui_utils.collapsing_header("Load", default=True)
        imgui.indent()
        self.load_widget(expanded)
        imgui.unindent()

        expanded, _visible = imgui_utils.collapsing_header("Performance", default=False)
        imgui.indent()
        self.perf_widget(expanded)
        imgui.unindent()

        expanded, _visible = imgui_utils.collapsing_header("Render", default=False)
        imgui.indent()
        self.rener_widget(expanded, decoder=self.use_gan_decoder)
        imgui.unindent()

        expanded, _visible = imgui_utils.collapsing_header("Camera", default=False)
        imgui.indent()
        self.cam_widget(expanded)
        imgui.unindent()

        if self.use_gan_decoder:
            expanded, _visible = imgui_utils.collapsing_header("Latent", default=False)
            imgui.indent()
            self.latent_widget(expanded)
            imgui.unindent()

        expanded, _visible = imgui_utils.collapsing_header("Video", default=False)
        imgui.indent()
        self.video_widget(expanded)
        imgui.unindent()

        expanded, _visible = imgui_utils.collapsing_header("Save", default=False)
        imgui.indent()
        self.capture_widget(expanded)
        imgui.unindent()

        # with imgui_utils.change_font(self.code_font):
        expanded, _visible = imgui_utils.collapsing_header("Edit", default=False)
        imgui.indent()
        self.edit_widget(expanded)
        imgui.unindent()

        expanded, _visible = imgui_utils.collapsing_header("Eval", default=False)
        imgui.indent()
        self.eval_widget(expanded)
        imgui.unindent()

        # imgui.show_style_editor()

        # Render.
        if self.is_skipping_frames():
            pass
        elif self._defer_rendering > 0:
            self._defer_rendering -= 1
        else:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None:
                self.result = result

        # Display.
        max_w = self.content_width - self.pane_w
        max_h = self.content_height
        pos = np.array([self.pane_w + max_w / 2, max_h / 2])
        if "image" in self.result:
            if self._tex_img is not self.result.image:
                self._tex_img = self.result.image
                if self._tex_obj is None or not self._tex_obj.is_compatible(image=self._tex_img):
                    self._tex_obj = gl_utils.Texture(image=self._tex_img, bilinear=False, mipmap=False)
                else:
                    self._tex_obj.update(self._tex_img)
            zoom = min(max_w / self._tex_obj.width, max_h / self._tex_obj.height)
            self._tex_obj.draw(pos=pos, zoom=zoom, align=0.5, rint=True)
        if "error" in self.result:
            self.print_error(self.result.error)
            if "message" not in self.result:
                self.result.message = str(self.result.error)
        if "message" in self.result:
            tex = text_utils.get_texture(
                self.result.message, size=self.font_size, max_width=max_w, max_height=max_h, outline=2
            )
            tex.draw(pos=pos, align=0.5, rint=True, color=1)
        if "eval" in self.result:
            self.eval_result = self.result.eval
        else:
            self.eval_result = None

        # End frame.
        self._adjust_font_size()
        imgui.end()
        self.end_frame()


@click.command()
@click.option("--data_path", help="Where to search for .ply files", metavar="PATH", default="./sample_scenes")
@click.option("--use_decoder", help="Visualizes the results of a decoder", is_flag=True)
def main(data_path, use_decoder):
    viz = Visualizer(data_path=data_path, use_gan_decoder=use_decoder)
    while not viz.should_close():
        viz.draw_frame()
    viz.close()


if __name__ == "__main__":
    main()
