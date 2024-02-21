import click
import imgui
import numpy as np
import torch
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)

from gui_utils import imgui_window
from gui_utils import imgui_utils
from gui_utils import gl_utils
from gui_utils import text_utils
from viz_utils.dict import EasyDict
from widgets import pose_widget, zoom_widget, edit_widget, eval_widget, performance_widget, pickle_widget, video_widget
from viz.async_renderer import AsyncRenderer


class Visualizer(imgui_window.ImguiWindow):
    def __init__(self, capture_dir=None, data_path=None):
        super().__init__(title="Gaussian Machine", window_width=1920, window_height=1200, font="fonts/JetBrainsMono-Regular.ttf")

        # Internals.
        self._last_error_print = None
        self._async_renderer = AsyncRenderer()
        self._defer_rendering = 0
        self._tex_img = None
        self._tex_obj = None
        self.eval_result = ""

        # Widget interface.
        self.args = EasyDict()
        self.result = EasyDict()

        # Widgets.
        self.pckl_widget = pickle_widget.PickleWidget(self, data_path)
        self.pose_widget = pose_widget.PoseWidget(self)
        self.zoom_widget = zoom_widget.ZoomWidget(self)
        self.edit_widget = edit_widget.EditWidget(self)
        self.eval_widget = eval_widget.EvalWidget(self)
        self.perf_widget = performance_widget.PerformanceWidget(self)
        self.video_widget = video_widget.VideoWidget(self)


        """
        self.latent_widget = latent_widget.LatentWidget(self)
        self.stylemix_widget = stylemix_widget.StyleMixingWidget(self)
        self.trunc_noise_widget = trunc_noise_widget.TruncationNoiseWidget(self)
        self.capture_widget = capture_widget.CaptureWidget(self)
        self.backbone_cache_widget = backbone_cache_widget.BackboneCacheWidget(self)
        self.layer_widget = layer_widget.LayerWidget(self)
        self.conditioning_pose_widget = conditioning_pose_widget.ConditioningPoseWidget(self)
        self.render_type_widget = render_type_widget.RenderTypeWidget(self)
        self.render_depth_sample_widget = render_depth_sample_widget.RenderDepthSampleWidget(self)
        self.gaussian_widget = gaussian_widget.GaussianWidget(self)
        # self.stats_widget             = stats_widget.StatsWidget(self)
        self.camera_widget = camera_widget.CameraWidget(self)
        """

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
        self.pane_w = self.font_size * 50
        self.button_w = self.font_size * 5
        self.label_w = round(self.font_size * 5.5)

        # Detect mouse dragging in the result area.
        dragging, dx, dy = imgui_utils.drag_hidden_window(
            "##result_area", x=self.pane_w, y=0, width=self.content_width - self.pane_w, height=self.content_height
        )
        if dragging:
            self.pose_widget.drag(dx, dy)

        # Begin control pane.
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.pane_w, self.content_height)
        imgui.begin(
            "##control_pane",
            closable=False,
            flags=(imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE),
        )

        # Widgets.
        self.pckl_widget(True)
        expanded, _visible = imgui_utils.collapsing_header("Performance & capture", default=False)
        self.perf_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Camera", default=False)
        self.pose_widget(expanded)
        self.zoom_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Video", default=False)
        self.video_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Edit", default=True)
        self.edit_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Eval", default=True)
        self.eval_widget(expanded)


        """
        self.pickle_widget(expanded)
        self.conditioning_pose_widget(expanded)
        self.render_type_widget(expanded)
        self.render_depth_sample_widget(expanded)
        self.latent_widget(expanded)
        self.stylemix_widget(expanded)
        self.trunc_noise_widget(expanded)

        self.capture_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Layers & channels", default=False)
        self.backbone_cache_widget(expanded)
        self.layer_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Scale Gaussians", default=True)
        self.gaussian_widget(expanded)
        # expanded, _visible = imgui_utils.collapsing_header('Stats', default=True)
        # self.stats_widget(expanded)
        expanded, _visible = imgui_utils.collapsing_header("Camera", default=True)
        self.camera_widget(expanded)
        """

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
@click.option("--capture-dir", help="Where to save screenshot captures", metavar="PATH", default=None)
@click.option("--data_path", help="Where to search for .ply files", metavar="PATH", default="C:/Users/fbarthel/Documents/CVGGaussianSplatting/download_output/relevant_runs")
def main(capture_dir, data_path):
    viz = Visualizer(capture_dir=capture_dir, data_path=data_path)
    while not viz.should_close():
        viz.draw_frame()
    viz.close()


if __name__ == "__main__":
    main()
