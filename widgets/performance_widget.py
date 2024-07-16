import array
import numpy as np
from imgui_bundle import imgui
from gui_utils import imgui_utils
from gui_utils.constants import *


class PerformanceWidget:
    def __init__(self, viz):
        self.viz = viz
        self.gui_times = [float("nan")] * 100
        self.render_times = [float("nan")] * 100
        self.render_times_smooth = 0
        self.fps_limit = 180
        self.use_vsync = False
        self.fast_render_mode = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self.gui_times = self.gui_times[1:] + [viz.frame_delta]
        if "render_time" in viz.result:
            self.render_times = self.render_times[1:] + [viz.result.render_time]
            del viz.result.render_time

        if show:
            imgui.text("GUI")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                imgui.plot_lines(
                    "##gui_times",
                    np.array(array.array("f", self.gui_times)),
                    scale_min=0,
                )
            imgui.same_line(viz.label_w + viz.font_size * 9)
            t = [x for x in self.gui_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            imgui.text(f"{t * 1e3:.1f} ms" if t > 0 else "N/A")
            imgui.same_line(viz.label_w + viz.font_size * 14)
            imgui.text(f"{1 / t:.1f} FPS" if t > 0 else "N/A")
            with imgui_utils.item_width(viz.font_size * 6):
                _changed, self.fps_limit = imgui.input_int("FPS limit", self.fps_limit)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)
            imgui.same_line(viz.label_w + viz.font_size * 9)
            _clicked, self.use_vsync = imgui.checkbox("Vertical sync", self.use_vsync)

            imgui.text("Render")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 8):
                imgui.plot_lines(
                    "##render_times",
                    np.array(array.array("f", self.render_times)),
                    scale_min=0,
                )
            imgui.same_line(viz.label_w + viz.font_size * 9)
            t = [x for x in self.render_times if x > 0]
            t = np.mean(t) if len(t) > 0 else 0
            self.render_times_smooth = self.render_times_smooth * 0.99 + 1 / t * 0.01
            imgui.text(f"{t * 1e3:.1f} ms" if t > 0 else "N/A")
            imgui.same_line(viz.label_w + viz.font_size * 14)
            imgui.text(f"{1 / t:.1f} FPS" if t > 0 else "N/A")
            imgui.text(f"{self.render_times_smooth:.1f} FPS smooth")

            _clicked, self.fast_render_mode = imgui.checkbox("Fast Render mode", self.fast_render_mode)

        viz.args.fast_render_mode = self.fast_render_mode

        viz.set_fps_limit(self.fps_limit)
        viz.set_vsync(self.use_vsync)
