from threading import Thread
import numpy as np
from imgui_bundle import imgui
import torch
import GPUtil
import time
from imgui_bundle import implot

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()
        self.gpu = GPUtil.getGPUs()[0]

    def run(self):
        while not self.stopped:
            self.gpu = GPUtil.getGPUs()[0]
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


class PerformanceWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Performance")
        self.num_elements = 500
        self.gui_FPS = [0] * 10
        self.gui_FPS_smooth = [0] * self.num_elements
        self.render_FPS = [0] * 10
        self.render_FPS_smooth = [0] * self.num_elements
        self.fps_limit = 180
        self.use_vsync = False
        self.fast_render_mode = False

        # CUDA
        self.device_properties = torch.cuda.get_device_properties(0)
        self.device_capability = (self.device_properties.major, self.device_properties.minor)
        self.device_name = self.device_properties.name
        self.device_memory = self.device_properties.total_memory
        self.gpu_monitor = Monitor(1)
        self.cuda_version = torch.version.cuda

    def close(self):
        self.gpu_monitor.stop()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if viz.frame_delta > 0:
            self.gui_FPS = self.gui_FPS[1:] + [1 / viz.frame_delta]
        if "render_time" in viz.result:
            self.render_FPS = self.render_FPS[1:] + [1 / viz.result.render_time]
            del viz.result.render_time

        self.gui_FPS_smooth = self.gui_FPS_smooth[1:] + [np.mean(self.gui_FPS)]
        self.render_FPS_smooth = self.render_FPS_smooth[1:] + [np.mean(self.render_FPS)]

        if show:
            plot_size = imgui.ImVec2(viz.pane_w - 150, 200)
            implot.set_next_axes_to_fit()
            if implot.begin_plot("FPS", plot_size):
                values_gui = np.array(self.gui_FPS_smooth)
                implot.plot_line("GUI", values=values_gui)
                values_render = np.array(self.render_FPS_smooth)
                implot.plot_line("Render", values=values_render)
                implot.end_plot()

            with imgui_utils.item_width(viz.font_size * 6):
                label("GUI FPS Limit:", viz.label_w)
                _changed, self.fps_limit = imgui.input_int("##FPS_limit", self.fps_limit)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)

                label("Vertical sync:", viz.label_w)
                _clicked, self.use_vsync = imgui.checkbox("##Vertical_sync", self.use_vsync)

            label("FPS GUI:", viz.label_w)
            imgui.text(f"{self.gui_FPS_smooth[-1]:.2f}")

            label("FPS Render:", viz.label_w)
            imgui.text(f"{self.render_FPS_smooth[-1]:.2f}")
            imgui.new_line()

            # CUDA
            label("Device:", viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu.name}")

            label("Device Capability:", viz.label_w)
            imgui.text(f"{self.device_capability[0]}.{self.device_capability[1]}")

            label("Driver:", viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu.driver}")

            label("CUDA Version:", viz.label_w)
            imgui.text(f"{self.cuda_version}")

            label("Clock Rate:", viz.label_w)
            imgui.text(f"{torch.cuda.clock_rate()}")

            label("Temperature:", viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu.temperature}° C")

            label("Memory Used:", viz.label_w)
            imgui.progress_bar(self.gpu_monitor.gpu.memoryUsed / self.gpu_monitor.gpu.memoryTotal, imgui.ImVec2(300, 30), f"{self.gpu_monitor.gpu.memoryUsed / 1024:.2f}GB / {self.gpu_monitor.gpu.memoryTotal / 1024:.2f}GB")

            if imgui.button("Empty Cache"):
                torch.cuda.empty_cache()

        viz.args.fast_render_mode = self.fast_render_mode
        viz.set_fps_limit(self.fps_limit)
        viz.set_vsync(self.use_vsync)
