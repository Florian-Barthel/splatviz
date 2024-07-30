from threading import Thread
import numpy as np
from imgui_bundle import imgui
from gui_utils import imgui_utils
import torch
import GPUtil
import time
from imgui_bundle import implot


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


class PerformanceWidget:
    def __init__(self, viz):
        self.num_elements = 500
        self.viz = viz
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
            if implot.begin_plot("FPS", plot_size):
                implot.setup_axis(implot.ImAxis_.x1.value, flags=implot.AxisFlags_.no_decorations.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, v_min=0, v_max=self.num_elements)
                # implot.setup_axis_zoom_constraints(implot.ImAxis_.x1.value, -10, 10)

                implot.setup_axis(implot.ImAxis_.y1.value)
                implot.setup_axis_limits(implot.ImAxis_.y1.value, v_min=0, v_max=200)

                values_gui = np.array(self.gui_FPS_smooth)
                implot.plot_line("GUI", values=values_gui)

                values_render = np.array(self.render_FPS_smooth)
                implot.plot_line("Render", values=values_render)
                implot.end_plot()

            with imgui_utils.item_width(viz.font_size * 6):
                imgui.text(f"GUI FPS Limit:")
                imgui.same_line(viz.label_w)
                _changed, self.fps_limit = imgui.input_int("##FPS_limit", self.fps_limit)
                self.fps_limit = min(max(self.fps_limit, 5), 1000)

                imgui.text(f"Vertical sync:")
                imgui.same_line(viz.label_w)
                _clicked, self.use_vsync = imgui.checkbox("##Vertical_sync", self.use_vsync)

            imgui.text(f"FPS GUI:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.gui_FPS_smooth[-1]:.2f}")

            imgui.text(f"FPS Render:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.render_FPS_smooth[-1]:.2f}")

            # CUDA
            imgui.new_line()
            imgui.text(f"Device:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu.name}")

            imgui.text(f"Device Capability:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.device_capability[0]}.{self.device_capability[1]}")

            imgui.text(f"Driver:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu.driver}")

            imgui.text(f"CUDA Version:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.cuda_version}")

            imgui.text(f"Clock Rate:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{torch.cuda.clock_rate()}")

            imgui.text(f"Temperature:")
            imgui.same_line(viz.label_w)
            imgui.text(f"{self.gpu_monitor.gpu.temperature}° C")

            imgui.text(f"Memory Used:")
            imgui.same_line(viz.label_w)
            imgui.progress_bar(self.gpu_monitor.gpu.memoryUsed / self.gpu_monitor.gpu.memoryTotal, imgui.ImVec2(300, 30), f"{self.gpu_monitor.gpu.memoryUsed / 1024:.2f}GB / {self.gpu_monitor.gpu.memoryTotal / 1024:.2f}GB")

            # imgui.text(f"Memory Free:")
            # imgui.same_line(viz.label_w)
            # imgui.progress_bar(self.gpu_monitor.gpu.memoryFree / self.gpu_monitor.gpu.memoryTotal, imgui.ImVec2(300, 30), f"{self.gpu_monitor.gpu.memoryFree / 1024:.2f}GB / {self.gpu_monitor.gpu.memoryTotal / 1024:.2f}GB")

            if imgui.button("Empty Cache"):
                torch.cuda.empty_cache()

        viz.args.fast_render_mode = self.fast_render_mode
        viz.set_fps_limit(self.fps_limit)
        viz.set_vsync(self.use_vsync)
