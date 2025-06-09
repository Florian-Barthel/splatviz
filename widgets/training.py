from imgui_bundle import imgui, ImVec2
from imgui_bundle import implot
import numpy as np

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.dict_utils import EasyDict
from widgets.widget import Widget


class TrainingWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Training")
        self.text = "gaussian"
        self.hist_cache = dict()
        self.use_cache_dict = dict()
        self.iterations = []
        self.plots = EasyDict(
            num_gaussians=dict(values=[], dtype=int),
            loss=dict(values=[], dtype=float),
            sh_degree=dict(values=[], dtype=int),
        )
        self.stop_at_value = -1
        self.stop_training = False
        self.stop_from_renderer = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            if self.stop_training or self.stop_from_renderer:
                if imgui.button("Resume Training", ImVec2(viz.label_w_large, 0)):
                    self.stop_training = False
                    if self.stop_from_renderer:
                        self.stop_at_value = -1
            else:
                if imgui.button("Pause Training", ImVec2(viz.label_w_large, 0)):
                    self.stop_training = True

        viz.args.do_training = not self.stop_training

        if "training_stats" in viz.result.keys():
            stats = viz.result["training_stats"]
        else:
            if show:
                label("No training stats send by the renderer.")
            return

        self.iterations.append(stats["iteration"])
        self.plots.num_gaussians["values"].append(stats["num_gaussians"])
        self.plots.loss["values"].append(stats["loss"])
        self.plots.sh_degree["values"].append(stats["sh_degree"])
        self.stop_from_renderer = stats["paused"]

        if show:
            if self.stop_training or self.stop_from_renderer:
                if imgui.button("Single Training Step", ImVec2(viz.label_w_large, 0)):
                    viz.args.single_training_step = True
                    self.stop_training = True

            label(f"Current Iteration", viz.label_w_large)
            imgui.text(str(stats['iteration']))
            label("Pause Training at", viz.label_w_large)
            _, self.stop_at_value = imgui.input_int("##stop_at", self.stop_at_value)

            for plot_name, plot_values in self.plots.items():
                plot_size = imgui.ImVec2(viz.pane_w - 150, 200)
                implot.set_next_axes_to_fit()
                if implot.begin_plot(plot_name, plot_size):
                    implot.plot_line(
                        plot_name,
                        ys=np.array(plot_values["values"], dtype=plot_values["dtype"]),
                        xs=np.array(self.iterations, dtype=plot_values["dtype"]),
                    )
                    implot.end_plot()

            imgui.text("Training Params:")
            with imgui_utils.indent():
                for param, value in stats["train_params"].items():
                    label(str(param), viz.label_w_large)
                    label(str(value), viz.label_w_large)
                    imgui.new_line()

        viz.args.stop_at_value = self.stop_at_value
