from imgui_bundle import imgui
from gui_utils import imgui_utils
from gui_utils.easy_imgui import label
from imgui_bundle import implot

from viz_utils.dict import EasyDict
from widgets.widget import Widget
import numpy as np


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

        self.pause_button_states = ["Resume Training", "Pause Training"]
        self.do_training = True

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            if imgui.button(self.pause_button_states[int(self.do_training)]):
                self.do_training = not self.do_training
        viz.args.do_training = self.do_training

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

        if show:
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

