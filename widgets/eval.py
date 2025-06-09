from imgui_bundle import imgui, implot
import numpy as np
import torch
import pprint

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.gui_utils import style
from splatviz_utils.dict_utils import EasyDict
from widgets.widget import Widget


class EvalWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Eval")
        self.text = "gs"
        self.hist_cache = dict()
        self.use_cache_dict = dict()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        viz.args.eval_text = ""

        if show:
            label("Eval code:")
            _changed, self.text = imgui.input_text("##input_text", self.text)
            imgui.new_line()
            with style.eval_color():
                with imgui_utils.change_font(self.viz._imgui_fonts_code[self.viz._cur_font_size]):
                    self.handle_type_rec(self.viz.eval_result, depth=0, obj_name="")
            viz.args.eval_text = self.text

    def handle_type_rec(self, result, depth, obj_name):
        if (
            hasattr(result, "__dict__")
            and len(result.__dict__.keys()) > 0
            or isinstance(result, dict)
            and len(result.keys()) > 0
        ):
            if isinstance(result, EasyDict):
                result = dict(result)
            elif hasattr(result, "__dict__"):
                result = result.__dict__

            sorted_keys = sorted(result.keys(), key=lambda x: type(result[x]).__name__)
            for key in sorted_keys:
                imgui.new_line()
                imgui.same_line(depth)
                info, primitive = self.get_short_info(key, result[key])
                if primitive:
                    imgui.new_line()
                    imgui.same_line(depth)
                    imgui.text(info)
                else:
                    expanded, _visible = imgui_utils.collapsing_header(info, default=False)
                    if expanded:
                        self.handle_type_rec(result[key], depth=depth + 20, obj_name=key)
        else:
            # write a non-primitive object that is not an object with __dict__
            if isinstance(result, torch.Tensor):
                self.handle_tensor(result, depth, obj_name)
            else:
                imgui.new_line()
                imgui.same_line(depth)
                imgui.text(pprint.pformat(result, compact=True))

    def handle_tensor(self, result, depth, var_name):
        orig_var_name = var_name
        var_name += self.viz.args.ply_file_paths[0]

        bins = 100
        if var_name not in self.hist_cache.keys() or not self.use_cache_dict[var_name]:
            hist = np.histogram(result.cpu().detach().numpy().reshape(-1), bins=bins)
            self.hist_cache[var_name] = hist
        if var_name not in self.use_cache_dict.keys():
            self.use_cache_dict[var_name] = True

        imgui.new_line()
        imgui.same_line(depth)
        plot_size = imgui.ImVec2(self.viz.pane_w - 150, 150)
        if implot.begin_plot(f"{orig_var_name}", plot_size):
            bar_size = (
                max(self.hist_cache[var_name][1].astype(np.float32))
                - min(self.hist_cache[var_name][1].astype(np.float32))
            ) / (bins + 1)
            implot.plot_bars(
                f"##hist{var_name}",
                ys=self.hist_cache[var_name][0].astype(np.float32),
                xs=self.hist_cache[var_name][1].astype(np.float32),
                bar_size=bar_size,
            )
            implot.end_plot()

        imgui.same_line()
        label("Cache")
        _, self.use_cache_dict[var_name] = imgui.checkbox(f"##cache{var_name}", self.use_cache_dict[var_name])

        if np.prod(result.shape) > 0:
            with imgui_utils.item_width(int(self.viz.pane_w  * 0.1)):
                imgui.new_line()
                imgui.same_line(depth)
                label("min")
                imgui.input_float(f"##min_{var_name}", result.min().cpu().numpy())
                imgui.same_line()
                label("mean")
                imgui.input_float(f"##mean_{var_name}", result.mean().cpu().numpy())
                imgui.same_line()
                label("max")
                imgui.input_float(f"##max_{var_name}", result.max().cpu().numpy())
                imgui.same_line()
                label("std")
                imgui.input_float(f"##std_{var_name}", result.std().cpu().numpy())

                imgui.new_line()
                imgui.same_line(depth)
                label("shape")
                imgui.text(str(list(result.shape)))


    @staticmethod
    def get_short_info(key, value):
        readable_type = type(value).__name__
        primitives = (bool, str, int, float, type(None))
        spacing_type = 10
        spacing_name = 30

        if isinstance(value, primitives):
            return f"   {readable_type:<{spacing_type}} {key:<{spacing_name}} {value}", True
        elif isinstance(value, torch.Tensor):
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}} shape={list(value.shape)}", False
        elif isinstance(value, dict) and len(value.keys()) == 0:
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}} {value}", True
        elif callable(value):
            readable_type = "function"
            return f"   {readable_type:<{spacing_type}} {key:<{spacing_name}} {value}", True
        else:
            return f"   {readable_type:<{spacing_type}} {key:<{spacing_name}}", False
