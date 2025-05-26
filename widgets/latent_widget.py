import numpy as np
import torch
from imgui_bundle import imgui
from imgui_bundle._imgui_bundle import implot

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.dict_utils import EasyDict
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")
        self.latent = EasyDict(x=0, y=0)
        self.truncation_psi = 1.0
        self.latent_spaces  = ["Z", "W"]
        self.latent_space = 1
        self.mapping_conditioning_modes = ["frontal", "zero", "current"]
        self.mapping_conditioning = 0
        self.seed = 0

    def drag(self, dx, dy):
        self.latent.x += dx / 5000
        self.latent.y -= dy / 5000

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Latent")
            with imgui_utils.item_width(viz.font_size * 8):
                changed, (x_man, y_man) = imgui.input_float2("##xy", v=[self.latent.x, self.latent.y])
                if changed:
                    self.latent.x = x_man
                    self.latent.y = y_man

            imgui.same_line()
            _clicked, dragging, dx, dy = imgui_utils.drag_button("Drag", width=viz.button_w)
            if dragging:
                self.drag(dx, dy)

            if implot.begin_plot("Latent Space", [500, 500]):
                implot.setup_axes_limits(-1, 1, -1, 1, True)
                _changed, self.latent.x, self.latent.y, _, _, _ = implot.drag_point(0, self.latent.x, self.latent.y, imgui.ImVec4(1, 1, 1, 1), 10, out_clicked=True)
                implot.end_plot()
            self.latent.x = np.clip(self.latent.x, -1, 1)
            self.latent.y = np.clip(self.latent.y, -1, 1)

            label("Truncation PSI", width=viz.label_w)
            _changed, self.truncation_psi = imgui.slider_float("##truncation", self.truncation_psi, 0.0, 1.0)

            label("Camera Conditioning", width=viz.label_w)
            _, self.mapping_conditioning = imgui.combo("##mapping_modes", self.mapping_conditioning, self.mapping_conditioning_modes)

            label("Latent Space", width=viz.label_w)
            _, self.latent_space = imgui.combo("##latent_space", self.latent_space, self.latent_spaces)

            label("Seed", width=viz.label_w)
            _, self.seed = imgui.input_int("##seed", self.seed)

        viz.args.truncation_psi = self.truncation_psi
        viz.args.latent_x = self.latent.x
        viz.args.latent_y = self.latent.y
        viz.args.seed = self.seed
        viz.args.latent_space = self.latent_spaces[self.latent_space]
        viz.args.mapping_conditioning = self.mapping_conditioning_modes[self.mapping_conditioning]

