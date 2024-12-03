import numpy as np
import torch
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.dict_utils import EasyDict
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")
        self.latent = EasyDict(x=0, y=0)
        self.truncation_psi = 1.0
        self.c_gen_conditioning_zero = True
        self.render_seg = False

        self.cond_hat = False
        self.cond_glasses = False
        self.cond_microphone = False
        self.cond_hand = False
        self.cond_multi_id = False
        self.cond_misc_group = False


    def drag(self, dx, dy):
        self.latent.x += dx / 1000
        self.latent.y += dy / 1000

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

            label("Truncation PSI")
            _changed, self.truncation_psi = imgui.slider_float(
                "##truncation", self.truncation_psi, 0.0, 1.0
            )

            label("c_gen_conditioning_zero")
            _changed, self.c_gen_conditioning_zero = imgui.checkbox("##c_gen_conditioning_zero", self.c_gen_conditioning_zero)

            label("render segmentation")
            _changed, self.render_seg = imgui.checkbox("##render_segmentation", self.render_seg)


            # conditional input
            label("cond hat")
            _changed, self.cond_hat = imgui.checkbox("##cond_hat", self.cond_hat)

            label("cond glasses")
            _changed, self.cond_glasses = imgui.checkbox("##cond_glasses", self.cond_glasses)

            label("cond microphone")
            _changed, self.cond_microphone = imgui.checkbox("##cond_microphone", self.cond_microphone)

            label("cond hand")
            _changed, self.cond_hand = imgui.checkbox("##cond_hand", self.cond_hand)

            label("cond multi_id")
            _changed, self.cond_multi_id = imgui.checkbox("##cond_multi_id", self.cond_multi_id)

            label("cond misc_group")
            _changed, self.cond_misc_group = imgui.checkbox("##cond_misc_group", self.cond_misc_group)

        viz.args.conditional_vector = torch.tensor([[
            self.cond_hat,
            self.cond_glasses,
            self.cond_microphone,
            self.cond_hand,
            self.cond_multi_id,
            self.cond_misc_group,
        ]]).int().to("cuda")
        viz.args.truncation_psi = self.truncation_psi
        viz.args.latent_x = self.latent.x
        viz.args.latent_y = self.latent.y
        viz.args.c_gen_conditioning_zero = self.c_gen_conditioning_zero
        viz.args.render_seg = self.render_seg

