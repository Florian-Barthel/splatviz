import torch
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget


class RenderWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Render")
        self.render_alpha = False
        self.render_depth = False
        self.render_gan_image = False
        self.resolution = 1024
        self.background_color = torch.tensor([0.0, 0.0, 0.0])

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True, decoder=False):
        viz = self.viz
        if show:
            label("Resolution", viz.label_w)
            _changed, self.resolution = imgui.input_int("##Resolution", self.resolution, 64)

            label("Render Alpha", viz.label_w)
            alpha_changed, self.render_alpha = imgui.checkbox("##RenderAlpha", self.render_alpha)

            label("Render Depth", viz.label_w)
            depth_changed, self.render_depth = imgui.checkbox("##RenderDepth", self.render_depth)
            if decoder:
                label("Render GAN", viz.label_w)
                _, self.render_gan_image = imgui.checkbox("##RenderGAN", self.render_gan_image)

            if self.render_alpha and alpha_changed:
                self.render_depth = False
            if self.render_depth and depth_changed:
                self.render_alpha = False

            label("Background Color", viz.label_w)
            _changed, background_color = imgui.input_float3("##background_color", v=self.background_color.tolist(), format="%.1f")
            if _changed:
                self.background_color = torch.tensor(background_color)

        viz.args.background_color = self.background_color
        viz.args.resolution = self.resolution
        viz.args.render_alpha = self.render_alpha
        viz.args.render_depth = self.render_depth
        viz.args.render_gan_image = self.render_gan_image
