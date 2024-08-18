from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.dict_utils import EasyDict
from widgets.widget import Widget


class LatentWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Latent")
        self.latent = EasyDict(x=0, y=0)

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

        viz.args.latent_x = self.latent.x
        viz.args.latent_y = self.latent.y
