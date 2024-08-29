from imgui_bundle import imgui, immvision
from gui_utils import imgui_utils
import numpy as np
from widgets.widget import Widget


class CompressionWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Compression")
        self.viz = viz
        self.prune_grid = False
        self.attr_names = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_opacity",
            "_rotation_rgba",
            "_rotation_euler_form_rgba",
            "_rotation_euler_angles_rgb"
        ]
        self.current_attr = 0
        self.current_sh_index = 1

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            if "grid_image" not in viz.result.keys(): 
                # show image indicating no singal incoming
                return

            attr_changed, self.current_attr = imgui.combo(
                "Grid attribute",
                self.current_attr,
                self.attr_names,
            )

            if self.attr_names[self.current_attr] == "_features_rest":
                changed, self.current_sh_index = imgui.slider_int(
                    "SH coefficient index", self.current_sh_index, 1, 15
                )
                viz.args.grid_sh_index = self.current_sh_index - 1
            viz.args.grid_attr = self.attr_names[self.current_attr]

            im_params = immvision.ImageParams()
            grid_img = viz.result["grid_image"]
            im_params.image_display_size = (grid_img.shape[0], grid_img.shape[1])
            immvision.image("2D Grid", grid_img, params=im_params)
