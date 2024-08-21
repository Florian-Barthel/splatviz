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
            "_rotation",
            "_opacity",
        ]
        self.current_attr = 1

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            if "grid_image" not in viz.result.keys(): 
                # show image indicating no singal incoming
                return

            prune_changed, self.prune_grid = imgui.checkbox("Prune 2d grid", self.prune_grid)
            viz.args.prune_grid = self.prune_grid

            attr_names = list(map(lambda x: x[1:], self.attr_names))
            attr_changed, self.current_attr = imgui.combo(
                "Grid Attribute" if viz.result["attributes_activated"] else "Grid Attribute (preactivated)",
                self.current_attr,
                attr_names,
            )

            viz.args.grid_attr = self.attr_names[self.current_attr]

            im_params = immvision.ImageParams()

            grid_img = viz.result["grid_image"]
            im_params.image_display_size = (grid_img.shape[0], grid_img.shape[1])

            immvision.image("2D Grid", grid_img, params=im_params)
