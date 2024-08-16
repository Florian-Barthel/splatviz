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
        self.current_attr = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            prune_changed, self.prune_grid = imgui.checkbox("Prune 2d grid", self.prune_grid)
            viz.args.prune_grid = self.prune_grid

            attr_changed, self.current_attr = imgui.combo(
                "Grid Attribute", self.current_attr, self.attr_names
            )

            viz.args.grid_attr = self.attr_names[self.current_attr]

            if "2D Grid" in viz.result.keys():
                im_params = immvision.ImageParams()
                im_params.image_display_size = (512, 512)

                grid_img = viz.result["2D Grid"]

                immvision.image("2D Grid", grid_img, params=im_params)