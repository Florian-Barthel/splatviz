import os
import re
import traceback
import PIL
from imgui_bundle import imgui
import numpy as np

from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget


class CaptureWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Save")
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_screenshots"))
        self.path_ply = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_ply_files"))

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Save Screenshot", viz.label_w)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != "":
                imgui.set_tooltip(self.path)
            if imgui_utils.button("Save img", width=viz.button_w):
                if "image" in viz.result:
                    self.save_png(viz.result.image)

            label("Save PLY", viz.label_w)
            if imgui_utils.button("Save ply", width=viz.button_w):
                viz.args.save_ply_path = self.path_ply
            else:
                viz.args.save_ply_path = None

    def save_png(self, image):
        viz = self.viz
        try:
            _height, _width, channels = image.shape
            assert channels in [1, 3]
            assert image.dtype == np.uint8
            os.makedirs(self.path, exist_ok=True)
            file_id = 0
            for entry in os.scandir(self.path):
                if entry.is_file():
                    match = re.fullmatch(r"(\d+).*", entry.name)
                    if match:
                        file_id = max(file_id, int(match.group(1)) + 1)
            if channels == 1:
                pil_image = PIL.Image.fromarray(image[:, :, 0], "L")
            else:
                pil_image = PIL.Image.fromarray(image, "RGB")
            pil_image.save(os.path.join(self.path, f"{file_id:05d}.png"))
        except Exception as e:
            viz.result.error = traceback.format_exception(e)
            viz.result.error += str(e)
