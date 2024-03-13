import os
import re
import PIL
import imgui
import numpy as np

from gui_utils import imgui_utils
from viz import renderer


class CaptureWidget:
    def __init__(self, viz):
        self.viz = viz
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_screenshots"))
        self.dump_image = False
        self.dump_gui = False
        self.defer_frames = 0
        self.disabled_time = 0

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Capture')
                imgui.same_line(viz.label_w)
                _changed, self.path = imgui_utils.input_text('##path', self.path, 1024,
                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                    width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                    help_text='PATH')
                if imgui.is_item_hovered() and not imgui.is_item_active() and self.path != '':
                    imgui.set_tooltip(self.path)
                imgui.same_line()
                if imgui_utils.button('Save image', width=viz.button_w, enabled=(self.disabled_time == 0 and 'image' in viz.result)):
                    if 'image' in viz.result:
                        self.save_png(viz.result.image)

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
        except:
            viz.result.error = renderer.CapturedException()
