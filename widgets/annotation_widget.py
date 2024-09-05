import torch
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget


class AnnotationWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Annotation")
        self.create_segmentation = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True, decoder=False):
        viz = self.viz
        if show:
            label("Create Segmentation", viz.label_w)
            _changed, self.create_segmentation = imgui.checkbox("##create_segmentation", self.create_segmentation)

        viz.args.create_segmentation = self.create_segmentation
