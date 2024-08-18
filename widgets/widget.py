from splatviz_utils.gui_utils import imgui_utils


class Widget:
    def __init__(self, viz, name):
        self.viz = viz
        self.name = name

    @imgui_utils.scoped_by_object_id
    def __call__(self, show):
        raise NotImplementedError

    def close(self):
        pass
