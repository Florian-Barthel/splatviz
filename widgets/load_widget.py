import os
from imgui_bundle import imgui
from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget


class LoadWidget(Widget):
    def __init__(self, viz, root, file_ending):
        super().__init__(viz, "Load")
        self.root = root
        self.filter = ""
        self.file_ending = file_ending
        self.items = self.list_runs_and_pkls()
        if len(self.items) == 0:
            raise FileNotFoundError(f"No .pkl found in '{root}'")
        self.ply = self.items[0]

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            _changed, self.filter = imgui.input_text("Filter", self.filter)
            if imgui_utils.button("Browse", width=viz.button_w, enabled=True):
                imgui.open_popup("browse_pkls_popup")
                self.items = self.list_runs_and_pkls()

            if imgui.begin_popup("browse_pkls_popup"):
                for item in self.items:
                    clicked = imgui.menu_item_simple(os.path.relpath(item, self.root))
                    if clicked:
                        self.ply = item
                imgui.end_popup()

            imgui.same_line()
            imgui.text(self.ply)
        viz.args.ply_file_paths = [self.ply]
        viz.args.current_ply_names = self.ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")

    def list_runs_and_pkls(self) -> list[str]:
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(self.file_ending):
                    current_path = os.path.join(root, file)
                    if all([filter in current_path for filter in self.filter.split(",")]):
                        self.items.append(str(current_path))
        return sorted(self.items)
