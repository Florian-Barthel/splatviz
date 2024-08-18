import os
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget


class LoadWidget(Widget):
    def __init__(self, viz, root):
        super().__init__(viz, "Load")
        self.root = root
        self.filter = ""
        self.items = self.list_runs_and_pkls()
        if len(self.items) == 0:
            raise FileNotFoundError(f"No .ply or compression_config.yml found in '{root}' with filter 'f{self.filter}'")
        self.plys: list[str] = [self.items[0]]
        self.use_splitscreen = False
        self.highlight_border = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Search Filters (comma separated)")
            _changed, self.filter = imgui.input_text("##Filter", self.filter)
            plys_to_remove = []

            for i, ply in enumerate(self.plys):
                if imgui.begin_popup(f"browse_pkls_popup{i}"):
                    for item in self.items:
                        clicked = imgui.menu_item_simple(os.path.relpath(item, self.root))
                        if clicked:
                            self.plys[i] = item
                    imgui.end_popup()

                if imgui_utils.button(f"Browse {i + 1}", width=viz.button_w):
                    imgui.open_popup(f"browse_pkls_popup{i}")
                    self.items = self.list_runs_and_pkls()
                imgui.same_line()
                if i > 0:
                    if imgui_utils.button(f"Remove {i + 1}", width=viz.button_w):
                        plys_to_remove.append(i)
                    imgui.same_line()
                imgui.text(f"Scene {i + 1}: " + ply[len(self.root) :])

            for i in plys_to_remove[::-1]:
                self.plys.pop(i)
            if imgui_utils.button("Add Scene", width=viz.button_w):
                self.plys.append(self.plys[-1])

            use_splitscreen, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
            highlight_border, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)

        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = self.plys
        viz.args.current_ply_names = [
            ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_") for ply in self.plys
        ]

    def list_runs_and_pkls(self) -> list[str]:
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".ply") or file.endswith("compression_config.yml"):
                    current_path = os.path.join(root, file)
                    if all([filter in current_path for filter in self.filter.split(",")]):
                        self.items.append(str(current_path))
        return sorted(self.items)
