import os
from imgui_bundle import imgui
from gui_utils import imgui_utils


class LoadWidget:
    def __init__(self, viz, root):
        self.viz = viz
        self.root = root
        self.filter = ""
        self.items = self.list_runs_and_pkls()
        if len(self.items) == 0:
            raise FileNotFoundError(f"No .pkl found in '{root}' with filter 'f{self.filter}'")
        self.ply = self.items[0]
        self.use_splitscreen = False
        self.highlight_border = False
        self.on_top = False

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
            _, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
            _, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)
            _, self.on_top = imgui.checkbox("On Top", self.on_top)

        viz.args.on_top = self.on_top
        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = [self.ply]
        viz.args.current_ply_names = self.ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")

    def list_runs_and_pkls(self) -> list[str]:
        self.items = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    current_path = os.path.join(root, file)
                    if all([filter in current_path for filter in self.filter.split(",")]):
                        self.items.append(str(current_path))
        return sorted(self.items)
