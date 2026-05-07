import os
from imgui_bundle import imgui

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils import easy_json
from imgui_bundle._imgui_bundle import portable_file_dialogs
from widgets.widget import Widget


class LoadWidget(Widget):
    _settings_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".splatviz_settings.json"))
    _scene_extensions = (".ply", ".yml", ".yaml")

    def __init__(self, viz):
        super().__init__(viz, "Load")
        self.plys: list[str] = [""]
        self.use_splitscreen = False
        self.highlight_border = False
        self.last_folder = self._load_last_folder()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        self._handle_dropped_files()
        if show:
            plys_to_remove = []
            for i, ply in enumerate(self.plys):
                if imgui_utils.button(f"Browse {i + 1}", width=viz.button_w):
                    files_from_dialog = self._open_ply_dialog()
                    if len(files_from_dialog) > 0:
                        self.plys[i] = files_from_dialog[0]
                imgui.same_line()
                if len(self.plys) > 1:
                    if imgui_utils.button(f"Remove {i + 1}", width=viz.button_w):
                        plys_to_remove.append(i)
                    imgui.same_line()
                imgui.text(f"Scene {i + 1}: {ply}")

            for i in plys_to_remove[::-1]:
                self.plys.pop(i)
            if imgui_utils.button("Add Scene", width=viz.button_w):
                files_from_dialog = self._open_ply_dialog()
                if len(files_from_dialog) > 0:
                    self.plys.append(files_from_dialog[0])

            if len(self.plys) > 1:
                use_splitscreen, self.use_splitscreen = imgui.checkbox("Splitscreen", self.use_splitscreen)
                highlight_border, self.highlight_border = imgui.checkbox("Highlight Border", self.highlight_border)

        viz.args.highlight_border = self.highlight_border
        viz.args.use_splitscreen = self.use_splitscreen
        viz.args.ply_file_paths = self.plys
        viz.args.current_ply_names = [
            ply.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_") for ply in self.plys
        ]

    def _open_ply_dialog(self):
        files_from_dialog = portable_file_dialogs.open_file(
            "Select .ply or .yml",
            self.last_folder,
            filters=[],
        ).result()
        if len(files_from_dialog) > 0:
            self.last_folder = os.path.dirname(os.path.abspath(files_from_dialog[0]))
            self._save_last_folder()
        return files_from_dialog

    def _handle_dropped_files(self):
        dropped_paths = self.viz._drag_and_drop_paths
        if dropped_paths is None:
            return

        self.viz._drag_and_drop_paths = None
        scene_paths = [path for path in dropped_paths if self._is_scene_path(path)]
        if len(scene_paths) == 0:
            return

        if self.plys == [""]:
            self.plys = scene_paths
        else:
            self.plys.extend(scene_paths)
        self.last_folder = os.path.dirname(os.path.abspath(scene_paths[0]))
        self._save_last_folder()

    def _is_scene_path(self, path):
        return os.path.isfile(path) and path.lower().endswith(self._scene_extensions)

    def _load_last_folder(self):
        try:
            data = easy_json.load_json(self._settings_file)
            folder = data.get("last_ply_folder", os.getcwd())
            if os.path.isdir(folder):
                return folder
        except (OSError, ValueError, TypeError):
            pass
        return os.getcwd()

    def _save_last_folder(self):
        try:
            easy_json.save_json({"last_ply_folder": self.last_folder}, self._settings_file, indent=2)
        except OSError:
            pass
