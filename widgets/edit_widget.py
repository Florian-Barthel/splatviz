import os.path
import time
import uuid
from imgui_bundle import imgui, imgui_color_text_edit as edit
import inspect

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.gui_utils.easy_json import load_json, save_json
from splatviz_utils.dict_utils import EasyDict
from scene.cameras import CustomCam
from renderer.gaussian_renderer import GaussianRenderer
from scene.gaussian_model import GaussianModel
from widgets.widget import Widget

default_preset = """gaussian._xyz = gaussian._xyz
gaussian._rotation = gaussian._rotation
gaussian._scaling = gaussian._scaling
gaussian._opacity = gaussian._opacity
gaussian._features_dc = gaussian._features_dc
gaussian._features_rest = gaussian._features_rest
self.bg_color[:] = 0
"""


def get_description(obj):
    attr_list = sorted(inspect.getmembers(obj), key=lambda x: x[0])
    res_string = str(obj.__name__) + "\n"
    for attr in attr_list:
        if attr[0].startswith("__"):
            continue
        res_string += "\t" + attr[0] + "\n"
    return res_string


class Slider(object):
    def __init__(self, key, value, min_value, max_value, _id=None):
        if _id is None:
            _id = str(uuid.uuid4())
        self.key = key
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self._id = _id

    def render(self, viz):
        _changed, self.value = imgui.slider_float(
            f"##slider-{self.key}-{self._id}",
            self.value,
            self.min_value,
            self.max_value,
        )
        with imgui_utils.item_width(viz.font_size * 4):
            imgui.same_line()
            min_changed, self.min_value = imgui.input_float(f"##min-{self._id}", self.min_value, )
            imgui.same_line()
            max_changed, self.max_value = imgui.input_float(f"##max-{self._id}", self.max_value)
            imgui.same_line()
            text_changed, self.key = imgui.input_text(f"##text-{self._id}", self.key)
        if min_changed or max_changed:
            self.value = min(self.value, self.max_value)
            self.value = max(self.value, self.min_value)


class EditWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Edit")
        cur_time = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        self.current_session_name = f"Restore Session {cur_time}"
        self.presets = {}
        self.history = {}
        self.history_size = 5
        self.safe_load = False
        self.preset_path = "./presets.json"
        self.history_path = "./history.json"
        self.load_presets()

        self.editor = edit.TextEditor()
        self.setup_editor()

        self.last_text = ""
        self.sliders = [Slider(**dict_values) for dict_values in self.presets["Default"]["slider"]]

        self.var_names = "xyzijklmnuvwabcdefghopqrst"
        self.var_name_index = 1
        self._cur_min_slider = -10
        self._cur_max_slider = 10
        self._cur_val_slider = 0
        self._cur_name_slider = self.var_names[self.var_name_index]
        self._cur_preset_name = ""

    def setup_editor(self):
        language = edit.TextEditor.LanguageDefinition.python()
        custom_identifiers = {
            "self": edit.TextEditor.Identifier(m_declaration=get_description(GaussianRenderer)),
            "gs": edit.TextEditor.Identifier(m_declaration=get_description(GaussianModel)),
            "render_cam": edit.TextEditor.Identifier(m_declaration=get_description(CustomCam)),
            "render": edit.TextEditor.Identifier(
                m_declaration=get_description(
                    EasyDict(render=0, viewspace_points=0, visibility_filter=0, radii=0, alpha=0, depth=0)
                )
            ),
            "slider": edit.TextEditor.Identifier(m_declaration=get_description(Slider)),
        }
        copy_identifiers = language.m_identifiers.copy()
        copy_identifiers.update(custom_identifiers)
        language.m_identifiers = copy_identifiers
        self.editor.set_language_definition(language)
        self.editor.set_text(self.presets["Default"]["edit_text"])

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            self.render_sliders()
            imgui.new_line()

            _changed, self.safe_load = imgui.checkbox("Safe Load", self.safe_load)

            if imgui_utils.button("Browse Presets", width=self.viz.button_large_w):
                imgui.open_popup("browse_presets")
            if imgui.begin_popup("browse_presets"):
                for preset_key in sorted(self.presets.keys()):
                    clicked = imgui.menu_item_simple(preset_key)
                    if clicked:
                        edit_text = self.presets[preset_key]["edit_text"]
                        self.sliders = [Slider(**dict_values) for dict_values in self.presets[preset_key]["slider"]]

                        if self.safe_load:
                            edit_text = f"''' # REMOVE THIS LINE\n{edit_text}\n''' # REMOVE THIS LINE"
                        self.editor.set_text(edit_text)
                imgui.end_popup()

            imgui.same_line(viz.button_large_w * 2)
            if imgui_utils.button("Browse History", width=self.viz.button_large_w):
                imgui.open_popup("browse_history")
            if imgui.begin_popup("browse_history"):
                for history_key in sorted(self.history.keys()):
                    name = "Current Session" if history_key == self.current_session_name else history_key
                    clicked = imgui.menu_item_simple(name)
                    if clicked:
                        edit_text = self.history[history_key]["edit_text"]
                        self.sliders = [Slider(**dict_values) for dict_values in self.history[history_key]["slider"]]
                        if self.safe_load:
                            edit_text = f"''' # REMOVE THIS LINE\n{edit_text}\n''' # REMOVE THIS LINE"
                        self.editor.set_text(edit_text)
                imgui.end_popup()

            increase_font_size = 5
            with imgui_utils.change_font(self.viz._imgui_fonts_code[self.viz._cur_font_size + increase_font_size]):
                line_height = self.editor.get_total_lines() * (self.viz._cur_font_size + increase_font_size)
                max_height = (self.viz._cur_font_size + increase_font_size) * 30
                editor_height = min(line_height, max_height)
                self.editor.render("Python Edit Code", a_size=imgui.ImVec2(viz.pane_w - 50, editor_height))

            imgui.new_line()
            label("Preset Name")
            _, self._cur_preset_name = imgui.input_text("##preset_name", self._cur_preset_name)
            imgui.same_line()
            if imgui_utils.button("Save as Preset", width=self.viz.button_large_w):
                self.presets[self._cur_preset_name] = dict(
                    edit_text=self.editor.get_text(), slider=[vars(slider) for slider in self.sliders]
                )
                save_json(filename=self.preset_path, data=self.presets, indent=2)
                self._cur_preset_name = ""

            edit_text = self.editor.get_text()
            if self.last_text != edit_text:
                self.history[self.current_session_name] = dict(
                    edit_text=self.editor.get_text(), slider=[vars(slider) for slider in self.sliders]
                )
                save_json(filename=self.history_path, data=self.history)
            self.last_text = edit_text

        viz.args.edit_text = self.last_text
        viz.args.slider = {slider.key: slider.value for slider in self.sliders}

    def load_presets(self):
        if not os.path.exists(self.preset_path):
            save_dict = dict(Default=dict(edit_text=default_preset, slider=[vars(Slider("x", 1, 0, 10))]))
            save_json(data=save_dict, filename=self.preset_path)

        self.presets = load_json(self.preset_path)

        if os.path.exists(self.history_path):
            history_all = load_json(self.history_path)
            keys = sorted(history_all.keys())
            num_keep = min(len(keys), self.history_size)
            keys = keys[-num_keep:]
            self.history = {key: history_all[key] for key in keys}

    def render_sliders(self):
        delete_keys = []
        for i, slider in enumerate(self.sliders):
            slider.render(self.viz)
            imgui.same_line()
            if imgui_utils.button(f"Remove##{slider._id}"):
                delete_keys.append(i)

        for i in delete_keys[::-1]:
            del self.sliders[i]

        imgui.push_item_width(70)
        label("Var name")
        _, self._cur_name_slider = imgui.input_text("##input_name", self._cur_name_slider)

        imgui.same_line()
        label("min")
        _, self._cur_min_slider = imgui.input_int("##input_min", self._cur_min_slider, 0)

        imgui.same_line()
        label("val")
        _, self._cur_val_slider = imgui.input_int("##input_val", self._cur_val_slider, 0)

        imgui.same_line()
        label("max")
        _, self._cur_max_slider = imgui.input_int("##input_max", self._cur_max_slider, 0)
        imgui.pop_item_width()

        imgui.same_line()
        if imgui_utils.button("Add Slider", width=self.viz.button_w):
            self.sliders.append(
                Slider(
                    key=self._cur_name_slider,
                    value=self._cur_val_slider,
                    min_value=self._cur_min_slider,
                    max_value=self._cur_max_slider,
                )
            )
            self.var_name_index += 1
            self._cur_name_slider = self.var_names[self.var_name_index % len(self.var_names)]
