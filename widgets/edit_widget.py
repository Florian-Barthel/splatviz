import os.path

from gui_utils import imgui_utils
import json

from imgui_bundle import imgui, imgui_color_text_edit as edit


default_preset = """gaussian._xyz = gaussian._xyz
gaussian._rotation = gaussian._rotation
gaussian._scaling = gaussian._scaling
gaussian._opacity = gaussian._opacity
gaussian._features_dc = gaussian._features_dc
gaussian._features_rest = gaussian._features_rest
self.bg_color[:] = 0
"""


class Slider:
    def __init__(self, key, value, min_value, max_value):
        self.key = key
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

    def render(self):
        _changed, self.value = imgui.slider_float(
            self.key,
            self.value,
            self.min_value,
            self.max_value,
        )


class EditWidget:
    def __init__(self, viz):
        self.viz = viz

        self.presets = {}
        self.load_presets()

        self.editor = edit.TextEditor()
        self.editor.set_language_definition(edit.TextEditor.LanguageDefinition.python())
        self.editor.set_text(self.presets["Default"])

        self.var_names = "xyzijklmnuvwabcdefghopqrst"
        self.var_name_index = 1
        self._cur_min_slider = -10
        self._cur_max_slider = 10
        self._cur_val_slider = 0
        self._cur_name_slider = self.var_names[self.var_name_index]
        self._cur_preset_name = ""

        self.sliders: list[Slider] = [Slider(key=self.var_names[0], value=5, min_value=0, max_value=10)]

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            self.render_sliders()
            imgui.new_line()

            if imgui_utils.button("Browse Presets", width=self.viz.button_large_w):
                imgui.open_popup("browse_presets")
                self.all_presets = self.presets.keys()

            if imgui.begin_popup("browse_presets"):
                for preset in self.all_presets:
                    clicked = imgui.menu_item_simple(preset)
                    if clicked:
                        self.editor.set_text(self.presets[preset])
                imgui.end_popup()

            with imgui_utils.change_font(self.viz._imgui_fonts_code[self.viz._cur_font_size]):
                line_height = self.editor.get_total_lines() * viz._cur_font_size
                max_height = viz._cur_font_size * 30
                editor_height = min(line_height, max_height)
                self.editor.render("Python Edit Code", a_size=imgui.ImVec2(viz.pane_w - 50, editor_height))

            imgui.new_line()
            imgui.text("Preset Name")
            imgui.same_line()
            _changed, self._cur_preset_name = imgui.input_text("##preset_name", self._cur_preset_name)
            imgui.same_line()
            if imgui_utils.button("Save as Preset", width=self.viz.button_large_w):
                self.presets[self._cur_preset_name] = self.editor.get_text()
                with open("./presets.json", "w", encoding="utf-8") as f:
                    json.dump(self.presets, f)
                self._cur_preset_name = ""

        viz.args.edit_text = self.editor.get_text()
        viz.args.update({slider.key: slider.value for slider in self.sliders})

    def load_presets(self):
        if not os.path.exists("./presets.json"):
            with open("./presets.json", "w", encoding="utf-8") as f:
                json.dump(dict(default=default_preset), f)

        with open("./presets.json", "r", encoding="utf-8") as f:
            self.presets = json.load(f)

    def render_sliders(self):
        delete_keys = []
        for i, slider in enumerate(self.sliders):
            slider.render()
            imgui.same_line()
            if imgui_utils.button("Remove " + slider.key, width=self.viz.button_w):
                delete_keys.append(i)

        for i in delete_keys[::-1]:
            del self.sliders[i]

        imgui.push_item_width(70)
        imgui.text("Var name")
        imgui.same_line()
        _changed, self._cur_name_slider = imgui.input_text("##input_name", self._cur_name_slider)

        imgui.same_line()
        imgui.text("min")
        imgui.same_line()
        _changed, self._cur_min_slider = imgui.input_int("##input_min", self._cur_min_slider, 0)

        imgui.same_line()
        imgui.text("val")
        imgui.same_line()
        _changed, self._cur_val_slider = imgui.input_int("##input_val", self._cur_val_slider, 0)

        imgui.same_line()
        imgui.text("max")
        imgui.same_line()
        _changed, self._cur_max_slider = imgui.input_int("##input_max", self._cur_max_slider, 0)
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
