from imgui_bundle import imgui

from splatviz_utils.gui_utils.easy_imgui import checkbox, label, slider
from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget


class FeatureWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Features")
        self.enabled = False
        self.model_index = 0
        self.dino_models = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]
        self.dino_index = 0
        self.sam_types = ["vit_b", "vit_l", "vit_h"]
        self.sam_type_index = 0
        self.sam_checkpoint = ""
        self.max_size = 518
        self.opacity = 1.0
        self.pca_stability = 0.95

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            label("Enabled", viz.label_w)
            self.enabled = checkbox(self.enabled, "features_enabled")

            label("DINO", viz.label_w)
            _changed, self.dino_index = imgui.combo("##dino_model", self.dino_index, self.dino_models)

            with imgui_utils.item_width(viz.font_size * 7):
                label("Max Size", viz.label_w)
                _changed, self.max_size = imgui.input_int("##feature_max_size", self.max_size)
                self.max_size = min(max(int(self.max_size), 64), 2048)

                label("Opacity", viz.label_w)
                self.opacity = slider(self.opacity, "feature_opacity", 0.0, 1.0, format="%.2f")

                label("PCA Stability", viz.label_w)
                self.pca_stability = slider(self.pca_stability, "feature_pca_stability", 0.0, 0.99, format="%.2f")

            if "feature_status" in viz.result:
                label("Status", viz.label_w)
                imgui.text_wrapped(viz.result.feature_status)

        viz.args.feature = {
            "enabled": self.enabled,
            "dino_model": self.dino_models[self.dino_index],
            "sam_type": self.sam_types[self.sam_type_index],
            "sam_checkpoint": self.sam_checkpoint,
            "max_size": self.max_size,
            "opacity": self.opacity,
            "pca_stability": self.pca_stability,
        }
