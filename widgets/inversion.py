from imgui_bundle import imgui, ImVec2
from imgui_bundle import portable_file_dialogs, immvision
import numpy as np
from PIL import Image
from imgui_bundle._imgui_bundle import implot

from splatviz_utils.dict_utils import EasyDict
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget


class Hyperparams:
    def __init__(self, name, default_value):
        self.name = name
        self.default_value = default_value



class InversionWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Inversion")
        self.loaded_images = []
        self.run_inversion = False
        self.run_pti = False
        self.inversion_step = 0
        self.pti_step = 0
        self.inversion_total_steps = 1000
        self.pti_total_steps = 500
        self.loss_settings = EasyDict(
            id_loss=Hyperparams("ID Similarity", 0.1),
            mse_loss=Hyperparams("MSE", 0.01),
            ssim_loss=Hyperparams("SSIM", 0.1),
            lpips_loss=Hyperparams("LPIPS", 1.0),
            learning_rate=Hyperparams("Learning Rate", 0.01),
            lr_decay=Hyperparams("Learning Rate Decay", 0.99),
        )

        self.plots = EasyDict(
            loss=dict(values=[], dtype=float),
        )
        self.steps = []


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text("1. Load Images\n")
            if imgui.button("Open Image", ImVec2(viz.label_w_large, 0)):
                files = portable_file_dialogs.open_file("Select Image", "./", filters=[], options=portable_file_dialogs.opt.multiselect).result()
                if len(files) > 0:
                    self.loaded_images = []
                    for file in files:
                        self.loaded_images.append(np.array(Image.open(file).convert("RGB")))

            im_size = self.viz.pane_w // (len(self.loaded_images) + 1)
            for i, image in enumerate(self.loaded_images):
                immvision.image_display_resizable(f"image_{i}", image, ImVec2(im_size, im_size), is_bgr_or_bgra=False, refresh_image=False)
                imgui.same_line(spacing=0)

            imgui.new_line()
            imgui.text("\n2. Image Preprocessing")
            if imgui.button("Preprocess", ImVec2(viz.label_w_large, 0)):
                pass

            imgui.text("\n3. Configure Loss")
            for key in self.loss_settings.keys():
                label(self.loss_settings[key].name, width=viz.label_w_large)
                changed, self.loss_settings[key].default_value = imgui.input_float(f"##{key}", self.loss_settings[key].default_value)

            imgui.text("\n4. Run Inversion")

            imgui.push_item_width(viz.label_w_large)
            changed, self.inversion_total_steps = imgui.input_int("##inversion_total_steps", self.inversion_total_steps, step=100)
            imgui.same_line()
            if imgui.button("Run Inversion", ImVec2(viz.label_w_large, 0)):
                self.run_inversion = True
            imgui.same_line()
            imgui.progress_bar(self.inversion_step / self.inversion_total_steps, overlay=f"{self.inversion_step} / {self.inversion_total_steps} Steps")

            imgui.push_item_width(viz.label_w_large)
            changed, self.pti_total_steps = imgui.input_int("##pti_total_steps", self.pti_total_steps, step=100)
            imgui.same_line()
            if imgui.button("Run Generator Tuning", ImVec2(viz.label_w_large, 0)):
                self.run_pti = True
            imgui.same_line()
            imgui.progress_bar(self.pti_step / self.pti_total_steps, overlay=f"{self.pti_step} / {self.pti_total_steps} Steps")

            self.steps.append(self.inversion_step)

            if self.run_inversion:
                self.inversion_step += 1
                self.plots.loss["values"].append(np.random.randn(1)[0])
            if self.run_pti:
                self.pti_step += 1

            for plot_name, plot_values in self.plots.items():
                implot.set_next_axes_to_fit()
                if implot.begin_plot(plot_name, imgui.ImVec2(viz.pane_w - 200, viz.pane_w // 4)):
                    implot.plot_line(
                        plot_name,
                        ys=np.array(plot_values["values"], dtype=plot_values["dtype"]),
                        xs=np.array(self.steps, dtype=plot_values["dtype"]),
                    )
                    implot.end_plot()

