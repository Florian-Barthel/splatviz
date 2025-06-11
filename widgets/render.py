import torch
from imgui_bundle import imgui
import cv2
from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from widgets.widget import Widget

colormaps = [
    ("NONE", None),
    ("COLORMAP_AUTUMN", cv2.COLORMAP_AUTUMN),
    ("COLORMAP_BONE", cv2.COLORMAP_BONE),
    ("COLORMAP_JET", cv2.COLORMAP_JET),
    ("COLORMAP_WINTER", cv2.COLORMAP_WINTER),
    ("COLORMAP_RAINBOW", cv2.COLORMAP_RAINBOW),
    ("COLORMAP_OCEAN", cv2.COLORMAP_OCEAN),
    ("COLORMAP_SUMMER", cv2.COLORMAP_SUMMER),
    ("COLORMAP_SPRING", cv2.COLORMAP_SPRING),
    ("COLORMAP_COOL", cv2.COLORMAP_COOL),
    ("COLORMAP_HSV", cv2.COLORMAP_HSV),
    ("COLORMAP_PINK", cv2.COLORMAP_PINK),
    ("COLORMAP_HOT", cv2.COLORMAP_HOT),
    ("COLORMAP_PARULA", cv2.COLORMAP_PARULA),
    ("COLORMAP_MAGMA", cv2.COLORMAP_MAGMA),
    ("COLORMAP_INFERNO", cv2.COLORMAP_INFERNO),
    ("COLORMAP_PLASMA", cv2.COLORMAP_PLASMA),
    ("COLORMAP_VIRIDIS", cv2.COLORMAP_VIRIDIS),
    ("COLORMAP_CIVIDIS", cv2.COLORMAP_CIVIDIS),
    ("COLORMAP_TWILIGHT", cv2.COLORMAP_TWILIGHT),
    ("COLORMAP_TWILIGHT_SHIFTED", cv2.COLORMAP_TWILIGHT_SHIFTED),
    ("COLORMAP_TURBO", cv2.COLORMAP_TURBO),
    ("COLORMAP_DEEPGREEN", cv2.COLORMAP_DEEPGREEN),
]

class RenderWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Render")
        self.render_alpha = False
        self.render_depth = False
        self.render_gan_image = False
        self.resolution = 1024
        self.background_color = torch.tensor([1.0, 1.0, 1.0])
        self.img_normalize = False
        self.current_colormap = 0
        self.colormap_dict = dict(colormaps)
        self.colormaps_names = [key for key, _ in colormaps]
        self.invert = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True, decoder=False):
        viz = self.viz
        if show:
            label("Resolution", viz.label_w)
            _changed, self.resolution = imgui.input_int("##Resolution", self.resolution, 64)

            label("Background Color", viz.label_w)
            _changed, background_color = imgui.color_edit3("##background_color_edit", self.background_color.tolist())
            if _changed:
                self.background_color = torch.tensor(background_color)

            label("Normalize", viz.label_w)
            _changed, self.img_normalize = imgui.checkbox("##Normalize", self.img_normalize)

            label("Render Alpha", viz.label_w)
            alpha_changed, self.render_alpha = imgui.checkbox("##RenderAlpha", self.render_alpha)

            label("Render Depth", viz.label_w)
            depth_changed, self.render_depth = imgui.checkbox("##RenderDepth", self.render_depth)

            label("Invert Colors", viz.label_w)
            depth_changed, self.invert = imgui.checkbox("##invert", self.invert)

            label("Colormap", viz.label_w)
            _, self.current_colormap = imgui.combo("##colormap", self.current_colormap, self.colormaps_names)

            if self.render_alpha and alpha_changed:
                self.render_depth = False
            if self.render_depth and depth_changed:
                self.render_alpha = False

        viz.args.background_color = self.background_color
        viz.args.resolution = self.resolution
        viz.args.render_alpha = self.render_alpha
        viz.args.render_depth = self.render_depth
        viz.args.colormap = self.colormap_dict[self.colormaps_names[self.current_colormap]]
        viz.args.invert = self.invert
        viz.args.img_normalize = self.img_normalize
