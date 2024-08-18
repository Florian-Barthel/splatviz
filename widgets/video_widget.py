from imgui_bundle import imgui
import numpy as np

from gui_utils import imgui_utils
from gui_utils.easy_imgui import label
from scene.cameras import CustomCam
from viz_utils.camera_utils import LookAtPoseSampler
from widgets.widget import Widget


class VideoWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Video")
        self.num_frames = 1000
        self.cam_height = 0.3
        self.radius = 6
        self.resolution = 1024
        self.fov = 40

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        viz.args.video_cams = []
        if show:
            label("Num Frames", viz.label_w)
            _changed, self.num_frames = imgui.input_int("##num_frames", self.num_frames)

            label("Camera Height", viz.label_w)
            _changed, self.cam_height = imgui.input_float("##camera_height", self.cam_height)

            label("Radius", viz.label_w)
            _changed, self.radius = imgui.input_float("##radius", self.radius)

            label("Resolution", viz.label_w)
            _changed, self.resolution = imgui.input_int("##resolution", self.resolution)

            label("FOV", viz.label_w)
            _changed, self.fov = imgui.input_int("##fov", self.fov)

            if imgui_utils.button("Render", viz.button_w):
                xs = np.linspace(0, 2 * np.pi, self.num_frames, endpoint=False)
                for x in xs:
                    extrinsic = LookAtPoseSampler.sample(
                        horizontal_mean=x,
                        vertical_mean=np.pi / 2 + self.cam_height,
                        lookat_position=self.viz.args.lookat_point,
                        radius=self.radius,
                        up_vector=self.viz.args.up_vector,
                    )
                    viz.args.video_cams.append(
                        CustomCam(
                            width=self.resolution,
                            height=self.resolution,
                            fovy=self.fov / 360 * 2 * np.pi,
                            fovx=self.fov / 360 * 2 * np.pi,
                            znear=0.01,
                            zfar=100,
                            extr=extrinsic[0].to("cuda"),
                        )
                    )
