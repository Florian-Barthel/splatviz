from imgui_bundle import imgui
import numpy as np

from splatviz_utils.gui_utils import imgui_utils
from splatviz_utils.gui_utils.easy_imgui import label
from splatviz_utils.cam_utils import LookAtPoseSampler
from widgets.widget import Widget
from scene.cameras import CustomCam

import datetime
import os
import imageio


class VideoRecordingWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Video Recording")
        self.is_recording = False
        self.cur_frames_recording = 0
        self.save_path = "videos"



    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):

        viz = self.viz
        viz.args.video_cams = []

        if show:
            
            label("Save Path", viz.label_w)
            _changed, self.save_path = imgui.input_text("##save_path", self.save_path, 100)

            if self.is_recording:
                
                imgui.text("Recording frames: {}".format(self.cur_frames_recording))
                imgui.text("Video saving to: {}".format(self.video_path))
                
                if imgui_utils.button("Stop Recording", viz.button_w):
                    self.stop_recording()
            else:
                if imgui_utils.button("Start Recording", viz.button_w):
                    self.start_recording()

            if self.is_recording:
                self.record(viz.result.image)
    
    def start_recording(self):
        self.is_recording = True

        self.cur_frames_recording = 0

        video_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp4")

        self.video_path = os.path.abspath(os.path.join(self.save_path, video_name))
        os.makedirs(self.save_path, exist_ok=True)
        self.video = imageio.get_writer(self.video_path, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)

    def record(self, image):
        self.video.append_data(image)
        self.cur_frames_recording += 1

    def stop_recording(self):
        self.video.close()
        self.is_recording = False


    def close(self):
        if self.is_recording:
            self.stop_recording()
        
            
