import json
import socket
import time

import numpy as np
import torch
import torch.nn

from scene.cameras import CustomCam
from viz_renderer.base_renderer import Renderer
from viz_utils.dict import EasyDict


class AttachRenderer(Renderer):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = None
        self.try_connect()

    def try_connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
        except Exception as e:
            self.socket = None
            print(e)

    def read(self, resolution):
        try:
            current_bytes = 0
            expected_bytes = resolution * resolution * 3
            while current_bytes < expected_bytes:
                message = self.socket.recv(expected_bytes - current_bytes)
                current_bytes = len(message)

            verify_len = self.socket.recv(4)
            verify_len = int.from_bytes(verify_len, "little")
            verify_data = self.socket.recv(verify_len)
            try:
                verify_dict = json.loads(verify_data)
            except Exception:
                verify_dict = {}

            image = np.frombuffer(message, dtype=np.uint8).reshape(resolution, resolution, 3)
            image = torch.from_numpy(image) / 255.0
            image = image.permute(2, 0, 1)
            return image, verify_dict
        except Exception as e:
            print(e)
            return torch.zeros([3, resolution, resolution]), {}

    def send(self, message):
        try:
            message_encode = json.dumps(message).encode()
            message_len_bytes = len(message_encode).to_bytes(4, 'little')
            self.socket.sendall(message_len_bytes + bytes(message_encode))
        except Exception as e:
            self.socket = None
            print(e)

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        resolution,
        cam_params,
        slider={},
        img_normalize=False,
        save_ply_path=None,
        **other_args,
    ):
        if self.socket is None:
            self.try_connect()
            if self.socket is None:
                time.sleep(1)
                return

        # slider = EasyDict(slider)
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(
            resolution,
            resolution,
            fovy=fov_rad,
            fovx=fov_rad,
            znear=0.01,
            zfar=10,
            extr=cam_params,
        )

        # Invert all operations from network_gui.py
        world_view_transform = render_cam.world_view_transform
        world_view_transform[:, 1] = -world_view_transform[:, 1]
        world_view_transform[:, 2] = -world_view_transform[:, 2]

        full_proj_transform = render_cam.full_proj_transform
        full_proj_transform[:, 1] = -full_proj_transform[:, 1]
        message = {
            "resolution_x": resolution,
            "resolution_y": resolution,
            "train": True,
            "fov_y": fov_rad,
            "fov_x": fov_rad,
            "z_near": 0.01,
            "z_far": 10.0,
            "shs_python": False,
            "rot_scale_python": False,
            "keep_alive": True,
            "scaling_modifier": 1,
            "view_matrix": world_view_transform.cpu().numpy().flatten().tolist(),
            "view_projection_matrix": full_proj_transform.cpu().numpy().flatten().tolist(),
            "edit_text": self.sanitize_command(edit_text),
            "slider": slider
        }
        self.send(message)
        image, stats = self.read(resolution)
        if len(stats.keys()) > 0:
            res.training_stats = stats

        self._return_image(
            image,
            res,
            normalize=img_normalize,
        )

