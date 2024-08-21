import json
import socket
import time
from threading import Thread

from matplotlib.pyplot import grid
import numpy as np
import torch
import torch.nn

from scene.cameras import CustomCam
from viz_renderer.base_renderer import Renderer
from viz_utils.dict import EasyDict


class AsyncConnector(Thread):
    def __init__(self, delay, host, port):
        super(AsyncConnector, self).__init__()
        self.delay = delay
        self.host = host
        self.port = port
        self._socket = None
        self.socket = None
        self.finished = False
        self.start()

    def run(self):
        while self.socket is None:
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.connect((self.host, self.port))
                self.socket = self._socket
                self.finished = True
                return
            except Exception as e:
                self._socket = None
                self.socket = None
                time.sleep(self.delay)

    def restart(self):
        self._socket = None
        self.socket = None
        self.run()


class AttachRenderer(Renderer):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.connector = AsyncConnector(1, host, port)
        self.socket = self.connector.socket
        self.next_bytes = bytes()

    def restart_connector(self):
        self.connector = AsyncConnector(1, self.host, self.port)

    def read(self, resolution):
        try:
            current_bytes = 0
            expected_bytes = resolution * resolution * 3
            try_counter = 10
            counter = 0
            message = bytes()
            while current_bytes < expected_bytes:
                message += self.socket.recv(expected_bytes - current_bytes)
                current_bytes = len(message)
                counter += 1
                if counter > try_counter:
                    print("Package loss")
                    break
            
            verify_len = self.socket.recv(4)
            verify_len = int.from_bytes(verify_len, "little")
            verify_data = self.socket.recv(verify_len)
            
            grid_image_nbytes = int.from_bytes(self.socket.recv(4), "little")
            if grid_image_nbytes != 0:
                grid_side_len = int.from_bytes(self.socket.recv(4), "little")
                grid_channels = int.from_bytes(self.socket.recv(4), "little")
                grid_data = bytes()
                while len(grid_data) < grid_image_nbytes:
                    grid_data += self.socket.recv(grid_image_nbytes - len(grid_data))
                grid_image = np.frombuffer(grid_data, dtype=np.uint8).copy() # copy to make it writable (needed by immvision)
                grid_image = grid_image.reshape(grid_side_len, grid_side_len, grid_channels)
            else:
                grid_image = np.zeros([grid_image_nbytes, grid_image_nbytes, 3], dtype=np.uint8)

            try:
                verify_dict = json.loads(verify_data)
            except Exception:
                verify_dict = {}

            image = np.frombuffer(message, dtype=np.uint8).reshape(resolution, resolution, 3)
            image = torch.from_numpy(image) / 255.0
            image = image.permute(2, 0, 1)

            return image, verify_dict, grid_image
        except Exception as e:
            print("Read Error", e)
            self.restart_connector()
            return torch.zeros([3, resolution, resolution]), {}

    def send(self, message):
        try:
            message_encode = json.dumps(message).encode()
            message_len_bytes = len(message_encode).to_bytes(4, 'little')
            self.socket.sendall(message_len_bytes + bytes(message_encode))
        except Exception as e:
            self.restart_connector()
            print("Send Error", e)

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        resolution,
        cam_params,
        do_training,
        stop_at_value=-1,
        single_training_step=False,
        slider={},
        img_normalize=False,
        save_ply_path=None,
        grid_attr = None,
        **other_args,
    ):
        self.socket = self.connector.socket
        if self.socket is None:
            if self.connector.finished:
                self.restart_connector()
            res.message = f"Waiting for connection\n{self.host}:{self.port}"
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
            "train": do_training,
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
            "slider": slider,
            "single_training_step": single_training_step,
            "stop_at_value": stop_at_value,
            "grid_attr": grid_attr
        }
        self.send(message)
        image, stats, grid_image = self.read(resolution)
        res["2D Grid"] = grid_image
        if len(stats.keys()) > 0:
            res.training_stats = stats
            res.error = res.training_stats["error"]
        self._return_image(
            image,
            res,
            normalize=img_normalize,
        )
