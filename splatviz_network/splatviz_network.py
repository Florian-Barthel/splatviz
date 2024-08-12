import copy
from typing import Any

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam


class SplatvizNetwork:
    def __init__(self, host="127.0.0.1", port=6009):
        self.slider = None
        self.edit_text = None
        self.custom_cam = None
        self.scaling_modifier = None
        self.keep_alive = None
        self.do_rot_scale_python = None
        self.do_shs_python = None
        self.do_training = None
        self.host = host
        self.port = port
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)
        self.conn = None
        self.addr = None
        print(f"Creating splatviz network connector for host={host} and port={port}")
        self.stop_at_value = -1

    def try_connect(self):
        try:
            self.conn, self.addr = self.listener.accept()
            print(f"\nConnected to splatviz at {self.addr}")
            self.conn.settimeout(None)
        except Exception as inst:
            pass

    def read(self):
        messageLength = self.conn.recv(4)
        messageLength = int.from_bytes(messageLength, 'little')
        message = self.conn.recv(messageLength)
        return json.loads(message.decode("utf-8"))

    def send(self, message_bytes, training_stats):
        if message_bytes != None:
            self.conn.sendall(message_bytes)
        self.conn.sendall(len(training_stats).to_bytes(4, 'little'))
        self.conn.sendall(training_stats.encode())

    def receive(self):
        message = self.read()
        width = message["resolution_x"]
        height = message["resolution_y"]
        if width != 0 and height != 0:
            try:
                self.do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                self.do_shs_python = bool(message["shs_python"])
                self.do_rot_scale_python = bool(message["rot_scale_python"])
                self.keep_alive = bool(message["keep_alive"])
                self.scaling_modifer = message["scaling_modifier"]
                world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
                world_view_transform[:, 1] = -world_view_transform[:, 1]
                world_view_transform[:, 2] = -world_view_transform[:, 2]
                full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
                full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                self.custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                self.edit_text = message["edit_text"]
                self.slider = message["slider"]
                self.stop_at_value = message["stop_at_value"]
                self.single_training_step = message["single_training_step"]
            except Exception as e:
                traceback.print_exc()
                raise e

    def render(self, pipe, gaussians, loss, render, background, iteration, opt):
        if self.conn == None:
            self.try_connect()
        while self.conn != None:
            edit_error = ""
            try:
                net_image_bytes = None
                self.receive()
                pipe.convert_SHs_python = self.do_shs_python
                pipe.compute_cov3D_python = self.do_rot_scale_python
                if len(self.edit_text) > 0:
                    gs = copy.deepcopy(gaussians)
                    slider = EasyDict(self.slider)
                    try:
                        exec(self.edit_text)
                    except Exception as e:
                        edit_error = str(e)
                else:
                    gs = gaussians

                if self.custom_cam != None:
                    with torch.no_grad():
                        net_image = render(self.custom_cam, gs, pipe, background, self.scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                training_stats = json.dumps({
                    "loss": loss,
                    "iteration": iteration,
                    "num_gaussians": gaussians.get_xyz.shape[0],
                    "sh_degree": gaussians.active_sh_degree,
                    "train_params": vars(opt),
                    "error": edit_error,
                    "paused": self.stop_at_value == iteration
                })
                self.send(net_image_bytes, training_stats)
                if self.do_training and ((iteration < int(opt.iterations)) or not self.keep_alive) and self.stop_at_value != iteration:
                    break
                if self.single_training_step:
                    break

            except Exception as e:
                print(e)
                self.conn = None


class EasyDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

