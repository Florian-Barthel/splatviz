import copy
from typing import Any
import torch
import traceback
import socket
import json

from scene.cameras import MiniCam


class SplatvizNetwork:
    """
    1. Copy this file into your gaussian splatting project. Then import the file as follows:
    from splatviz_network import SplatvizNetwork

    2. At the beginning of the train script, create the network connector:
    network = SplatvizNetwork()

    3. Call the render function inside the training loop:
    network.render(pipe, gaussians, ema_loss_for_log, render, background, iteration, opt)
    """

    def __init__(self, host="127.0.0.1", port=6077):
        self.host = host
        self.port = port
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)
        self.conn = None
        self.addr = None
        print(f"Creating splatviz network connector for host={host} and port={port}")

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
                do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                do_shs_python = bool(message["shs_python"])
                do_rot_scale_python = bool(message["rot_scale_python"])
                keep_alive = bool(message["keep_alive"])
                scaling_modifier = message["scaling_modifier"]
                world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
                world_view_transform[:, 1] = -world_view_transform[:, 1]
                world_view_transform[:, 2] = -world_view_transform[:, 2]
                full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
                full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                edit_text = message["edit_text"]
                slider = message["slider"]
            except Exception as e:
                print("")
                traceback.print_exc()
                raise e
            return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier, edit_text, slider
        else:
            return None, None, None, None, None, None, None, None

    def render(self, pipe, gaussians, loss, render, background, iteration, opt):
        if self.conn == None:
            self.try_connect()
        while self.conn != None:
            edit_error = ""
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, edit_text, slider = self.receive()
                if len(edit_text) > 0:
                    gs = copy.deepcopy(gaussians)
                    slider = EasyDict(slider)
                    try:
                        exec(edit_text)
                    except Exception as e:
                        edit_error = str(e)
                else:
                    gs = gaussians

                if custom_cam != None:
                    with torch.no_grad():
                        net_image = render(custom_cam, gs, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                training_stats = json.dumps({
                    "loss": loss,
                    "iteration": iteration,
                    "num_gaussians": gaussians.get_xyz.shape[0],
                    "sh_degree": gaussians.active_sh_degree,
                    "train_params": vars(opt),
                    "error": edit_error
                })
                self.send(net_image_bytes, training_stats)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
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

