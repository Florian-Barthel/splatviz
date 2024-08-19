import copy
import os
import pickle
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.cam_utils import fov_to_intrinsics
from splatviz_utils.dict_utils import EasyDict


class GaussianDecoderRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.decoder = None
        self.position_prediction = None
        self.last_z = torch.zeros([1, 512], device=self._device)
        self.last_command = ""
        self.latent_map = torch.randn([1, 512, 10, 10], device=self._device, dtype=torch.float)
        self.reload_model = True
        self._current_ply_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        current_ply_names,
        background_color,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        latent_x=0.0,
        latent_y=0.0,
        render_gan_image=False,
        save_ply_path=None,
        slider={},
        **other_args
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        self.load_decoder(ply_file_paths[0])

        # create videos
        if len(video_cams) > 0:
            self.render_video("./_videos", video_cams)

        # create camera
        intrinsics = fov_to_intrinsics(fov, device=self._device)[None, :]
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        gan_camera_params = torch.concat([cam_params.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        z = self.create_z(latent_x, latent_y)
        if not torch.equal(self.last_z, z) or self.reload_model or render_gan_image:
            result = self.position_prediction.get_data(z=z, camera_params=gan_camera_params)
            gaussian_attr = self.decoder(z, gan_camera_params, result.vertices, truncation_psi=1.0)
            self.gaussian_model._xyz = gaussian_attr.xyz
            self.gaussian_model._scaling = gaussian_attr.scale
            self.gaussian_model._rotation = gaussian_attr.rotation
            self.gaussian_model._opacity = gaussian_attr.opacity
            self.gaussian_model._features_dc = gaussian_attr.color.unsqueeze(1)
            self.last_z = z
            self.reload_model = False

        gs = copy.deepcopy(self.gaussian_model)
        exec(self.sanitize_command(edit_text))

        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)

        img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))["render"]
        if render_gan_image:
            gan_image = torch.nn.functional.interpolate(result.img, size=[img.shape[1], img.shape[2]])[0]
            img = torch.concat([img, gan_image], dim=2)
        self._return_image(img, res, normalize=img_normalize)

        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)

    def create_z(self, latent_x, latent_y):
        latent_x = torch.tensor(latent_x, device="cuda", dtype=torch.float)
        latent_y = torch.tensor(latent_y, device="cuda", dtype=torch.float)
        position = torch.stack([latent_x, latent_y]).reshape(1, 1, 1, 2)
        # todo: interpolate in w
        z = torch.nn.functional.grid_sample(self.latent_map, position, padding_mode="reflection")
        return z.reshape(1, 512)

    def render_video(self, save_path, video_cams):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=self.gaussian_model, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")

    def load_decoder(self, ply_file_path):
        if ply_file_path != self._current_ply_file_path:
            if ply_file_path.endswith(".pkl"):
                with open(ply_file_path, "rb") as input_file:
                    save_file = pickle.load(input_file)
                    self.decoder = save_file["decoder"]
                    self.position_prediction = save_file["dataloader"]
                    self.reload_model = True
                    self._current_ply_file_path = ply_file_path
