import copy
import pickle
import numpy as np
import torch
import torch.nn
from scene import GaussianModel
from gaussian_renderer import render_simple
from renderer.base_renderer import Renderer
from scene.cameras import CustomCam
from splatviz_utils.cam_utils import fov_to_intrinsics
from splatviz_utils.dict_utils import EasyDict


class GANRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.generator = None
        self.position_prediction = None
        self.last_z = torch.zeros([1, 512], device=self._device)
        self.last_command = ""
        self.latent_map = torch.randn([1, 512, 10, 10], device=self._device, dtype=torch.float)
        self.reload_model = True
        self._current_pkl_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=3, disable_xyz_log_activation=True)
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.last_gan_result = None

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
        save_ply_path=None,
        truncation_psi=1.0,
        c_gen_conditioning_zero=True,
        slider={},
        **other_args
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        self.load(ply_file_paths[0])
        self.generator.rendering_kwargs["c_gen_conditioning_zero"] = c_gen_conditioning_zero

        # create camera
        intrinsics = fov_to_intrinsics(fov, device=self._device)[None, :]
        fov_rad = fov / 360 * 2 * np.pi

        z = self.create_z(latent_x, latent_y)
        if not torch.equal(self.last_z, z) or self.reload_model:
            gan_camera_params = torch.concat([cam_params.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            self.last_gan_result = self.generator(z, gan_camera_params, truncation_psi=truncation_psi)
            self.last_z = z

        gan_model = EasyDict(self.last_gan_result["gaussian_params"])
        self.gaussian_model._xyz = gan_model._xyz
        self.gaussian_model._features_dc = gan_model._features_dc
        self.gaussian_model._features_rest = gan_model._features_rest
        self.gaussian_model._scaling = gan_model._scaling
        self.gaussian_model._rotation = gan_model._rotation
        self.gaussian_model._opacity = gan_model._opacity

        gs = self.gaussian_model
        exec(self.sanitize_command(edit_text))

        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)

        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        result = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))
        img = result["render"]
        #img = (img + 1) / 2
        self._return_image(img, res, normalize=img_normalize)

        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def create_z(self, latent_x, latent_y):
        latent_x = torch.tensor(latent_x, device="cuda", dtype=torch.float)
        latent_y = torch.tensor(latent_y, device="cuda", dtype=torch.float)
        position = torch.stack([latent_x, latent_y]).reshape(1, 1, 1, 2)
        # todo: interpolate in w
        z = torch.nn.functional.grid_sample(self.latent_map, position, padding_mode="reflection")
        return z.reshape(1, 512)

    def load(self, pkl_file_path):
        if pkl_file_path != self._current_pkl_file_path:
            if pkl_file_path.endswith(".pkl"):
                with open(pkl_file_path, "rb") as input_file:
                    save_file = pickle.load(input_file)
                    self.generator = save_file["G_ema"]
                    self._current_pkl_file_path = pkl_file_path
                    self.generator.rendering_kwargs["c_gen_conditioning_zero"] = True
                    self.generator = self.generator.to("cuda")
