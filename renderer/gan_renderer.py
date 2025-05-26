import pickle
import numpy as np
import torch
import torch.nn
from tqdm import tqdm

from scene import GaussianModel
from gaussian_renderer import render_simple
from renderer.base_renderer import Renderer
from scene.cameras import CustomCam
from splatviz_utils.dict_utils import EasyDict
from gan_helper.latent_vector import LatentMap
from gan_helper.view_conditioning import view_conditioning


class GANRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.generator = None
        self.last_latent = torch.zeros([1, 512], device=self._device)
        self._current_pkl_file_path = ""
        self.gaussian_model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        self.latent_map = LatentMap()
        self.device = torch.device("cuda")
        self.last_truncation_psi = 1.0
        self.last_mapping_conditioning = "frontal"
        self.last_seed = 0

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
        latent_space="W",
        img_normalize=False,
        latent_x=0.0,
        latent_y=0.0,
        save_ply_path=None,
        truncation_psi=1.0,
        mapping_conditioning="frontal",
        save_ply_grid_path=None,
        seed=0,
        slider={},
        **other_args
    ):
        slider = EasyDict(slider)
        cam_params = cam_params.to(self.device)
        mapping_conditioning_changed = mapping_conditioning != self.last_mapping_conditioning
        seed_changed = seed != self.last_seed
        self.last_mapping_conditioning = mapping_conditioning
        if seed_changed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.latent_map = LatentMap()
            self.latent_map.load_w_map(self.generator.mapping, truncation_psi)
            self.last_seed = seed

        # generator
        model_changed = self.load(ply_file_paths[0])
        truncation_psi_changed = self.last_truncation_psi != truncation_psi
        if truncation_psi_changed and latent_space == "W":
            self.latent_map.load_w_map(self.generator.mapping, truncation_psi)

        latent = self.latent_map.get_latent(latent_x, latent_y, latent_space=latent_space)
        latent_changed = not torch.equal(self.last_latent, latent)

        if seed_changed or latent_changed or model_changed or truncation_psi_changed or mapping_conditioning_changed or mapping_conditioning == "current":
            gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
            if latent_space == "Z":
                mapped_latent = self.generator.mapping(latent, mapping_camera_params, truncation_psi=truncation_psi)
            elif latent_space == "W":
                mapped_latent = latent[:, None, :] .repeat(1, self.generator.mapping_network.num_ws, 1)
            gan_result = self.generator.synthesis(mapped_latent, gan_camera_params, render_output=False)
            self.last_latent = latent
            self.extract_gaussians(gan_result)

        # edit 3DGS scene
        gs = self.gaussian_model
        exec(self.sanitize_command(edit_text))

        # render 3DGS scene
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to(self.device))["render"]

        # return / eval / save scene
        self._return_image(img, res, normalize=img_normalize)
        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

        if save_ply_grid_path is not None:
            self.save_ply_grid(cam_params, fov, latent_space, mapped_latent, mapping_conditioning, truncation_psi)

    def save_ply_grid(self, cam_params, fov, latent_space, mapped_latent, mapping_conditioning, truncation_psi, steps=16):
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, steps), np.linspace(-0.5, 0.5, steps))
        for i in tqdm(range(steps)):
            for j in range(steps):
                x = xs[i, j]
                y = ys[i, j]
                latent = self.latent_map.get_latent(x, y, latent_space=latent_space)
                gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
                if latent_space == "Z":
                    mapped_latent = self.generator.mapping(latent, mapping_camera_params, truncation_psi=truncation_psi)
                elif latent_space == "W":
                    mapped_latent = latent[:, None, :].repeat(1, self.generator.mapping_network.num_ws, 1)
                gan_result = self.generator.synthesis(mapped_latent, gan_camera_params)
                self.last_latent = latent
                self.extract_gaussians(gan_result)
                self.save_ply(self.gaussian_model, f"./_ply_grid/model_c{i:02d}_r{j:02d}.ply")

    def extract_gaussians(self, gan_result):
        gan_model = EasyDict(gan_result["gaussian_params"][0])
        self.gaussian_model._xyz = gan_model._xyz
        self.gaussian_model._features_dc = gan_model._features_dc
        self.gaussian_model._features_rest = gan_model._features_dc[:, 0:0]
        self.gaussian_model._scaling = gan_model._scaling
        self.gaussian_model._rotation = gan_model._rotation
        self.gaussian_model._opacity = gan_model._opacity

    def load(self, pkl_file_path):
        if pkl_file_path == self._current_pkl_file_path:
            return False
        if not pkl_file_path.endswith(".pkl"):
            return False

        with open(pkl_file_path, "rb") as input_file:
            save_file = pickle.load(input_file)
        self.generator = save_file["G_ema"]
        self.generator = self.generator.to(self.device)
        self._current_pkl_file_path = pkl_file_path
        self.latent_map.load_w_map(self.generator.mapping, self.last_truncation_psi)
        return True
