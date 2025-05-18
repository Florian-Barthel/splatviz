import pickle
import numpy as np
import torch
import torch.nn

from gan_helper.latent_vector import LatentMap
from gan_helper.view_conditioning import view_conditioning
from scene import GaussianModel
from gaussian_renderer import render_simple
from renderer.base_renderer import Renderer
from scene.cameras import CustomCam
from splatviz_utils.dict_utils import EasyDict


class GANRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self._current_pkl_file_path = ""
        self.generator = None
        self.gaussian_model = GaussianModel(sh_degree=1, disable_xyz_log_activation=True)
        self.gaussian_model.active_sh_degree = 1
        self.last_gan_result = None
        self.latent_map = LatentMap()
        self.last_z = torch.zeros([1, 512], device=self._device)

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        resolution,
        ply_file_paths,
        cam_params,
        background_color,
        img_normalize,
        latent_x,
        latent_y,
        save_ply_path=None,
        truncation_psi=1.0,
        mapping_conditioning="frontal",
        slider={},
        **other_args
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)

        # generator
        z = self.latent_map.get_latent(latent_x, latent_y, "W")
        latent_changed = not torch.equal(self.last_z, z)
        model_changed = self.load(ply_file_paths[0])
        if latent_changed or model_changed:
            gan_camera_params, mapping_camera_params = view_conditioning(cam_params, fov, mapping_conditioning)
            ws = self.generator.mapping(z, mapping_camera_params, truncation_psi=truncation_psi)
            gan_result = self.generator.synthesis(ws, gan_camera_params, noise_mode='const')
            self.last_z = z
            self.extract_gaussians(gan_result)

        # edit 3DGS scene
        gs = self.gaussian_model
        exec(edit_text)

        # render 3DGS
        fov_rad = fov / 360 * 2 * np.pi
        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        img = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"))["render"]

        # return / eval / save 3DGS
        self._return_image(img, res, normalize=img_normalize)
        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def extract_gaussians(self, gan_result):
        gan_model = gan_result.gaussian_attribute_output.gaussian_attributes
        num_gaussians = gan_model["POSITION"].shape[1]
        self.gaussian_model._xyz = gan_model["POSITION"][0]
        color = gan_model["COLOR"][0].reshape(num_gaussians, -1, 3)
        self.gaussian_model._features_dc = color[:, :1]
        self.gaussian_model._features_rest = color[:, 1:]
        self.gaussian_model._scaling = gan_model["SCALE"][0]
        self.gaussian_model._rotation = gan_model["ROTATION"][0].contiguous()
        self.gaussian_model._opacity = gan_model["OPACITY"][0]


    def load(self, pkl_file_path):
        if pkl_file_path == self._current_pkl_file_path:
            return False
        if not pkl_file_path.endswith(".pkl"):
            return False

        with open(pkl_file_path, "rb") as input_file:
            save_file = pickle.load(input_file)
        self.generator = save_file["G_ema"]
        self._current_pkl_file_path = pkl_file_path
        self.generator = self.generator.to("cuda")

        # GGHead specific backward compatibility
        if not hasattr(self.generator, '_n_uv_channels_background'):
            self.generator._n_uv_channels_background = 0
        if self.generator._config.use_initial_scales and not hasattr(self.generator, '_initial_gaussian_scales_head'):
            self.generator.register_buffer("_initial_gaussian_scales_head", self.generator._initial_gaussian_scales, persistent=False)
        if not hasattr(self.generator, '_n_uv_channels_per_shell'):
            # n_uv_channels was renamed into n_uv_channels_per_shell
            self.generator._n_uv_channels_per_shell = self.generator._n_uv_channels
        if not hasattr(self.generator, '_n_uv_channels_decoded'):
            self.generator._n_uv_channels_decoded = self.generator._n_uv_channels
        if not hasattr(self.generator, '_n_uv_channels_per_shell_decoded'):
            self.generator._n_uv_channels_per_shell_decoded = self.generator._n_uv_channels_per_shell
        if not hasattr(self.generator._config, 'template_update_attributes'):
            # template_update_attributes was added to config and used in forward pass
            self.generator._config.template_update_attributes = []
        if (self.generator._config.super_resolution_config.use_superresolution
                and self.generator._config.super_resolution_config.superresolution_version == 2
                and not hasattr(self.generator.super_resolution, 'n_downsampling_layers')):
            # Number of downsampling layers was made variable
            self.generator.super_resolution.n_downsampling_layers = 1
        return True