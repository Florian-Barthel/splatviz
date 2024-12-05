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
from face_parsing.face_segment import SegmentationModel


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
        self.gghead = False
        sh_degree = 1 if self.gghead else 0
        self.gaussian_model = GaussianModel(sh_degree=sh_degree, disable_xyz_log_activation=True)
        self.gaussian_model.active_sh_degree = sh_degree
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.last_gan_result = None
        self.segmentation_model = SegmentationModel()

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
        render_seg=False,
        img_normalize=False,
        latent_x=0.0,
        latent_y=0.0,
        save_ply_path=None,
        truncation_psi=1.0,
        c_gen_conditioning_zero=True,
        conditional_vector=None,
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
            # if not self.gghead:
            gan_camera_params = torch.concat([cam_params.reshape(-1, 16), intrinsics.reshape(-1, 9), conditional_vector], 1)
            # else:
            # gan_camera_params = torch.concat([cam_params.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            self.last_gan_result = self.generator(z, gan_camera_params, truncation_psi=truncation_psi, noise_mode='const')
            self.last_z = z

        if self.gghead:
            self.generator.rendering_config.c_gen_conditioning_zero = c_gen_conditioning_zero
            gan_model = self.last_gan_result.gaussian_attribute_output.gaussian_attributes
            num_gaussians = gan_model["POSITION"].shape[1]
            self.gaussian_model._xyz = gan_model["POSITION"][0]
            color = gan_model["COLOR"][0].reshape(num_gaussians, -1, 3)
            self.gaussian_model._features_dc = color[:, :1]
            self.gaussian_model._features_rest = color[:, 1:]
            self.gaussian_model._scaling = gan_model["SCALE"][0]
            self.gaussian_model._rotation = gan_model["ROTATION"][0].contiguous()
            self.gaussian_model._opacity = gan_model["OPACITY"][0]
        else:
            gan_model = EasyDict(self.last_gan_result["gaussian_params"][0])
            self.gaussian_model._xyz = gan_model._xyz
            self.gaussian_model._features_dc = gan_model._features_dc
            # self.gaussian_model._features_rest = gan_model._features_rest
            self.gaussian_model._scaling = gan_model._scaling
            self.gaussian_model._rotation = gan_model._rotation
            self.gaussian_model._opacity = gan_model._opacity

        gs = self.gaussian_model
        exec(self.sanitize_command(edit_text))

        if save_ply_path is not None:
            self.save_ply(gs, save_ply_path)

        render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
        if self.gghead:
            result = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda")) #, override_color=gan_model._features_dc)
        else:
            result = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"), override_color=gan_model._features_dc) #, override_color=gan_model._features_dc)

        # img = self.last_gan_result.images[0]
        img = result["render"]

        #img = img * 2 - 1
        # img = (img + 1) / 2
        # img = torch.clip(img, 0.0, 1.0)

        if render_seg:
            img = self.segmentation_model(img)

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

                    if self.gghead:
                        # Backward compatibility
                        if not hasattr(self.generator, '_n_uv_channels_background'):
                            self.generator._n_uv_channels_background = 0

                        if self.generator._config.use_initial_scales and not hasattr(self.generator, '_initial_gaussian_scales_head'):
                            self.generator.register_buffer(
                                "_initial_gaussian_scales_head", self.generator._initial_gaussian_scales, persistent=False
                            )

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
