import copy
import os
from typing import List
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
from pathlib import Path

from compression.compression_exp import run_single_decompression
from decalib.deca import DECA
from decalib.models.FLAME import FLAME
from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from viz.base_renderer import Renderer
from viz_utils.dict import EasyDict


class FlameRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0
        self.deca = DECA()
        self.gaussian_model = None
        self.input_image = None
        self.last_image_path = None
        self.codedict = None

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
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        use_splitscreen=False,
        highlight_border=False,
        save_ply_path=None,
        use_flame_cam=True,
        **slider,
    ):
        slider = EasyDict(slider)

        images = []
        # Load
        if ply_file_paths[0] != self.last_image_path:
            self.codedict, out_dict, self.input_image = self.deca.run_image(ply_file_paths[0])
            self.gaussian_model = self._load_model(out_dict)
            self.last_image_path = ply_file_paths[0]

        if self.input_image.shape[-1] != resolution:
            self.input_image = torch.nn.functional.interpolate(self.input_image, size=(resolution, resolution))
        images.append(self.input_image[0])

        # Edit
        gaussian: GaussianModel = copy.deepcopy(self.gaussian_model)
        try:
            exec(self.sanitize_command(edit_text))
        except Exception as e:
            res.error = e

        # Render video
        if len(video_cams) > 0:
            self.render_video("./_videos", video_cams, gaussian)

        if use_flame_cam:
            pose = self.codedict['pose']
            rotation_matrices = batch_rodrigues(pose[:, :3])
            R_deca = rotation_matrices
            T_deca = R_deca[:, :, -1]
            cam_params = torch.zeros([4, 4], device="cuda")
            cam_params[:3, :3] = R_deca
            cam_params[3:, :3] = T_deca
            cam_params[3, 3] = 1

        # Render current view
        fov_rad = fov / 360 * 2 * np.pi
        # fov_rad = 2 * np.arctan(112. / 1015) * 180 / np.pi
        # fov_rad = 2 * np.pi * fov_rad / 360
        render_cam = CustomCam(
            resolution,
            resolution,
            fovy=fov_rad,
            fovx=fov_rad,
            znear=0.01,
            zfar=15,
            extr=cam_params,
        )
        render = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)
        if render_alpha:
            images.append(render["alpha"])
        elif render_depth:
            images.append(render["depth"] / render["depth"].max())
        else:
            images.append(render["render"])

        # Save ply
        if save_ply_path is not None:
            os.makedirs(save_ply_path, exist_ok=True)
            save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
            print("Model saved in", save_path)
            gaussian.save_ply(save_path)

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )

        res.mean_xyz = torch.mean(gaussian.get_xyz, dim=0)
        res.std_xyz = torch.std(gaussian.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def _load_model(self, out_dict):
        model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)

        verts = out_dict["trans_verts"] # (1, 5023, 3)
        num_verts = verts.shape[1]

        model._xyz = verts[0]
        model._xyz[:, -1] = - model._xyz[:, -1]
        # model._xyz[:, -1] *= -1

        model._features_dc = torch.ones([num_verts, 1, 3], device="cuda")
        model._features_rest = torch.zeros([num_verts, (model.max_sh_degree + 1) ** 2 - 1, 3], device="cuda")
        model._rotation = torch.zeros([num_verts, 4], device="cuda")
        model._rotation[:, 0] = 1
        model._scaling = torch.ones([num_verts, 3], device="cuda") * -5
        model._opacity = torch.ones([num_verts, 1], device="cuda") * 1
        return model

    def render_video(self, save_path, video_cams, gaussian):
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/rotate_{len(os.listdir(save_path))}.mp4"
        video = imageio.get_writer(filename, mode="I", fps=30, codec="libx264", bitrate="16M", quality=10)
        for render_cam in tqdm(video_cams):
            img = render_simple(viewpoint_camera=render_cam, pc=gaussian, bg_color=self.bg_color)["render"]
            img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            video.append_data(img)
        video.close()
        print(f"Video saved in {filename}.")


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    '''  same as batch_matrix2axis
    Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat