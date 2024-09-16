import copy
import os
import traceback
from typing import List
import imageio
import numpy as np
import torch
import torch.nn

from c3dgs.gaussian_renderer import render
from c3dgs.scene import GaussianModel
from scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict


class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

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
        use_splitscreen=False,
        highlight_border=False,
        save_ply_path=None,
        slider={},
        **other_args,
    ):
        cam_params = cam_params.to("cuda")
        slider = EasyDict(slider)
        if len(ply_file_paths) == 0:
            res.error = "Select a .npz file"
            return

        # Remove old scenes
        if len(ply_file_paths) < self._last_num_scenes:
            for i in range(ply_file_paths, self.num_parallel_scenes):
                self.gaussian_models[i] = None
            self._last_num_scenes = len(ply_file_paths)

        images = []
        for scene_index, ply_file_path in enumerate(ply_file_paths):
            # Load
            if ply_file_path != self._current_ply_file_paths[scene_index]:
                self.gaussian_models[scene_index] = self._load_model(ply_file_path)
                self._current_ply_file_paths[scene_index] = ply_file_path

            # Edit
            gs: GaussianModel = copy.deepcopy(self.gaussian_models[scene_index])
            try:
                exec(self.sanitize_command(edit_text))
            except Exception as e:
                error = traceback.format_exc()
                error += str(e)
                res.error = error

            # Render current view
            fov_rad = fov / 360 * 2 * np.pi
            render_cam = CustomCam(resolution, resolution, fovy=fov_rad, fovx=fov_rad, extr=cam_params)
            render_res = render(viewpoint_camera=render_cam, pc=gs, bg_color=background_color.to("cuda"), pipe=EasyDict(debug=False, compute_cov3D_python=False, convert_SHs_python=False))
            if render_alpha:
                images.append(render_res["alpha"])
            elif render_depth:
                images.append(render_res["depth"] / render_res["depth"].max())
            else:
                images.append(render_res["render"])

            # Save ply
            if save_ply_path is not None:
                self.save_ply(gs, save_ply_path)

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            highlight_border=highlight_border,
        )

        res.mean_xyz = torch.mean(gs.get_xyz, dim=0)
        res.std_xyz = torch.std(gs.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def _load_model(self, ply_file_path):
        if ply_file_path.endswith(".npz"):
            gaussians = GaussianModel(3, quantization=True)
            gaussians.load_npz(ply_file_path)
        else:
            raise NotImplementedError("Only .npz files are supported.")
        return gaussians

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)
