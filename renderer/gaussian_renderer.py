import copy
import os
import traceback
from typing import List
import cv2
import numpy as np
import torch
import torch.nn
from pathlib import Path

from compression.compression_exp import run_single_decompression
from gaussian_splatting.gaussian_renderer import render_simple
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import CustomCam
from renderer.base_renderer import Renderer
from splatviz_utils.dict_utils import EasyDict


class GaussianRenderer(Renderer):
    def __init__(self, num_parallel_scenes=16):
        super().__init__()
        self.num_parallel_scenes = num_parallel_scenes
        self.gaussian_models: List[GaussianModel | None] = [None] * num_parallel_scenes
        self._current_ply_file_paths: List[str | None] = [None] * num_parallel_scenes
        self._model_stats: List[EasyDict | None] = [None] * num_parallel_scenes
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32).to("cuda")
        self._last_num_scenes = 0

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        render_width,
        render_height,
        ply_file_paths,
        cam_params,
        current_ply_names,
        background_color,
        video_cams=None,
        video_width=1920,
        video_height=1080,
        video_fps=30,
        video_path=None,
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        use_splitscreen=False,
        layout="side_by_side",
        highlight_border=False,
        save_ply_path=None,
        colormap=None,
        invert=False,
        slider={},
        **other_args,
    ):
        cam_params = cam_params.to("cuda")
        background_color = background_color.to("cuda")
        slider = EasyDict(slider)
        ply_file_paths = [path for path in ply_file_paths if path]
        if len(ply_file_paths) == 0:
            res.message = "Load a 3D Gaussian Splatting scene by dragging a supported  \n.ply or .yml file into the window, or select Browse to choose one."
            return

        # Remove old scenes
        if len(ply_file_paths) < self._last_num_scenes:
            for i in range(len(ply_file_paths), self.num_parallel_scenes):
                self.gaussian_models[i] = None
                self._current_ply_file_paths[i] = None
                self._model_stats[i] = None
            self._last_num_scenes = len(ply_file_paths)

        render_width = max(int(render_width), 1)
        render_height = max(int(render_height), 1)
        num_scenes = len(ply_file_paths)
        if use_splitscreen:
            layout = "splitscreen"
        if num_scenes <= 1:
            layout = "side_by_side"
        use_splitscreen = layout == "splitscreen"

        grid_shape = None
        if layout == "grid":
            cols = int(np.ceil(np.sqrt(num_scenes)))
            rows = int(np.ceil(num_scenes / cols))
            grid_shape = (rows, cols)
            col_widths = self._split_extent(render_width, cols)
            row_heights = self._split_extent(render_height, rows)
            scene_widths = [col_widths[scene_index % cols] for scene_index in range(num_scenes)]
            scene_heights = [row_heights[scene_index // cols] for scene_index in range(num_scenes)]
        elif layout == "splitscreen":
            scene_widths = [render_width] * num_scenes
            scene_heights = [render_height] * num_scenes
        else:
            scene_widths = self._split_extent(render_width, num_scenes)
            scene_heights = [render_height] * num_scenes

        images = []
        video_paths = []
        for scene_index, ply_file_path in enumerate(ply_file_paths):
            # Load
            if ply_file_path != self._current_ply_file_paths[scene_index]:
                self.gaussian_models[scene_index] = self._load_model(ply_file_path)
                self._current_ply_file_paths[scene_index] = ply_file_path
                self._model_stats[scene_index] = self._collect_model_stats(self.gaussian_models[scene_index])

            # Edit only needs an isolated copy when user code can mutate the scene.
            if edit_text.strip():
                gs: GaussianModel = copy.deepcopy(self.gaussian_models[scene_index])
                try:
                    exec(edit_text)
                except Exception as e:
                    error = traceback.format_exc()
                    error += str(e)
                    res.error = error
            else:
                gs: GaussianModel = self.gaussian_models[scene_index]

            # Render video
            if video_cams is not None and len(video_cams) > 0 and video_path is not None:
                video_paths.append(
                    self.render_video(
                        video_path,
                        video_cams,
                        gs,
                        fov=fov,
                        width=video_width,
                        height=video_height,
                        video_fps=video_fps,
                        background_color=background_color,
                        scene_index=scene_index,
                        num_scenes=num_scenes,
                    )
                )

            # Render current view
            fov_rad = fov / 360 * 2 * np.pi
            scene_width = scene_widths[scene_index]
            scene_height = scene_heights[scene_index]
            fov_x = 2 * np.arctan(np.tan(fov_rad * 0.5) * scene_width / scene_height)
            render_cam = CustomCam(scene_width, scene_height, fovy=fov_rad, fovx=fov_x, extr=cam_params)
            with torch.no_grad():
                render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color)
            if render_alpha:
                images.append(render["alpha"])
            elif render_depth:
                images.append((render["depth"] - render["depth"].min()) / (render["depth"].max()))
            else:
                images.append(render["render"])

            # Save ply
            if save_ply_path is not None:
                self.save_ply(gs, save_ply_path)

        self._last_num_scenes = len(ply_file_paths)

        if edit_text.strip():
            stats = self._collect_model_stats(gs)
        else:
            stats = self._model_stats[scene_index]
        res.mean_xyz = stats.mean_xyz
        res.std_xyz = stats.std_xyz
        if len(eval_text) > 0:
            res.eval = eval(eval_text)
        if len(video_paths) == 1:
            res.video_path = video_paths[0]
        elif len(video_paths) > 1:
            res.video_path = ", ".join(video_paths)

        self._return_image(
            images,
            res,
            normalize=img_normalize,
            use_splitscreen=use_splitscreen,
            layout=layout,
            grid_shape=grid_shape,
            target_size=(render_width, render_height),
            highlight_border=highlight_border,
            colormap=colormap,
            invert=invert,
        )

    def _load_model(self, ply_file_path):
        ply_file_path_lower = ply_file_path.lower()
        if ply_file_path_lower.endswith(".ply"):
            model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
            model.load_ply(ply_file_path)
        elif ply_file_path_lower.endswith((".yml", ".yaml")):
            model = run_single_decompression(Path(ply_file_path).parent.absolute())
        else:
            raise NotImplementedError("Select a .ply or .yml file.")
        return model

    @staticmethod
    def _collect_model_stats(gaussian):
        xyz = gaussian.get_xyz
        return EasyDict(mean_xyz=torch.mean(xyz, dim=0), std_xyz=torch.std(xyz))

    @staticmethod
    def render_video(
        video_path,
        video_cams,
        gs,
        fov,
        width,
        height,
        video_fps,
        background_color,
        scene_index,
        num_scenes,
    ):
        width = max(int(width), 16)
        height = max(int(height), 16)
        video_fps = max(int(video_fps), 1)
        video_path = GaussianRenderer._scene_video_path(video_path, scene_index, num_scenes)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")

        fov_rad = fov / 360 * 2 * np.pi
        fov_x = 2 * np.arctan(np.tan(fov_rad * 0.5) * width / height)
        try:
            with torch.no_grad():
                for cam_params in video_cams:
                    extr = torch.as_tensor(cam_params, device="cuda", dtype=torch.float32)
                    render_cam = CustomCam(width, height, fovy=fov_rad, fovx=fov_x, extr=extr)
                    render = render_simple(viewpoint_camera=render_cam, pc=gs, bg_color=background_color)
                    image = (render["render"] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return video_path

    @staticmethod
    def _scene_video_path(video_path, scene_index, num_scenes):
        if num_scenes <= 1:
            return video_path
        root, ext = os.path.splitext(video_path)
        return f"{root}_scene{scene_index + 1}{ext or '.mp4'}"

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        os.makedirs(save_ply_path, exist_ok=True)
        save_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_path)
        gaussian.save_ply(save_path)
