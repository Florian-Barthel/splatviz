import copy
import os
import imageio
import numpy as np
import torch
import torch.nn
from tqdm import tqdm
import json
import re

from decalib.datasets.datasets import process_single_image
from decalib.deca import DECA
from gaussian_renderer import render_simple
from scene import GaussianModel
from scene.cameras import CustomCam
from utils.graphics_utils import focal2fov
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
        self.cam_params_ffhq = None
        self.cam_params = None
        self.fov_rad_ffhq = None
        with open("/home/barthel/datasets/FFHQ/label/dataset_old.json", "r") as f:
            self.label_dict = json.load(f)

        with open("/home/barthel/datasets/FFHQ/label/deca_predictions.json", "r") as f:
            self.deca_predictions = json.load(f)

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
        on_top=False,
        highlight_border=False,
        save_ply_path=None,
        use_ffhq_cam=True,
        **slider,
    ):
        slider = EasyDict(slider)

        images = []
        # Load
        if ply_file_paths[0] != self.last_image_path:
            self.input_image = process_single_image(ply_file_paths[0])
            # _, self.input_image = self.deca.encode_image(ply_file_paths[0])
            self.codedict = self.deca_predictions[ply_file_paths[0].split("/")[-1]]

            for key in ["pose", "exp", "shape"]:
                self.codedict[key] = torch.tensor(self.codedict[key], dtype=torch.float, device="cuda")

            self.last_image_path = ply_file_paths[0]

            # load ffhq cam
            index = int(re.findall(r'\d+', ply_file_paths[0])[-1])
            cam = self.label_dict["labels"][index][1]
            print(f"Using {self.label_dict['labels'][index][0]}")
            ffhq_cam_params = np.array(cam[:16], dtype=float).reshape(4, 4)
            self.cam_params_ffhq = torch.tensor(ffhq_cam_params, dtype=torch.float, device="cuda")
            self.fov_rad_ffhq = focal2fov(float(cam[16]))

        if self.input_image.shape[-1] != resolution:
            self.input_image = torch.nn.functional.interpolate(self.input_image, size=(resolution, resolution))
        images.append(self.input_image[0])

        # remove flame head rotation
        self.codedict["pose"][0, 0:3] = 0

        # Edit
        edit_before_flame = False
        codedict = copy.deepcopy(self.codedict)

        fov_rad = fov / 360 * 2 * np.pi
        if use_ffhq_cam:
            scale = 1
            fov_rad = self.fov_rad_ffhq / scale
            self.cam_params = copy.deepcopy(self.cam_params_ffhq)
            self.cam_params[:3, 3:] = self.cam_params_ffhq[:3, 3:] * scale

        else:
            self.cam_params = cam_params

        if edit_before_flame:
            try:
                exec(self.sanitize_command(edit_text))
            except Exception as e:
                res.error = e
        verts = self.deca.decode_flame(codedict)
        self.gaussian_model = self._load_model(verts)
        gaussian: GaussianModel = copy.deepcopy(self.gaussian_model)
        if not edit_before_flame:
            try:
                exec(self.sanitize_command(edit_text))
            except Exception as e:
                res.error = e

        # Render video
        if len(video_cams) > 0:
            self.render_video("./_videos", video_cams, gaussian)


        render_cam = CustomCam(
            resolution,
            resolution,
            fovy=fov_rad,
            fovx=fov_rad,
            znear=0.01,
            zfar=15,
            extr=self.cam_params,
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
            on_top=on_top
        )

        res.mean_xyz = torch.mean(gaussian.get_xyz, dim=0)
        res.std_xyz = torch.std(gaussian.get_xyz)
        if len(eval_text) > 0:
            res.eval = eval(eval_text)

    def _load_model(self, verts): # (1, 5023, 3)
        model = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        num_verts = verts.shape[1]

        """
        flame_to_bfm_neutral_37_face_only = torch.tensor([
            [2.682716929557429, 0.010446918125791843, 0.04746649927210553, 0.0030014961233934233],
            [-0.009421598630294275, 2.682515580606053, -0.05790466856909523, 0.04740944787628047],
            [-0.047680604263683285, 0.05772857372357329, 2.682112266656848, -0.0024605516344398115],
            [0.0, 0.0, 0.0, 1.0]
        ], device="cuda")
        xyz_h = torch.ones([num_verts, 4], dtype=torch.float32, device="cuda")
        xyz_h[:, :3] = verts[0]

        xyz_h= torch.bmm(flame_to_bfm_neutral_37_face_only[None, ...].tile([len(xyz_h), 1, 1]), xyz_h[..., None])
        new_xyz = xyz_h[:, :3, 0] / xyz_h[:, 3:4, 0]
        """

        flame_to_bfm_neutral_37_face_only = torch.tensor([
            [2.682716929557429, 0.010446918125791843, 0.04746649927210553, 0.0030014961233934233],
            [-0.009421598630294275, 2.682515580606053, -0.05790466856909523, 0.04740944787628047],
            [-0.047680604263683285, 0.05772857372357329, 2.682112266656848, -0.0024605516344398115],
            [0.0, 0.0, 0.0, 1.0]
        ], device="cuda")

        flame_vertices = (to_homogeneous(verts) @ flame_to_bfm_neutral_37_face_only.T)[..., :3]

        # xyz_h = torch.ones([num_verts, 4], dtype=torch.float32, device="cuda")
        # xyz_h[:, :3] = verts[0]

        # xyz_h= torch.bmm(flame_to_bfm_neutral_37_face_only[None, ...].tile([len(xyz_h), 1, 1]), xyz_h[..., None])
        # new_xyz = xyz_h[:, :3, 0] / xyz_h[:, 3:4, 0]

        model._xyz = flame_vertices[0]
        model._features_dc = torch.ones([num_verts, 1, 3], device="cuda")
        model._features_rest = torch.zeros([num_verts, (model.max_sh_degree + 1) ** 2 - 1, 3], device="cuda")
        model._rotation = torch.zeros([num_verts, 4], device="cuda")
        model._rotation[:, 0] = 1
        model._scaling = torch.ones([num_verts, 3], device="cuda") * -8
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

def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    points = torch.concat([points, ones], dim=-1)

    return points