#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation
from enum import Enum


class ColorMode(Enum):
    NOT_INDEXED = 0
    ALL_INDEXED = 1


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(
            scaling, scaling_modifier, rotation, strip_sym=True
        ):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if strip_sym:
                return strip_symmetric(actual_covariance)
            else:
                return actual_covariance

        self.scaling_activation = lambda x: torch.nn.functional.normalize(
            torch.nn.functional.relu(x)
        )
        self.scaling_inverse_activation = lambda x: x
        self.scaling_factor_activation = torch.exp
        self.scaling_factor_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, quantization=True):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._scaling_factor = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # quantization related stuff
        self._feature_indices = None
        self._gaussian_indices = None

        self.quantization = quantization
        self.color_index_mode = ColorMode.NOT_INDEXED

        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.opacity_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.scaling_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.rotation_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).cuda()
        self.xyz_qa = FakeQuantizationHalf.apply

        if not self.quantization:


            self.features_dc_qa.disable_fake_quant()
            self.features_dc_qa.disable_observer()
            self.features_rest_qa.disable_fake_quant()
            self.features_rest_qa.disable_observer()
            
            self.scaling_qa.disable_fake_quant()
            self.scaling_qa.disable_observer()

            self.rotation_qa.disable_fake_quant()
            self.rotation_qa.disable_observer()
            self.xyz_qa = lambda x: x

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        scaling_n = self.scaling_qa(self.scaling_activation(self._scaling))
        if self.is_gaussian_indexed:
            return scaling_n[self._gaussian_indices]
        else:
            return scaling_n

    @property
    def get_scaling_normalized(self):
        return self.scaling_qa(self.scaling_activation(self._scaling))


    @property
    def get_rotation(self):
        rotation = self.rotation_activation(self.rotation_qa(self._rotation))
        if self.is_gaussian_indexed:
            return rotation[self._gaussian_indices]
        else:
            return rotation

    @property
    def _rotation_post_activation(self):
        return self.rotation_activation(self.rotation_qa(self._rotation))

    @property
    def get_xyz(self):
        return self.xyz_qa(self._xyz)

    @property
    def get_features(self):
        features_dc = self.features_dc_qa(self._features_dc)
        features_rest = self.features_rest_qa(self._features_rest)

        if self.color_index_mode == ColorMode.ALL_INDEXED:
            return torch.cat((features_dc, features_rest), dim=1)[self._feature_indices]
        else:
            return torch.cat((features_dc, features_rest), dim=1)
        
    @property
    def _get_features_raw(self):
        features_dc = self.features_dc_qa(self._features_dc)
        features_rest = self.features_rest_qa(self._features_rest)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_qa(self.opacity_activation(self._opacity))

    def get_covariance(self, scaling_modifier=1, strip_sym: bool = True):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation, strip_sym
        )

    def get_normalized_covariance(self, scaling_modifier=1, strip_sym: bool = True):
        scaling_n = self.scaling_qa(self.scaling_activation(self._scaling))
        return self.covariance_activation(
            scaling_n, scaling_modifier, self.get_rotation, strip_sym
        )

    @property
    def is_color_indexed(self):
        return self._feature_indices is not None

    @property
    def is_gaussian_indexed(self):
        return self._gaussian_indices is not None

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._scaling_factor],
                "lr": training_args.scaling_lr,
                "name": "scaling_factor",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        # l.append("scale_factor")
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self.is_gaussian_indexed or self.is_color_indexed:
            print(
                "WARNING: indexed colors/gaussians are not supported for ply files and are converted to dense attributes"
            )

        color_features = self.get_features.detach()

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            color_features[:, :1]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            color_features[:, 1:]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = (
            self.scaling_factor_inverse_activation(self.get_scaling.detach())
            .cpu()
            .numpy()
        )

        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load(self, path: str,override_quantization=False):
        ext = os.path.splitext(path)[1]
        if ext == ".ply":
            self.load_ply(path)
        elif ext == ".npz":
            self.load_npz(path,override_quantization)
        else:
            raise NotImplementedError(f"file ending '{ext}' not supported")

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_") and not p.name.startswith("scale_factor")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        scaling = self.scaling_factor_activation(torch.tensor(scales, dtype=torch.float, device="cuda"))
        scaling_norm = scaling.norm(2, -1, keepdim=True)
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(scaling / scaling_norm).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def save_npz(
        self,
        path,
        compress: bool = True,
        half_precision: bool = False,
        sort_morton=False,
    ):
        with torch.no_grad():
            if sort_morton:
                self._sort_morton()
            if isinstance(path, str):
                mkdir_p(os.path.dirname(os.path.abspath(path)))

            dtype = torch.half if half_precision else torch.float32

            save_dict = dict()

            save_dict["quantization"] = self.quantization

            # save position
            if self.quantization:
                save_dict["xyz"] = self.get_xyz.detach().half().cpu().numpy()
            else:
                save_dict["xyz"] = self._xyz.detach().cpu().numpy()

            # save color features
            if self.quantization:
                features_dc_q = torch.quantize_per_tensor(
                    self._features_dc.detach(),
                    self.features_dc_qa.scale,
                    self.features_dc_qa.zero_point,
                    self.features_dc_qa.dtype,
                ).int_repr()
                save_dict["features_dc"] = features_dc_q.cpu().numpy()
                save_dict["features_dc_scale"] = self.features_dc_qa.scale.cpu().numpy()
                save_dict[
                    "features_dc_zero_point"
                ] = self.features_dc_qa.zero_point.cpu().numpy()

                features_rest_q = torch.quantize_per_tensor(
                    self._features_rest.detach(),
                    self.features_rest_qa.scale,
                    self.features_rest_qa.zero_point,
                    self.features_rest_qa.dtype,
                ).int_repr()
                save_dict["features_rest"] = features_rest_q.cpu().numpy()
                save_dict["features_rest_scale"] = self.features_rest_qa.scale.cpu().numpy()
                save_dict[
                    "features_rest_zero_point"
                ] = self.features_rest_qa.zero_point.cpu().numpy()
            else:
                save_dict["features_dc"] = self._features_dc.detach().cpu().numpy()
                save_dict["features_rest"] = self._features_rest.detach().cpu().numpy()

            # save opacity
            if self.quantization:
                opacity = self.opacity_activation(self._opacity).detach()
                opacity_q = torch.quantize_per_tensor(
                    opacity,
                    scale=self.opacity_qa.scale,
                    zero_point=self.opacity_qa.zero_point,
                    dtype=self.opacity_qa.dtype,
                ).int_repr()
                save_dict["opacity"] = opacity_q.cpu().numpy()
                save_dict["opacity_scale"] = self.opacity_qa.scale.cpu().numpy()
                save_dict[
                    "opacity_zero_point"
                ] = self.opacity_qa.zero_point.cpu().numpy()
            else:
                save_dict["opacity"] = self._opacity.detach().to(dtype).cpu().numpy()

            # save indices
            if self.is_color_indexed:
                save_dict["feature_indices"] = (
                    self._feature_indices.detach().contiguous().cpu().int().numpy()
                )
            if self.is_gaussian_indexed:
                save_dict["gaussian_indices"] = (
                    self._gaussian_indices.detach().contiguous().cpu().int().numpy()
                )

            # save scaling
            if self.quantization:
                scaling = self.scaling_activation(self._scaling.detach())
                scaling_q = torch.quantize_per_tensor(
                    scaling,
                    scale=self.scaling_qa.scale,
                    zero_point=self.scaling_qa.zero_point,
                    dtype=self.scaling_qa.dtype,
                ).int_repr()
                save_dict["scaling"] = scaling_q.cpu().numpy()
                save_dict["scaling_scale"] = self.scaling_qa.scale.cpu().numpy()
                save_dict[
                    "scaling_zero_point"
                ] = self.scaling_qa.zero_point.cpu().numpy()

            else:
                save_dict["scaling"] = self._scaling.detach().to(dtype).cpu().numpy()


            # save rotation
            if self.quantization:
                rotation = self.rotation_activation(self._rotation).detach()
                rotation_q = torch.quantize_per_tensor(
                    rotation,
                    scale=self.rotation_qa.scale,
                    zero_point=self.rotation_qa.zero_point,
                    dtype=self.rotation_qa.dtype,
                ).int_repr()
                save_dict["rotation"] = rotation_q.cpu().numpy()
                save_dict["rotation_scale"] = self.rotation_qa.scale.cpu().numpy()
                save_dict[
                    "rotation_zero_point"
                ] = self.rotation_qa.zero_point.cpu().numpy()
            else:
                save_dict["rotation"] = self._rotation.detach().to(dtype).cpu().numpy()

            save_fn = np.savez_compressed if compress else np.savez
            save_fn(path, **save_dict)

    def load_npz(self, path,override_quantization=False):
        state_dict = np.load(path)

        quantization = state_dict["quantization"]
        if not override_quantization and self.quantization != quantization:
            print("WARNING: model is not quantisation aware but loaded model is")
        if override_quantization:
            self.quantization = quantization

        # load position
        self._xyz = nn.Parameter(
            torch.from_numpy(state_dict["xyz"]).float().cuda(), requires_grad=True
        )

        # load color
        if quantization:
            features_rest_q = torch.from_numpy(state_dict["features_rest"]).int().cuda()
            features_rest_scale = torch.from_numpy(
                state_dict["features_rest_scale"]
            ).cuda()
            features_rest_zero_point = torch.from_numpy(
                state_dict["features_rest_zero_point"]
            ).cuda()
            features_rest = (
                features_rest_q - features_rest_zero_point
            ) * features_rest_scale
            self._features_rest = nn.Parameter(features_rest, requires_grad=True)
            self.features_rest_qa.scale = features_rest_scale
            self.features_rest_qa.zero_point = features_rest_zero_point
            self.features_rest_qa.activation_post_process.min_val = features_rest.min()
            self.features_rest_qa.activation_post_process.max_val = features_rest.max()

            features_dc_q = torch.from_numpy(state_dict["features_dc"]).int().cuda()
            features_dc_scale = torch.from_numpy(state_dict["features_dc_scale"]).cuda()
            features_dc_zero_point = torch.from_numpy(
                state_dict["features_dc_zero_point"]
            ).cuda()
            features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
            self._features_dc = nn.Parameter(features_dc, requires_grad=True)

            self.features_dc_qa.scale = features_dc_scale
            self.features_dc_qa.zero_point = features_dc_zero_point
            self.features_dc_qa.activation_post_process.min_val = features_dc.min()
            self.features_dc_qa.activation_post_process.max_val = features_dc.max()

        else:
            features_dc = torch.from_numpy(state_dict["features_dc"]).float().cuda()
            features_rest = torch.from_numpy(state_dict["features_rest"]).float().cuda()
            self._features_dc = nn.Parameter(features_dc, requires_grad=True)
            self._features_rest = nn.Parameter(features_rest, requires_grad=True)

        # load opacity
        if quantization:
            opacity_q = torch.from_numpy(state_dict["opacity"]).int().cuda()
            opacity_scale = torch.from_numpy(state_dict["opacity_scale"]).cuda()
            opacity_zero_point = torch.from_numpy(
                state_dict["opacity_zero_point"]
            ).cuda()
            opacity = (opacity_q - opacity_zero_point) * opacity_scale
            self._opacity = nn.Parameter(
                self.inverse_opacity_activation(opacity), requires_grad=True
            )
            self.opacity_qa.scale = opacity_scale
            self.opacity_qa.zero_point = opacity_zero_point
            self.opacity_qa.activation_post_process.min_val = opacity.min()
            self.opacity_qa.activation_post_process.max_val = opacity.max()

        else:
            self._opacity = nn.Parameter(
                torch.from_numpy(state_dict["opacity"]).float().cuda(),
                requires_grad=True,
            )

        # load scaling
        if quantization:
            scaling_q = torch.from_numpy(state_dict["scaling"]).int().cuda()
            scaling_scale = torch.from_numpy(state_dict["scaling_scale"]).cuda()
            scaling_zero_point = torch.from_numpy(
                state_dict["scaling_zero_point"]
            ).cuda()
            scaling = (scaling_q - scaling_zero_point) * scaling_scale
            self._scaling = nn.Parameter(
                self.scaling_inverse_activation(scaling), requires_grad=True
            )
            self.scaling_qa.scale = scaling_scale
            self.scaling_qa.zero_point = scaling_zero_point
            self.scaling_qa.activation_post_process.min_val = scaling.min()
            self.scaling_qa.activation_post_process.max_val = scaling.max()
        else:
            self._scaling = torch.tensor(1.0).cuda()

        # load rotation
        if quantization:
            rotation_q = torch.from_numpy(state_dict["rotation"]).int().cuda()
            rotation_scale = torch.from_numpy(state_dict["rotation_scale"]).cuda()
            rotation_zero_point = torch.from_numpy(
                state_dict["rotation_zero_point"]
            ).cuda()
            rotation = (rotation_q - rotation_zero_point) * rotation_scale
            self._rotation = nn.Parameter(rotation, requires_grad=True)
            self.rotation_qa.scale = rotation_scale
            self.rotation_qa.zero_point = rotation_zero_point
            self.rotation_qa.activation_post_process.min_val = rotation.min()
            self.rotation_qa.activation_post_process.max_val = rotation.max()
        else:
            self._rotation = nn.Parameter(
                torch.from_numpy(state_dict["rotation"]).float().cuda(),
                requires_grad=True,
            )

        if "gaussian_indices" in list(state_dict.keys()):
            self._gaussian_indices = nn.Parameter(
                torch.from_numpy(state_dict["gaussian_indices"]).long().to("cuda"),
                requires_grad=False,
            )

        self.color_index_mode = ColorMode.NOT_INDEXED
        if "feature_indices" in list(state_dict.keys()):
            self._feature_indices = nn.Parameter(
                torch.from_numpy(state_dict["feature_indices"]).long().to("cuda"),
                requires_grad=False,
            )
            self.color_index_mode = ColorMode.ALL_INDEXED

        self.active_sh_degree = self.max_sh_degree

    def _sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)

            if self.is_color_indexed:
                self._feature_indices = nn.Parameter(
                    self._feature_indices[order], requires_grad=False
                )
            else:
                self._features_rest = nn.Parameter(
                    self._features_rest[order], requires_grad=True
                )
                self._features_dc = nn.Parameter(
                    self._features_dc[order], requires_grad=True
                )

            if self.is_gaussian_indexed:
                self._gaussian_indices = nn.Parameter(
                    self._gaussian_indices[order], requires_grad=False
                )
            else:
                self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
                self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)

    def mask_splats(self, mask: torch.Tensor):
        with torch.no_grad():
            self._xyz = nn.Parameter(self._xyz[mask], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[mask], requires_grad=True)


            if self.is_color_indexed:
                self._feature_indices = nn.Parameter(
                    self._feature_indices[mask], requires_grad=False
                )
            else:
                self._features_dc = nn.Parameter(
                    self._features_dc[mask], requires_grad=True
                )
                self._features_rest = nn.Parameter(
                    self._features_rest[mask], requires_grad=True
                )
            if self.is_gaussian_indexed:
                self._gaussian_indices = nn.Parameter(
                    self._gaussian_indices[mask], requires_grad=False
                )
            else:
                self._scaling = nn.Parameter(self._scaling[mask], requires_grad=True)
                self._rotation = nn.Parameter(self._rotation[mask], requires_grad=True)

    def set_color_indexed(self, features: torch.Tensor, indices: torch.Tensor):
        self._feature_indices = nn.Parameter(indices, requires_grad=False)
        self._features_dc = nn.Parameter(features[:, :1].detach(), requires_grad=True)
        self._features_rest = nn.Parameter(features[:, 1:].detach(), requires_grad=True)
        self.color_index_mode = ColorMode.ALL_INDEXED

    def set_gaussian_indexed(
        self, rotation: torch.Tensor, scaling: torch.Tensor, indices: torch.Tensor
    ):
        self._gaussian_indices = nn.Parameter(indices.detach(), requires_grad=False)
        self._rotation = nn.Parameter(rotation.detach(), requires_grad=True)
        self._scaling = nn.Parameter(scaling.detach(), requires_grad=True)


class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def splitBy3(a):
    x = a & 0x1FFFFF  # we only look at the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor) -> torch.Tensor:
    x, y, z = pos.unbind(-1)
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer