# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import imgui
import numpy as np
import torch
import pprint

from gui_utils import imgui_utils
from viz_utils.dict import EasyDict


class EvalWidget:
    def __init__(self, viz):
        self.viz = viz
        self.text = "gaussian"
        self.hist_cache = dict()
        self.use_cache_dict = dict()

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            _changed, self.text = imgui.input_text("", self.text)
            self.format()
        viz.args.eval_text = self.text

    def format(self):
        self.handle_type_rec(self.viz.eval_result, depth=20, obj_name="")

    def handle_type_rec(self, result, depth, obj_name):
        if hasattr(result, "__dict__") and len(result.__dict__.keys()) > 0 or isinstance(result, dict) and len(
                result.keys()) > 0:
            if isinstance(result, EasyDict):
                result = dict(result)
            elif hasattr(result, "__dict__"):
                result = result.__dict__

            sorted_keys = sorted(result.keys(), key=lambda x: type(result[x]).__name__)
            for key in sorted_keys:
                imgui.new_line()
                imgui.same_line(depth)
                info, primitive = self.get_short_info(key, result[key])
                if primitive:
                    imgui.new_line()
                    imgui.same_line(depth)
                    imgui.text(info)
                else:
                    expanded, _visible = imgui_utils.collapsing_header(info, default=False)
                    if expanded:
                        self.handle_type_rec(result[key], depth=depth + 20, obj_name=key)

        else:
            # write a non-primitive object that is not an object with __dict__
            if isinstance(result, torch.Tensor):
                self.handle_tensor(result, depth, obj_name)
            else:
                imgui.new_line()
                imgui.same_line(depth)
                imgui.text(pprint.pformat(result, compact=True))

    def handle_tensor(self, result, depth, var_name):
        imgui.new_line()
        imgui.same_line(depth)
        imgui.text(pprint.pformat(result, compact=True))
        if np.prod(list(result.shape)) > 0:
            imgui.new_line()
            imgui.same_line(depth)
            imgui.text(pprint.pformat(list(result.shape), compact=True))
            imgui.new_line()
            imgui.same_line(depth)
            imgui.text(pprint.pformat(
                f"min: {result.min().item():.2f}, max: {result.max().item():.2f}, mean:{result.std().mean():.2f}, std:{result.std().item():.2f}",
                compact=True))

        var_name += self.viz.args.ply_file_path
        if var_name not in self.use_cache_dict.keys():
            self.use_cache_dict[var_name] = True
        imgui.new_line()
        imgui.same_line(depth)
        _, self.use_cache_dict[var_name] = imgui.checkbox("Use Cache", self.use_cache_dict[var_name])
        if var_name not in self.hist_cache.keys() or not self.use_cache_dict[var_name]:
            hist = np.histogram(result.cpu().detach().numpy().reshape(-1), bins=50)
            self.hist_cache[var_name] = hist
        imgui.same_line()
        imgui.core.plot_histogram("", self.hist_cache[var_name][0].astype(np.float32))

    @staticmethod
    def get_short_info(key, value):
        readable_type = type(value).__name__
        primitives = (bool, str, int, float, type(None))
        spacing_type = 10
        spacing_name = 30

        if isinstance(value, primitives):
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}} {value}", True
        elif isinstance(value, torch.Tensor):
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}} shape={list(value.shape)}", False
        elif isinstance(value, dict) and len(value.keys()) == 0:
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}} {value}", True
        elif callable(value):
            readable_type = "function"
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}} {value}", True
        else:
            return f"{readable_type:<{spacing_type}} {key:<{spacing_name}}", False
