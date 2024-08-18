import numpy as np
import torch
import os
from scene import GaussianModel
import yaml
import pandas as pd

from compression.codecs.jpeg_xl import JpegXlCodec
from compression.codecs.npz import NpzCodec
from compression.codecs.exr import EXRCodec
from compression.codecs.png import PNGCodec

codecs = {
    "jpeg-xl": JpegXlCodec,
    "npz": NpzCodec,
    "exr": EXRCodec,
    "png": PNGCodec,
    # 'png': compress_png,
    # 'webp': compress_webp,
    # 'bzip2': compress_bzip2,
}


def inverse_log_transform(transformed_coords):
    positive = transformed_coords > 0
    negative = transformed_coords < 0
    zero = transformed_coords == 0
    original_coords = np.zeros_like(transformed_coords)
    original_coords[positive] = np.expm1(transformed_coords[positive])
    original_coords[negative] = -np.expm1(-transformed_coords[negative])
    # For zero, no change is needed as original_coords is already initialized to zeros
    return original_coords


def decompress_attr(gaussians, attr_config, compressed_file, min_val, max_val):
    attr_name = attr_config["name"]
    attr_method = attr_config["method"]
    codec = codecs[attr_method]()

    if attr_config.get("normalize", False):
        decompressed_attr = codec.decode_with_normalization(compressed_file, min_val, max_val)
    else:
        decompressed_attr = codec.decode(compressed_file)

    if attr_config.get("contract", False):
        decompressed_attr = inverse_log_transform(decompressed_attr)

    if isinstance(decompressed_attr, np.ndarray):
        decompressed_attr = torch.tensor(decompressed_attr).to("cuda")
    gaussians.set_attr_from_grid_img(attr_name, decompressed_attr)


def run_single_decompression(compressed_dir):
    compr_info = pd.read_csv(os.path.join(compressed_dir, "compression_info.csv"), index_col=0)
    with open(os.path.join(compressed_dir, "compression_config.yml"), "r") as stream:
        experiment_config = yaml.safe_load(stream)
    disable_xyz_log_activation = experiment_config.get("disable_xyz_log_activation")
    if disable_xyz_log_activation is None:
        disable_xyz_log_activation = True
    decompressed_gaussians = GaussianModel(experiment_config["max_sh_degree"], disable_xyz_log_activation)
    decompressed_gaussians.active_sh_degree = experiment_config["active_sh_degree"]

    for attribute in experiment_config["attributes"]:
        attr_name = attribute["name"]
        # compressed_bytes = compressed_attrs[attr_name]
        compressed_file = os.path.join(compressed_dir, compr_info.loc[attr_name, "file"])
        decompress_attr(
            decompressed_gaussians,
            attribute,
            compressed_file,
            compr_info.loc[attr_name, "min"],
            compr_info.loc[attr_name, "max"],
        )
    return decompressed_gaussians
