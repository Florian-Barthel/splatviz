# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import functools
from typing import Optional

import numpy as np
import PIL.Image
import PIL.ImageFont
import scipy.ndimage

from splatviz_utils.gui_utils.url import open_url
from . import gl_utils


# ----------------------------------------------------------------------------


def get_default_font():
    url = "http://fonts.gstatic.com/s/opensans/v17/mem8YaGs126MiZpBA-U1UpcaXcl0Aw.ttf"  # Open Sans regular
    return open_url(url, return_filename=True)


# ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def get_pil_font(font=None, size=32):
    if font is None:
        font = get_default_font()
    return PIL.ImageFont.truetype(font=font, size=size)


# ----------------------------------------------------------------------------


def get_array(string, *, dropshadow_radius: int = None, **kwargs):
    if dropshadow_radius is not None:
        offset_x = int(np.ceil(dropshadow_radius * 2 / 3))
        offset_y = int(np.ceil(dropshadow_radius * 2 / 3))
        return _get_array_priv(
            string, dropshadow_radius=dropshadow_radius, offset_x=offset_x, offset_y=offset_y, **kwargs
        )
    else:
        return _get_array_priv(string, **kwargs)


@functools.lru_cache(maxsize=10000)
def _get_array_priv(
    string: str,
    *,
    size: int = 32,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    min_size=10,
    shrink_coef=0.8,
    dropshadow_radius: int = None,
    offset_x: int = None,
    offset_y: int = None,
    **kwargs
):
    def render(render_size):
        if dropshadow_radius is not None:
            # separate implementation for dropshadow text rendering
            return _get_array_impl_dropshadow(
                string, size=render_size, radius=dropshadow_radius, offset_x=offset_x, offset_y=offset_y, **kwargs
            )
        return _get_array_impl(string, size=render_size, **kwargs)

    def fits(array):
        height, width, _ = array.shape
        return (max_width is None or width <= max_width) and (max_height is None or height <= max_height)

    size = max(int(size), int(min_size))
    if max_width is None and max_height is None:
        return render(size)

    array = render(size)
    if fits(array):
        best_size, best_array = size, array
        hard_cap = max(size, int(max_width or 0), int(max_height or 0), int(min_size)) * 4
        hi = size
        while hi < hard_cap:
            hi = min(max(hi + 1, int(np.ceil(hi / shrink_coef))), hard_cap)
            array = render(hi)
            if not fits(array):
                break
            best_size, best_array = hi, array
        lo = best_size + 1
    else:
        best_array = None
        lo = int(min_size)
        hi = size

    while lo < hi:
        mid = (lo + hi) // 2
        array = render(mid)
        if fits(array):
            best_array = array
            lo = mid + 1
        else:
            hi = mid

    return best_array if best_array is not None else render(int(min_size))


# ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=10000)
def _get_array_impl(
    string, *, font=None, size=32, outline=0, outline_pad=3, outline_coef=3, outline_exp=2, line_pad: int = None
):
    pil_font = get_pil_font(font=font, size=size)
    lines = [pil_font.getmask(line, "L") for line in string.split("\n")]
    lines = [np.array(line, dtype=np.uint8).reshape([line.size[1], line.size[0]]) for line in lines]
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1])), mode="constant") for line in lines]
    line_spacing = line_pad if line_pad is not None else size // 2
    lines = [np.pad(line, ((0, line_spacing), (0, 0)), mode="constant") for line in lines[:-1]] + lines[-1:]
    mask = np.concatenate(lines, axis=0)
    alpha = mask
    if outline > 0:
        mask = np.pad(mask, int(np.ceil(outline * outline_pad)), mode="constant", constant_values=0)
        alpha = mask.astype(np.float32) / 255
        alpha = scipy.ndimage.gaussian_filter(alpha, outline)
        alpha = 1 - np.maximum(1 - alpha * outline_coef, 0) ** outline_exp
        alpha = (alpha * 255 + 0.5).clip(0, 255).astype(np.uint8)
        alpha = np.maximum(alpha, mask)
    return _compose_text_rgba(mask, alpha, shadow=outline > 0)


# ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=10000)
def _get_array_impl_dropshadow(
    string, *, font=None, size=32, radius: int, offset_x: int, offset_y: int, line_pad: int = None, **kwargs
):
    assert (offset_x > 0) and (offset_y > 0)
    pil_font = get_pil_font(font=font, size=size)
    lines = [pil_font.getmask(line, "L") for line in string.split("\n")]
    lines = [np.array(line, dtype=np.uint8).reshape([line.size[1], line.size[0]]) for line in lines]
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1])), mode="constant") for line in lines]
    line_spacing = line_pad if line_pad is not None else size // 2
    lines = [np.pad(line, ((0, line_spacing), (0, 0)), mode="constant") for line in lines[:-1]] + lines[-1:]
    mask = np.concatenate(lines, axis=0)
    alpha = mask

    mask = np.pad(mask, 2 * radius + max(abs(offset_x), abs(offset_y)), mode="constant", constant_values=0)
    alpha = mask.astype(np.float32) / 255
    alpha = scipy.ndimage.gaussian_filter(alpha, radius)
    alpha = 1 - np.maximum(1 - alpha * 1.5, 0) ** 1.4
    alpha = (alpha * 255 + 0.5).clip(0, 255).astype(np.uint8)
    alpha = np.pad(alpha, [(offset_y, 0), (offset_x, 0)], mode="constant")[:-offset_y, :-offset_x]
    alpha = np.maximum(alpha, mask)
    return _compose_text_rgba(mask, alpha, shadow=True)


def _compose_text_rgba(mask, alpha, *, shadow=False):
    mask = np.asarray(mask, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    rgb = np.full(mask.shape + (3,), 255, dtype=np.uint8)
    if shadow:
        rgb[mask == 0] = 0
    return np.concatenate([rgb, alpha[:, :, np.newaxis]], axis=-1)


# ----------------------------------------------------------------------------


@functools.lru_cache(maxsize=10000)
def get_texture(string, bilinear=True, mipmap=True, **kwargs):
    return gl_utils.Texture(image=get_array(string, **kwargs), bilinear=bilinear, mipmap=mipmap)


# ----------------------------------------------------------------------------
