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
import contextlib
import numpy as np
import OpenGL.GL as gl
import OpenGL.GL.ARB.texture_float
from splatviz_utils.dict_utils import EasyDict


_texture_formats = {
    ("uint8", 1): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE, internalformat=gl.GL_LUMINANCE8),
    ("uint8", 2): EasyDict(
        type=gl.GL_UNSIGNED_BYTE, format=gl.GL_LUMINANCE_ALPHA, internalformat=gl.GL_LUMINANCE8_ALPHA8
    ),
    ("uint8", 3): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGB, internalformat=gl.GL_RGB8),
    ("uint8", 4): EasyDict(type=gl.GL_UNSIGNED_BYTE, format=gl.GL_RGBA, internalformat=gl.GL_RGBA8),
    ("float32", 1): EasyDict(
        type=gl.GL_FLOAT, format=gl.GL_LUMINANCE, internalformat=OpenGL.GL.ARB.texture_float.GL_LUMINANCE32F_ARB
    ),
    ("float32", 2): EasyDict(
        type=gl.GL_FLOAT,
        format=gl.GL_LUMINANCE_ALPHA,
        internalformat=OpenGL.GL.ARB.texture_float.GL_LUMINANCE_ALPHA32F_ARB,
    ),
    ("float32", 3): EasyDict(type=gl.GL_FLOAT, format=gl.GL_RGB, internalformat=gl.GL_RGB32F),
    ("float32", 4): EasyDict(type=gl.GL_FLOAT, format=gl.GL_RGBA, internalformat=gl.GL_RGBA32F),
}


def get_texture_format(dtype, channels):
    return _texture_formats[(np.dtype(dtype).name, int(channels))]


def prepare_texture_data(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.dtype.name == "float64":
        image = image.astype("float32")
    return image


class Texture:
    def __init__(self, *, image=None, width=None, height=None, channels=None, dtype=None, bilinear=True, mipmap=True):
        self.gl_id = None
        self.bilinear = bilinear
        self.mipmap = mipmap

        # Determine size and dtype.
        if image is not None:
            image = prepare_texture_data(image)
            self.height, self.width, self.channels = image.shape
            self.dtype = image.dtype
        else:
            assert width is not None and height is not None
            self.width = width
            self.height = height
            self.channels = channels if channels is not None else 3
            self.dtype = np.dtype(dtype) if dtype is not None else np.uint8

        # Validate size and dtype.
        assert isinstance(self.width, int) and self.width >= 0
        assert isinstance(self.height, int) and self.height >= 0
        assert isinstance(self.channels, int) and self.channels >= 1
        assert self.is_compatible(width=width, height=height, channels=channels, dtype=dtype)

        # Create texture object.
        self.gl_id = gl.glGenTextures(1)
        with self.bind():
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR if self.bilinear else gl.GL_NEAREST
            )
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR if self.mipmap else gl.GL_NEAREST
            )
        self.update(image)

    def delete(self):
        if self.gl_id is not None:
            gl.glDeleteTextures([self.gl_id])
            self.gl_id = None

    def __del__(self):
        try:
            self.delete()
        except:
            pass

    @contextlib.contextmanager
    def bind(self):
        prev_id = gl.glGetInteger(gl.GL_TEXTURE_BINDING_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.gl_id)
        yield
        gl.glBindTexture(gl.GL_TEXTURE_2D, prev_id)

    def update(self, image):
        if image is not None:
            image = prepare_texture_data(image)
            assert self.is_compatible(image=image)
        with self.bind():
            fmt = get_texture_format(self.dtype, self.channels)
            gl.glPushClientAttrib(gl.GL_CLIENT_PIXEL_STORE_BIT)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, fmt.internalformat, self.width, self.height, 0, fmt.format, fmt.type, image
            )
            if self.mipmap:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glPopClientAttrib()

    def draw(self, *, pos=0, zoom=1, align=0, rint=False, color=1, alpha=1, rounding=0):
        zoom = np.broadcast_to(np.asarray(zoom, dtype="float32"), [2])
        size = zoom * [self.width, self.height]
        with self.bind():
            gl.glPushAttrib(gl.GL_ENABLE_BIT)
            gl.glEnable(gl.GL_TEXTURE_2D)
            draw_rect(pos=pos, size=size, align=align, rint=rint, color=color, alpha=alpha, rounding=rounding)
            gl.glPopAttrib()

    def is_compatible(
        self, *, image=None, width=None, height=None, channels=None, dtype=None
    ):  # pylint: disable=too-many-return-statements
        if image is not None:
            if image.ndim != 3:
                return False
            ih, iw, ic = image.shape
            if not self.is_compatible(width=iw, height=ih, channels=ic, dtype=image.dtype):
                return False
        if width is not None and self.width != width:
            return False
        if height is not None and self.height != height:
            return False
        if channels is not None and self.channels != channels:
            return False
        if dtype is not None and self.dtype != dtype:
            return False
        return True


def draw_shape(vertices, *, mode=gl.GL_TRIANGLE_FAN, pos=0, size=1, color=1, alpha=1):
    assert vertices.ndim == 2 and vertices.shape[1] == 2
    pos = np.broadcast_to(np.asarray(pos, dtype="float32"), [2])
    size = np.broadcast_to(np.asarray(size, dtype="float32"), [2])
    color = np.broadcast_to(np.asarray(color, dtype="float32"), [3])
    alpha = np.clip(np.broadcast_to(np.asarray(alpha, dtype="float32"), []), 0, 1)

    gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
    gl.glPushAttrib(gl.GL_CURRENT_BIT | gl.GL_TRANSFORM_BIT)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()

    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
    gl.glVertexPointer(2, gl.GL_FLOAT, 0, vertices)
    gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, vertices)
    gl.glTranslate(pos[0], pos[1], 0)
    gl.glScale(size[0], size[1], 1)
    gl.glColor4f(color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha)
    gl.glDrawArrays(mode, 0, vertices.shape[0])

    gl.glPopMatrix()
    gl.glPopAttrib()
    gl.glPopClientAttrib()


def draw_rect(*, pos=0, pos2=None, size=None, align=0, rint=False, color=1, alpha=1, rounding=0):
    assert pos2 is None or size is None
    pos = np.broadcast_to(np.asarray(pos, dtype="float32"), [2])
    pos2 = np.broadcast_to(np.asarray(pos2, dtype="float32"), [2]) if pos2 is not None else None
    size = np.broadcast_to(np.asarray(size, dtype="float32"), [2]) if size is not None else None
    size = size if size is not None else pos2 - pos if pos2 is not None else np.array([1, 1], dtype="float32")
    pos = pos - size * align
    if rint:
        pos = np.rint(pos)
    rounding = np.broadcast_to(np.asarray(rounding, dtype="float32"), [2])
    rounding = np.minimum(np.abs(rounding) / np.maximum(np.abs(size), 1e-8), 0.5)
    if np.min(rounding) == 0:
        rounding *= 0
    vertices = _setup_rect(float(rounding[0]), float(rounding[1]))
    draw_shape(vertices, mode=gl.GL_TRIANGLE_FAN, pos=pos, size=size, color=color, alpha=alpha)


@functools.lru_cache(maxsize=10000)
def _setup_rect(rx, ry):
    t = np.linspace(0, np.pi / 2, 1 if max(rx, ry) == 0 else 64)
    s = 1 - np.sin(t)
    c = 1 - np.cos(t)
    x = [c * rx, 1 - s * rx, 1 - c * rx, s * rx]
    y = [s * ry, c * ry, 1 - s * ry, 1 - c * ry]
    v = np.stack([x, y], axis=-1).reshape(-1, 2)
    return v.astype("float32")
