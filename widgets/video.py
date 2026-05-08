import os
import time

import numpy as np
import torch
from imgui_bundle import imgui

from splatviz_utils.gui_utils.easy_imgui import label, slider
from splatviz_utils.gui_utils import imgui_utils
from widgets.widget import Widget


class VideoWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Video")
        self.checkpoints = []
        self.speed = 1.0
        self.fps = 30
        self.width = 1920
        self.height = 1080
        self.smoothing = 1.0
        self.momentum = 0.55
        self.preview_t = 0.0
        self.preview = False
        self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_videos"))
        self._pending_request = None
        self._status = ""

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        current_cam = self._current_cam()
        if "video_path" in viz.result:
            self._status = f"Saved to {viz.result.video_path}"

        if show:
            label("Checkpoints", viz.label_w)
            if imgui_utils.button("Add", width=viz.button_w) and current_cam is not None:
                self.checkpoints.append(current_cam.copy())
            imgui.same_line()
            if imgui_utils.button("Clear", width=viz.button_w):
                self.checkpoints = []

            for index, checkpoint in enumerate(self.checkpoints):
                label(f"{index + 1:02d}", viz.label_w)
                imgui.text(self._checkpoint_label(checkpoint))
                imgui.same_line()
                if imgui_utils.button(f"Use##{index}", width=viz.button_w, enabled=current_cam is not None):
                    self.checkpoints[index] = current_cam.copy()
                imgui.same_line()
                if imgui_utils.button(f"Up##{index}", width=viz.button_w, enabled=index > 0):
                    self.checkpoints[index - 1], self.checkpoints[index] = self.checkpoints[index], self.checkpoints[index - 1]
                    break
                imgui.same_line()
                if imgui_utils.button(f"Down##{index}", width=viz.button_w, enabled=index < len(self.checkpoints) - 1):
                    self.checkpoints[index + 1], self.checkpoints[index] = self.checkpoints[index], self.checkpoints[index + 1]
                    break
                imgui.same_line()
                if imgui_utils.button(f"Del##{index}", width=viz.button_w):
                    del self.checkpoints[index]
                    break

            with imgui_utils.item_width(viz.font_size * 7):
                label("Speed", viz.label_w)
                self.speed = slider(self.speed, "video_speed", 0.01, 20.0, format="%.2f u/s", log=True)

                label("FPS", viz.label_w)
                _changed, self.fps = imgui.input_int("##video_fps", self.fps)
                self.fps = min(max(int(self.fps), 1), 240)

                label("Width", viz.label_w)
                _changed, self.width = imgui.input_int("##video_width", self.width)
                self.width = min(max(int(self.width), 16), 8192)

                label("Height", viz.label_w)
                _changed, self.height = imgui.input_int("##video_height", self.height)
                self.height = min(max(int(self.height), 16), 8192)

                label("Smoothing", viz.label_w)
                self.smoothing = slider(self.smoothing, "video_smoothing", 0.0, 1.0, format="%.2f")

                label("Momentum", viz.label_w)
                self.momentum = slider(self.momentum, "video_momentum", 0.0, 0.95, format="%.2f")

                label("Preview", viz.label_w)
                _changed, self.preview = imgui.checkbox("##video_preview", self.preview)
                imgui.same_line()
                self.preview_t = slider(self.preview_t, "video_preview_t", 0.0, 1.0, format="%.3f")

            cams, duration = self._build_path()
            label("Duration", viz.label_w)
            imgui.text(f"{duration:.2f}s / {len(cams)} frames" if len(cams) else "Add at least 2 checkpoints")

            label("Export", viz.label_w)
            can_export = len(cams) > 1
            if imgui_utils.button("Render mp4", width=viz.button_large_w, enabled=can_export):
                os.makedirs(self.path, exist_ok=True)
                name = time.strftime("camera_path_%Y%m%d_%H%M%S.mp4")
                self._pending_request = (cams, os.path.join(self.path, name))
                self._status = f"Rendering {name}"
            if self._status:
                imgui.same_line()
                imgui.text(self._status)

        if self.preview:
            cams, _duration = self._build_path()
            if len(cams):
                index = min(int(round(self.preview_t * (len(cams) - 1))), len(cams) - 1)
                viz.args.cam_params = torch.tensor(cams[index])

        if self._pending_request is not None:
            cams, path = self._pending_request
            viz.args.video_cams = cams
            viz.args.video_width = self.width
            viz.args.video_height = self.height
            viz.args.video_fps = self.fps
            viz.args.video_path = path
            self._pending_request = None
        else:
            viz.args.video_cams = None
            viz.args.video_width = self.width
            viz.args.video_height = self.height
            viz.args.video_fps = self.fps
            viz.args.video_path = None

    def _current_cam(self):
        if "cam_params" not in self.viz.args:
            return None
        return self.viz.args.cam_params.detach().cpu().numpy().astype(np.float32)

    def _checkpoint_label(self, checkpoint):
        pos = checkpoint[:3, 3]
        return f"x {pos[0]:.2f}  y {pos[1]:.2f}  z {pos[2]:.2f}"

    def _build_path(self):
        if len(self.checkpoints) < 2:
            return np.empty((0, 4, 4), dtype=np.float32), 0.0

        positions = np.stack([checkpoint[:3, 3] for checkpoint in self.checkpoints], axis=0)
        dense_positions, dense_u = self._dense_position_path(positions)
        distances = np.linalg.norm(np.diff(dense_positions, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        path_length = float(cumulative[-1])
        if path_length <= 1e-6:
            return np.stack(self.checkpoints, axis=0).astype(np.float32), 0.0

        duration = path_length / max(self.speed, 1e-6)
        frame_count = max(int(np.ceil(duration * self.fps)) + 1, 2)
        target_distances = np.linspace(0.0, path_length, frame_count)

        cams = np.empty((frame_count, 4, 4), dtype=np.float32)
        rotations = np.stack([checkpoint[:3, :3] for checkpoint in self.checkpoints], axis=0)
        quats = np.stack([_quat_from_matrix(rot) for rot in rotations], axis=0)
        quats = _continuous_quats(quats)
        key_distances = np.arange(len(self.checkpoints), dtype=np.float32)

        for frame_index, distance in enumerate(target_distances):
            dense_index = min(np.searchsorted(cumulative, distance, side="right") - 1, len(distances) - 1)
            local = 0.0
            if distances[dense_index] > 1e-8:
                local = (distance - cumulative[dense_index]) / distances[dense_index]
            position = dense_positions[dense_index] * (1.0 - local) + dense_positions[dense_index + 1] * local
            key_u = np.interp(distance, cumulative, dense_u)
            rotation = _matrix_from_quat(_interpolate_quats(quats, key_distances, key_u, self.smoothing))
            cams[frame_index] = np.eye(4, dtype=np.float32)
            cams[frame_index, :3, :3] = rotation
            cams[frame_index, :3, 3] = position

        return _apply_drone_momentum(cams, self.fps, self.momentum), duration

    def _dense_position_path(self, positions):
        samples = []
        sample_u = []
        segments = len(positions) - 1
        for segment in range(segments):
            p0 = positions[max(segment - 1, 0)]
            p1 = positions[segment]
            p2 = positions[segment + 1]
            p3 = positions[min(segment + 2, len(positions) - 1)]
            count = 64
            for step in range(count):
                t = step / count
                linear = p1 * (1.0 - t) + p2 * t
                smooth = _centripetal_catmull_rom(p0, p1, p2, p3, t)
                samples.append(linear * (1.0 - self.smoothing) + smooth * self.smoothing)
                sample_u.append(segment + t)
        samples.append(positions[-1])
        sample_u.append(float(segments))
        return np.asarray(samples, dtype=np.float32), np.asarray(sample_u, dtype=np.float32)


def _centripetal_catmull_rom(p0, p1, p2, p3, t):
    knot0 = 0.0
    knot1 = _next_knot(knot0, p0, p1)
    knot2 = _next_knot(knot1, p1, p2)
    knot3 = _next_knot(knot2, p2, p3)
    if knot2 - knot1 <= 1e-8:
        return p1.copy()

    t = knot1 + (knot2 - knot1) * t
    a1 = _safe_lerp(p0, p1, knot0, knot1, t)
    a2 = _safe_lerp(p1, p2, knot1, knot2, t)
    a3 = _safe_lerp(p2, p3, knot2, knot3, t)
    b1 = _safe_lerp(a1, a2, knot0, knot2, t)
    b2 = _safe_lerp(a2, a3, knot1, knot3, t)
    return _safe_lerp(b1, b2, knot1, knot2, t)


def _next_knot(knot, p0, p1):
    return knot + max(float(np.linalg.norm(p1 - p0)), 1e-6) ** 0.5


def _safe_lerp(a, b, knot_a, knot_b, knot):
    if abs(knot_b - knot_a) <= 1e-8:
        return b.copy()
    t = (knot - knot_a) / (knot_b - knot_a)
    return a * (1.0 - t) + b * t


def _apply_drone_momentum(cams, fps, momentum):
    if len(cams) < 3 or momentum <= 1e-4:
        return cams

    momentum = min(max(float(momentum), 0.0), 0.95)
    dt = 1.0 / max(float(fps), 1.0)
    positions = cams[:, :3, 3]
    quats = _continuous_quats(np.stack([_quat_from_matrix(cam[:3, :3]) for cam in cams], axis=0))

    smooth_positions = np.empty_like(positions)
    smooth_positions[0] = positions[0]
    current = positions[0].copy()
    velocity = (positions[1] - positions[0]) / dt
    velocity_response = 1.0 - np.exp(-dt / (0.015 + momentum * 0.42))
    drift_response = 1.0 - np.exp(-dt / (0.08 + momentum * 1.2))

    for index in range(1, len(positions) - 1):
        desired_velocity = (positions[index + 1] - positions[index - 1]) / (2.0 * dt)
        velocity += (desired_velocity - velocity) * velocity_response
        current = current + velocity * dt
        current += (positions[index] - current) * drift_response
        smooth_positions[index] = current

    smooth_positions[-1] = positions[-1]
    smooth_positions = _ease_endpoints(smooth_positions, positions)

    smooth_quats = np.empty_like(quats)
    smooth_quats[0] = quats[0]
    current_quat = quats[0]
    attitude_response = 1.0 - np.exp(-dt / (0.02 + momentum * 0.55))
    for index in range(1, len(quats) - 1):
        current_quat = _slerp(current_quat, quats[index], attitude_response)
        smooth_quats[index] = current_quat
    smooth_quats[-1] = quats[-1]

    drone_cams = cams.copy()
    for index in range(len(cams)):
        drone_cams[index, :3, 3] = smooth_positions[index]
        drone_cams[index, :3, :3] = _matrix_from_quat(smooth_quats[index])
    return drone_cams


def _ease_endpoints(smooth_positions, target_positions):
    count = len(smooth_positions)
    if count < 4:
        return smooth_positions
    eased = smooth_positions.copy()
    edge_count = max(min(count // 6, 30), 2)
    for index in range(edge_count):
        blend = 1.0 - index / edge_count
        blend = blend * blend * (3.0 - 2.0 * blend)
        eased[index] = smooth_positions[index] * (1.0 - blend) + target_positions[index] * blend
        end_index = count - index - 1
        eased[end_index] = smooth_positions[end_index] * (1.0 - blend) + target_positions[end_index] * blend
    return eased


def _quat_from_matrix(matrix):
    trace = np.trace(matrix)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ]
            )
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ]
            )
    return _quat_normalize(quat)


def _matrix_from_quat(quat):
    w, x, y, z = _quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _continuous_quats(quats):
    quats = quats.copy()
    for index in range(1, len(quats)):
        if np.dot(quats[index - 1], quats[index]) < 0.0:
            quats[index] = -quats[index]
    return quats


def _interpolate_quats(quats, key_distances, u, smoothing):
    segment = min(np.searchsorted(key_distances, u, side="right") - 1, len(quats) - 2)
    segment = max(segment, 0)
    span = key_distances[segment + 1] - key_distances[segment]
    t = 0.0 if span <= 1e-8 else (u - key_distances[segment]) / span
    eased_t = t * (1.0 - smoothing) + (t * t * (3.0 - 2.0 * t)) * smoothing
    return _slerp(quats[segment], quats[segment + 1], eased_t)


def _slerp(a, b, t):
    a = _quat_normalize(a)
    b = _quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(a + t * (b - a))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    return a * (np.cos(theta) - dot * sin_theta / sin_theta_0) + b * (sin_theta / sin_theta_0)


def _quat_normalize(quat):
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm
