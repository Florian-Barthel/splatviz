from imgui_bundle import imgui
from gui_utils.easy_imgui import label, slider, checkbox
import torch
import numpy as np

from gui_utils import imgui_utils
from viz_utils.dict import EasyDict
from viz_utils.camera_utils import (
    get_forward_vector,
    create_cam2world_matrix,
    get_origin,
    normalize_vecs,
)
from widgets.widget import Widget


class CamWidget(Widget):
    def __init__(self, viz):
        super().__init__(viz, "Camera")
        self.fov = 45
        self.radius = 3
        self.lookat_point = torch.tensor((0.0, 0.0, 0.0), device="cuda")
        self.cam_pos = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        self.up_vector = torch.tensor([0.0, -1.0, 0.0], device="cuda")
        self.forward = torch.tensor([0.0, 0.0, -1.0], device="cuda")

        # controls
        self.pose = EasyDict(yaw=0, pitch=0)
        self.invert_x = False
        self.invert_y = False
        self.move_speed = 0.02
        self.wasd_move_speed = 0.1
        self.drag_speed = 0.005
        self.rotate_speed = 0.02
        self.control_modes = ["Orbit", "WASD"]
        self.current_control_mode = 0
        self.last_drag_delta = imgui.ImVec2(0, 0)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show: bool):
        viz = self.viz
        active_region = EasyDict(x=viz.pane_w, y=0, width=viz.content_width - viz.pane_w, height=viz.content_height)
        self.handle_dragging_in_window(**active_region)
        self.handle_mouse_wheel()
        self.handle_wasd()

        if show:
            label("Camera Mode", viz.label_w)
            _, self.current_control_mode = imgui.combo("##cam_modes", self.current_control_mode, self.control_modes)

            if self.control_modes[self.current_control_mode] == "WASD":
                label("Move Speed", viz.label_w)
                self.wasd_move_speed = slider(self.wasd_move_speed, "move_speed", 0.001, 1, log=True)

            label("Drag Speed", viz.label_w)
            self.drag_speed = slider(self.drag_speed, "drag_speed", 0.001, 0.1, log=True)

            label("Rotate Speed", viz.label_w)
            self.rotate_speed = slider(self.rotate_speed, "rot_speed", 0.001, 0.1, log=True)

            imgui.push_item_width(200)
            label("Up Vector", viz.label_w)
            _changed, up_vector_tuple = imgui.input_float3("##up_vector", v=self.up_vector.tolist(), format="%.1f")
            if _changed:
                self.up_vector = torch.tensor(up_vector_tuple, device="cuda")

            imgui.same_line()
            if imgui_utils.button("Set current direction", width=viz.button_large_w):
                self.up_vector = self.forward
                self.pose.yaw = 0
                self.pose.pitch = 0

            imgui.same_line()
            if imgui_utils.button("Flip", width=viz.button_w):
                self.up_vector = -self.up_vector

            label("FOV", viz.label_w)
            self.fov = slider(self.fov, "##fov", 1, 180, format="%.1f °")

            if self.control_modes[self.current_control_mode] == "Orbit":
                label("Radius", viz.label_w)
                self.radius = slider(self.radius, "##radius", 1, 20, format="%.1f °")

                imgui.same_line()
                if imgui_utils.button("Set to xyz stddev", width=viz.button_large_w) and "std_xyz" in viz.result.keys():
                    self.radius = viz.result.std_xyz.item()

                label("Look at Point", viz.label_w)
                _, look_at_point_tuple = imgui.input_float3("##lookat", self.lookat_point.tolist(), format="%.1f")
                self.lookat_point = torch.tensor(look_at_point_tuple, device=torch.device("cuda"))
                imgui.same_line()
                if imgui_utils.button("Set to xyz mean", width=viz.button_large_w) and "mean_xyz" in viz.result.keys():
                    self.lookat_point = viz.result.mean_xyz
            imgui.pop_item_width()

            label("Invert X", viz.label_w)
            self.invert_x = checkbox(self.invert_x, "invert_x")
            label("Invert Y", viz.label_w)
            self.invert_y = checkbox(self.invert_y, "invert_y")

        self.cam_params = create_cam2world_matrix(self.forward, self.cam_pos, self.up_vector).to("cuda")[0]
        viz.args.yaw = self.pose.yaw
        viz.args.pitch = self.pose.pitch
        viz.args.fov = self.fov
        viz.args.cam_params = self.cam_params

    def handle_dragging_in_window(self, x, y, width, height):
        x_dir = -1 if self.invert_x else 1
        y_dir = -1 if self.invert_y else 1

        if imgui.is_mouse_dragging(0):  # left mouse button
            new_delta = imgui.get_mouse_drag_delta(0)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta
                self.pose.yaw += x_dir * delta.x * self.rotate_speed * 0.1
                self.pose.pitch += y_dir * delta.y * self.rotate_speed * 0.1
                self.pose.pitch = np.clip(self.pose.pitch, -np.pi / 2, np.pi / 2)
        elif imgui.is_mouse_dragging(1):  # middle mouse button
            # TODO: dragging with the middle mouse button could be used for yet another purpose
            pass
        elif imgui.is_mouse_dragging(2):  # right mouse button
            new_delta = imgui.get_mouse_drag_delta(2)
            if imgui_utils.did_drag_start_in_window(x, y, width, height, new_delta):
                delta = new_delta - self.last_drag_delta
                self.last_drag_delta = new_delta

                right = torch.linalg.cross(self.forward, self.up_vector)
                right = right / torch.linalg.norm(right)
                cam_up = torch.linalg.cross(right, self.forward)
                cam_up = cam_up / torch.linalg.norm(cam_up)

                x_change = x_dir * right * -delta.x * self.drag_speed
                y_change = y_dir * cam_up * delta.y * self.drag_speed
                self.cam_pos += x_change
                self.cam_pos += y_change
                if self.control_modes[self.current_control_mode] == "Orbit":
                    self.lookat_point += x_change
                    self.lookat_point += y_change
        else:
            self.last_drag_delta = imgui.ImVec2(0, 0)

    def handle_wasd(self):
        if self.control_modes[self.current_control_mode] == "WASD":
            self.forward = get_forward_vector(
                lookat_position=self.cam_pos,
                horizontal_mean=self.pose.yaw + np.pi / 2,
                vertical_mean=self.pose.pitch + np.pi / 2,
                radius=0.01,
                up_vector=self.up_vector,
            )
            self.sideways = torch.linalg.cross(self.forward, self.up_vector)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.cam_pos += self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.cam_pos -= self.sideways * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.cam_pos -= self.forward * self.wasd_move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.cam_pos += self.sideways * self.wasd_move_speed

        elif self.control_modes[self.current_control_mode] == "Orbit":
            self.cam_pos = get_origin(
                self.pose.yaw + np.pi / 2,
                self.pose.pitch + np.pi / 2,
                self.radius,
                self.lookat_point,
                device=torch.device("cuda"),
                up_vector=self.up_vector,
            )
            self.forward = normalize_vecs(self.lookat_point - self.cam_pos)
            if imgui.is_key_down(imgui.Key.up_arrow) or "w" in self.viz.current_pressed_keys:
                self.pose.pitch += self.move_speed
            if imgui.is_key_down(imgui.Key.left_arrow) or "a" in self.viz.current_pressed_keys:
                self.pose.yaw += self.move_speed
            if imgui.is_key_down(imgui.Key.down_arrow) or "s" in self.viz.current_pressed_keys:
                self.pose.pitch -= self.move_speed
            if imgui.is_key_down(imgui.Key.right_arrow) or "d" in self.viz.current_pressed_keys:
                self.pose.yaw -= self.move_speed

    def handle_mouse_wheel(self):
        mouse_pos = imgui.get_io().mouse_pos
        if mouse_pos.x >= self.viz.pane_w:
            wheel = imgui.get_io().mouse_wheel
            if self.control_modes[self.current_control_mode] == "WASD":
                self.cam_pos += self.forward * self.move_speed * wheel
            elif self.control_modes[self.current_control_mode] == "Orbit":
                self.radius -= wheel / 10
