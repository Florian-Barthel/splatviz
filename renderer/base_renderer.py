import os
import traceback

import cv2
import torch
import torch.nn

from splatviz_utils.dict_utils import EasyDict


class Renderer:
    def __init__(self):
        self._device = torch.device("cuda")
        self._pinned_bufs = dict()
        self._is_timing = False
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = EasyDict()
        try:
            self._render_impl(res, **args)
        except Exception as e:
            res.error = "".join(traceback.format_exception(e))
            res.error += str(e)
        self._end_event.record(torch.cuda.current_stream(self._device))
        if "image" in res:
            res.image = res.image
        if "error" in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def _render_impl(self, **args):
        raise NotImplementedError

    def _load_model(self, path):
        raise NotImplementedError

    @staticmethod
    def _return_image(
        images,
        res: dict,
        normalize: bool,
        use_splitscreen: bool = False,
        layout: str = "side_by_side",
        grid_shape = None,
        target_size = None,
        highlight_border: bool = False,
        on_top: bool = False,
        colormap = None,
        invert = False,
        scene_texts = None,
    ) -> None:

        if not isinstance(images, list):
            images = [images]

        scene_regions = Renderer._scene_regions(images, layout, use_splitscreen, grid_shape, target_size)
        if layout == "grid":
            img = Renderer._compose_grid(images, grid_shape, target_size, highlight_border)
        elif use_splitscreen:
            img = torch.zeros_like(images[0])
            split_size = img.shape[-1] // len(images)
            offset = 0
            for i in range(len(images)):
                img[..., offset : offset + split_size] = images[i][..., offset : offset + split_size]
                offset += split_size
                if highlight_border and i != len(images) - 1:
                    img[..., offset - 1 : offset] = 1

        elif on_top:
            mask = torch.mean(images[1], dim=0)
            img = images[0] * (1 - mask) + images[1] * mask
        elif len(images) == 1:
            img = images[0]
        else:
            img = torch.concat(images, dim=2)

        # Scale and convert to uint8.
        if normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if invert:
            img = 255 - img
        img = img.cpu().numpy().copy()
        if colormap is not None:
            img = cv2.applyColorMap(img, colormap)
        Renderer._draw_scene_texts(img, scene_texts, scene_regions)
        res.image = img

    @staticmethod
    def _compose_grid(images, grid_shape, target_size, highlight_border):
        if grid_shape is None or target_size is None:
            raise ValueError("Grid layout requires grid_shape and target_size.")

        rows, cols = grid_shape
        target_width, target_height = target_size
        row_heights = Renderer._split_extent(target_height, rows)
        col_widths = Renderer._split_extent(target_width, cols)
        img = images[0].new_zeros((images[0].shape[0], target_height, target_width))

        for scene_index, scene_img in enumerate(images):
            row = scene_index // cols
            col = scene_index % cols
            x = sum(col_widths[:col])
            y = sum(row_heights[:row])
            img[..., y : y + row_heights[row], x : x + col_widths[col]] = scene_img

        if highlight_border:
            for col in range(1, cols):
                x = sum(col_widths[:col])
                img[..., :, x - 1 : x] = 1
            for row in range(1, rows):
                y = sum(row_heights[:row])
                img[..., y - 1 : y, :] = 1

        return img

    @staticmethod
    def _scene_regions(images, layout, use_splitscreen, grid_shape, target_size):
        if len(images) == 0:
            return []

        if layout == "grid":
            rows, cols = grid_shape
            target_width, target_height = target_size
            row_heights = Renderer._split_extent(target_height, rows)
            col_widths = Renderer._split_extent(target_width, cols)
            regions = []
            for scene_index in range(len(images)):
                row = scene_index // cols
                col = scene_index % cols
                x = sum(col_widths[:col])
                y = sum(row_heights[:row])
                regions.append((x, y, col_widths[col], row_heights[row]))
            return regions

        if use_splitscreen:
            height, width = images[0].shape[-2:]
            split_size = width // len(images)
            return [(scene_index * split_size, 0, split_size, height) for scene_index in range(len(images))]

        regions = []
        x = 0
        for image in images:
            height, width = image.shape[-2:]
            regions.append((x, 0, width, height))
            x += width
        return regions

    @staticmethod
    def _draw_scene_texts(img, scene_texts, scene_regions):
        if scene_texts is None:
            return

        for text, (x, y, width, height) in zip(scene_texts, scene_regions):
            text = str(text).strip()
            if not text:
                continue

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(min(width, height) / 360, 0.8)
            margin = max(round(scale * 14), 12)
            max_text_width = max(width - margin * 2, 1)
            thickness = max(round(scale * 2), 2)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

            if text_width > max_text_width:
                scale *= max_text_width / text_width
                thickness = max(round(scale * 2), 1)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

            text_x = x + max((width - text_width) // 2, margin)
            text_y = y + margin + text_height
            border_thickness = thickness + max(round(scale * 2), 2)
            cv2.putText(img, text, (text_x, text_y), font, scale, (255, 255, 255), border_thickness, cv2.LINE_AA)
            cv2.putText(img, text, (text_x, text_y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    @staticmethod
    def _split_extent(extent, parts):
        base = extent // parts
        remainder = extent % parts
        return [base + (1 if index < remainder else 0) for index in range(parts)]

    @staticmethod
    def save_ply(gaussian, save_ply_path):
        if not save_ply_path.endswith(".ply"):
            os.makedirs(save_ply_path, exist_ok=True)
            save_ply_path = os.path.join(save_ply_path, f"model_{len(os.listdir(save_ply_path))}.ply")
        print("Model saved in", save_ply_path)
        gaussian.save_ply(save_ply_path)
