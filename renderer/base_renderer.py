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
        if "stats" in res:
            res.stats = res.stats.cpu().detach().numpy()
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
        invert = False
    ) -> None:

        if not isinstance(images, list):
            images = [images]

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
        else:
            img = torch.concat(images, dim=2)

        res.stats = torch.stack([img.mean(), img.std()])

        # Scale and convert to uint8.
        if normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if invert:
            img = 255 - img
        img = img.cpu().numpy()
        if colormap is not None:
            img = cv2.applyColorMap(img, colormap)
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
