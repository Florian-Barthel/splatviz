import re
from typing import List
import torch
import torch.nn

from viz.render_utils import CapturedException
from viz_utils.dict import EasyDict


class Renderer:
    def __init__(self):
        self._device = torch.device("cuda")
        self._pinned_bufs = dict()
        self._is_timing = False
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._net_layers = dict()
        self._last_model_input = None

    def render(self, **args):
        self._is_timing = True
        self._start_event.record(torch.cuda.current_stream(self._device))
        res = EasyDict()
        try:
            with torch.no_grad():
                self._render_impl(res, **args)
        except:
            res.error = CapturedException()
        self._end_event.record(torch.cuda.current_stream(self._device))
        if "image" in res:
            res.image = self.to_cpu(res.image).detach().numpy()
        if "stats" in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if "error" in res:
            res.error = str(res.error)
        if self._is_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    @staticmethod
    def sanitize_command(edit_text):
        command = re.sub(";+", ";", edit_text.replace("\n", ";"))
        while command.startswith(";"):
            command = command[1:]
        return command

    def _render_impl(
        self,
        res,
        fov,
        edit_text,
        eval_text,
        size,
        ply_file_path,
        cam_params,
        current_ply_names,
        video_cams=[],
        render_depth=False,
        render_alpha=False,
        img_normalize=False,
        **slider,
    ):
        raise NotImplementedError

    def _load_model(self, path):
        raise NotImplementedError

    @staticmethod
    def _return_image(
        images: List[torch.Tensor],
        res: dict,
        normalize: bool,
        use_splitscreen: bool,
        highlight_border: bool,
        on_top: bool = False
    ) -> None:
        if use_splitscreen:
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

        res.stats = torch.stack(
            [
                img.mean(),
                img.mean(),
                img.std(),
                img.std(),
                img.norm(float("inf")),
                img.norm(float("inf")),
            ]
        )

        # Scale and convert to uint8.
        if normalize:
            img = img / img.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        res.image = img
