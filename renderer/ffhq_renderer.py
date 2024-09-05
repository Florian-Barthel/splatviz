from PIL import Image
from renderer.base_renderer import Renderer
import torchvision.transforms as transforms
from annotations.create_segmentations import api

class FFHQRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.last_command = ""
        self.reload_model = True
        self._current_img_path = ""
        self.image = None
        self.to_tensor = transforms.ToTensor()

    def _render_impl(
        self,
        res,
        resolution,
        ply_file_paths,
        current_ply_names,
        eval_text,
        create_segmentation=False,
        img_normalize=False,
        **other_args
    ):
        self.load(ply_file_paths[0])
        self._return_image(self.image, res, normalize=img_normalize)

        if create_segmentation:
            result = api.evaluate(self.image)[0]
            result = result.argmax(0)[None, ...]
            result = result / result.max()
            self._return_image(result, res, normalize=img_normalize)
        else:
            self._return_image(self.image, res, normalize=img_normalize)

        if len(eval_text) > 0:
            res.eval = eval(eval_text)


    def load(self, img_path):
        if img_path != self._current_img_path:
            self.image = Image.open(img_path)
            self.image = self.to_tensor(self.image)
            self._current_img_path = img_path

