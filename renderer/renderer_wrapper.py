import copy

from renderer.feature_processor import FeatureProcessor
from splatviz_utils.dict_utils import equal_dicts


class RendererWrapper:
    def __init__(self, renderer, update_all_the_time):
        self.renderer = renderer
        self._cur_args = None
        self.result = None
        self.update_all_the_time = update_all_the_time
        self.feature_processor = FeatureProcessor()

    def set_args(self, **args):
        something_changed = not equal_dicts(args, self._cur_args)
        if something_changed or self.update_all_the_time:
            self.result = self.renderer.render(**args)
            self._apply_features(args.get("feature", {}))
            self._cur_args = copy.deepcopy(args)

    def _apply_features(self, feature_settings):
        if self.result is None or "image" not in self.result:
            return
        try:
            self.result.image = self.feature_processor.apply(self.result.image, feature_settings)
            if feature_settings.get("enabled", False):
                self.result.feature_status = "Features active"
        except Exception as exc:
            self.result.feature_status = str(exc)

