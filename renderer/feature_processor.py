import os
import itertools

import cv2
import numpy as np
import torch


class FeatureProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dino_name = None
        self._dino_model = None
        self._sam_key = None
        self._sam_predictor = None
        self._pca_key = None
        self._pca_components = None
        self._pca_low = None
        self._pca_high = None
        self.status = ""

    def apply(self, image, settings):
        if image is None or not settings.get("enabled", False):
            return image

        image = self._ensure_rgb(image)
        model = settings.get("model", "DINO")
        max_size = int(settings.get("max_size", 518))
        if model == "DINO":
            features = self._extract_dino(image, settings.get("dino_model", "dinov2_vits14"), max_size)
        elif model == "SAM":
            features = self._extract_sam(
                image,
                settings.get("sam_type", "vit_b"),
                settings.get("sam_checkpoint", ""),
                max_size,
            )
        else:
            raise RuntimeError(f"Unknown feature model: {model}")

        pca_key = (
            model,
            settings.get("dino_model", ""),
            settings.get("sam_type", ""),
            settings.get("sam_checkpoint", ""),
        )
        feature_image = self._pca_to_rgb(
            features,
            image.shape[:2],
            pca_key=pca_key,
            stability=float(settings.get("pca_stability", 0.95)),
        )
        opacity = float(settings.get("opacity", 1.0))
        opacity = min(max(opacity, 0.0), 1.0)
        if opacity < 1.0:
            feature_image = (feature_image.astype(np.float32) * opacity + image.astype(np.float32) * (1.0 - opacity))
            feature_image = feature_image.clip(0, 255).astype(np.uint8)
        return feature_image

    @staticmethod
    def _ensure_rgb(image):
        if image.ndim != 3:
            raise RuntimeError("Feature models require an HxWxC image.")
        if image.shape[2] == 3:
            return image
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        raise RuntimeError(f"Feature models require 1 or 3 image channels, got {image.shape[2]}.")

    def _extract_dino(self, image, model_name, max_size):
        model = self._load_dino(model_name)
        tensor = self._image_to_tensor(image, max_size=max_size, patch_size=14)
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                output = model.forward_features(tensor)
                if isinstance(output, dict) and "x_norm_patchtokens" in output:
                    tokens = output["x_norm_patchtokens"]
                    h = tensor.shape[-2] // 14
                    w = tensor.shape[-1] // 14
                    return tokens.reshape(tokens.shape[0], h, w, tokens.shape[-1]).permute(0, 3, 1, 2)[0]
            if hasattr(model, "get_intermediate_layers"):
                output = model.get_intermediate_layers(tensor, n=1, reshape=True)[0]
                return output[0]
            output = model(tensor)
            if output.ndim == 4:
                return output[0]
        raise RuntimeError("DINO model did not return a spatial feature map.")

    def _load_dino(self, model_name):
        if self._dino_model is not None and self._dino_name == model_name:
            return self._dino_model
        try:
            model = torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not load DINO from torch.hub. Install/cache the DINOv2 weights first, "
                "or run once with network access."
            ) from exc
        model.eval().to(self.device)
        self._dino_model = model
        self._dino_name = model_name
        return model

    def _extract_sam(self, image, sam_type, checkpoint, max_size):
        predictor = self._load_sam(sam_type, checkpoint)
        resized = self._resize_image_np(image, max_size)
        predictor.set_image(resized)
        features = predictor.features
        if features is None:
            raise RuntimeError("SAM did not produce image features.")
        return features[0]

    def _load_sam(self, sam_type, checkpoint):
        checkpoint = os.path.abspath(os.path.expanduser(checkpoint)) if checkpoint else ""
        if checkpoint == "":
            raise RuntimeError("Set a SAM checkpoint path before enabling SAM features.")
        if not os.path.isfile(checkpoint):
            raise RuntimeError(f"SAM checkpoint does not exist: {checkpoint}")
        key = (sam_type, checkpoint)
        if self._sam_predictor is not None and self._sam_key == key:
            return self._sam_predictor
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise RuntimeError("SAM support requires the `segment_anything` package.") from exc

        sam = sam_model_registry[sam_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        self._sam_predictor = predictor
        self._sam_key = key
        return predictor

    def _image_to_tensor(self, image, max_size, patch_size):
        resized = self._resize_image_np(image, max_size)
        h = max((resized.shape[0] // patch_size) * patch_size, patch_size)
        w = max((resized.shape[1] // patch_size) * patch_size, patch_size)
        resized = cv2.resize(resized, (w, h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).to(self.device, dtype=torch.float32) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    @staticmethod
    def _resize_image_np(image, max_size):
        h, w = image.shape[:2]
        scale = min(float(max_size) / max(h, w), 1.0)
        if scale >= 1.0:
            return image
        new_size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _pca_to_rgb(self, features, target_shape, pca_key=None, stability=0.95):
        features = features.detach().float()
        c, h, w = features.shape
        flat = features.reshape(c, h * w).T
        flat = flat - flat.mean(dim=0, keepdim=True)
        if flat.shape[0] < 3 or flat.shape[1] < 3:
            raise RuntimeError("Feature map is too small for 3-channel PCA.")

        _, _, current_components = torch.pca_lowrank(flat, q=3, center=False)
        current_components = current_components[:, :3]
        components = self._stable_pca_components(current_components, pca_key, stability)

        projected = flat @ components
        current_low = torch.quantile(projected, 0.01, dim=0, keepdim=True)
        current_high = torch.quantile(projected, 0.99, dim=0, keepdim=True)
        low, high = self._stable_pca_range(current_low, current_high, pca_key, stability)
        projected = (projected - low) / (high - low).clamp_min(1e-6)
        projected = projected.clamp(0.0, 1.0)

        rgb = projected.reshape(h, w, 3).detach().cpu().numpy()
        rgb = (rgb * 255.0).astype(np.uint8)
        return cv2.resize(rgb, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

    def _stable_pca_components(self, current_components, pca_key, stability):
        stability = min(max(float(stability), 0.0), 0.99)
        if self._pca_key != pca_key or self._pca_components is None or self._pca_components.shape != current_components.shape:
            self._pca_key = pca_key
            self._pca_components = current_components.detach()
            return current_components

        previous = self._pca_components.to(current_components.device)
        aligned = self._match_component_order(previous, current_components)
        for channel in range(aligned.shape[1]):
            if torch.dot(previous[:, channel], aligned[:, channel]) < 0:
                aligned[:, channel] = -aligned[:, channel]

        mixed = previous * stability + aligned * (1.0 - stability)
        mixed, _ = torch.linalg.qr(mixed, mode="reduced")
        for channel in range(mixed.shape[1]):
            if torch.dot(previous[:, channel], mixed[:, channel]) < 0:
                mixed[:, channel] = -mixed[:, channel]
        self._pca_components = mixed.detach()
        return mixed

    @staticmethod
    def _match_component_order(previous, current):
        score = torch.abs(previous.T @ current)
        best_order = None
        best_score = None
        for order in itertools.permutations(range(current.shape[1])):
            order_score = sum(score[index, channel] for index, channel in enumerate(order))
            if best_score is None or order_score > best_score:
                best_score = order_score
                best_order = order
        return current[:, list(best_order)]

    def _stable_pca_range(self, current_low, current_high, pca_key, stability):
        stability = min(max(float(stability), 0.0), 0.99)
        if self._pca_key != pca_key or self._pca_low is None or self._pca_high is None:
            self._pca_low = current_low.detach()
            self._pca_high = current_high.detach()
            return current_low, current_high

        low = self._pca_low.to(current_low.device) * stability + current_low * (1.0 - stability)
        high = self._pca_high.to(current_high.device) * stability + current_high * (1.0 - stability)
        self._pca_low = low.detach()
        self._pca_high = high.detach()
        return low, high
