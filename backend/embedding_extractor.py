"""Appearance embedding extractors for person re-identification.

Provides an abstract base and two implementations:
- OSNetExtractor: GPU-accelerated OSNet-x0_25 via torchreid
- DummyExtractor: returns controlled embeddings for testing
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingExtractor(ABC):
    """Extract appearance embeddings from cropped person images."""

    @abstractmethod
    def extract(self, rgb_frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        """Extract one embedding per bounding box.

        Args:
            rgb_frame: (H, W, 3) RGB image.
            bboxes: list of (x1, y1, x2, y2) pixel bounding boxes.

        Returns:
            List of (512,) L2-normalized float32 embeddings, one per bbox.
        """
        ...


class OSNetExtractor(EmbeddingExtractor):
    """OSNet-x0_25 appearance embedding via torchreid."""

    def __init__(self, device: str = "cuda"):
        import torch
        try:
            from torchreid.utils import FeatureExtractor
        except (ImportError, ModuleNotFoundError):
            from torchreid.reid.utils.feature_extractor import FeatureExtractor

        self._device = device if torch.cuda.is_available() else "cpu"
        self._extractor = FeatureExtractor(
            model_name="osnet_x0_25",
            device=self._device,
        )

    def extract(self, rgb_frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        import cv2
        import torch

        if not bboxes:
            return []

        crops = []
        for x1, y1, x2, y2 in bboxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(rgb_frame.shape[1], x2)
            y2 = min(rgb_frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue
            crop = rgb_frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            crops.append(crop)

        features = self._extractor(crops)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        embeddings = []
        for i in range(features.shape[0]):
            emb = features[i].astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        return embeddings


class DummyExtractor(EmbeddingExtractor):
    """Returns pre-configured or zero embeddings for testing."""

    def __init__(self, embeddings: dict[int, np.ndarray] | None = None, dim: int = 512):
        self._embeddings = embeddings or {}
        self._dim = dim
        self._call_count = 0

    def extract(self, rgb_frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        result = []
        for i, _bbox in enumerate(bboxes):
            idx = self._call_count + i
            if idx in self._embeddings:
                emb = self._embeddings[idx].astype(np.float32)
            else:
                emb = np.zeros(self._dim, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            result.append(emb)
        self._call_count += len(bboxes)
        return result
