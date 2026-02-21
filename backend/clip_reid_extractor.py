"""CLIP-based re-identification extractor for cross-cut person matching.

Uses CLIP ViT-B/16 to extract 768D appearance embeddings from person crops.
Only invoked during the first few frames after a scene cut (not every frame),
providing richer appearance features than OSNet for cross-view matching.

Falls back gracefully to zero embeddings if CLIP is unavailable.
"""

from __future__ import annotations

import numpy as np

from embedding_extractor import EmbeddingExtractor


class CLIPReIDExtractor(EmbeddingExtractor):
    """CLIP ViT-B/16 based person appearance embedding extractor.

    Produces 768D L2-normalized embeddings from person crops.
    """

    EMBEDDING_DIM = 768

    def __init__(self, device: str = "cuda", model_name: str = "ViT-B/16"):
        import torch

        self._device = device if torch.cuda.is_available() else "cpu"
        self._model_name = model_name
        self._model = None
        self._preprocess = None
        self._available = False

        try:
            import clip as openai_clip

            model, preprocess = openai_clip.load(model_name, device=self._device)
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._available = True
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            # Try transformers CLIP as fallback
            try:
                from transformers import CLIPModel, CLIPProcessor

                self._hf_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch16"
                )
                self._hf_model = self._hf_model.to(self._device).eval()
                self._hf_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch16"
                )
                self._available = True
                self._model = None  # sentinel: use HF path
            except Exception:
                print(
                    f"[clip_reid] CLIP unavailable ({e}), "
                    "CLIPReIDExtractor will return zero embeddings",
                    flush=True,
                )

    @property
    def available(self) -> bool:
        """Whether a CLIP model was successfully loaded."""
        return self._available

    def extract(
        self,
        rgb_frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """Extract 768D CLIP embeddings from person crops.

        Args:
            rgb_frame: (H, W, 3) RGB image.
            bboxes: list of (x1, y1, x2, y2) pixel bounding boxes.

        Returns:
            List of (768,) L2-normalized float32 embeddings, one per bbox.
            Returns zero vectors if CLIP is unavailable.
        """
        if not bboxes:
            return []

        if not self._available:
            return [
                np.zeros(self.EMBEDDING_DIM, dtype=np.float32) for _ in bboxes
            ]

        crops = self._extract_crops(rgb_frame, bboxes)

        if self._model is not None and self._preprocess is not None:
            return self._extract_openai_clip(crops)
        else:
            return self._extract_hf_clip(crops)

    def _extract_crops(
        self,
        rgb_frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """Extract and clamp person crops from the frame."""
        import cv2

        crops = []
        for x1, y1, x2, y2 in bboxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(rgb_frame.shape[1], x2)
            y2 = min(rgb_frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            crop = rgb_frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))
            crops.append(crop)
        return crops

    def _extract_openai_clip(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """Extract embeddings using openai/clip package."""
        import torch
        from PIL import Image

        images = []
        for crop in crops:
            pil_img = Image.fromarray(crop)
            images.append(self._preprocess(pil_img))

        batch = torch.stack(images).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(batch)

        features = features.cpu().numpy().astype(np.float32)
        return self._normalize_embeddings(features)

    def _extract_hf_clip(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """Extract embeddings using HuggingFace transformers CLIP."""
        import torch
        from PIL import Image

        pil_images = [Image.fromarray(c) for c in crops]
        inputs = self._hf_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._hf_model.get_image_features(**inputs)

        features = outputs.cpu().numpy().astype(np.float32)
        return self._normalize_embeddings(features)

    @staticmethod
    def _normalize_embeddings(features: np.ndarray) -> list[np.ndarray]:
        """L2-normalize each embedding vector."""
        embeddings = []
        for i in range(features.shape[0]):
            emb = features[i]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
        return embeddings
