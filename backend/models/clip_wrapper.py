import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPObjectLabeler:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fallback = False
        self.model = None
        self.processor = None

        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))

        self.candidate_labels: List[str] = [
            "person",
            "car",
            "bicycle",
            "motorcycle",
            "bus",
            "truck",
            "dog",
            "cat",
            "bird",
            "horse",
            "cow",
            "sheep",
            "bottle",
            "cup",
            "chair",
            "table",
            "laptop",
            "phone",
            "book",
            "bag",
            "tree",
            "flower",
            "building",
            "road",
            "sky",
        ]
        self.prompt_labels = [f"a photo of a {label}" for label in self.candidate_labels]

        if os.getenv("SAM_GC_USE_CLIP", "1") != "1":
            self.use_fallback = True
            print("[WARN] SAM_GC_USE_CLIP is disabled. Using heuristic labels.")
            return

        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"[INFO] CLIP labeler loaded: {model_name}")
        except Exception as exc:
            self.use_fallback = True
            print(f"[WARN] CLIP unavailable, using heuristic labels. Reason: {exc}")

    def _dominant_color_name(self, rgb_mean: np.ndarray) -> str:
        r, g, b = [float(x) for x in rgb_mean]
        if max(r, g, b) - min(r, g, b) < 20:
            return "gray"
        if r >= g and r >= b:
            return "red"
        if g >= r and g >= b:
            return "green"
        return "blue"

    def _size_bucket(self, area_ratio: float) -> str:
        if area_ratio < 0.1:
            return "small"
        if area_ratio < 0.35:
            return "medium"
        return "large"

    def _heuristic_label(self, image_np: np.ndarray, mask_np: np.ndarray) -> Tuple[str, float]:
        selected = image_np[mask_np > 0]
        if selected.size == 0:
            return "uncertain object", 0.0
        rgb_mean = selected.reshape(-1, 3).mean(axis=0)
        area_ratio = float(mask_np.mean())
        label = f"{self._size_bucket(area_ratio)} {self._dominant_color_name(rgb_mean)} object"
        return label, 0.25

    def label_mask(self, image_np: np.ndarray, mask_np: np.ndarray) -> Tuple[str, float]:
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            return "uncertain object", 0.0

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image_np[y1 : y2 + 1, x1 : x2 + 1]
        mask_crop = mask_np[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0 or mask_crop.size == 0:
            return "uncertain object", 0.0

        if self.use_fallback or self.model is None or self.processor is None:
            return self._heuristic_label(image_np, mask_np)

        masked_crop = (crop * mask_crop[..., None]).astype(np.uint8)
        pil_img = Image.fromarray(masked_crop)

        try:
            with torch.no_grad():
                inputs = self.processor(
                    text=self.prompt_labels,
                    images=pil_img,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).detach().cpu().numpy()
                best_idx = int(np.argmax(probs))
                label = self.candidate_labels[best_idx]
                score = float(probs[best_idx])
                return label, score
        except Exception:
            return self._heuristic_label(image_np, mask_np)
