
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


class SAMProposer:
    def __init__(
        self,
        model_type: str = "vit_h",
        model_path=None,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
    ):
        """
        Default behavior: lightweight fallback masks.
        Full SAM is loaded only when SAM_GC_FORCE_SAM=1.
        """
        self.use_fallback = True
        self.mask_generator = None
        self.default_iou_threshold = float(pred_iou_thresh)
        self.points_per_side = int(points_per_side)
        self.min_mask_region_area = int(min_mask_region_area)

        force_real = os.getenv("SAM_GC_FORCE_SAM", "0") == "1"

        if model_path is None:
            model_path = Path(__file__).resolve().parent / "sam_vit_h_4b8939.pth"
        else:
            model_path = Path(model_path)

        # Fallback is the default mode unless full SAM is explicitly requested.
        if not force_real:
            print("[INFO] SAM fallback mode active. Set SAM_GC_FORCE_SAM=1 to load full SAM.")
            return

        # Full SAM path starts here.
        if not model_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at: {model_path}. "
                "Place sam_vit_h_4b8939.pth in backend/models/ or pass an explicit path."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[WARN] Loading full SAM on CPU. This may be very slow.")

        sam = sam_model_registry[model_type](checkpoint=str(model_path))
        sam.to(device=device)
        sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=min_mask_region_area,
        )
        self.use_fallback = False
        print(f"[INFO] Full SAM loaded on {device}.")

    def _nms_filter(self, masks: List[Dict], nms_iou_threshold: float) -> List[Dict]:
        if len(masks) <= 1:
            return masks

        ordered = sorted(
            masks,
            key=lambda m: (float(m.get("score", 0.0)), int(m["area"])),
            reverse=True,
        )
        kept: List[Dict] = []
        for candidate in ordered:
            suppress = False
            for selected in kept:
                if _mask_iou(candidate["mask"], selected["mask"]) > nms_iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)
        return kept

    def _fallback_masks(self, image_np: np.ndarray) -> List[Dict]:
        h, w = image_np.shape[:2]
        candidate_masks = []

        for c in range(3):
            ch = image_np[:, :, c]
            th = np.percentile(ch, 70)
            candidate_masks.append((ch >= th).astype(np.uint8))

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidate_masks.append(otsu.astype(np.uint8))

        fallback_scores = [0.93, 0.84, 0.76, 0.62]
        clean_masks = []
        for idx, mask in enumerate(candidate_masks):
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            area = len(xs)
            if area > self.min_mask_region_area:
                clean_masks.append(
                    {
                        "mask": mask,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "area": int(area),
                        "score": fallback_scores[min(idx, len(fallback_scores) - 1)],
                    }
                )

        if not clean_masks:
            full = np.ones((h, w), dtype=np.uint8)
            clean_masks.append(
                {
                    "mask": full,
                    "bbox": [0, 0, w - 1, h - 1],
                    "area": int(h * w),
                    "score": 0.5,
                }
            )
        return clean_masks[:50]

    def get_masks(
        self,
        image_np: np.ndarray,
        iou_threshold: float | None = None,
        nms_iou_threshold: float = 0.45,
        presence_detection: bool = True,
    ) -> List[Dict]:
        effective_iou = self.default_iou_threshold if iou_threshold is None else float(iou_threshold)
        effective_iou = float(np.clip(effective_iou, 0.0, 1.0))
        nms_iou_threshold = float(np.clip(nms_iou_threshold, 0.0, 1.0))

        if self.use_fallback or self.mask_generator is None:
            masks = self._fallback_masks(image_np)
        else:
            raw_masks = self.mask_generator.generate(image_np)
            masks = []
            for m in raw_masks:
                score = float(m.get("predicted_iou", 0.0))
                if score < effective_iou:
                    continue
                mask = m["segmentation"].astype(np.uint8)
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                x1, y1 = xs.min(), ys.min()
                x2, y2 = xs.max(), ys.max()
                area = len(xs)
                if area > self.min_mask_region_area:
                    masks.append(
                        {
                            "mask": mask,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "area": int(area),
                            "score": score,
                        }
                    )

        masks = [m for m in masks if float(m.get("score", 0.0)) >= effective_iou]

        if not presence_detection:
            masks = [m for m in masks if m["area"] >= int(0.05 * image_np.shape[0] * image_np.shape[1])]

        masks = self._nms_filter(masks, nms_iou_threshold)

        if not masks:
            h, w = image_np.shape[:2]
            full = np.ones((h, w), dtype=np.uint8)
            masks = [{"mask": full, "bbox": [0, 0, w - 1, h - 1], "area": int(h * w), "score": 0.5}]

        return masks[:50]
