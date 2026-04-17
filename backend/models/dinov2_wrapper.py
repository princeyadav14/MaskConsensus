import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class DinoV2Wrapper:
    def __init__(self, model_name: str = "dinov2_vits14", allow_fallback: bool = True):
        """
        Load pretrained DINOv2 from TorchHub.
        Falls back to lightweight hand-crafted features if DINOv2 cannot be loaded.
        """
        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "torch"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_dir))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.model = None
        self.use_fallback = False

        # Keep default startup fast/offline-friendly.
        # Set SAM_GC_USE_DINO=1 to load full DINOv2 from TorchHub.
        if os.getenv("SAM_GC_USE_DINO", "0") != "1":
            self.use_fallback = True
            print("[WARN] SAM_GC_USE_DINO is not enabled. Using fallback embeddings.")
            return

        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model=model_name,
                pretrained=True,
            )
            self.model.eval()
            self.model.to(self.device)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to load DINOv2. Ensure internet access for first download "
                    "or pre-cache model in TORCH_HOME."
                ) from exc
            self.use_fallback = True
            print(f"[WARN] DINOv2 unavailable, using fallback embeddings. Reason: {exc}")

    def _fallback_embedding(self, crop: np.ndarray, mask_crop: np.ndarray) -> torch.Tensor:
        masked = crop[mask_crop > 0]
        if masked.size == 0:
            return torch.zeros(7, dtype=torch.float32)

        rgb_mean = masked.mean(axis=0) / 255.0
        rgb_std = masked.std(axis=0) / 255.0
        area_ratio = float(mask_crop.mean())
        feat = np.concatenate([rgb_mean, rgb_std, np.array([area_ratio])], axis=0)
        return torch.tensor(feat, dtype=torch.float32)

    def get_embeddings(self, image_np: np.ndarray, masks: list) -> List[torch.Tensor]:
        """
        Get embeddings for masked crops.
        Returns list of 1D tensors with the same embedding dimensionality.
        """
        embeddings: List[torch.Tensor] = []

        for mask_info in masks:
            x1, y1, x2, y2 = mask_info["bbox"]
            crop = image_np[y1 : y2 + 1, x1 : x2 + 1]
            mask_crop = mask_info["mask"][y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)

            if crop.size == 0 or mask_crop.size == 0:
                continue

            if self.use_fallback or self.model is None:
                embeddings.append(self._fallback_embedding(crop, mask_crop))
                continue

            masked_crop = crop * mask_crop[..., None]
            crop_img = Image.fromarray(masked_crop.astype(np.uint8))
            tensor = self.transform(crop_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)

            if isinstance(output, (tuple, list)):
                output = output[0]
            if isinstance(output, dict):
                if "x_norm_clstoken" in output:
                    output = output["x_norm_clstoken"]
                elif "cls_token" in output:
                    output = output["cls_token"]
                else:
                    output = next(iter(output.values()))

            emb = output.squeeze(0).flatten().detach().cpu().float()
            embeddings.append(emb)

        return embeddings
