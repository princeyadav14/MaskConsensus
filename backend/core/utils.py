import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List
from fastapi import UploadFile

def load_image(path: str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def preprocess_image(image, size=1024):
    h, w = image.shape[:2]
    # Keep original resolution unless image is larger than the target size.
    scale = min(1.0, size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
    return image

def process_uploaded_images(images: List[UploadFile]) -> List[np.ndarray]:
    processed = []
    for image in images:
        contents = image.file.read()
        img_pil = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(img_pil)
        processed.append(preprocess_image(img_np))
    return processed

