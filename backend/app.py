from collections import Counter
from statistics import mean
from typing import List

import base64
import io
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from core.utils import process_uploaded_images
from models.sam_wrapper import SAMProposer
from models.dinov2_wrapper import DinoV2Wrapper
from models.clip_wrapper import CLIPObjectLabeler
from core.graph_solver import solve_consensus_with_stats

app = FastAPI(title="SAM-GC Project")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# These are created at startup, so full/fallback mode is determined when uvicorn starts.
sam_proposer = SAMProposer(pred_iou_thresh=0.0)
dino = DinoV2Wrapper()
clip_labeler = CLIPObjectLabeler()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/process-group")
async def process_group(
    images: List[UploadFile] = File(...),
    iou_threshold: float = Form(0.88),
    nms_threshold: float = Form(0.45),
    presence_detection: bool = Form(True),
):
    start_time = time.time()

    if len(images) < 3:
        raise HTTPException(status_code=400, detail="Please upload at least 3 images.")
    if not (0.0 <= iou_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="iou_threshold must be in [0, 1].")
    if not (0.0 <= nms_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="nms_threshold must be in [0, 1].")

    # Step 1: preprocess
    processed_images = process_uploaded_images(images)
    image_shapes = [img.shape[:2] for img in processed_images]  # (H, W)

    # Step 2: SAM proposals
    sam_masks = []
    for img in processed_images:
        masks = sam_proposer.get_masks(
            img,
            iou_threshold=iou_threshold,
            nms_iou_threshold=nms_threshold,
            presence_detection=presence_detection,
        )
        if len(masks) == 0:
            raise HTTPException(status_code=422, detail="No valid SAM masks found for one or more images.")
        sam_masks.append(masks)

    # Step 3: DINOv2 embeddings
    embeddings_list = []
    for i, img in enumerate(processed_images):
        embeddings = dino.get_embeddings(img, sam_masks[i])
        if len(embeddings) == 0:
            raise HTTPException(status_code=422, detail="No embeddings generated for one or more images.")
        embeddings_list.append(embeddings)

    # Step 4: Graph consensus (now mask-aware, not embedding-only)
    graph_start = time.time()
    consensus = solve_consensus_with_stats(
        embeddings_list=embeddings_list,
        masks_list=sam_masks,
        image_shapes=image_shapes,
    )
    graph_time = time.time() - graph_start
    winning_indices = consensus["winning_indices"]

    # Step 5: encode winning masks
    winning_masks_base64 = []
    consensus_scores = consensus["winning_scores"]

    for i in range(len(processed_images)):
        win_idx = winning_indices[i]
        win_mask = sam_masks[i][win_idx]["mask"]

        pil_mask = Image.fromarray((win_mask * 255).astype("uint8")).convert("RGB")
        buffered = io.BytesIO()
        pil_mask.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        winning_masks_base64.append(f"data:image/png;base64,{img_b64}")

    # Step 6: label winning masks
    similar_object_labels = []
    similar_object_confidences = []
    for i in range(len(processed_images)):
        win_idx = winning_indices[i]
        win_mask = sam_masks[i][win_idx]["mask"]
        label, score = clip_labeler.label_mask(processed_images[i], win_mask)
        similar_object_labels.append(label)
        similar_object_confidences.append(float(score))

    # Safer group label logic:
    # do not claim a "common object" if labels are tied or confidence is too weak.
    common_object_label = "Common Similar Object: Uncertain"
    if similar_object_labels:
        counts = Counter(similar_object_labels)
        best_label, best_count = counts.most_common(1)[0]
        majority_needed = max(2, (len(similar_object_labels) + 1) // 2)

        best_label_scores = [
            s for lbl, s in zip(similar_object_labels, similar_object_confidences) if lbl == best_label
        ]
        avg_best_conf = mean(best_label_scores) if best_label_scores else 0.0

        if best_count == len(similar_object_labels):
            common_object_label = f"Common Similar Object: {best_label}"
        elif best_count >= majority_needed and avg_best_conf >= 0.08:
            common_object_label = f"Common Similar Object: {best_label}"

    process_time = time.time() - start_time
    global_affinity = float(consensus["global_affinity"])

    if global_affinity >= 0.85:
        group_cohesion = "High"
    elif global_affinity >= 0.65:
        group_cohesion = "Medium"
    else:
        group_cohesion = "Low"

    return {
        "status": "complete",
        "image_count": len(images),
        "winning_masks": winning_masks_base64,
        "winning_indices": winning_indices,
        "consensus_scores": [float(x) for x in consensus_scores],
        "global_affinity": global_affinity,
        "group_cohesion": group_cohesion,
        "nodes_evaluated": int(consensus["nodes_evaluated"]),
        "graph_solver_time": round(graph_time, 3),
        "masks_per_image": [len(m) for m in sam_masks],
        "similar_object_labels": similar_object_labels,
        "similar_object_confidences": [round(float(x), 3) for x in similar_object_confidences],
        "common_object_label": common_object_label,
        "thresholds": {
            "iou_threshold": round(iou_threshold, 3),
            "nms_threshold": round(nms_threshold, 3),
            "presence_detection": bool(presence_detection),
        },
        "process_time": round(process_time, 3),
        "solver_name": consensus.get("solver_name", "unknown"),
        "objective_value": float(consensus.get("objective_value", 0.0)),
        "restarts": int(consensus.get("restarts", 0)),
        "valid_candidates_per_image": consensus.get("valid_candidates_per_image", []),
    }


@app.post("/predict")
async def predict(image: bytes):  # Placeholder
    return {"result": "prediction"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
