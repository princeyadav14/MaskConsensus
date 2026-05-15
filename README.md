# SAM-GC: Group Image Co-Segmentation

This project is a FastAPI + browser-based demo for finding the common object across a group of images. It combines mask proposals, per-mask feature extraction, graph-based consensus scoring, and object labeling to return one selected mask per image along with group-level similarity metrics.

The repository currently works as a technical preview with two operating modes:
- `Fallback mode` for quick local demos without heavyweight model downloads
- `Full model mode` using SAM, DINOv2, and CLIP when resources are available

## Project Overview

Given a set of at least 3 images, the pipeline:

1. preprocesses each uploaded image
2. generates candidate masks for each image
3. extracts an embedding for every candidate mask
4. selects one consensus object per image using a mask-aware graph solver
5. labels the selected object in each image
6. returns masks, scores, and group-level summary statistics

## Repository Structure

```text
backend/
  app.py                    FastAPI app and pipeline orchestration
  requirements.txt          Python dependencies
  core/
    utils.py                Image decoding and preprocessing
    graph_solver.py         Mask-aware consensus solver
  models/
    sam_wrapper.py          SAM mask generator + fallback masks
    dinov2_wrapper.py       DINOv2 embeddings + fallback features
    clip_wrapper.py         CLIP labeling + heuristic fallback

frontend/
  index.html                Main UI
  assets/
    css/style.css           Frontend styling
    js/app.js               Upload flow, API calls, result rendering
```

## How It Works In This Repo

### Backend

The backend is implemented in [backend/app.py](/e:/EE655/EE655_CourseProject-main/EE655_CourseProject-main/backend/app.py).

- `GET /health` returns a simple health payload
- `POST /api/process-group` runs the full multi-image pipeline
- `POST /predict` exists but is only a placeholder

The main endpoint validates:
- minimum 3 uploaded images
- `iou_threshold` in `[0, 1]`
- `nms_threshold` in `[0, 1]`

### Model behavior

The wrappers are designed to degrade gracefully when full models are unavailable:

- `SAMProposer` uses simple fallback masks by default unless `SAM_GC_FORCE_SAM=1`
- `DinoV2Wrapper` uses fallback handcrafted features unless `SAM_GC_USE_DINO=1`
- `CLIPObjectLabeler` tries to load CLIP by default, but falls back to heuristic labels if loading fails

### Consensus solver

The solver in [backend/core/graph_solver.py](/e:/EE655/EE655_CourseProject-main/EE655_CourseProject-main/backend/core/graph_solver.py) is a mask-aware consensus method, not a plain embedding-only baseline. It combines:

- semantic similarity between mask embeddings
- area compatibility
- SAM score compatibility
- unary plausibility scoring for individual masks
- coordinate-ascent optimization over per-image candidate selections

## Frontend

The frontend is a static single-page interface in [frontend/index.html](/e:/EE655/EE655_CourseProject-main/EE655_CourseProject-main/frontend/index.html) with logic in [frontend/assets/js/app.js](/e:/EE655/EE655_CourseProject-main/EE655_CourseProject-main/frontend/assets/js/app.js).

It supports:
- drag-and-drop image upload
- image preview cards
- IoU threshold control
- NMS threshold control
- presence detection toggle
- pipeline step indicators
- result metrics and common object label
- mask overlay rendering for the selected consensus mask

The frontend resolves the backend URL in this order:

1. `?api=<backend-url>` query parameter
2. saved `localStorage` value `SAM_GC_API_BASE`
3. default `http://localhost:8000`

## Setup

### 1. Create a Python environment

From the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the SAM checkpoint

You can download pretrained SAM weights using https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### 3. Start the backend

```powershell
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the frontend

Open [frontend/index.html](/e:/EE655/EE655_CourseProject-main/EE655_CourseProject-main/frontend/index.html) in a browser.

If your browser blocks local file API usage or you prefer a local web server, you can serve the frontend folder with any simple static server.

## Runtime Modes

### Default local mode

This is the easiest way to run the project. In this mode:

- SAM uses fallback mask generation
- DINOv2 uses fallback embeddings
- CLIP will try to load, then fall back if unavailable

Start normally:

```powershell
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Full model mode

Use this when you have the required model resources and want higher-fidelity results.

```powershell
cd backend
$env:SAM_GC_FORCE_SAM="1"
$env:SAM_GC_USE_DINO="1"
$env:SAM_GC_USE_CLIP="1"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- SAM checkpoint is expected at `backend/models/sam_vit_h_4b8939.pth`
- DINOv2 is loaded through Torch Hub
- CLIP is loaded through Hugging Face Transformers
- first-time model loading may require internet access and local cache creation
- CPU-only full mode may be slow

## API

### `GET /health`

Example response:

```json
{
  "status": "healthy"
}
```

### `POST /api/process-group`

Multipart form fields:

- `images`: repeated image file input, minimum 3 files
- `iou_threshold`: float, default `0.88`
- `nms_threshold`: float, default `0.45`
- `presence_detection`: boolean, default `true`

Typical response fields:

- `status`
- `image_count`
- `winning_masks`
- `winning_indices`
- `consensus_scores`
- `global_affinity`
- `group_cohesion`
- `nodes_evaluated`
- `graph_solver_time`
- `masks_per_image`
- `similar_object_labels`
- `similar_object_confidences`
- `common_object_label`
- `thresholds`
- `process_time`
- `solver_name`
- `objective_value`
- `restarts`
- `valid_candidates_per_image`


## Frontend Controls

### IoU threshold

Passed to the backend as `iou_threshold`. Higher values keep fewer, stronger mask candidates.

### NMS threshold

Passed as `nms_threshold`. Lower values suppress overlapping masks more aggressively.

### Presence detection

When disabled, smaller masks are filtered more aggressively in the SAM wrapper.

## Remote Backend Usage

If you run the backend on another machine or a tunnelled environment, open the frontend with:

```text
frontend/index.html?api=http://your-backend-url:8000
```

Example:

```text
http://127.0.0.1:5500/?api=https://example.ngrok-free.app
```

## Google Colab Setup

If you want to run the backend on Google Colab and use the frontend locally, this is the clearest flow:

### 1. Open a Colab notebook

Set the runtime to `GPU` if you want to use full SAM/DINOv2/CLIP mode with better performance.

### 2. Clone the repository and move into the backend

Run this in a Colab cell:

```python
!git clone https://github.com/vd-0711/EE655_CourseProject.git
%cd /content/EE655_CourseProject/backend
```

### 3. Install dependencies

Install the dependencies plus `pyngrok`:

```python
!pip install -r requirements.txt pyngrok
```

### 4. Download the SAM checkpoint

```python
!wget -P /content/Automatic-Co-Segmentation-using-SAM/backend/models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 5. Set your ngrok token and enable the full models

Replace `"YOUR_NGROK_TOKEN"` with your own token from ngrok:

```python
import os
from pyngrok import ngrok

# Get a free token at https://dashboard.ngrok.com/get-started/your-authtoken
os.environ["NGROK_AUTHTOKEN"] = "YOUR_NGROK_TOKEN"
ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])

os.environ["SAM_GC_FORCE_SAM"] = "1"
os.environ["SAM_GC_USE_DINO"] = "1"
os.environ["SAM_GC_USE_CLIP"] = "1"
```

### 6. Create the public tunnel

```python
public_url = ngrok.connect(8000).public_url
print("Public API URL:", public_url)
```

Copy the printed `public_url`. You will use it in the frontend later.

### 7. Start the backend server

```python
!uvicorn app:app --host 0.0.0.0 --port 8000
```

## Using The Local Frontend With The Colab Backend

After Colab prints the public API URL:

1. start your frontend locally
2. open the frontend in your browser
3. append `?api=` to the frontend URL
4. paste the Colab/ngrok public URL after it

Example:

```text
http://127.0.0.1:5500/?api=https://abc123.ngrok-free.app
```

This tells the local frontend to send requests to the backend running in Colab instead of `http://localhost:8000`.

## Current Limitations

- no automated tests are included yet
- fallback mode is convenient for demos, but not equivalent to full-model quality
- full-model setup is only partially automated because weights and first-run downloads are external
- the solver is a practical mask-aware consensus implementation, not a fully productionized research pipeline

## Suggested Next Improvements

- add unit and integration tests for the API
- add startup logging that clearly reports active runtime mode
- add a reproducible model setup script
- improve failure-state messaging in the frontend
- add result export for masks and metrics

## Quick Start

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open the frontend, upload at least 3 images, and run the pipeline.
