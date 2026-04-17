# SAM-GC Project

Segment Anything Model (SAM) + Graph Cut (GC) for advanced image segmentation.

## Structure
- `backend/`: Python FastAPI server with SAM/DINOv2 models and SCO graph solver.
- `frontend/`: Web UI for image upload and prediction.

## Setup & Run
1. Backend:
   ```
   cd sam-gc-project/backend
   pip install -r requirements.txt
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```
2. Frontend: Open `frontend/index.html` in browser (connects to localhost:8000).

### Runtime Modes
- Default (fast/offline demo):
  - Uses lightweight fallback masks/embeddings/labels when GPU or online model loading is unavailable.
- Full SAM + DINOv2 (slower, needs model resources):
  - Set env vars before starting backend:
  ```
  set SAM_GC_FORCE_SAM=1
  set SAM_GC_USE_DINO=1
  set SAM_GC_USE_CLIP=1
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
  ```
  - `SAM` checkpoint path defaults to `backend/models/sam_vit_h_4b8939.pth`.

### Run On Google Colab GPU
1. In Colab: set runtime to `GPU`.
2. In a Colab cell:
```python
!git clone <YOUR_REPO_URL>
%cd sam-gc-project/backend
!pip install -r requirements.txt pyngrok
```
3. Start backend + tunnel:
```python
import os
from pyngrok import ngrok

# Optional: set your token if needed
# os.environ["NGROK_AUTHTOKEN"] = "YOUR_TOKEN"
# ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])

os.environ["SAM_GC_FORCE_SAM"] = "1"
os.environ["SAM_GC_USE_DINO"] = "1"
os.environ["SAM_GC_USE_CLIP"] = "1"

public_url = ngrok.connect(8000).public_url
print("Public API URL:", public_url)

!uvicorn app:app --host 0.0.0.0 --port 8000
```
4. Open your local frontend with this query param:
```text
http://127.0.0.1:5500/?api=<PASTE_PUBLIC_API_URL>
```
Example:
```text
http://127.0.0.1:5500/?api=https://abc123.ngrok-free.app
```

## API
- GET `/health`
- POST `/api/process-group` (multipart `images`, minimum 3 files)
- POST `/predict` (placeholder)
- `/api/process-group` response now includes:
  - `similar_object_labels`: per-image label for the selected consensus object.
  - `similar_object_confidences`: CLIP confidence (0 to 1) for each label.
  - `common_object_label`: compact group-level summary label.

### Thresholds (frontend controls)
- `iou_threshold`:
  - Mask confidence gate in `[0, 1]`. Higher values keep fewer, higher-confidence masks.
- `nms_threshold`:
  - Mask overlap suppression IoU in `[0, 1]`. Lower values suppress overlapping masks more aggressively.
- `presence_detection`:
  - When disabled, small/weak masks are filtered more aggressively.

## Next
- Download SAM/DINOv2 weights to `models/`.
- Implement full SCO in `core/graph_solver.py`.
- Add interactive points/mask overlay in JS.

