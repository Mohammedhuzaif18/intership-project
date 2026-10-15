"""
FastAPI service: chest X-ray classification with Grad-CAM.

  uvicorn backend.main:app --host 127.0.0.1 --port 8000     # UI at http://127.0.0.1:8000

GET  /health                 model status, classes, Grad-CAM layer
POST /predict                image file -> class, confidence, all probabilities, Grad-CAM overlay (base64)
GET  /samples/{class_name}   a random held-out test X-ray of that class (for the UI's sample buttons)
"""
import base64
import io
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import BEST_MODEL_PATH, CLASS_NAMES_PATH, SPLITS_DIR  # noqa: E402
from src.data.preprocessing import preprocess_image  # noqa: E402
from src.gate import ChestGate  # noqa: E402
from src.gradcam import GradCAM, colorize_heatmap, overlay_heatmap  # noqa: E402
from src.utils import absolute, load_json  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", BEST_MODEL_PATH))
CLASS_NAMES_FILE = Path(os.environ.get("CLASS_NAMES_FILE", CLASS_NAMES_PATH))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DISCLAIMER = "Educational research prototype. Not a medical device; not for diagnosis or treatment decisions."

state = {"model": None, "cam": None, "meta": None, "gate": None, "error": None}


def load_model():
    try:
        meta = load_json(CLASS_NAMES_FILE)
        model = tf.keras.models.load_model(MODEL_PATH)
        outputs = model.output_shape[-1]
        expected = 1 if len(meta["classes"]) == 2 else len(meta["classes"])
        if outputs != expected:
            raise ValueError(f"Model has {outputs} outputs but class_names.json lists {meta['classes']}")
        state.update(model=model, cam=GradCAM(model, meta.get("target_layer")), meta=meta, error=None)
        try:
            state["gate"] = ChestGate.load(model, MODEL_PATH)
        except Exception as e:
            state["gate"] = None
            print(f"Chest gate disabled: {type(e).__name__}: {e}")
        print(f"Loaded {meta['model']} ({meta['task']}) from {MODEL_PATH}; Grad-CAM layer {state['cam'].layer_name}")
    except Exception as e:
        state.update(model=None, cam=None, meta=None, gate=None, error=f"{type(e).__name__}: {e}")
        print(f"Model not loaded: {state['error']}")


@asynccontextmanager
async def lifespan(_app):
    load_model()
    yield


app = FastAPI(
    title="Interpretable Chest X-Ray Classifier",
    description="Respiratory disorder classification from chest X-rays with Grad-CAM explanations.",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    """Browsers must re-check the UI files, so an updated frontend is never served from a stale cache."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _jpeg_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health():
    if state["model"] is None and state["error"]:
        load_model()  # picks up a model trained after the server started
    meta = state["meta"] or {}
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model": meta.get("model"),
        "task": meta.get("task"),
        "classes": meta.get("classes", []),
        "display_names": meta.get("display_names", {}),
        "target_layer": state["cam"].layer_name if state["cam"] else None,
        "test_metrics": meta.get("test_metrics"),
        "error": state["error"],
        "chest_gate": {"enabled": state["gate"] is not None,
                       "threshold": round(state["gate"].threshold, 4) if state["gate"] else None},
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {state['error']}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 20 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            image = im.copy()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as an image (use JPEG or PNG).")

    meta = state["meta"]
    classes = meta["classes"]
    # Identical to the training preprocessing; models trained before the crop ablation carry no variant (squash)
    model_input = preprocess_image(image, meta.get("preprocessing", {}).get("variant", "squash"))
    gate = state["gate"]
    is_chest, chest_score = gate.is_chest_xray(model_input) if gate else (True, None)
    if not is_chest:
        # Not a chest X-ray: no disease result, no probabilities, no Grad-CAM
        return {
            "success": True, "is_chest_xray": False, "chest_score": round(chest_score, 4),
            "chest_threshold": round(gate.threshold, 4), "filename": file.filename,
            "predicted_class": None, "predicted_label": None, "confidence": None, "class_probabilities": {},
            "display_names": meta["display_names"], "gradcam_overlay_base64": None, "heatmap_base64": None,
            "model_input_base64": _jpeg_data_url(model_input), "target_layer": state["cam"].layer_name,
            "model": meta["model"], "task": meta["task"], "disclaimer": DISCLAIMER,
            "message": "This image does not look like a chest X-ray, so no disease result is given.",
        }
    try:
        probs, heat, idx = state["cam"](model_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {type(e).__name__}: {e}")

    size = model_input.shape[1::-1]
    return {
        "success": True,
        "is_chest_xray": True,
        "chest_score": round(chest_score, 4) if chest_score is not None else None,
        "filename": file.filename,
        "predicted_class": classes[idx],
        "predicted_label": meta["display_names"].get(classes[idx], classes[idx]),
        "confidence": round(float(probs[idx]), 4),
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "display_names": meta["display_names"],
        "gradcam_overlay_base64": _jpeg_data_url(overlay_heatmap(model_input, heat)),
        "heatmap_base64": _jpeg_data_url(colorize_heatmap(heat, size)),
        "model_input_base64": _jpeg_data_url(model_input),
        "target_layer": state["cam"].layer_name,
        "model": meta["model"],
        "task": meta["task"],
        "disclaimer": DISCLAIMER,
    }


@app.get("/samples/{class_name}")
def sample(class_name: str):
    meta = state["meta"]
    if meta is None or class_name not in meta["classes"]:
        raise HTTPException(status_code=404, detail=f"Unknown class '{class_name}'.")
    test_csv = SPLITS_DIR / meta["task"] / "test.csv"
    if not test_csv.exists():
        raise HTTPException(status_code=404, detail="No test split available.")
    rows = pd.read_csv(test_csv, usecols=["path", "label"])
    paths = rows.loc[rows["label"] == class_name, "path"].tolist()
    if not paths:
        raise HTTPException(status_code=404, detail=f"No test images for '{class_name}'.")
    path = absolute(random.choice(paths))
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, filename=path.name)


frontend_dir = ROOT / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
