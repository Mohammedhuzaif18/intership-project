import os
import io
import sys
import json
import glob
import random
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gradcam import find_target_conv_layer, make_gradcam_heatmap, overlay_gradcam, colorize_heatmap

app = FastAPI(
    title="Explainable AI Chest X-Ray Diagnostic Support",
    description="FastAPI service for respiratory disorder classification with Grad-CAM visual interpretability.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.abspath(os.environ.get("MODEL_PATH", "./models/best_model.keras"))
CLASS_INDICES_PATH = os.path.abspath("./models/class_indices.json")
# Held-out test images used by the UI's sample buttons, first match wins
SAMPLE_DIRS = ["./data/preprocessed_224/test", "./data/processed_multiclass/test", "./data/processed/test"]

model = None
target_conv_layer_name = None
class_names = ["NORMAL", "PNEUMONIA"]


def load_system_model():
    global model, target_conv_layer_name, class_names
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, "r") as f:
                mapping = json.load(f)
            if "classes" in mapping:
                class_names = mapping["classes"]
            else:
                keys = sorted([k for k in mapping.keys() if str(k).isdigit()], key=lambda x: int(x))
                class_names = [mapping[k] for k in keys]
        except Exception as e:
            print(f"Warning loading class indices: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            target_conv_layer_name = find_target_conv_layer(model)
            print(f"Model loaded from {MODEL_PATH}. Target layer: {target_conv_layer_name}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            model = None
    else:
        print(f"Model file not found yet at {MODEL_PATH}.")


@app.on_event("startup")
def startup_event():
    load_system_model()


@app.get("/health")
def health_check():
    global model
    if model is None:
        load_system_model()
    return {
        "status": "healthy",
        "service": "Explainable Respiratory Disorder Classifier API",
        "model_loaded": model is not None,
        "model_name": model.name if model is not None else None,
        "model_path": MODEL_PATH,
        "target_conv_layer": target_conv_layer_name,
        "classes": class_names,
        "num_classes": len(class_names),
        "disclaimer": "Educational research prototype only. Not intended for clinical diagnosis.",
    }


@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    global model, target_conv_layer_name

    if model is None:
        load_system_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model is still training or not available.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_resized = pil_img.resize((224, 224))
        img_array = np.array(img_resized)
        img_input = np.expand_dims(img_array, axis=0)

        raw_pred = model.predict(img_input, verbose=0)

        if raw_pred.shape[-1] == 1:
            pneumonia_prob = float(raw_pred[0][0])
            normal_prob = 1.0 - pneumonia_prob
            pred_idx = 1 if pneumonia_prob >= 0.5 else 0
            predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else ("PNEUMONIA" if pred_idx == 1 else "NORMAL")
            confidence = pneumonia_prob if pred_idx == 1 else normal_prob
            class_probs = {}
            if len(class_names) >= 2:
                class_probs[class_names[0]] = round(normal_prob, 4)
                class_probs[class_names[1]] = round(pneumonia_prob, 4)
            else:
                class_probs["PNEUMONIA"] = round(pneumonia_prob, 4)
        else:
            probs = raw_pred[0]
            pred_idx = int(np.argmax(probs))
            predicted_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            confidence = float(probs[pred_idx])
            class_probs = {
                class_names[i] if i < len(class_names) else str(i): round(float(probs[i]), 4)
                for i in range(len(probs))
            }

        heatmap = make_gradcam_heatmap(
            img_input,
            model,
            last_conv_layer_name=target_conv_layer_name,
            pred_index=pred_idx,
        )
        overlay, _ = overlay_gradcam(img_array, heatmap, alpha=0.45)

        overlay_b64 = _jpeg_data_url(Image.fromarray(overlay))
        orig_b64 = _jpeg_data_url(img_resized)
        # Bare colormap, so the UI can blend it over the original at any opacity
        jet_rgb, _ = colorize_heatmap(heatmap, img_resized.size)
        heatmap_b64 = _jpeg_data_url(Image.fromarray(jet_rgb))

        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probs,
            "gradcam_overlay_base64": overlay_b64,
            "heatmap_base64": heatmap_b64,
            "original_image_base64": orig_b64,
            "target_layer_used": target_conv_layer_name,
            "interpretation": (
                "Grad-CAM visual attribution: Warm regions (red, orange, yellow) indicate focal features "
                "in the lung fields that strongly drove the network's classification decision."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


def _jpeg_data_url(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/samples/{class_name}")
def sample_image(class_name: str):
    """A random held-out test X-ray of the given class, for the UI's sample buttons."""
    if class_name not in class_names:
        raise HTTPException(status_code=404, detail=f"Unknown class '{class_name}'.")
    for base in SAMPLE_DIRS:
        files = glob.glob(os.path.join(os.path.abspath(base), class_name, "*.*"))
        if files:
            path = random.choice(files)
            return FileResponse(path, media_type="image/jpeg", filename=os.path.basename(path))
    raise HTTPException(status_code=404, detail=f"No test images found for '{class_name}'.")


frontend_path = os.path.abspath("./frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
