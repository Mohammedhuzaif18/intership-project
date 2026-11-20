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
from urllib.parse import quote

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import BEST_MODEL_PATH, CLASS_NAMES_PATH, SPLITS_DIR  # noqa: E402
from src.data.preprocessing import preprocess_image  # noqa: E402
from src.gate import ChestGate  # noqa: E402
from src.gradcam import GradCAM, colorize_heatmap, overlay_heatmap  # noqa: E402
from src.utils import absolute, load_json  # noqa: E402
from backend.image_quality import assess_image_quality, thresholds as quality_thresholds  # noqa: E402
from backend.patient_history import (  # noqa: E402
    PatientHistoryError,
    compare_records,
    initialize_store,
    list_records,
    patient_exists,
    record_image,
    save_record,
    validate_patient_id,
)
from backend.auth import AuthenticationError, create_access_token, decode_access_token  # noqa: E402
from backend.review_store import (  # noqa: E402
    AccountConflictError,
    AccountNotFoundError,
    AccountStoreError,
    add_prediction,
    authenticate,
    claim_review_request,
    create_review_request,
    doctor_queue,
    doctor_review_image,
    get_user,
    initialize_review_store,
    patient_prediction_record,
    patient_prediction_report,
    patient_predictions,
    register_doctor,
    register_patient,
    submit_doctor_review,
)

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
    initialize_store()
    initialize_review_store()
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
        "image_quality_thresholds": quality_thresholds(),
        "disclaimer": DISCLAIMER,
    }


async def _read_image(file: UploadFile) -> tuple[bytes, Image.Image, str]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 20 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            image_format = im.format or "JPEG"
            image = im.copy()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as an image (use JPEG or PNG).")
    return data, image, image_format


@app.post("/quality")
async def check_quality(file: UploadFile = File(...)):
    _, image, _ = await _read_image(file)
    return {"success": True, "quality": assess_image_quality(image)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: str | None = Form(None),
    authorization: str | None = Header(default=None),
):
    if patient_id is not None:
        user = require_role(current_user(authorization), "PATIENT")
        if patient_id.strip() != user["patient_id"]:
            raise HTTPException(status_code=403, detail="You may only save predictions under your own Patient ID.")
    return await _run_prediction(file, patient_id)


async def _run_prediction(file: UploadFile, patient_id: str | None):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {state['error']}")
    if patient_id is not None:
        try:
            patient_id = validate_patient_id(patient_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    data, image, image_format = await _read_image(file)
    quality = assess_image_quality(image)
    if not quality["passed"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Image quality checks failed; upload a clearer image before prediction.",
                "quality": quality,
            },
        )

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
            "quality": quality,
            "message": "This image does not look like a chest X-ray, so no disease result is given.",
        }
    try:
        probs, heat, idx = state["cam"](model_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {type(e).__name__}: {e}")

    size = model_input.shape[1::-1]
    result = {
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
        "quality": quality,
        "disclaimer": DISCLAIMER,
    }
    if patient_id is not None:
        try:
            record, previous_id = save_record(patient_id, data, file.filename or "upload", image_format, result)
        except PatientHistoryError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        result["patient_record"] = {
            "patient_id": patient_id,
            "record_id": record["record_id"],
            "previous_record_id": previous_id,
            "uploaded_at": record["uploaded_at"],
        }
    return result


class RegistrationBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=6, max_length=256)


class DoctorRegistrationBody(RegistrationBody):
    verification_code: str = Field(min_length=1, max_length=256)


class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=6, max_length=256)
    role: str | None = Field(default=None, pattern="^(PATIENT|DOCTOR)$")


class DoctorReviewBody(BaseModel):
    decision: str = Field(pattern="^(APPROVE|REJECT)$")
    observations: str = Field(min_length=1, max_length=5000)
    comments: str = Field(default="", max_length=5000)


def _account_error(exc: Exception):
    if isinstance(exc, AccountConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, AccountNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, AccountStoreError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="A valid bearer token is required.",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_access_token(authorization[7:].strip())
        user = get_user(claims["sub"])
        if user is None or user["role"] != claims["role"]:
            raise AuthenticationError("Account is inactive or token role is no longer valid.")
        return user
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc),
                            headers={"WWW-Authenticate": "Bearer"}) from exc
    except AccountStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def require_role(user: dict, role: str) -> dict:
    if user["role"] != role:
        raise HTTPException(status_code=403, detail=f"This endpoint requires the {role} role.")
    return user


def _prediction_image_urls(prediction_id: str) -> dict:
    return {
        kind: f"/auth/patient/predictions/{prediction_id}/images/{kind}"
        for kind in ("original", "gradcam", "heatmap")
    }


@app.post("/auth/patient/register", status_code=201)
def patient_register(body: RegistrationBody):
    try:
        user = register_patient(body.email, body.full_name, body.password)
        token, expires = create_access_token(user["user_id"], user["role"])
        return {"user": user, "access_token": token, "token_type": "bearer", "expires_at": expires}
    except (ValueError, AccountConflictError, AccountStoreError) as exc:
        _account_error(exc)


@app.post("/auth/doctor/register", status_code=201)
def doctor_register(body: DoctorRegistrationBody):
    try:
        user = register_doctor(body.email, body.full_name, body.password, body.verification_code)
        token, expires = create_access_token(user["user_id"], user["role"])
        return {"user": user, "access_token": token, "token_type": "bearer", "expires_at": expires}
    except (PermissionError, ValueError, AccountConflictError, AccountStoreError) as exc:
        _account_error(exc)


@app.post("/auth/login")
def login(body: LoginBody):
    try:
        user = authenticate(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AccountStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if user is None or (body.role and user["role"] != body.role):
        raise HTTPException(status_code=401, detail="Email, password, or selected role is incorrect.")
    token, expires = create_access_token(user["user_id"], user["role"])
    return {"user": user, "access_token": token, "token_type": "bearer", "expires_at": expires}


@app.get("/auth/me")
def account_me(user: dict = Depends(current_user)):
    return {"user": user}


@app.post("/auth/patient/predict")
async def authenticated_patient_predict(
    file: UploadFile = File(...),
    user: dict = Depends(current_user),
):
    require_role(user, "PATIENT")
    result = await _run_prediction(file, patient_id=user["patient_id"])
    if result.get("is_chest_xray") is False:
        return result
    try:
        prediction_id = add_prediction(user, result["patient_record"]["record_id"], result)
    except (AccountNotFoundError, AccountStoreError) as exc:
        _account_error(exc)
    result["prediction_id"] = prediction_id
    result["report_type"] = "AI_ONLY"
    return result


@app.get("/auth/patient/predictions")
def patient_report_list(user: dict = Depends(current_user)):
    require_role(user, "PATIENT")
    try:
        reports = patient_predictions(user)
        for report in reports:
            report["image_urls"] = _prediction_image_urls(report["prediction_id"])
        return {"patient_id": user["patient_id"], "predictions": reports}
    except AccountStoreError as exc:
        _account_error(exc)


@app.get("/auth/patient/predictions/{prediction_id}/report")
def patient_report(prediction_id: str, user: dict = Depends(current_user)):
    require_role(user, "PATIENT")
    try:
        report = patient_prediction_report(user, prediction_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Prediction not found for this patient.")
        record = patient_prediction_record(user, prediction_id)
        report["image_urls"] = _prediction_image_urls(prediction_id)
        review_request = report["review_request"]
        report["report_type"] = (
            "DOCTOR_REVIEWED"
            if review_request and review_request["status"] == "COMPLETED"
            else "AI_ONLY"
        )
        report["doctor_reviewed_report"] = (
            review_request["review"]
            if review_request and review_request["status"] == "COMPLETED"
            else None
        )
        report["patient_id"] = record["patient_id"]
        return report
    except (AccountStoreError, AccountNotFoundError) as exc:
        _account_error(exc)


@app.post("/auth/patient/predictions/{prediction_id}/review-request", status_code=201)
def patient_request_review(prediction_id: str, user: dict = Depends(current_user)):
    require_role(user, "PATIENT")
    try:
        return create_review_request(user, prediction_id)
    except (AccountNotFoundError, AccountConflictError, AccountStoreError) as exc:
        _account_error(exc)


@app.get("/auth/patient/predictions/{prediction_id}/images/{kind}")
def authenticated_patient_image(
    prediction_id: str,
    kind: str,
    user: dict = Depends(current_user),
):
    require_role(user, "PATIENT")
    try:
        record = patient_prediction_record(user, prediction_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Prediction not found for this patient.")
        path = record_image(record["patient_id"], record["record_id"], kind)
    except (AccountStoreError, PatientHistoryError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Prediction image was not found.")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg")


@app.get("/auth/doctor/reviews")
def doctor_review_queue(user: dict = Depends(current_user)):
    require_role(user, "DOCTOR")
    try:
        return {"reviews": doctor_queue(user)}
    except AccountStoreError as exc:
        _account_error(exc)


@app.post("/auth/doctor/reviews/{request_id}/claim")
def doctor_claim_review(request_id: str, user: dict = Depends(current_user)):
    require_role(user, "DOCTOR")
    try:
        request = claim_review_request(user, request_id)
        request["image_urls"] = {
            kind: f"/auth/doctor/reviews/{request_id}/images/{kind}"
            for kind in ("original", "gradcam", "heatmap")
        }
        return request
    except (AccountNotFoundError, AccountConflictError, AccountStoreError) as exc:
        _account_error(exc)


@app.post("/auth/doctor/reviews/{request_id}/submit")
def doctor_submit_review(
    request_id: str,
    body: DoctorReviewBody,
    user: dict = Depends(current_user),
):
    require_role(user, "DOCTOR")
    try:
        return submit_doctor_review(user, request_id, body.decision, body.observations, body.comments)
    except (AccountNotFoundError, AccountConflictError, AccountStoreError, ValueError) as exc:
        _account_error(exc)


@app.get("/auth/doctor/reviews/{request_id}/images/{kind}")
def authenticated_doctor_image(
    request_id: str,
    kind: str,
    user: dict = Depends(current_user),
):
    require_role(user, "DOCTOR")
    try:
        record = doctor_review_image(user, request_id, kind)
        if record is None:
            raise HTTPException(status_code=404, detail="Review image is not assigned to this doctor.")
        path = record_image(record["patient_id"], record["record_id"], kind)
    except PatientHistoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Review image was not found.")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg")


def _record_with_image_urls(record: dict) -> dict:
    record = dict(record)
    patient_path = quote(record["patient_id"], safe="")
    record["image_urls"] = {
        kind: f"/patients/{patient_path}/records/{record['record_id']}/images/{kind}"
        for kind in ("original", "gradcam", "heatmap")
    }
    return record


@app.get("/patients/{patient_id}/history")
def patient_history(
    patient_id: str,
    limit: int = Query(500, ge=1, le=5000),
    user: dict = Depends(current_user),
):
    try:
        patient_id = validate_patient_id(patient_id)
        require_role(user, "PATIENT")
        if patient_id != user["patient_id"]:
            raise HTTPException(status_code=404, detail="No history found for this Patient ID.")
        if not patient_exists(patient_id):
            raise HTTPException(status_code=404, detail="No history found for this Patient ID.")
        records = list_records(patient_id)
        return {
            "patient_id": patient_id,
            "records": [_record_with_image_urls(record) for record in (records or [])[-limit:]],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PatientHistoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/patients/{patient_id}/compare")
def patient_comparison(
    patient_id: str,
    record_a_id: str = Query(..., min_length=1, max_length=64),
    record_b_id: str = Query(..., min_length=1, max_length=64),
    user: dict = Depends(current_user),
):
    try:
        patient_id = validate_patient_id(patient_id)
        require_role(user, "PATIENT")
        if patient_id != user["patient_id"]:
            raise HTTPException(status_code=404, detail="One or both records were not found for this Patient ID.")
        comparison = compare_records(patient_id, record_a_id, record_b_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PatientHistoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if comparison is None:
        raise HTTPException(status_code=404, detail="One or both records were not found for this Patient ID.")
    comparison["previous"] = _record_with_image_urls(comparison["previous"])
    comparison["current"] = _record_with_image_urls(comparison["current"])
    return comparison


@app.get("/patients/{patient_id}/records/{record_id}/images/{kind}")
def patient_record_image(
    patient_id: str,
    record_id: str,
    kind: str,
    user: dict = Depends(current_user),
):
    try:
        patient_id = validate_patient_id(patient_id)
        require_role(user, "PATIENT")
        if patient_id != user["patient_id"]:
            raise HTTPException(status_code=404, detail="Patient image was not found.")
        path = record_image(patient_id, record_id, kind)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PatientHistoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Patient image was not found.")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)


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
