"""Persistent, patient-scoped storage and visual comparison for uploaded X-rays."""
import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = Path(os.environ.get("PATIENT_HISTORY_DIR", ROOT / "data" / "patient_history")).resolve()
IMAGE_DIR = STORE_DIR / "records"
DATABASE_PATH = STORE_DIR / "history.sqlite3"
PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
IMAGE_KINDS = {"original", "gradcam", "heatmap"}


class PatientHistoryError(Exception):
    """Raised when patient history cannot be persisted or read."""


def validate_patient_id(patient_id: str | None) -> str:
    value = (patient_id or "").strip()
    if not PATIENT_ID_PATTERN.fullmatch(value):
        raise ValueError("Patient ID must be 1–64 letters, numbers, dots, underscores or hyphens.")
    return value


def _connect() -> sqlite3.Connection:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def _connection():
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_store() -> None:
    try:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        with _connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS patient_records (
                    record_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                    original_path TEXT NOT NULL,
                    gradcam_path TEXT NOT NULL,
                    heatmap_path TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    class_probabilities TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task TEXT NOT NULL
                )"""
            )
            schema = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'patient_records'"
            ).fetchone()["sql"]
            normalized_schema = "".join(schema.upper().split())
            if "UNIQUE(PATIENT_ID,IMAGE_SHA256)" in normalized_schema:
                conn.execute(
                    """CREATE TABLE patient_records_new (
                        record_id TEXT PRIMARY KEY,
                        patient_id TEXT NOT NULL REFERENCES patients(patient_id),
                        original_path TEXT NOT NULL,
                        gradcam_path TEXT NOT NULL,
                        heatmap_path TEXT NOT NULL,
                        image_sha256 TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        predicted_class TEXT NOT NULL,
                        predicted_label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        class_probabilities TEXT NOT NULL,
                        uploaded_at TEXT NOT NULL,
                        model TEXT NOT NULL,
                        task TEXT NOT NULL
                    )"""
                )
                conn.execute(
                    """INSERT INTO patient_records_new (
                        record_id, patient_id, original_path, gradcam_path, heatmap_path,
                        image_sha256, filename, predicted_class, predicted_label, confidence,
                        class_probabilities, uploaded_at, model, task
                    )
                    SELECT record_id, patient_id, original_path, gradcam_path, heatmap_path,
                        image_sha256, filename, predicted_class, predicted_label, confidence,
                        class_probabilities, uploaded_at, model, task
                    FROM patient_records"""
                )
                conn.execute("DROP TABLE patient_records")
                conn.execute("ALTER TABLE patient_records_new RENAME TO patient_records")
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_patient_records_time
                   ON patient_records(patient_id, uploaded_at)"""
            )
    except (OSError, sqlite3.Error) as exc:
        raise PatientHistoryError(f"Could not initialize patient history storage: {exc}") from exc


def _decode_data_url(value: str) -> bytes:
    try:
        header, encoded = value.split(",", 1)
        if not header.startswith("data:image/"):
            raise ValueError("Expected an image data URL.")
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise PatientHistoryError("Could not save generated Grad-CAM image data.") from exc


def _extension(image_format: str) -> str:
    return {"PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg"}.get(image_format.upper(), ".jpg")


def save_record(
    patient_id: str,
    image_bytes: bytes,
    filename: str,
    image_format: str,
    prediction: dict,
) -> tuple[dict, str | None]:
    """Persist an unchanged upload and its prediction artifacts; return it and its prior record ID."""
    patient_id = validate_patient_id(patient_id)
    record_id = uuid.uuid4().hex
    record_dir = IMAGE_DIR / record_id
    digest = hashlib.sha256(image_bytes).hexdigest()
    paths = {
        "original": record_dir / f"original{_extension(image_format)}",
        "gradcam": record_dir / "gradcam.jpg",
        "heatmap": record_dir / "heatmap.jpg",
    }
    previous_id = None
    created_dir = False
    try:
        gradcam_bytes = _decode_data_url(prediction["gradcam_overlay_base64"])
        heatmap_bytes = _decode_data_url(prediction["heatmap_base64"])
        uploaded_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute(
                "SELECT record_id FROM patient_records WHERE patient_id = ? "
                "ORDER BY uploaded_at DESC, record_id DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO patients(patient_id, created_at) VALUES (?, ?)",
                (patient_id, uploaded_at),
            )
            record_dir.mkdir(parents=True, exist_ok=False)
            created_dir = True
            paths["original"].write_bytes(image_bytes)
            paths["gradcam"].write_bytes(gradcam_bytes)
            paths["heatmap"].write_bytes(heatmap_bytes)
            conn.execute(
                """
                INSERT INTO patient_records(
                    record_id, patient_id, original_path, gradcam_path, heatmap_path,
                    image_sha256, filename, predicted_class, predicted_label, confidence,
                    class_probabilities, uploaded_at, model, task
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    patient_id,
                    str(paths["original"].relative_to(STORE_DIR)),
                    str(paths["gradcam"].relative_to(STORE_DIR)),
                    str(paths["heatmap"].relative_to(STORE_DIR)),
                    digest,
                    Path(filename or "upload").name[:255],
                    prediction["predicted_class"],
                    prediction["predicted_label"],
                    float(prediction["confidence"]),
                    json.dumps(prediction["class_probabilities"]),
                    uploaded_at,
                    prediction["model"],
                    prediction["task"],
                ),
            )
            previous_id = prior["record_id"] if prior else None
        return get_record(patient_id, record_id), previous_id
    except (OSError, sqlite3.Error, KeyError, ValueError, TypeError) as exc:
        if created_dir:
            shutil.rmtree(record_dir, ignore_errors=True)
        raise PatientHistoryError(f"Could not save patient history: {exc}") from exc


def patient_exists(patient_id: str) -> bool:
    try:
        with _connection() as conn:
            return conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone() is not None
    except sqlite3.Error as exc:
        raise PatientHistoryError(f"Could not read patient history: {exc}") from exc


def _record_dict(row: sqlite3.Row) -> dict:
    return {
        "record_id": row["record_id"],
        "patient_id": row["patient_id"],
        "filename": row["filename"],
        "predicted_class": row["predicted_class"],
        "predicted_label": row["predicted_label"],
        "confidence": row["confidence"],
        "class_probabilities": json.loads(row["class_probabilities"]),
        "uploaded_at": row["uploaded_at"],
        "model": row["model"],
        "task": row["task"],
    }


def list_records(patient_id: str) -> list[dict] | None:
    try:
        with _connection() as conn:
            patient = conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
            if patient is None:
                return None
            rows = conn.execute(
                "SELECT * FROM patient_records WHERE patient_id = ? ORDER BY uploaded_at, record_id",
                (patient_id,),
            ).fetchall()
            return [_record_dict(row) for row in rows]
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise PatientHistoryError(f"Could not read patient history: {exc}") from exc


def get_record(patient_id: str, record_id: str) -> dict | None:
    try:
        with _connection() as conn:
            row = conn.execute(
                "SELECT * FROM patient_records WHERE patient_id = ? AND record_id = ?",
                (patient_id, record_id),
            ).fetchone()
            if row is None:
                return None
            record = _record_dict(row)
            record["_paths"] = {
                kind: (STORE_DIR / row[f"{kind}_path"]).resolve()
                for kind in ("original", "gradcam", "heatmap")
            }
            return record
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise PatientHistoryError(f"Could not read patient record: {exc}") from exc


def record_image(patient_id: str, record_id: str, kind: str) -> Path | None:
    if kind not in IMAGE_KINDS:
        return None
    record = get_record(patient_id, record_id)
    if record is None:
        return None
    path = record["_paths"][kind]
    if not path.is_relative_to(STORE_DIR) or not path.is_file():
        raise PatientHistoryError("Stored patient image is unavailable or outside the history store.")
    return path


def _image_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise PatientHistoryError("Could not encode the comparison image.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _comparison_map(previous_path: Path, current_path: Path) -> tuple[str, dict]:
    previous = cv2.imread(str(previous_path), cv2.IMREAD_GRAYSCALE)
    current = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)
    if previous is None or current is None:
        raise PatientHistoryError("A stored X-ray could not be decoded for comparison.")

    size = (512, 512)
    previous = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    aligned = False
    matched_features = 0
    try:
        detector = cv2.ORB_create(nfeatures=1200)
        keypoints_a, descriptors_a = detector.detectAndCompute(previous, None)
        keypoints_b, descriptors_b = detector.detectAndCompute(current, None)
        if descriptors_a is not None and descriptors_b is not None:
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors_a, descriptors_b, k=2)
            good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
            matched_features = len(good)
            if len(good) >= 8:
                source = np.float32([keypoints_a[m.queryIdx].pt for m in good])
                target = np.float32([keypoints_b[m.trainIdx].pt for m in good])
                transform, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if transform is not None and inliers is not None and int(inliers.sum()) >= 6:
                    previous = cv2.warpAffine(previous, transform, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    aligned = True
    except cv2.error:
        aligned = False

    difference = cv2.absdiff(previous, current)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    color_map = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
    return _image_data_url(color_map), {
        "aligned": aligned,
        "matched_features": matched_features,
        "message": (
            "Images were feature-aligned before pixel comparison."
            if aligned else "Reliable feature alignment was not available; images were resized only."
        ),
    }


def describe_prediction_change(previous: dict, current: dict) -> dict:
    """Describe the model-category shift without presenting it as a clinical conclusion."""
    previous_class = previous["predicted_class"]
    current_class = current["predicted_class"]
    previous_label = previous["predicted_label"]
    current_label = current["predicted_label"]
    previous_confidence = float(previous["confidence"])
    current_confidence = float(current["confidence"])
    confidence_change = current_confidence - previous_confidence

    if previous_class != "NORMAL" and current_class == "NORMAL":
        assessment = "possible_improvement_signal"
        summary = (
            f"Possible improvement signal from the model: the previous image was classified as "
            f"{previous_label} and the new image as Normal. The model's prediction changed in a "
            f"direction that may be consistent with improvement, but it does not confirm that the "
            f"disease improved."
        )
    elif previous_class == "NORMAL" and current_class != "NORMAL":
        assessment = "possible_new_or_worsening_signal"
        summary = (
            f"Possible new or worsening signal from the model: the previous image was classified "
            f"as Normal and the new image as {current_label}. This prediction change does not confirm "
            f"that disease has developed or worsened."
        )
    elif previous_class == current_class == "NORMAL":
        assessment = "no_predicted_category_change_normal"
        summary = (
            "The model classified both images as Normal, so its predicted category did not change. "
            "This does not prove that the patient is healthy or that an illness is absent."
        )
    elif previous_class == current_class:
        assessment = "no_predicted_category_change_disease"
        summary = (
            f"The model classified both images as {current_label}, so its predicted disease category "
            f"did not change. This does not establish whether the condition improved, stayed the same, "
            f"or worsened."
        )
    else:
        assessment = "predicted_disease_category_changed"
        summary = (
            f"The model's predicted category changed from {previous_label} to {current_label}. "
            "Because both predictions are disease categories, this change alone does not indicate "
            "improvement or worsening."
        )

    summary += (
        f" Model confidence changed from {previous_confidence:.1%} to {current_confidence:.1%} "
        f"({confidence_change:+.1%}); confidence is not a measure of disease severity."
    )
    return {"assessment": assessment, "summary": summary}


def compare_records(patient_id: str, first_id: str, second_id: str) -> dict | None:
    first = get_record(patient_id, first_id)
    second = get_record(patient_id, second_id)
    if first is None or second is None:
        return None
    previous, current = sorted(
        (first, second),
        key=lambda item: (item["uploaded_at"], item["record_id"]),
    )
    difference_map, alignment = _comparison_map(
        previous["_paths"]["original"],
        current["_paths"]["original"],
    )
    prediction_change = describe_prediction_change(previous, current)
    for record in (previous, current):
        record.pop("_paths", None)
    return {
        "patient_id": patient_id,
        "previous": previous,
        "current": current,
        "prediction_change": prediction_change,
        "difference_map_base64": difference_map,
        "alignment": alignment,
        "disclaimer": (
            "This comparison is based on the model's predicted categories, not a clinical assessment. "
            "It cannot confirm improvement, worsening, or disease severity. A qualified clinician "
            "must interpret the X-rays alongside the patient's symptoms and other tests."
        ),
    }
