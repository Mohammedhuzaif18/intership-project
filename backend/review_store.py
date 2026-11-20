"""Persistent account, prediction, and doctor-review records in the app's SQLite database."""
import hmac
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from backend.auth import hash_password, verify_password
from backend.patient_history import PatientHistoryError, _connection

EMAIL_PATTERN = re.compile(r"^[^@\s]{1,128}@[^@\s]{1,190}$")
DOCTOR_REGISTRATION_CODE = os.environ.get("DOCTOR_REGISTRATION_CODE", "")
DOCTOR_REGISTRATION_CODE_PATTERN = re.compile(r"^[0-9]{4}$")


class AccountStoreError(Exception):
    """Raised when account or review storage fails."""


class AccountConflictError(Exception):
    """Raised for duplicate accounts or already-claimed/completed review requests."""


class AccountNotFoundError(Exception):
    """Raised when a user, prediction, or review request is not available to the caller."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _email(value: str) -> str:
    result = (value or "").strip().lower()
    if len(result) > 254 or not EMAIL_PATTERN.fullmatch(result):
        raise ValueError("Enter a valid email address.")
    return result


def initialize_review_store() -> None:
    try:
        with _connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('PATIENT', 'DOCTOR')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS patient_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
                    patient_id TEXT NOT NULL UNIQUE REFERENCES patients(patient_id)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS doctor_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
                    verified_at TEXT NOT NULL,
                    verification_method TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ai_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL UNIQUE REFERENCES patient_records(record_id),
                    patient_user_id TEXT NOT NULL REFERENCES users(user_id),
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS doctor_review_requests (
                    request_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL UNIQUE REFERENCES ai_predictions(prediction_id),
                    patient_user_id TEXT NOT NULL REFERENCES users(user_id),
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'IN_REVIEW', 'COMPLETED')),
                    requested_at TEXT NOT NULL,
                    assigned_doctor_user_id TEXT REFERENCES users(user_id),
                    claimed_at TEXT,
                    completed_at TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS doctor_reviews (
                    review_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE REFERENCES doctor_review_requests(request_id),
                    doctor_user_id TEXT NOT NULL REFERENCES users(user_id),
                    decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'REJECT')),
                    observations TEXT NOT NULL,
                    comments TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_review_queue
                   ON doctor_review_requests(status, requested_at)"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_predictions_patient
                   ON ai_predictions(patient_user_id, created_at)"""
            )
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not initialize authentication and review storage: {exc}") from exc


def _public_user(row: sqlite3.Row) -> dict:
    user = {
        "user_id": row["user_id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
    }
    if row["role"] == "PATIENT":
        user["patient_id"] = row["patient_id"]
    return user


def register_patient(email: str, full_name: str, password: str) -> dict:
    email = _email(email)
    name = (full_name or "").strip()
    if not 2 <= len(name) <= 120:
        raise ValueError("Full name must be between 2 and 120 characters.")
    password_hash = hash_password(password)
    user_id = uuid.uuid4().hex
    patient_id = f"PT-{uuid.uuid4().hex[:16].upper()}"
    now = _now()
    try:
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO users(user_id, email, full_name, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, 'PATIENT', ?)",
                (user_id, email, name, password_hash, now),
            )
            conn.execute(
                "INSERT INTO patients(patient_id, created_at) VALUES (?, ?)",
                (patient_id, now),
            )
            conn.execute(
                "INSERT INTO patient_profiles(user_id, patient_id) VALUES (?, ?)",
                (user_id, patient_id),
            )
            return {
                "user_id": user_id,
                "email": email,
                "full_name": name,
                "role": "PATIENT",
                "patient_id": patient_id,
            }
    except sqlite3.IntegrityError as exc:
        raise AccountConflictError("An account with this email already exists.") from exc
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not register patient: {exc}") from exc


def register_doctor(email: str, full_name: str, password: str, verification_code: str) -> dict:
    if not DOCTOR_REGISTRATION_CODE:
        raise PermissionError("Doctor registration is disabled until the server administrator configures DOCTOR_REGISTRATION_CODE.")
    if not DOCTOR_REGISTRATION_CODE_PATTERN.fullmatch(DOCTOR_REGISTRATION_CODE):
        raise AccountStoreError("DOCTOR_REGISTRATION_CODE must be configured as exactly four digits.")
    if not hmac.compare_digest(verification_code or "", DOCTOR_REGISTRATION_CODE):
        raise PermissionError("Doctor verification failed.")
    email = _email(email)
    name = (full_name or "").strip()
    if not 2 <= len(name) <= 120:
        raise ValueError("Full name must be between 2 and 120 characters.")
    password_hash = hash_password(password)
    user_id = uuid.uuid4().hex
    now = _now()
    try:
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO users(user_id, email, full_name, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, 'DOCTOR', ?)",
                (user_id, email, name, password_hash, now),
            )
            conn.execute(
                "INSERT INTO doctor_profiles(user_id, verified_at, verification_method) VALUES (?, ?, 'registration_code')",
                (user_id, now),
            )
        return {"user_id": user_id, "email": email, "full_name": name, "role": "DOCTOR"}
    except sqlite3.IntegrityError as exc:
        raise AccountConflictError("An account with this email already exists.") from exc
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not register doctor: {exc}") from exc


def authenticate(email: str, password: str) -> dict | None:
    email = _email(email)
    try:
        with _connection() as conn:
            row = conn.execute(
                """SELECT u.*, p.patient_id
                   FROM users u LEFT JOIN patient_profiles p ON p.user_id = u.user_id
                   WHERE u.email = ? AND u.is_active = 1""",
                (email,),
            ).fetchone()
            if row is None or not verify_password(row["password_hash"], password):
                return None
            return _public_user(row)
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not authenticate account: {exc}") from exc


def get_user(user_id: str) -> dict | None:
    try:
        with _connection() as conn:
            row = conn.execute(
                """SELECT u.*, p.patient_id
                   FROM users u LEFT JOIN patient_profiles p ON p.user_id = u.user_id
                   WHERE u.user_id = ? AND u.is_active = 1""",
                (user_id,),
            ).fetchone()
            return _public_user(row) if row else None
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not load account: {exc}") from exc


def add_prediction(user: dict, record_id: str, result: dict) -> str:
    prediction_id = uuid.uuid4().hex
    persisted_result = {
        key: value for key, value in result.items()
        if key not in {"gradcam_overlay_base64", "heatmap_base64", "model_input_base64"}
    }
    try:
        with _connection() as conn:
            record = conn.execute(
                "SELECT patient_id FROM patient_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if record is None or record["patient_id"] != user["patient_id"]:
                raise AccountNotFoundError("Prediction record does not belong to this patient.")
            conn.execute(
                """INSERT INTO ai_predictions(prediction_id, record_id, patient_user_id, result_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (prediction_id, record_id, user["user_id"], json.dumps(persisted_result), _now()),
            )
        return prediction_id
    except AccountNotFoundError:
        raise
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not save AI prediction: {exc}") from exc


def create_review_request(user: dict, prediction_id: str) -> dict:
    request_id = uuid.uuid4().hex
    requested_at = _now()
    try:
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            prediction = conn.execute(
                "SELECT 1 FROM ai_predictions WHERE prediction_id = ? AND patient_user_id = ?",
                (prediction_id, user["user_id"]),
            ).fetchone()
            if prediction is None:
                raise AccountNotFoundError("Prediction was not found for this patient.")
            conn.execute(
                """INSERT INTO doctor_review_requests(
                    request_id, prediction_id, patient_user_id, status, requested_at
                ) VALUES (?, ?, ?, 'PENDING', ?)""",
                (request_id, prediction_id, user["user_id"], requested_at),
            )
        return {
            "request_id": request_id,
            "prediction_id": prediction_id,
            "status": "PENDING",
            "requested_at": requested_at,
        }
    except AccountNotFoundError:
        raise
    except sqlite3.IntegrityError as exc:
        raise AccountConflictError("A doctor review has already been requested for this prediction.") from exc
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not create doctor review request: {exc}") from exc


def patient_predictions(user: dict) -> list[dict]:
    try:
        with _connection() as conn:
            rows = conn.execute(
                """SELECT p.prediction_id, p.record_id, p.result_json, p.created_at,
                          r.request_id, r.status, r.requested_at, r.completed_at,
                          d.decision, d.observations, d.comments, d.reviewed_at,
                          du.full_name AS doctor_name
                   FROM ai_predictions p
                   LEFT JOIN doctor_review_requests r ON r.prediction_id = p.prediction_id
                   LEFT JOIN doctor_reviews d ON d.request_id = r.request_id
                   LEFT JOIN users du ON du.user_id = d.doctor_user_id
                   WHERE p.patient_user_id = ?
                   ORDER BY p.created_at DESC""",
                (user["user_id"],),
            ).fetchall()
            results = []
            for row in rows:
                results.append({
                    "prediction_id": row["prediction_id"],
                    "record_id": row["record_id"],
                    "created_at": row["created_at"],
                    "result": {
                        key: value for key, value in json.loads(row["result_json"]).items()
                        if key not in {"gradcam_overlay_base64", "heatmap_base64", "model_input_base64"}
                    },
                    "review_request": {
                        "request_id": row["request_id"],
                        "status": row["status"],
                        "requested_at": row["requested_at"],
                        "completed_at": row["completed_at"],
                        "review": ({
                            "decision": row["decision"],
                            "observations": row["observations"],
                            "comments": row["comments"],
                            "reviewed_at": row["reviewed_at"],
                            "doctor_name": row["doctor_name"],
                        } if row["decision"] else None),
                    } if row["request_id"] else None,
                })
            return results
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not load patient reports: {exc}") from exc


def patient_prediction_report(user: dict, prediction_id: str) -> dict | None:
    for prediction in patient_predictions(user):
        if prediction["prediction_id"] == prediction_id:
            return prediction
    return None


def patient_prediction_record(user: dict, prediction_id: str) -> dict | None:
    try:
        with _connection() as conn:
            row = conn.execute(
                """SELECT p.record_id, pp.patient_id
                   FROM ai_predictions p
                   JOIN patient_profiles pp ON pp.user_id = p.patient_user_id
                   WHERE p.prediction_id = ? AND p.patient_user_id = ?""",
                (prediction_id, user["user_id"]),
            ).fetchone()
            return dict(row) if row else None
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not authorize patient image: {exc}") from exc


def _request_details(conn: sqlite3.Connection, request_id: str) -> dict | None:
    row = conn.execute(
        """SELECT r.request_id, r.status, r.requested_at, r.assigned_doctor_user_id,
                  r.claimed_at, r.completed_at, p.prediction_id, p.record_id,
                  p.result_json, p.created_at AS prediction_created_at,
                  u.user_id AS patient_user_id, u.full_name AS patient_name,
                  pp.patient_id
           FROM doctor_review_requests r
           JOIN ai_predictions p ON p.prediction_id = r.prediction_id
           JOIN users u ON u.user_id = r.patient_user_id
           JOIN patient_profiles pp ON pp.user_id = u.user_id
           WHERE r.request_id = ?""",
        (request_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "request_id": row["request_id"],
        "status": row["status"],
        "requested_at": row["requested_at"],
        "assigned_doctor_user_id": row["assigned_doctor_user_id"],
        "claimed_at": row["claimed_at"],
        "completed_at": row["completed_at"],
        "prediction_id": row["prediction_id"],
        "record_id": row["record_id"],
        "prediction_created_at": row["prediction_created_at"],
        "result": json.loads(row["result_json"]),
        "patient_user_id": row["patient_user_id"],
        "patient_name": row["patient_name"],
        "patient_id": row["patient_id"],
    }


def _patient_review_history(conn: sqlite3.Connection, patient_user_id: str, current_prediction_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT p.prediction_id, p.record_id, p.result_json, p.created_at,
                  r.status AS review_status, r.requested_at, d.decision, d.observations,
                  d.comments, d.reviewed_at, du.full_name AS doctor_name
           FROM ai_predictions p
           LEFT JOIN doctor_review_requests r ON r.prediction_id = p.prediction_id
           LEFT JOIN doctor_reviews d ON d.request_id = r.request_id
           LEFT JOIN users du ON du.user_id = d.doctor_user_id
           WHERE p.patient_user_id = ? AND p.prediction_id != ?
           ORDER BY p.created_at DESC LIMIT 20""",
        (patient_user_id, current_prediction_id),
    ).fetchall()
    history = []
    for row in rows:
        prediction = json.loads(row["result_json"])
        history.append({
            "prediction_id": row["prediction_id"],
            "created_at": row["created_at"],
            "ai_prediction": prediction.get("predicted_label"),
            "confidence": prediction.get("confidence"),
            "review_status": row["review_status"],
            "review_requested_at": row["requested_at"],
            "doctor_review": ({
                "decision": row["decision"],
                "observations": row["observations"],
                "comments": row["comments"],
                "reviewed_at": row["reviewed_at"],
                "doctor_name": row["doctor_name"],
            } if row["decision"] else None),
        })
    return history


def doctor_queue(doctor: dict) -> list[dict]:
    try:
        with _connection() as conn:
            rows = conn.execute(
                """SELECT r.request_id, r.status, r.requested_at, r.assigned_doctor_user_id,
                          p.prediction_id, p.record_id, p.result_json, u.full_name AS patient_name,
                          pp.patient_id
                   FROM doctor_review_requests r
                   JOIN ai_predictions p ON p.prediction_id = r.prediction_id
                   JOIN users u ON u.user_id = r.patient_user_id
                   JOIN patient_profiles pp ON pp.user_id = u.user_id
                   WHERE r.status IN ('PENDING', 'IN_REVIEW')
                     AND (r.assigned_doctor_user_id IS NULL OR r.assigned_doctor_user_id = ?)
                   ORDER BY r.requested_at""",
                (doctor["user_id"],),
            ).fetchall()
            queue = []
            for row in rows:
                result = json.loads(row["result_json"])
                queue.append({
                    "request_id": row["request_id"],
                    "status": row["status"],
                    "requested_at": row["requested_at"],
                    "assigned_doctor_user_id": row["assigned_doctor_user_id"],
                    "prediction_id": row["prediction_id"],
                    "record_id": row["record_id"],
                    "patient_name": row["patient_name"],
                    "patient_id": row["patient_id"],
                    "predicted_label": result["predicted_label"],
                    "confidence": result["confidence"],
                })
            return queue
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not load doctor review queue: {exc}") from exc


def claim_review_request(doctor: dict, request_id: str) -> dict:
    try:
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, assigned_doctor_user_id FROM doctor_review_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise AccountNotFoundError("Review request was not found.")
            if row["status"] == "COMPLETED":
                raise AccountConflictError("This review has already been completed.")
            if row["assigned_doctor_user_id"] not in (None, doctor["user_id"]):
                raise AccountNotFoundError("This review request is assigned to another doctor.")
            conn.execute(
                """UPDATE doctor_review_requests
                   SET assigned_doctor_user_id = ?, status = 'IN_REVIEW', claimed_at = COALESCE(claimed_at, ?)
                   WHERE request_id = ?""",
                (doctor["user_id"], _now(), request_id),
            )
            details = _request_details(conn, request_id)
            details["result"].pop("gradcam_overlay_base64", None)
            details["result"].pop("heatmap_base64", None)
            details["result"].pop("model_input_base64", None)
            details["image_urls"] = {
                kind: f"/auth/doctor/reviews/{request_id}/images/{kind}"
                for kind in ("original", "gradcam", "heatmap")
            }
            details["previous_review_history"] = _patient_review_history(
                conn, details["patient_user_id"], details["prediction_id"]
            )
            details.pop("patient_user_id", None)
            return details
    except (AccountNotFoundError, AccountConflictError):
        raise
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not open doctor review request: {exc}") from exc


def doctor_review_image(doctor: dict, request_id: str, kind: str) -> dict | None:
    try:
        with _connection() as conn:
            row = conn.execute(
                """SELECT r.assigned_doctor_user_id, p.record_id, pp.patient_id
                   FROM doctor_review_requests r
                   JOIN ai_predictions p ON p.prediction_id = r.prediction_id
                   JOIN patient_profiles pp ON pp.user_id = r.patient_user_id
                   WHERE r.request_id = ?""",
                (request_id,),
            ).fetchone()
            if row is None or row["assigned_doctor_user_id"] != doctor["user_id"]:
                return None
            return {"record_id": row["record_id"], "patient_id": row["patient_id"]}
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not authorize doctor image: {exc}") from exc


def submit_doctor_review(
    doctor: dict,
    request_id: str,
    decision: str,
    observations: str,
    comments: str,
) -> dict:
    decision = (decision or "").upper()
    observations = (observations or "").strip()
    comments = (comments or "").strip()
    if decision not in {"APPROVE", "REJECT"}:
        raise ValueError("Decision must be APPROVE or REJECT.")
    if not 1 <= len(observations) <= 5000 or len(comments) > 5000:
        raise ValueError("Observations are required (up to 5,000 characters); comments may be up to 5,000 characters.")
    reviewed_at = _now()
    review_id = uuid.uuid4().hex
    try:
        with _connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, assigned_doctor_user_id FROM doctor_review_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None or row["assigned_doctor_user_id"] != doctor["user_id"]:
                raise AccountNotFoundError("Review request is not assigned to this doctor.")
            if row["status"] != "IN_REVIEW":
                raise AccountConflictError("This request is not currently open for review.")
            conn.execute(
                """INSERT INTO doctor_reviews(
                    review_id, request_id, doctor_user_id, decision, observations, comments, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (review_id, request_id, doctor["user_id"], decision, observations, comments, reviewed_at),
            )
            conn.execute(
                "UPDATE doctor_review_requests SET status = 'COMPLETED', completed_at = ? WHERE request_id = ?",
                (reviewed_at, request_id),
            )
            return {
                "review_id": review_id,
                "request_id": request_id,
                "decision": decision,
                "observations": observations,
                "comments": comments,
                "doctor_name": doctor["full_name"],
                "reviewed_at": reviewed_at,
            }
    except (AccountNotFoundError, AccountConflictError, ValueError):
        raise
    except sqlite3.IntegrityError as exc:
        raise AccountConflictError("A doctor review has already been submitted for this request.") from exc
    except sqlite3.Error as exc:
        raise AccountStoreError(f"Could not submit doctor review: {exc}") from exc
