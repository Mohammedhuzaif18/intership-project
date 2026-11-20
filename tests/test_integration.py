"""
End-to-end check of the running app: starts uvicorn, sends real held-out test X-rays to
/predict and verifies the response, including that the API reproduces the offline
test-set evaluation (same preprocessing, same model, same prediction).

Run: python tests/test_integration.py
"""
import base64
import hashlib
import hmac
import json
import os
import io
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES_PATH, RESULTS_DIR  # noqa: E402
from src.utils import absolute, load_json  # noqa: E402
from backend.patient_history import describe_prediction_change  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_model(base: str, proc, timeout: float = 240):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        try:
            h = requests.get(f"{base}/health", timeout=5).json()
            if h["model_loaded"]:
                return h
            if h.get("error"):
                raise RuntimeError(f"Model failed to load: {h['error']}")
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError("Server did not become ready in time")


def decode_data_url(url: str) -> Image.Image:
    assert url.startswith("data:image/jpeg;base64,"), url[:40]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def drawn_portrait() -> bytes:
    """A colour, photo-like portrait (face, hair, clothes): clearly not a chest X-ray."""
    yy, xx = np.mgrid[0:600, 0:480]
    img = np.stack([90 + xx * 120 // 480, 127 + yy * 80 // 600, np.full_like(xx, 168)], axis=-1).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.ellipse([90, 50, 390, 390], fill=(43, 29, 20))
    d.ellipse([115, 130, 365, 450], fill=(201, 154, 120))
    d.ellipse([177, 251, 213, 269], fill=(27, 27, 27))
    d.ellipse([267, 251, 303, 269], fill=(27, 27, 27))
    d.arc([195, 305, 285, 395], 30, 150, fill=(122, 59, 46), width=6)
    d.rectangle([70, 470, 410, 600], fill=(31, 42, 68))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def encoded_image(array: np.ndarray, image_format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=image_format)
    return buf.getvalue()


def upload(path: Path, base: str, patient_id: str | None = None):
    with open(path, "rb") as f:
        form = {"file": (path.name, f, "image/jpeg")}
        fields = {"patient_id": patient_id} if patient_id is not None else None
        return requests.post(f"{base}/predict", files=form, data=fields, timeout=120)


def portal_upload(path: Path, base: str, token: str):
    with open(path, "rb") as f:
        return requests.post(
            f"{base}/auth/patient/predict",
            files={"file": (path.name, f, "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )


def expired_token(token: str, secret_path: Path) -> str:
    header, encoded_payload, _ = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(encoded_payload + "=" * (-len(encoded_payload) % 4)))
    payload["exp"] = int(time.time()) - 10
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=").decode()
    signing_input = f"{header}.{encoded}"
    signature = hmac.new(secret_path.read_bytes(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def create_legacy_history_database(path: str) -> None:
    """Recreate the previous duplicate-rejecting schema to exercise startup migration."""
    database = Path(path) / "history.sqlite3"
    conn = sqlite3.connect(database)
    try:
        conn.executescript(
            """
            CREATE TABLE patients (patient_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
            CREATE TABLE patient_records (
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
                task TEXT NOT NULL,
                UNIQUE(patient_id, image_sha256)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def main():
    cases = [
        ("PNEUMONIA", "NORMAL", "possible_improvement_signal"),
        ("NORMAL", "COVID", "possible_new_or_worsening_signal"),
        ("TB", "TB", "no_predicted_category_change_disease"),
        ("NORMAL", "NORMAL", "no_predicted_category_change_normal"),
        ("COVID", "TB", "predicted_disease_category_changed"),
    ]
    for previous_class, current_class, expected in cases:
        change = describe_prediction_change(
            {"predicted_class": previous_class, "predicted_label": previous_class, "confidence": 0.8},
            {"predicted_class": current_class, "predicted_label": current_class, "confidence": 0.7},
        )
        assert change["assessment"] == expected, change
    print("[PASS] model-change wording covers possible improvement, worsening, unchanged and changed disease categories")

    meta = load_json(CLASS_NAMES_PATH)
    classes = meta["classes"]
    results_dir = RESULTS_DIR if meta["task"] == "binary" else RESULTS_DIR / meta["task"]
    offline = pd.read_csv(results_dir / f"test_predictions_{meta['model_file'].removesuffix('.keras')}.csv")
    # One test image per class, plus a misclassified one if the model has any
    picks = [offline[offline["true"] == c].iloc[0] for c in classes]
    wrong = offline[offline["true"] != offline["pred"]]
    if len(wrong):
        picks.append(wrong.iloc[0])

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    history_dir = tempfile.mkdtemp(prefix="pulmoscan-history-test-")
    create_legacy_history_database(history_dir)
    server_env = os.environ.copy()
    server_env["PATIENT_HISTORY_DIR"] = history_dir
    server_env["DOCTOR_REGISTRATION_CODE"] = "7391"
    server_env.pop("AUTH_JWT_SECRET", None)
    server_env["AUTH_TOKEN_LIFETIME_SECONDS"] = "3600"
    server_args = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
                   "--port", str(port)]
    proc = subprocess.Popen(server_args, cwd=ROOT, env=server_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        health = wait_for_model(base, proc)
        print(f"[PASS] /health: {health['model']} ({health['task']}), classes={health['classes']}, "
              f"layer={health['target_layer']}")
        assert health["classes"] == classes

        for row in picks:
            path = absolute(row["path"])
            with open(path, "rb") as image_file:
                quality_response = requests.post(
                    f"{base}/quality",
                    files={"file": (path.name, image_file, "image/jpeg")},
                    timeout=30,
                )
            assert quality_response.status_code == 200, quality_response.text[:300]
            assert quality_response.json()["quality"]["passed"], quality_response.json()
            r = upload(path, base)
            assert r.status_code == 200, f"{r.status_code}: {r.text[:300]}"
            d = r.json()
            probs = d["class_probabilities"]
            assert d["success"] is True
            assert d.get("is_chest_xray", True) is True, f"real test X-ray rejected by the chest gate: {d}"
            assert set(probs) == set(classes)
            assert abs(sum(probs.values()) - 1) < 1e-3, probs
            assert d["predicted_class"] in classes
            assert abs(d["confidence"] - max(probs.values())) < 1e-4
            for key in ("gradcam_overlay_base64", "heatmap_base64", "model_input_base64"):
                img = decode_data_url(d[key])
                assert img.size == (224, 224), (key, img.size)
            # Same preprocessing as training: the API must reproduce the offline test prediction
            assert d["predicted_class"] == row["pred"], (d["predicted_class"], row["pred"])
            for c in classes:
                assert abs(probs[c] - row[f"p_{c}"]) < 2e-3, (c, probs[c], row[f"p_{c}"])
            print(f"[PASS] /predict {path.name}: true={row['true']} pred={d['predicted_class']} "
                  f"conf={d['confidence']:.4f} (matches offline evaluation)")

        quality_cases = [
            ("dark", np.zeros((256, 256), dtype=np.uint8), "too_dark"),
            ("bright", np.full((256, 256), 255, dtype=np.uint8), "too_bright"),
            ("small", np.arange(32 * 32, dtype=np.uint8).reshape(32, 32), "low_resolution"),
        ]
        noise = np.random.default_rng(42).integers(50, 200, size=(256, 256), dtype=np.uint8)
        quality_cases.append(("blurred", cv2.GaussianBlur(noise, (41, 41), 12), "blurry"))
        wide_noise = np.random.default_rng(43).integers(40, 220, size=(200, 512), dtype=np.uint8)
        quality_cases.append(("unusual-framing", wide_noise, "framing"))
        for label, pixels, issue in quality_cases:
            image_bytes = encoded_image(pixels)
            response = requests.post(
                f"{base}/quality",
                files={"file": (f"{label}.png", image_bytes, "image/png")},
                timeout=30,
            )
            assert response.status_code == 200, response.text[:300]
            report = response.json()["quality"]
            assert not report["passed"] and issue in {item["code"] for item in report["issues"]}, report
            print(f"[PASS] /quality detects {label} image ({issue})")
            blocked = requests.post(
                f"{base}/predict",
                files={"file": (f"{label}.png", image_bytes, "image/png")},
                timeout=30,
            )
            assert blocked.status_code == 422, blocked.text[:300]
            assert "quality" in blocked.json()["detail"]
        print("[PASS] /predict blocks quality failures before running inference")

        first_path = absolute(picks[0]["path"])
        second_path = absolute(picks[1]["path"])
        assert requests.get(f"{base}/patients/PRIVATE/history", timeout=30).status_code == 401
        patient_registration = requests.post(
            f"{base}/auth/patient/register",
            json={
                "email": "patient1@example.test",
                "full_name": "Patient One",
                "password": "sixsix",
            },
            timeout=30,
        )
        assert patient_registration.status_code == 201, patient_registration.text[:300]
        patient_session = patient_registration.json()
        patient_token = patient_session["access_token"]
        patient_headers = {"Authorization": f"Bearer {patient_token}"}
        patient_id = patient_session["user"]["patient_id"]
        assert patient_session["user"]["role"] == "PATIENT"
        assert requests.post(
            f"{base}/auth/patient/register",
            json={
                "email": "short-password@example.test",
                "full_name": "Short Password",
                "password": "12345",
            },
            timeout=30,
        ).status_code == 422
        short_password_login = requests.post(
            f"{base}/auth/login",
            json={"email": "patient1@example.test", "password": "sixsix", "role": "PATIENT"},
            timeout=30,
        )
        assert short_password_login.status_code == 200
        assert short_password_login.json()["user"]["user_id"] == patient_session["user"]["user_id"]
        assert requests.get(f"{base}/auth/me", headers=patient_headers, timeout=30).json()["user"]["role"] == "PATIENT"
        assert requests.get(f"{base}/auth/doctor/reviews", headers=patient_headers, timeout=30).status_code == 403
        assert requests.get(f"{base}/auth/patient/predictions", timeout=30).status_code == 401
        bad_login = requests.post(
            f"{base}/auth/login",
            json={"email": "patient1@example.test", "password": "wrong-password-123", "role": "PATIENT"},
            timeout=30,
        )
        assert bad_login.status_code == 401
        wrong_role_login = requests.post(
            f"{base}/auth/login",
            json={"email": "patient1@example.test", "password": "sixsix", "role": "DOCTOR"},
            timeout=30,
        )
        assert wrong_role_login.status_code == 401
        assert requests.get(
            f"{base}/auth/me",
            headers={"Authorization": "Bearer invalid.token.value"},
            timeout=30,
        ).status_code == 401
        assert requests.get(
            f"{base}/auth/me",
            headers={
                "Authorization": (
                    f"Bearer {expired_token(patient_token, Path(history_dir) / 'jwt_signing_key')}"
                )
            },
            timeout=30,
        ).status_code == 401
        print("[PASS] patient registration/login, invalid/expired JWT, and role restrictions")

        first_response = portal_upload(first_path, base, patient_token)
        assert first_response.status_code == 200, first_response.text[:300]
        first_record = first_response.json()["patient_record"]
        assert first_record["previous_record_id"] is None, first_record
        first_prediction_id = first_response.json()["prediction_id"]
        first_history = requests.get(
            f"{base}/patients/{patient_id}/history",
            headers=patient_headers,
            timeout=30,
        )
        assert first_history.status_code == 200
        assert len(first_history.json()["records"]) == 1
        ai_report = requests.get(
            f"{base}/auth/patient/predictions/{first_prediction_id}/report",
            headers=patient_headers,
            timeout=30,
        )
        assert ai_report.status_code == 200 and ai_report.json()["report_type"] == "AI_ONLY"
        assert ai_report.json()["doctor_reviewed_report"] is None
        print("[PASS] authenticated first patient upload and immediate AI-only report")

        second_response = portal_upload(second_path, base, patient_token)
        assert second_response.status_code == 200, second_response.text[:300]
        second_record = second_response.json()["patient_record"]
        assert second_record["previous_record_id"] == first_record["record_id"], second_record
        second_prediction_id = second_response.json()["prediction_id"]
        review_request = requests.post(
            f"{base}/auth/patient/predictions/{first_prediction_id}/review-request",
            headers=patient_headers,
            timeout=30,
        )
        assert review_request.status_code == 201, review_request.text[:300]
        review_request = review_request.json()
        assert review_request["status"] == "PENDING"
        assert requests.post(
            f"{base}/auth/patient/predictions/{first_prediction_id}/review-request",
            headers=patient_headers,
            timeout=30,
        ).status_code == 409
        assert requests.post(
            f"{base}/auth/doctor/register",
            json={
                "email": "doctor1@example.test",
                "full_name": "Doctor One",
                "password": "secure-doctor-password-1",
                "verification_code": "incorrect-code",
            },
            timeout=30,
        ).status_code == 403
        malformed_registration_code = requests.post(
            f"{base}/auth/doctor/register",
            json={
                "email": "doctor-malformed-code@example.test",
                "full_name": "Doctor Malformed Code",
                "password": "secure-doctor-password-1",
                "verification_code": "48A7",
            },
            timeout=30,
        )
        assert malformed_registration_code.status_code == 403
        doctor_registration = requests.post(
            f"{base}/auth/doctor/register",
            json={
                "email": "doctor1@example.test",
                "full_name": "Doctor One",
                "password": "secure-doctor-password-1",
                "verification_code": "7391",
            },
            timeout=30,
        )
        assert doctor_registration.status_code == 201, doctor_registration.text[:300]
        doctor_session = doctor_registration.json()
        doctor_token = doctor_session["access_token"]
        doctor_headers = {"Authorization": f"Bearer {doctor_token}"}
        assert doctor_session["user"]["role"] == "DOCTOR"
        assert requests.get(f"{base}/auth/patient/predictions", headers=doctor_headers, timeout=30).status_code == 403
        assert requests.get(f"{base}/auth/doctor/reviews", headers=patient_headers, timeout=30).status_code == 403
        queue = requests.get(f"{base}/auth/doctor/reviews", headers=doctor_headers, timeout=30)
        assert queue.status_code == 200 and len(queue.json()["reviews"]) == 1
        queued_review = queue.json()["reviews"][0]
        assert queued_review["request_id"] == review_request["request_id"]
        assert queued_review["patient_id"] == patient_id
        claim = requests.post(
            f"{base}/auth/doctor/reviews/{review_request['request_id']}/claim",
            headers=doctor_headers,
            timeout=30,
        )
        assert claim.status_code == 200, claim.text[:300]
        claimed = claim.json()
        assert claimed["status"] == "IN_REVIEW"
        assert claimed["result"]["predicted_class"] == first_response.json()["predicted_class"]
        assert len(claimed["previous_review_history"]) == 1
        assert claimed["previous_review_history"][0]["prediction_id"] == second_prediction_id
        for kind, url in claimed["image_urls"].items():
            image_response = requests.get(f"{base}{url}", headers=doctor_headers, timeout=30)
            assert image_response.status_code == 200
            assert image_response.headers["content-type"].startswith("image/")
            assert requests.get(f"{base}{url}", headers=patient_headers, timeout=30).status_code == 403
        doctor_two = requests.post(
            f"{base}/auth/doctor/register",
            json={
                "email": "doctor2@example.test",
                "full_name": "Doctor Two",
                "password": "secure-doctor-password-2",
                "verification_code": "7391",
            },
            timeout=30,
        )
        assert doctor_two.status_code == 201, doctor_two.text[:300]
        doctor_two_headers = {"Authorization": f"Bearer {doctor_two.json()['access_token']}"}
        assert requests.get(
            f"{base}/auth/doctor/reviews",
            headers=doctor_two_headers,
            timeout=30,
        ).json()["reviews"] == []
        assert requests.post(
            f"{base}/auth/doctor/reviews/{review_request['request_id']}/claim",
            headers=doctor_two_headers,
            timeout=30,
        ).status_code == 404
        assert requests.get(
            f"{base}{claimed['image_urls']['original']}",
            headers=doctor_two_headers,
            timeout=30,
        ).status_code == 404
        submitted_review = requests.post(
            f"{base}/auth/doctor/reviews/{review_request['request_id']}/submit",
            headers=doctor_headers,
            json={
                "decision": "APPROVE",
                "observations": "The provided image is technically reviewable.",
                "comments": "Educational workflow test.",
            },
            timeout=30,
        )
        assert submitted_review.status_code == 200, submitted_review.text[:300]
        assert submitted_review.json()["decision"] == "APPROVE"
        reviewed_report_response = requests.get(
            f"{base}/auth/patient/predictions/{first_prediction_id}/report",
            headers=patient_headers,
            timeout=30,
        )
        assert reviewed_report_response.status_code == 200
        reviewed_report = reviewed_report_response.json()
        assert reviewed_report["report_type"] == "DOCTOR_REVIEWED"
        assert reviewed_report["result"]["predicted_class"] == first_response.json()["predicted_class"]
        assert reviewed_report["doctor_reviewed_report"]["decision"] == "APPROVE"
        assert reviewed_report["doctor_reviewed_report"]["doctor_name"] == "Doctor One"
        patient_report_list = requests.get(
            f"{base}/auth/patient/predictions",
            headers=patient_headers,
            timeout=30,
        )
        assert patient_report_list.status_code == 200
        completed_entry = next(
            report for report in patient_report_list.json()["predictions"]
            if report["prediction_id"] == first_prediction_id
        )
        assert completed_entry["review_request"]["status"] == "COMPLETED"
        assert completed_entry["review_request"]["review"]["observations"] == "The provided image is technically reviewable."
        print("[PASS] verified doctor registration, review queue/claim, protected images and AI-plus-doctor report")

        rejected_request = requests.post(
            f"{base}/auth/patient/predictions/{second_prediction_id}/review-request",
            headers=patient_headers,
            timeout=30,
        )
        assert rejected_request.status_code == 201
        rejected_request = rejected_request.json()
        rejected_claim = requests.post(
            f"{base}/auth/doctor/reviews/{rejected_request['request_id']}/claim",
            headers=doctor_headers,
            timeout=30,
        )
        assert rejected_claim.status_code == 200
        prior_reviews = rejected_claim.json()["previous_review_history"]
        assert len(prior_reviews) == 1
        assert prior_reviews[0]["doctor_review"]["decision"] == "APPROVE"
        rejected_review = requests.post(
            f"{base}/auth/doctor/reviews/{rejected_request['request_id']}/submit",
            headers=doctor_headers,
            json={
                "decision": "REJECT",
                "observations": "The model output is not approved.",
                "comments": "",
            },
            timeout=30,
        )
        assert rejected_review.status_code == 200
        assert rejected_review.json()["decision"] == "REJECT"
        rejected_report = requests.get(
            f"{base}/auth/patient/predictions/{second_prediction_id}/report",
            headers=patient_headers,
            timeout=30,
        )
        assert rejected_report.status_code == 200
        assert rejected_report.json()["doctor_reviewed_report"]["decision"] == "REJECT"
        print("[PASS] doctor rejection remains separate from the original AI prediction")

        history = requests.get(
            f"{base}/patients/{patient_id}/history",
            headers=patient_headers,
            timeout=30,
        ).json()["records"]
        assert len(history) == 2 and history[0]["record_id"] == first_record["record_id"]
        assert history[0]["uploaded_at"] <= history[1]["uploaded_at"]
        assert all(record["patient_id"] == patient_id for record in history)

        params = {"record_a_id": first_record["record_id"], "record_b_id": second_record["record_id"]}
        comparison_response = requests.get(
            f"{base}/patients/{patient_id}/compare",
            params=params,
            headers=patient_headers,
            timeout=60,
        )
        assert comparison_response.status_code == 200, comparison_response.text[:300]
        comparison = comparison_response.json()
        assert comparison["previous"]["record_id"] == first_record["record_id"]
        assert comparison["current"]["record_id"] == second_record["record_id"]
        assert comparison["patient_id"] == patient_id
        assert comparison["prediction_change"]["summary"]
        assert comparison["prediction_change"]["assessment"] == "possible_new_or_worsening_signal"
        assert "classified as Normal" in comparison["prediction_change"]["summary"]
        assert "does not confirm" in comparison["prediction_change"]["summary"]
        assert "cannot confirm improvement, worsening, or disease severity" in comparison["disclaimer"]
        assert comparison["difference_map_base64"].startswith("data:image/jpeg;base64,")
        for record in (comparison["previous"], comparison["current"]):
            for kind, url in record["image_urls"].items():
                image_response = requests.get(f"{base}{url}", timeout=30)
                assert image_response.status_code == 401
                image_response = requests.get(f"{base}{url}", headers=patient_headers, timeout=30)
                assert image_response.status_code == 200 and image_response.headers["content-type"].startswith("image/")
                Image.open(io.BytesIO(image_response.content)).verify()
        print("[PASS] same-patient history and OpenCV comparison return protected X-rays, Grad-CAMs and a difference map")

        patient_two = requests.post(
            f"{base}/auth/patient/register",
            json={
                "email": "patient2@example.test",
                "full_name": "Patient Two",
                "password": "secure-patient-password-2",
            },
            timeout=30,
        )
        assert patient_two.status_code == 201, patient_two.text[:300]
        patient_two_token = patient_two.json()["access_token"]
        own_data_attack = requests.get(
            f"{base}/auth/patient/predictions/{first_prediction_id}/report",
            headers={"Authorization": f"Bearer {patient_two_token}"},
            timeout=30,
        )
        assert own_data_attack.status_code == 404
        cross = requests.get(
            f"{base}/patients/{patient_id}/compare",
            params={"record_a_id": first_record["record_id"], "record_b_id": second_record["record_id"]},
            headers={"Authorization": f"Bearer {patient_two_token}"},
            timeout=30,
        )
        assert cross.status_code == 404, cross.text[:300]
        assert requests.get(f"{base}/patients/{patient_id}/history", timeout=30).status_code == 401
        repeated_image = portal_upload(first_path, base, patient_token)
        assert repeated_image.status_code == 200, repeated_image.text[:300]
        repeated_record = repeated_image.json()["patient_record"]
        assert repeated_record["previous_record_id"] == second_record["record_id"]
        updated_history = requests.get(
            f"{base}/patients/{patient_id}/history",
            headers=patient_headers,
            timeout=30,
        ).json()["records"]
        assert len(updated_history) == 3
        assert updated_history[-1]["record_id"] == repeated_record["record_id"]
        assert updated_history[-1]["filename"] == first_path.name
        with open(second_path, "rb") as image_file:
            ownership_attack = requests.post(
                f"{base}/predict",
                files={"file": (second_path.name, image_file, "image/jpeg")},
                data={"patient_id": "somebody-else"},
                timeout=60,
            )
        assert ownership_attack.status_code == 401, ownership_attack.text[:300]
        print("[PASS] patient record isolation and repeated same-image upload")

        r = requests.post(f"{base}/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}, timeout=30)
        assert r.status_code == 400, r.status_code
        print("[PASS] /predict rejects a non-image with 400")

        if health.get("chest_gate", {}).get("enabled"):
            r = requests.post(f"{base}/predict", files={"file": ("portrait.jpg", drawn_portrait(), "image/jpeg")},
                              timeout=120)
            assert r.status_code == 200, r.status_code
            d = r.json()
            assert d["is_chest_xray"] is False and d["predicted_class"] is None, d
            assert d["class_probabilities"] == {} and d["heatmap_base64"] is None
            print(f"[PASS] /predict gives no disease result for a non-chest photo "
                  f"(similarity {d['chest_score']:.2f} < {d['chest_threshold']:.2f})")
        else:
            print("[SKIP] chest gate not fitted (run python -m src.gate); non-chest check skipped")

        for c in classes:
            r = requests.get(f"{base}/samples/{c}", timeout=30)
            assert r.status_code == 200 and r.headers["content-type"].startswith("image/"), (c, r.status_code)
            Image.open(io.BytesIO(r.content)).verify()
        assert requests.get(f"{base}/samples/NOT_A_CLASS", timeout=30).status_code == 404
        print("[PASS] /samples returns real test images per class and 404 for unknown classes")

        r = requests.get(f"{base}/", timeout=30)
        assert r.status_code == 200 and "Educational prototype, not a diagnostic tool" in r.text
        assert "Patient history" in r.text and "/quality" not in r.text
        print("[PASS] frontend served at / with the disclaimer and patient history interface")
        portal = requests.get(f"{base}/auth.html", timeout=30)
        assert portal.status_code == 200 and "Patient registration" in portal.text
        assert "AI demo" not in portal.text and "Open legacy AI demo" not in portal.text
        index_page = requests.get(f"{base}/index.html", timeout=30)
        assert index_page.status_code == 200 and "intakeForm" in index_page.text
        assert "My X-ray reports and doctor recommendations" in index_page.text
        report_page = requests.get(f"{base}/results.html", timeout=30)
        assert report_page.status_code == 200 and "requestDoctorReviewBtn" in report_page.text
        print("[PASS] login is separate, patient analysis uses the main app, and report offers optional doctor review")

        proc.terminate()
        proc.wait(timeout=20)
        proc = subprocess.Popen(server_args, cwd=ROOT, env=server_env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_model(base, proc)
        restored = requests.get(
            f"{base}/patients/{patient_id}/history",
            headers=patient_headers,
            timeout=30,
        )
        assert restored.status_code == 200 and len(restored.json()["records"]) == 3
        restored_report = requests.get(
            f"{base}/auth/patient/predictions/{first_prediction_id}/report",
            headers=patient_headers,
            timeout=30,
        )
        assert restored_report.status_code == 200
        assert restored_report.json()["doctor_reviewed_report"]["decision"] == "APPROVE"
        restored_queue = requests.get(f"{base}/auth/doctor/reviews", headers=doctor_headers, timeout=30)
        assert restored_queue.status_code == 200
        print("[PASS] patient history, accounts, predictions and doctor reviews persist after restart")
        print("\nAll integration checks passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(history_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
