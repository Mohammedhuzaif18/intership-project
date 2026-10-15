"""
End-to-end check of the running app: starts uvicorn, sends real held-out test X-rays to
/predict and verifies the response, including that the API reproduces the offline
test-set evaluation (same preprocessing, same model, same prediction).

Run: python tests/test_integration.py
"""
import base64
import io
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES_PATH, RESULTS_DIR  # noqa: E402
from src.utils import absolute, load_json  # noqa: E402


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


def main():
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
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
                             "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        health = wait_for_model(base, proc)
        print(f"[PASS] /health: {health['model']} ({health['task']}), classes={health['classes']}, "
              f"layer={health['target_layer']}")
        assert health["classes"] == classes

        for row in picks:
            path = absolute(row["path"])
            with open(path, "rb") as f:
                r = requests.post(f"{base}/predict", files={"file": (path.name, f, "image/jpeg")}, timeout=120)
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
        print("[PASS] frontend served at / with the disclaimer")
        print("\nAll integration checks passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
