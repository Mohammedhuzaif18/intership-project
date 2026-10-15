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

import pandas as pd
import requests
from PIL import Image

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
