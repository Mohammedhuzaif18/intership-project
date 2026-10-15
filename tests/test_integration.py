import os
import sys
import glob
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.main import app


def test_api():
    print("Initializing TestClient for FastAPI backend...")
    client = TestClient(app)

    print("Testing GET /health...")
    health_resp = client.get("/health")
    print("Health response status:", health_resp.status_code)
    health_data = health_resp.json()
    print("Health response:", health_data)
    assert health_resp.status_code == 200
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True
    allowed = set(health_data.get("classes") or ["NORMAL", "PNEUMONIA"])
    print("[PASS] Health check verified.")

    test_images = glob.glob("./data/preprocessed_224/test/*/*.*")
    if not test_images:
        test_images = glob.glob("./data/processed_multiclass/test/*/*.*")
    if not test_images:
        test_images = glob.glob("./data/processed/test/*/*.*")

    assert len(test_images) > 0, "No test images found for integration test!"
    test_img_path = test_images[0]
    print(f"Testing POST /predict with real image: {test_img_path}")

    with open(test_img_path, "rb") as f:
        files = {"file": (os.path.basename(test_img_path), f, "image/jpeg")}
        predict_resp = client.post("/predict", files=files)

    print("Predict response status:", predict_resp.status_code)
    assert predict_resp.status_code == 200, f"Predict failed: {predict_resp.text}"
    pred_data = predict_resp.json()
    print("Predict response keys:", list(pred_data.keys()))
    print(f"Predicted class: {pred_data['predicted_class']}")
    print(f"Confidence: {pred_data['confidence']}")
    print(f"Class probabilities: {pred_data['class_probabilities']}")
    print(f"Target conv layer used: {pred_data['target_layer_used']}")
    print(f"Grad-CAM base64 prefix: {pred_data['gradcam_overlay_base64'][:50]}...")

    assert pred_data["success"] is True
    assert pred_data["predicted_class"] in allowed
    assert 0.0 <= pred_data["confidence"] <= 1.0
    assert "class_probabilities" in pred_data
    assert "gradcam_overlay_base64" in pred_data
    assert pred_data["gradcam_overlay_base64"].startswith("data:image/jpeg;base64,")
    print("[PASS] POST /predict and Grad-CAM generation fully verified!")


if __name__ == "__main__":
    test_api()
