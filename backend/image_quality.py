"""Technical image-quality screening for uploaded chest X-rays."""
import os

import cv2
import numpy as np
from PIL import Image


MIN_RESOLUTION = int(os.environ.get("XRAY_QUALITY_MIN_RESOLUTION", "128"))
MIN_SHARPNESS = float(os.environ.get("XRAY_QUALITY_MIN_SHARPNESS", "5.0"))
MIN_BRIGHTNESS = float(os.environ.get("XRAY_QUALITY_MIN_BRIGHTNESS", "20"))
MAX_BRIGHTNESS = float(os.environ.get("XRAY_QUALITY_MAX_BRIGHTNESS", "235"))
MIN_ASPECT_RATIO = float(os.environ.get("XRAY_QUALITY_MIN_ASPECT_RATIO", "0.5"))
MAX_ASPECT_RATIO = float(os.environ.get("XRAY_QUALITY_MAX_ASPECT_RATIO", "2.0"))


def thresholds() -> dict:
    return {
        "minimum_resolution_px": MIN_RESOLUTION,
        "minimum_laplacian_variance": MIN_SHARPNESS,
        "minimum_mean_brightness": MIN_BRIGHTNESS,
        "maximum_mean_brightness": MAX_BRIGHTNESS,
        "minimum_aspect_ratio": MIN_ASPECT_RATIO,
        "maximum_aspect_ratio": MAX_ASPECT_RATIO,
    }


def assess_image_quality(image: Image.Image) -> dict:
    """Return technical checks without altering the source image or model preprocessing."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    aspect_ratio = width / height

    issues = []
    suggestions = []
    if min(width, height) < MIN_RESOLUTION:
        issues.append({"code": "low_resolution", "message": f"Image resolution is low ({width}×{height}px)."})
        suggestions.append(f"Upload an image at least {MIN_RESOLUTION} pixels on each side.")
    if sharpness < MIN_SHARPNESS:
        issues.append({"code": "blurry", "message": "The image appears blurry or out of focus."})
        suggestions.append("Use a sharper, in-focus X-ray image without motion blur.")
    if brightness < MIN_BRIGHTNESS:
        issues.append({"code": "too_dark", "message": "The image is darker than the configured range."})
        suggestions.append("Upload a properly exposed image with visible chest detail.")
    elif brightness > MAX_BRIGHTNESS:
        issues.append({"code": "too_bright", "message": "The image is brighter than the configured range."})
        suggestions.append("Upload a properly exposed image with visible chest detail.")

    if not MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
        issues.append({"code": "framing", "message": "The image framing or crop has an unusual aspect ratio."})
        suggestions.append("Check that the full chest, including both lung fields, is visible and not cropped.")

    return {
        "status": "fail" if issues else "pass",
        "passed": not issues,
        "issues": issues,
        "warnings": [issue["message"] for issue in issues],
        "suggestions": suggestions,
        "measurements": {
            "width": width,
            "height": height,
            "laplacian_variance": round(sharpness, 2),
            "mean_brightness": round(brightness, 2),
            "aspect_ratio": round(aspect_ratio, 3),
        },
        "thresholds": thresholds(),
        "disclaimer": (
            "These are technical image checks only. Passing does not establish medical reliability "
            "or diagnostic accuracy."
        ),
    }
