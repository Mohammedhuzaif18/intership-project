"""Shared constants for the respiratory X-ray pipeline."""

PREFERRED_CLASSES = ["NORMAL", "PNEUMONIA", "COVID", "TB"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
BINARY_THRESHOLD = 0.5
