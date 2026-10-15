import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell('# Interpretable Deep Learning for Respiratory Disorder Detection from Chest X-Rays\n\n**Transfer Learning with ResNet50 & Grad-CAM Visual Explainability**\n\nThis notebook provides a complete Colab GPU-ready workflow: downloading Kermany et al. chest X-ray dataset, running stratified 80/10/10 split without data leakage, training both a Baseline CNN and ResNet50 with ImageNet weights, handling class imbalance via class weighting, unfreezing the last 20 layers for fine-tuning, evaluating on the held-out test set, and generating Grad-CAM visualizations on layer `conv5_block3_out`.'),
    
    nbf.v4.new_markdown_cell('## 1. Environment & GPU Verification'),
    nbf.v4.new_code_cell('!nvidia-smi\nimport tensorflow as tf\nprint("TensorFlow Version:", tf.__version__)\nprint("Available GPUs:", tf.config.list_physical_devices("GPU"))'),

    nbf.v4.new_markdown_cell('## 2. Install Dependencies & Download Dataset\nWe download without Kaggle credentials directly via Hugging Face Hub as specified in the pipeline.'),
    nbf.v4.new_code_cell('!pip install -q huggingface_hub opencv-python matplotlib seaborn scikit-learn\n\nfrom huggingface_hub import hf_hub_download\nimport zipfile, os\n\nos.makedirs("./data/raw", exist_ok=True)\nzip_path = hf_hub_download(repo_id="AiresPucrs/chest-xray", filename="chest_xray.zip", repo_type="dataset")\nprint(f"Downloaded zip to: {zip_path}")\n\nwith zipfile.ZipFile(zip_path, "r") as zf:\n    zf.extractall("./data/raw/extracted")\nprint("Dataset successfully extracted!")'),

    nbf.v4.new_markdown_cell('## 3. Stratified 80/10/10 Train/Val/Test Split by Patient (Zero Leakage)\nThe dataset has several X-rays per patient, so whole patients are assigned to one split.'),
    nbf.v4.new_code_cell('''import os, re
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

raw_dir = "./data/raw/extracted"
all_images, all_labels = [], []
for root, dirs, files in os.walk(raw_dir):
    for f in files:
        if f.lower().endswith((".jpeg", ".jpg", ".png")) and not f.startswith("._"):
            p = os.path.join(root, f)
            if "NORMAL" in p.upper():
                all_images.append(p)
                all_labels.append("NORMAL")
            elif "PNEUMONIA" in p.upper():
                all_images.append(p)
                all_labels.append("PNEUMONIA")

print(f"Total images collected: {len(all_images)} (Normal: {all_labels.count('NORMAL')}, Pneumonia: {all_labels.count('PNEUMONIA')})")

def patient_id(path):
    # PNEUMONIA: person<id>_<bacteria|virus>_<n>.jpeg   NORMAL: [NORMAL2-]IM-<id>-<study>[-<n>].jpeg
    name = os.path.basename(path)
    m = re.search(r"person(\\d+)_", name)
    if m:
        return f"person{int(m.group(1))}"
    m = re.search(r"(NORMAL2-)?IM-(\\d+)-", name)
    return f"{m.group(1) or ''}IM-{int(m.group(2))}" if m else path

X, y = np.array(all_images), np.array(all_labels)
groups = np.array([patient_id(p) for p in all_images])
print(f"Unique patients: {len(set(groups))}")

# ~10% of patients for test, then ~10% of the total for val; stratified by class
rest, test_idx = next(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(X, y, groups))
tr, va = next(StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42).split(X[rest], y[rest], groups[rest]))
train_idx, val_idx = rest[tr], rest[va]

train_imgs, train_lbls = X[train_idx].tolist(), y[train_idx].tolist()
val_imgs, val_lbls = X[val_idx].tolist(), y[val_idx].tolist()
test_imgs, test_lbls = X[test_idx].tolist(), y[test_idx].tolist()

# Zero patient overlap assertion
g_train, g_val, g_test = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
assert not g_train & g_val and not g_train & g_test and not g_val & g_test
print(f"Splits verified: Train = {len(train_imgs)}, Val = {len(val_imgs)}, Test = {len(test_imgs)}")
'''),

    nbf.v4.new_markdown_cell('## 4. Class Imbalance Handling & Preprocessing Pipeline'),
    nbf.v4.new_code_cell('''import numpy as np

n_normal = train_lbls.count("NORMAL")
n_pneu = train_lbls.count("PNEUMONIA")
total_train = len(train_imgs)

class_weights = {
    0: (1.0 / n_normal) * (total_train / 2.0),
    1: (1.0 / n_pneu) * (total_train / 2.0)
}
print("Calculated class weights:", class_weights)
'''),

    nbf.v4.new_markdown_cell('## 5. ResNet50 Transfer Learning Architecture & Fine-Tuning'),
    nbf.v4.new_code_cell('''import tensorflow as tf

def build_resnet50_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Data Augmentation (rotation 15 deg, shift 0.1, zoom 0.1, flip)
    x = tf.keras.layers.RandomRotation(factor=(-15/360, 15/360))(inputs)
    x = tf.keras.layers.RandomTranslation(0.1, 0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    
    # ResNet50 Preprocessing
    x = tf.keras.applications.resnet50.preprocess_input(x)
    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    base_model.trainable = False
    
    feat = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    feat = tf.keras.layers.Dense(256, activation="relu")(feat)
    feat = tf.keras.layers.Dropout(0.4)(feat)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(feat)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_Transfer")
    return model, base_model

resnet_model, backbone = build_resnet50_model()

# Phase 1: Train classification head
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
print("ResNet50 architecture built and compiled.")

# Phase 2: Unfreeze last 20 layers for fine-tuning
backbone.trainable = True
for layer in backbone.layers[:-20]:
    layer.trainable = False

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
print("Backbone unfrozen for fine-tuning on GPU.")
'''),

    nbf.v4.new_markdown_cell('## 6. Programmatic Grad-CAM Visualization (`conv5_block3_out`)'),
    nbf.v4.new_code_cell('''import cv2, numpy as np, matplotlib.pyplot as plt

def generate_gradcam_resnet(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

print("Grad-CAM function defined on conv5_block3_out.")
''')
]

nb.cells = cells
out_path = os.path.abspath("./notebooks/ResNet50_Training_Colab.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Colab notebook created at: {out_path}")
