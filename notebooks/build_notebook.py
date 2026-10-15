"""Generates notebooks/ResNet50_Training_Colab.ipynb, a GPU-ready runner for the same pipeline.

Run: python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Interpretable Deep Learning for Respiratory Disorder Detection: Colab training\n\n"
       "Runs the project's own modules on a Colab GPU: data preparation (download, quality checks, "
       "patient-level split, CLAHE preprocessing), baseline CNN + ResNet50 training, test-set evaluation "
       "and Grad-CAM. Results match the local CPU pipeline, only faster.\n\n"
       "**Runtime -> Change runtime type -> GPU** before running."),
    md("## 1. Check the GPU"),
    code("!nvidia-smi -L\nimport tensorflow as tf\nprint('TensorFlow', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"),
    md("## 2. Get the project code\n\n"
       "Either set `REPO_URL` to your git remote, or leave it empty and upload a zip of the project folder "
       "(without `data/`, `models/` and `.venv/`)."),
    code("import os, zipfile\n"
         "REPO_URL = ''  # e.g. 'https://github.com/<you>/respiratory-xray.git'\n"
         "if REPO_URL:\n"
         "    !git clone -q $REPO_URL project\n"
         "else:\n"
         "    from google.colab import files\n"
         "    uploaded = files.upload()\n"
         "    zipfile.ZipFile(next(iter(uploaded))).extractall('/content/upload')\n"
         "    # The project root is the folder that contains src/, wherever it sits in the zip\n"
         "    root = next(d for d, subdirs, _ in os.walk('/content/upload') if 'src' in subdirs)\n"
         "    !rm -rf project && cp -r \"{root}\" project\n"
         "%cd project"),
    md("## 3. Install the few packages Colab lacks\n\nColab already ships TensorFlow, scikit-learn, pandas and matplotlib."),
    code("!pip install -q opencv-python-headless huggingface_hub tabulate"),
    md("## 4. Data: download, quality checks, patient-level split, CLAHE preprocessing"),
    code("!python -m src.data.prepare"),
    md("## 5. Train the baseline CNN and ResNet50 (5 frozen + 3 fine-tune epochs)"),
    code("!python -m src.train --task binary"),
    md("## 6. Grad-CAM on 10 held-out test images"),
    code("!python -m src.gradcam"),
    md("## 7. Results (held-out test set)"),
    code("from IPython.display import Image, Markdown, display\n"
         "display(Markdown(open('results/model_comparison.md').read()))\n"
         "for f in ['results/roc_curves.png', 'results/confusion_matrix_resnet50.png',\n"
         "          'results/training_curves_resnet50.png', 'results/gradcam/gradcam_grid.png']:\n"
         "    display(Image(f, width=800))"),
    md("## 8. Download the trained models and results"),
    code("!zip -qr trained_outputs.zip models results splits\n"
         "from google.colab import files\nfiles.download('trained_outputs.zip')"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "kernelspec": {
    "name": "python3", "display_name": "Python 3"}, "colab": {"provenance": []}})
out = Path(__file__).resolve().parent / "ResNet50_Training_Colab.ipynb"
nbf.write(nb, out)
print(f"Wrote {out}")
