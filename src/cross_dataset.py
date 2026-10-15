"""
Cross-dataset generalization test (Phase 3, step 17).

Direction A (default): the Review-I binary model, trained only on Kermany (pediatric, one
centre), is tested on the Kermany test split (seen source) and on COVID-QU-Ex Normal and
Non-COVID infection images (unseen source: adults, many centres). The near-duplicate filter in
src.data.multiclass guarantees none of those COVID-QU-Ex images is a copy of a Kermany image.

Direction B (--reverse): ResNet50 trained on COVID-QU-Ex Normal / Non-COVID only, tested on
COVID-QU-Ex test images (seen) and the Kermany test split (unseen).

  python -m src.cross_dataset [--reverse]
"""
import argparse

import pandas as pd
import tensorflow as tf

from src.config import BINARY_CLASSES, DISPLAY_NAMES, MODELS_DIR, RESULTS_DIR, SEED, SPLITS_DIR
from src.data.loaders import class_weights, load_split, make_dataset
from src.evaluate import compute_metrics, plot_confusion_matrix, predict_probs
from src.utils import load_json, logger, save_json, set_seed

OUT = RESULTS_DIR / "cross_dataset"
CROSS_CSV = SPLITS_DIR / "cross_dataset" / "covidqu_normal_pneumonia.csv"


def _evaluate(model, df, train_source, test_source, seen, name):
    y, p = predict_probs(model, make_dataset(df, BINARY_CLASSES, training=False))
    m = compute_metrics(y, p, BINARY_CLASSES)
    slug = f"{train_source}_on_{test_source}".lower().replace(" ", "_").replace("-", "")
    plot_confusion_matrix(m["confusion_matrix"], BINARY_CLASSES,
                          f"{name}: trained on {train_source}, tested on {test_source}",
                          OUT / f"confusion_matrix_{slug}.png", DISPLAY_NAMES)
    return {"Model": name, "Trained on": train_source, "Tested on": test_source,
            "Source seen in training": "yes" if seen else "no", "n": m["n"], "Accuracy": m["accuracy"],
            "Precision": m["precision"], "Recall": m["recall"], "Specificity": m["specificity"],
            "F1": m["f1"], "ROC-AUC": m["roc_auc"]}


def main(reverse: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    if not CROSS_CSV.exists():
        msg = "No COVID-QU-Ex data prepared (run src.data.multiclass first); cross-dataset test skipped."
        logger.warning(msg)
        save_json({"status": "skipped", "reason": msg}, OUT / "status.json")
        return
    set_seed(SEED)
    unseen = pd.read_csv(CROSS_CSV)
    kermany_test = load_split("binary", "test")
    meta = load_json(MODELS_DIR / "binary" / "class_names.json")
    model = tf.keras.models.load_model(MODELS_DIR / "binary" / meta["model_file"])

    rows = [
        _evaluate(model, kermany_test, "Kermany", "Kermany", True, meta["model"]),
        _evaluate(model, unseen, "Kermany", "COVID-QU-Ex", False, meta["model"]),
    ]

    if reverse:
        from src.train import BUDGET, train_transfer

        train_df = unseen[unseen["split"] == "train"]
        val_df = unseen[unseen["split"] == "val"]
        test_df = unseen[unseen["split"] == "test"]
        logger.info(f"Reverse direction: training ResNet50 on {len(train_df)} COVID-QU-Ex images")
        rev, _, _ = train_transfer("resnet50", 2, make_dataset(train_df, BINARY_CLASSES, training=True),
                                   make_dataset(val_df, BINARY_CLASSES, training=False),
                                   class_weights(train_df, BINARY_CLASSES))
        rows += [
            _evaluate(rev, test_df, "COVID-QU-Ex", "COVID-QU-Ex", True, "ResNet50"),
            _evaluate(rev, kermany_test, "COVID-QU-Ex", "Kermany", False, "ResNet50"),
        ]
        logger.info(f"Reverse model budget: {BUDGET}")

    table = pd.DataFrame(rows)
    drops = []
    for train_src, g in table.groupby("Trained on", sort=False):
        seen = g[g["Source seen in training"] == "yes"].iloc[0]
        for _, r in g[g["Source seen in training"] == "no"].iterrows():
            drops.append({"Trained on": train_src, "Seen-source accuracy": seen["Accuracy"],
                          "Unseen-source accuracy": r["Accuracy"],
                          "Accuracy drop (points)": 100 * (seen["Accuracy"] - r["Accuracy"]),
                          "Seen-source F1": seen["F1"], "Unseen-source F1": r["F1"]})
    drops = pd.DataFrame(drops)
    table.to_csv(OUT / "cross_dataset_results.csv", index=False)
    drops.to_csv(OUT / "accuracy_drop.csv", index=False)
    with open(OUT / "cross_dataset_results.md", "w", encoding="utf-8") as f:
        f.write("# Cross-dataset generalization (test images only)\n\n")
        f.write(table.to_markdown(index=False, floatfmt=".4f") + "\n\n## Accuracy drop on the unseen source\n\n")
        f.write(drops.to_markdown(index=False, floatfmt=".4f") + "\n")
    save_json({"status": "done", "results": rows, "drops": drops.to_dict(orient="records")}, OUT / "status.json")
    logger.info(f"Cross-dataset results:\n{drops.to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse", action="store_true", help="Also train on COVID-QU-Ex and test on Kermany")
    main(parser.parse_args().reverse)
