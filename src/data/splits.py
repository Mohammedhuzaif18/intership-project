"""
Patient-grouped, stratified train/val/test splitting.

The Kermany set has several X-rays per patient, so splitting by image would put
the same patient in train and test and inflate test metrics. Every image gets a
patient key and each patient lands in exactly one split.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import logger

_PERSON = re.compile(r"person(\d+)_", re.IGNORECASE)
_NORMAL_IM = re.compile(r"(NORMAL2-)?IM-(\d+)-", re.IGNORECASE)


def kermany_patient_id(path) -> str | None:
    """
    Patient key parsed from a Kermany filename, or None if the name doesn't match.

    PNEUMONIA: person<id>_<bacteria|virus>_<n>.jpeg. 170 ids occur in both the
    original train and test folders; they are treated as the same patient, which
    can only make groups larger, never leak.
    NORMAL: [NORMAL2-]IM-<id>-<study>[-<n>].jpeg. IM and NORMAL2-IM are separate
    series (the original split uses disjoint id ranges within each series).
    """
    name = Path(path).name
    m = _PERSON.search(name)
    if m:
        return f"person{int(m.group(1))}"
    m = _NORMAL_IM.search(name)
    if m:
        return f"{(m.group(1) or '').upper()}IM-{int(m.group(2))}"
    return None


def assign_group_splits(df: pd.DataFrame, strat_col: str, group_col: str, seed: int,
                        val_frac: float = 0.10, test_frac: float = 0.10) -> pd.Series:
    """
    Stratified ~80/10/10 split in which every group lands in one split only.
    Returns a Series of "train" / "val" / "test" aligned with df.index.
    """
    y = df[strat_col].to_numpy()
    groups = df[group_col].to_numpy()
    idx = np.arange(len(df))

    outer = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    rest, test = next(outer.split(idx, y, groups))
    inner = StratifiedGroupKFold(n_splits=round((1 - test_frac) / val_frac), shuffle=True, random_state=seed)
    train_rel, val_rel = next(inner.split(rest, y[rest], groups[rest]))

    split = np.empty(len(df), dtype=object)
    split[rest[train_rel]] = "train"
    split[rest[val_rel]] = "val"
    split[test] = "test"
    out = pd.Series(split, index=df.index, name="split")
    assert_no_overlap(df.assign(split=out), group_col)
    return out


def assert_no_overlap(df: pd.DataFrame, col: str) -> None:
    """No value of `col` (patient id, file hash, ...) may appear in two splits."""
    per_value = df.groupby(col)["split"].nunique()
    shared = per_value[per_value > 1]
    assert shared.empty, f"Leakage: {len(shared)} values of '{col}' span splits, e.g. {list(shared.index[:3])}"
    logger.info(f"Verification passed: no '{col}' appears in more than one split.")


def save_split_files(df: pd.DataFrame, out_dir: Path, columns) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        part = df[df["split"] == name].sort_values("path")[columns]
        part.to_csv(out_dir / f"{name}.csv", index=False)
        logger.info(f"Saved {out_dir / f'{name}.csv'} ({len(part)} rows)")
