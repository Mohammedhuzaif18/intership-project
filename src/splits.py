"""
Patient-grouped, stratified train/val/test splitting.

The Kermany set has several X-rays per patient, so splitting by image puts the
same patient in train and test and inflates test metrics. Every image gets a
patient key and each patient lands in exactly one split.
"""
import os
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.utils import logger

_PERSON = re.compile(r"person(\d+)_", re.IGNORECASE)
_NORMAL_IM = re.compile(r"(NORMAL2-)?IM-(\d+)-", re.IGNORECASE)


def kermany_patient_id(path: str):
    """
    Patient key parsed from a Kermany filename, or None if the name doesn't match.

    PNEUMONIA: person<id>_<bacteria|virus>_<n>.jpeg. 170 ids occur in both the
    original train and test folders; they are treated as the same patient, which
    can only make groups larger, never leak.
    NORMAL: [NORMAL2-]IM-<id>-<study>[-<n>].jpeg. IM and NORMAL2-IM are separate series.
    Split prefixes added by our own copies (e.g. "train_") are ignored.
    """
    name = os.path.basename(path)
    m = _PERSON.search(name)
    if m:
        return f"person{int(m.group(1))}"
    m = _NORMAL_IM.search(name)
    if m:
        return f"{(m.group(1) or '').upper()}IM-{int(m.group(2))}"
    return None


def kermany_groups(paths):
    """Patient keys for Kermany paths; unparseable names become their own group."""
    groups = []
    unknown = 0
    for p in paths:
        pid = kermany_patient_id(p)
        if pid is None:
            unknown += 1
            pid = f"file:{p}"
        groups.append(pid)
    if unknown:
        logger.warning(f"{unknown} images had no parseable patient id; each is treated as its own patient.")
    return groups


def stratified_group_split(paths, labels, groups, seed: int, val_frac: float = 0.10, test_frac: float = 0.10):
    """
    Stratified ~80/10/10 split in which every group (patient) lands in one split only.
    Returns {"train": (paths, labels), "val": (...), "test": (...)}.
    """
    paths, labels, groups = np.asarray(paths), np.asarray(labels), np.asarray(groups)

    outer = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(paths, labels, groups))

    inner = StratifiedGroupKFold(n_splits=round((1 - test_frac) / val_frac), shuffle=True, random_state=seed)
    train_rel, val_rel = next(inner.split(paths[rest_idx], labels[rest_idx], groups[rest_idx]))

    split_idx = {"train": rest_idx[train_rel], "val": rest_idx[val_rel], "test": test_idx}
    assert_no_group_overlap({name: groups[idx] for name, idx in split_idx.items()})
    for name, idx in split_idx.items():
        logger.info(f"Split [{name.upper()}]: {len(idx)} images from {len(set(groups[idx]))} patients")

    return {name: (paths[idx].tolist(), labels[idx].tolist()) for name, idx in split_idx.items()}


def assert_no_group_overlap(split_groups: dict):
    names = list(split_groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(split_groups[a]) & set(split_groups[b])
            assert not shared, (
                f"Patient leakage: {len(shared)} patients in both {a} and {b}, e.g. {sorted(shared)[:3]}"
            )
    logger.info("Verification passed: no patient appears in more than one split.")


def assert_no_file_overlap(splits: dict):
    """Same file (name + size) must not be copied into two splits."""
    def key(p):
        return os.path.basename(p) + "_" + str(os.path.getsize(p))

    keys = {name: set(map(key, imgs)) for name, (imgs, _) in splits.items()}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not keys[a] & keys[b], f"Data leakage detected: {a} and {b} share files!"
    logger.info("Verification passed: no file appears in more than one split.")
