"""Extract the dataset and build the stratified train/val split."""
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets

from common import DATA_ROOT, SPLIT_PATH, ensure_data, load_params

p = load_params()["data"]
ensure_data(p["archive"])

ds = datasets.StanfordCars(DATA_ROOT, split="train", download=False)
labels = [t for _, t in ds._samples]
idx = np.arange(len(labels))
train_idx, val_idx = train_test_split(
    idx, test_size=p["val_fraction"], stratify=labels, random_state=p["seed"]
)

os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
with open(SPLIT_PATH, "w") as f:
    json.dump({
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "classes": ds.classes,
    }, f)

print(f"train: {len(train_idx)} | val: {len(val_idx)} | classes: {len(ds.classes)}")
