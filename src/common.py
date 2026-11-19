import json
import os
import tarfile

import torch
import yaml
from torch.utils.data import Subset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats
DATA_ROOT = "data"
SPLIT_PATH = "artifacts/split.json"


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_data(archive):
    """Extract the DVC-tracked archive if data/ is not there yet."""
    if not os.path.isdir(DATA_ROOT):
        print(f"extracting {archive} ...")
        with tarfile.open(archive) as t:
            t.extractall(".")


def get_device(pref="auto"):
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def eval_tfm(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_tfm(size, mode):
    if mode == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_train_val(p):
    t = p["train"]
    ensure_data(p["data"]["archive"])
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    tr_idx, va_idx = split["train_idx"], split["val_idx"]
    if t.get("subset"):
        n = int(t["subset"])
        tr_idx, va_idx = tr_idx[:n], va_idx[: max(1, n // 5)]
    aug = datasets.StanfordCars(DATA_ROOT, split="train", download=False,
                                transform=train_tfm(t["image_size"], t["augmentation"]))
    clean = datasets.StanfordCars(DATA_ROOT, split="train", download=False,
                                  transform=eval_tfm(t["image_size"]))
    return Subset(aug, tr_idx), Subset(clean, va_idx), split["classes"]


def build_test(p):
    t = p["train"]
    ensure_data(p["data"]["archive"])
    test = datasets.StanfordCars(DATA_ROOT, split="test", download=False,
                                 transform=eval_tfm(t["image_size"]))
    if t.get("subset"):
        test = Subset(test, list(range(int(t["subset"]))))
    return test
