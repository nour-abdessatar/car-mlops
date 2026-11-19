"""Evaluate the trained model on the official test set."""
import json
import os

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import build_test, get_device, load_params
from train import run_epoch


def main():
    p = load_params()
    t = p["train"]
    device = get_device(t["device"])

    with open("models/classes.json") as f:
        classes = json.load(f)

    test_set = build_test(p)
    loader = DataLoader(test_set, batch_size=t["batch_size"], shuffle=False,
                        num_workers=t["num_workers"], pin_memory=(device == "cuda"))

    model = timm.create_model(t["model"], pretrained=False, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("models/best.pt", map_location=device))

    loss, acc = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    print(f"TEST | loss {loss:.3f} | acc {acc:.4f}")

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval.json", "w") as f:
        json.dump({"test_acc": acc, "test_loss": loss, "test_size": len(test_set)}, f, indent=2)


if __name__ == "__main__":
    main()
