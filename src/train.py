"""Train ConvNeXt on Stanford Cars, log to MLflow, save best weights."""
import copy
import json
import os
import subprocess
import time

import mlflow
import timm
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from common import build_train_val, get_device, load_params

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI",
                            "https://dagshub.com/nour-abdessatar/car-mlops.mlflow")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = load_params()
    t = p["train"]
    device = get_device(t["device"])
    torch.manual_seed(p["data"]["seed"])
    print("device:", device)

    train_set, val_set, classes = build_train_val(p)
    pin = device == "cuda"
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"], pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"], pin_memory=pin)

    model = timm.create_model(t["model"], pretrained=True, num_classes=len(classes),
                              drop_path_rate=t["drop_path"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
                 if t["cosine"] else None)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # traceability: code + data versions
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "unknown"
    try:
        with open("data.tar.dvc") as f:
            data_md5 = yaml.safe_load(f)["outs"][0]["md5"]
    except Exception:
        data_md5 = "unknown"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("convnext-stanford-cars")
    with mlflow.start_run(run_name="dvc-pipeline-train"):
        mlflow.log_params({**t, "train_size": len(train_set), "val_size": len(val_set),
                           "seed": p["data"]["seed"]})
        mlflow.set_tags({"git_commit": git_commit, "data_version": data_md5,
                         "data_file": "data.tar", "data_remote": "dagshub"})

        best_val, best_state = 0.0, None
        last = (0.0, 0.0, 0.0, 0.0)
        for epoch in range(1, t["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            if scheduler:
                scheduler.step()
            mlflow.log_metrics({"train_loss": tr_loss, "train_acc": tr_acc,
                                "val_loss": va_loss, "val_acc": va_acc}, step=epoch)
            if va_acc > best_val:
                best_val, best_state = va_acc, copy.deepcopy(model.state_dict())
            last = (tr_loss, tr_acc, va_loss, va_acc)
            print(f"epoch {epoch}/{t['epochs']} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s")

        mlflow.log_metric("best_val_acc", best_val)

        os.makedirs("models", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)
        torch.save(best_state, "models/best.pt")
        with open("models/classes.json", "w") as f:
            json.dump(classes, f)
        with open("metrics/train.json", "w") as f:
            json.dump({"best_val_acc": best_val, "final_train_loss": last[0],
                       "final_train_acc": last[1], "final_val_loss": last[2],
                       "final_val_acc": last[3]}, f, indent=2)
        mlflow.log_artifact("models/best.pt")
        mlflow.log_artifact("models/classes.json")


if __name__ == "__main__":
    main()
