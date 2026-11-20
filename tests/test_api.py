import io, json

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main


class Dummy(nn.Module):
    def forward(self, x):
        out = torch.zeros(x.shape[0], 196)
        out[:, 0] = 5.0
        return out


def test_health_and_predict(monkeypatch):
    monkeypatch.setattr(main.mlflow.pytorch, "load_model", lambda *a, **k: Dummy())
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200

        buf = io.BytesIO()
        Image.new("RGB", (300, 200), "red").save(buf, format="PNG")
        r = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 3
        assert preds[0]["class"] == json.load(open("classes.json"))[0]

        r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
        assert r.status_code == 400