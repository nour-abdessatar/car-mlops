"""One-off: export the production model from the MLflow registry to ONNX.
Run where torch, timm and mlflow are installed (e.g. Colab), with DagsHub credentials set.
"""
import os

import mlflow
import mlflow.pytorch
import numpy as np
import onnxruntime as ort
import torch

URI = os.getenv("MODEL_URI", "models:/car-classifier@production")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",
                        "https://dagshub.com/nour-abdessatar/car-mlops.mlflow"))

model = mlflow.pytorch.load_model(URI, map_location="cpu").eval()
dummy = torch.randn(1, 3, 224, 224)
kw = dict(input_names=["input"], output_names=["logits"],
          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
          opset_version=17)
try:
    torch.onnx.export(model, dummy, "model.onnx", dynamo=False, **kw)
except TypeError:  # older torch has no `dynamo` argument
    torch.onnx.export(model, dummy, "model.onnx", **kw)

# parity check: ONNX must match PyTorch
x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    ref = model(x).numpy()
out = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"]) \
         .run(None, {"input": x.numpy()})[0]
print("max abs diff:", float(np.abs(ref - out).max()))
print("argmax equal:", bool((ref.argmax(1) == out.argmax(1)).all()))
print("size MB:", round(os.path.getsize("model.onnx") / 1e6, 1))