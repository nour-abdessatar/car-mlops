import io
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
MODEL_NAME = os.getenv("MODEL_NAME", "car-classifier@production")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
state = {}


def load_session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1        # small host: keep CPU/memory use low
    opts.enable_cpu_mem_arena = False
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - MEAN) / STD)[None].astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


@asynccontextmanager
async def lifespan(app):
    state["session"] = load_session(MODEL_PATH)
    state["classes"] = json.load(open("classes.json"))
    yield


app = FastAPI(title="Car classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image")
    logits = state["session"].run(None, {"input": preprocess(img)})[0][0]
    probs = softmax(logits)
    top = np.argsort(probs)[::-1][:3]
    return {"predictions": [
        {"class": state["classes"][int(i)], "confidence": round(float(probs[i]), 4)}
        for i in top
    ]}