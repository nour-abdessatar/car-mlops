import io, json, os
from contextlib import asynccontextmanager

import mlflow.pytorch, torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

MODEL_URI = os.getenv("MODEL_URI", "models:/car-classifier@production")
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
state = {}

@asynccontextmanager
async def lifespan(app):
    state["model"] = mlflow.pytorch.load_model(MODEL_URI, map_location="cpu").eval()
    state["classes"] = json.load(open("classes.json"))
    yield

app = FastAPI(title="Car classifier", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Not a valid image")
    with torch.no_grad():
        probs = state["model"](tfm(img).unsqueeze(0)).softmax(1)[0]
    top = probs.topk(3)
    return {"predictions": [
        {"class": state["classes"][i], "confidence": round(p, 4)}
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]}