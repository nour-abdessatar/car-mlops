FROM python:3.11-slim
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app/ app/
COPY classes.json .
# ~110 MB model, hosted as a GitHub Release asset (too big for a normal git file)
ADD https://github.com/nour-abdessatar/car-mlops/releases/download/v1/model.onnx model.onnx

ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]