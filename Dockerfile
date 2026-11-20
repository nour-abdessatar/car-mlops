FROM python:3.13-slim
WORKDIR /code
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY app/ app/
COPY classes.json .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]