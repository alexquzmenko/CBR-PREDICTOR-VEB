FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    MODEL_CONFIG_PATH=configs/modeling.yaml

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

EXPOSE 8080

# Yandex Serverless Container injects PORT at runtime; do not set PORT in revision env.
CMD uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
