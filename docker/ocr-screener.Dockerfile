FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAIN_OCR_DATA_DIR=/app/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[ocr]"

EXPOSE 8001

CMD ["uvicorn", "train_ocr.screener.api:app", "--host", "0.0.0.0", "--port", "8001"]
