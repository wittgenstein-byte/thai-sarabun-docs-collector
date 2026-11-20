FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAIN_OCR_DATA_DIR=/app/data

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[publisher]"

ENTRYPOINT ["train-ocr"]
CMD ["build-saraban", "--help"]
