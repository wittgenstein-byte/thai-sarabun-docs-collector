FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRAIN_OCR_DATA_DIR=/app/data \
    DISPLAY=:99

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    chromium \
    chromium-driver \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    && ln -sf /usr/bin/chromium /usr/bin/google-chrome \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY docker/google-entrypoint.sh /usr/local/bin/google-entrypoint.sh
RUN chmod +x /usr/local/bin/google-entrypoint.sh

RUN pip install --no-cache-dir ".[google]"

EXPOSE 7900

ENTRYPOINT ["/usr/local/bin/google-entrypoint.sh"]
CMD ["train-ocr", "collect-google", "--help"]
