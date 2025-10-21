# Face Pipeline Dockerfile
FROM python:3.11-slim

# System deps for opencv + insightface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libopencv-core-dev libopencv-imgproc-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INSIGHTFACE_HOME=/models/insightface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create model cache dir
RUN mkdir -p /models/insightface

# Warm model cache - download and cache models during build
# This pre-downloads InsightFace models to /models/insightface
# so subsequent container runs are instant (no download time)
RUN python - <<'PY'
from pipeline.detector import load_detector
from pipeline.embedder import load_model
load_detector(); load_model()
print("Models warmed")
PY

# Healthcheck (optional)
HEALTHCHECK CMD python -c "import sys; from pipeline.embedder import load_model; load_model(); sys.exit(0)"

CMD ["python", "main.py"]

