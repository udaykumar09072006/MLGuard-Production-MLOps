# ==============================================================================
# MLGuard Multi-Stage Dockerfile (Production & Local Development)
# ==============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create runtime directories
RUN mkdir -p /app/data/uploads /app/artifacts/models /app/artifacts/reference /app/data/sample

# Create unprivileged user for security
RUN useradd -m -u 1000 mlguard && \
    chown -R mlguard:mlguard /app

USER mlguard

EXPOSE 8000 8501

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
