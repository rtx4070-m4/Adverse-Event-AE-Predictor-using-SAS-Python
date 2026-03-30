# ============================================================
# Adverse Event Risk Predictor - Dockerfile
# Multi-stage build for production deployment
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY python/     ./python/
COPY api/        ./api/
COPY dashboard/  ./dashboard/
COPY models/     ./models/
COPY data/processed/ ./data/processed/

# Create non-root user for security
RUN useradd -m -u 1001 clinicai && \
    chown -R clinicai:clinicai /app

USER clinicai

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV API_URL=http://localhost:8000

# Expose ports
EXPOSE 8000 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
