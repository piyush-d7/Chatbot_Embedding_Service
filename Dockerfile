# Multi-stage Dockerfile for smaller, optimized embedding service
FROM python:3.11-slim as builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies
COPY embedding_requirements.txt .
RUN pip install --user --no-cache-dir --no-warn-script-location \
    -r embedding_requirements.txt

# Pre-download the model in the builder stage
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Final lightweight runtime stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

WORKDIR /app

# Copy installed packages and models from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY embedding_main.py .

# Update PATH to include user packages
ENV PATH=/root/.local/bin:$PATH

# Set Python optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security (optional)
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy app files to user directory
COPY --chown=app:app embedding_main.py .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "embedding_main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]