# Use multi-stage build for optimized image size
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Create final image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# Copy installed packages and source code from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /home/appuser/app

# Create necessary directories with correct permissions
RUN mkdir -p /home/appuser/data \
    mkdir -p /home/appuser/models \
    mkdir -p /home/appuser/results \
    mkdir -p /home/appuser/logs

# Set working directory to app
WORKDIR /home/appuser/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "main.py"]

# Documentation for the image
LABEL maintainer="Indserf Team <contact@indserf.com>" \
      version="1.0.0" \
      description="Unsupervised Trading Pattern Analysis" \
      org.opencontainers.image.source="https://github.com/indserf/unsupervised-trading"

# Expose ports (if needed)
EXPOSE 8000

# Volume configuration for persistent data
VOLUME ["/home/appuser/data", "/home/appuser/models", "/home/appuser/results", "/home/appuser/logs"]

# Environment variable defaults
ENV MODEL_DIR=/home/appuser/models \
    DATA_DIR=/home/appuser/data \
    RESULTS_DIR=/home/appuser/results \
    LOG_DIR=/home/appuser/logs \
    CONFIG_PATH=/home/appuser/app/config.json \
    CUDA_VISIBLE_DEVICES="" \
    LOG_LEVEL=INFO

# Add additional scripts
COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["docker-entrypoint.sh"]
