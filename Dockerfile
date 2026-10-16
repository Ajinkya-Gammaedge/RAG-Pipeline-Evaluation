FROM python:3.12-slim

WORKDIR /app

# Production Python environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HF_HOME=/home/appuser/.cache/huggingface

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and set up model cache directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

# Install python dependencies first to maximize layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]