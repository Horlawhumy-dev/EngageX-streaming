# Use lightweight Python base image
FROM python:3.11-slim

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    ffmpeg \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies first (cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN chmod +x /app/check-celery.sh

# Expose port 8000 by default
ENV APP_PORT=8000
CMD ["sh", "-c", "daphne -b 0.0.0.0 -p $APP_PORT EngageX_Streaming.asgi:application"]

# Optional health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:${APP_PORT}/health || exit 1