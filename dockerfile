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

# Collect static files (if you use Django staticfiles)
# RUN python manage.py collectstatic --noinput

# Expose port 8000 by default
EXPOSE 8000

# CMD to run Daphne server for ASGI Django app
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "EngageX_Streaming.asgi:application"]

# Optional health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1