FROM python:3.11-slim

# Create and switch to app directory
WORKDIR /app

# Install system dependencies (optional but common)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Ensure entrypoint script is executable
RUN chmod +x /app/docker-entrypoint.sh

# Environment variables for Django
ENV DJANGO_SETTINGS_MODULE=EngageX_Streaming.settings
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED=1

# Expose port for Django
EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]