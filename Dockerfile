FROM python:3.11-slim

# System settings
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and switch to app directory
WORKDIR /app

# Install system dependencies (optional but common)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Default command (can be overridden in ECS task definition)
CMD ["gunicorn", "app.EngageX_Streaming.wsgi:application", "--bind", "0.0.0.0:8000"]