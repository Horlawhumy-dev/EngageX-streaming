#!/bin/bash

# Ensure it's run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Run as root" >&2
    exit 1
fi

# Log all output
exec > >(tee /var/log/user-data.log | logger -t user-data) 2>&1

echo "Updating packages..."
apt update -y && apt upgrade -y

echo "Installing Docker..."
apt install -y docker.io unzip curl jq

echo "Starting and enabling Docker..."
systemctl start docker
systemctl enable docker

echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | jq -r .tag_name)
curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

echo "Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/aws*

echo "Adding ubuntu to docker group..."
usermod -aG docker ubuntu

echo "Creating app directory..."
mkdir -p /home/ubuntu/app
chown -R ubuntu:ubuntu /home/ubuntu/app

echo "Creating docker volume..."
docker volume create app_tmp-data

# Fix permissions on docker volume data directory
echo "Fixing permissions on docker volume data directory..."
chown -R 1000:1000 /var/lib/docker/volumes/app_tmp-data/_data || echo "Warning: Could not change ownership of docker volume data"

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=us-west-1

# Write deploy script for ubuntu user
cat > /home/ubuntu/app/deploy.sh <<'EOSCRIPT'
#!/bin/bash

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=us-west-1

=======
# Switch to ubuntu user to set up app
sudo -i -u ubuntu bash <<'EOF'
cd ~/app

echo "Creating .env file..."
cat > .env <<EOL
OPENAI_API_KEY=
POSTGRESQL_DATABASE_NAME=
POSTGRESQL_USERNAME=
POSTGRESQL_PASSWORD=
POSTGRESQL_SERVER_NAME=
DEEPGRAM_API_KEY=
EMAIL_USE_TLS=True
EMAIL_HOST_USER=
EMAIL_HOST_PASSWORD=
DEFAULT_FROM_EMAIL=
ALLOWED_HOSTS="localhost,*"
USE_S3=True
AWS_STORAGE_BUCKET_NAME=
AWS_S3_REGION_NAME=us-west-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AUTH_TOKEN_FOR_WEBSOCKET=
STRIPE_SECRET_KEY=
STRIPE_PUBLISHABLE_KEY=
STRIPE_WEBHOOK_SECRET=
INTUIT_VERIFIER_TOKEN=
INTUIT_CLIENT_ID=
INTUIT_CLIENT_SECRET=
NEW_INTUIT_REDIRECT_URI=
INTUIT_ENVIRONMENT=production
VITE_ENGAGEX_PASS=
REDIS_URL=
EOL


# Write your docker-compose.yml file
cat << 'EOF' > /app/docker-compose.yml
version: '3.8'

services:
  # Django Instances
  django:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8000 EngageX_Streaming.asgi:application"
    environment:
      - APP_PORT=8000
    ports:
      - "8000:8000"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-2:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8001
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8001 EngageX_Streaming.asgi:application"
    ports:
      - "8001:8001"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-3:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8002
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8002 EngageX_Streaming.asgi:application"
    ports:
      - "8002:8002"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-4:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8003
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8003 EngageX_Streaming.asgi:application"
    ports:
      - "8003:8003"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-5:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8004
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8004 EngageX_Streaming.asgi:application"
    ports:
      - "8004:8004"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-6:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8005
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8005 EngageX_Streaming.asgi:application"
    ports:
      - "8005:8005"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-7:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8006
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8006 EngageX_Streaming.asgi:application"
    ports:
      - "8006:8006"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-8:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8007
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8007 EngageX_Streaming.asgi:application"
    ports:
      - "8007:8007"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-9:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8008
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8008 EngageX_Streaming.asgi:application"
    ports:
      - "8008:8008"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-10:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8009
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8009 EngageX_Streaming.asgi:application"
    ports:
      - "8009:8009"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-11:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8010
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8010 EngageX_Streaming.asgi:application"
    ports:
      - "8010:8010"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-12:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8011
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8011 EngageX_Streaming.asgi:application"
    ports:
      - "8011:8011"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-13:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8012
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8012 EngageX_Streaming.asgi:application"
    ports:
      - "8012:8012"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-14:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8013
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8013 EngageX_Streaming.asgi:application"
    ports:
      - "8013:8013"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  django-15:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    environment:
      - APP_PORT=8014
    command: /bin/sh -c "python manage.py migrate && daphne -b 0.0.0.0 -p 8014 EngageX_Streaming.asgi:application"
    ports:
      - "8014:8014"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env

  # Celery Workers
  celery:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-2:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-3:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-4:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-5:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-6:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-7:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-8:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-9:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
  celery-10:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming worker --loglevel=info --pool=prefork
    healthcheck:
      test: ["CMD", "./check-celery.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env
      
  celery-flower:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    command: celery -A EngageX_Streaming flower --loglevel=info
    ports:
      - "5555:5555"
    env_file:
      - .env

volumes:
  tmp-data:
EOF

echo "Logging into AWS ECR..."
aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 266735827053.dkr.ecr.us-west-1.amazonaws.com

echo "Pulling latest Docker images..."
docker-compose pull

echo "Starting containers..."
docker-compose up -d --remove-orphans
EOF

echo "Startup completed successfully!"