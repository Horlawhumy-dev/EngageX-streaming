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
ALLOWED_HOSTS=localhost,*
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

echo "Logging into AWS ECR…"
aws ecr get-login-password --region us-west-1 docker login --username AWS --password-stdin 266735827053.dkr.ecr.us-west-1.amazonaws.com

echo "Generating docker-compose.yml…"
cat > docker-compose.yml <<EOF
version: '3.8'

services:
EOF

# Add 5 Django services
for i in $(seq 0 4); do
  PORT=$((9000 + i))
  cat >> docker-compose.yml <<EOF
  django-$((i+1)):
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:$PORT/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    environment:
      - APP_PORT=$PORT
    command: >
      /bin/sh -c "python manage.py migrate &&
        daphne -b 0.0.0.0 -p $PORT EngageX_Streaming.asgi:application"
    ports:
      - "$PORT:$PORT"
    volumes:
      - tmp-data:/tmp
    env_file:
      - .env

EOF
done

# Add 3 Celery workers
for i in $(seq 1 3); do
  cat >> docker-compose.yml <<EOF
  celery-$i:
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

EOF
done

# Add Flower service
cat >> docker-compose.yml <<EOF
  flower:
    image: 266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5556"]
      interval: 30s
      timeout: 10s
      retries: 3
    command: celery -A EngageX_Streaming flower --port=5556
    ports:
      - "5556:5556"
    env_file:
      - .env

volumes:
  tmp-data:
EOF

echo "Pulling images..."
docker-compose pull

aws ecr get-login-password --region us-west-1 | \
docker login --username AWS --password-stdin 266735827053.dkr.ecr.us-west-1.amazonaws.com

echo "Starting services with retries..."
max_retries=2
count=0
until docker-compose up -d --remove-orphans; do
  count=$((count + 1))
  if [ $count -ge $max_retries ]; then
    echo "docker-compose up failed after $count attempts."
    exit 1
  fi
  echo "docker-compose up failed. Retrying in 10 seconds... Attempt #$count"
  sleep 10
done

echo "Startup completed successfully!"
EOSCRIPT

chmod +x /home/ubuntu/app/deploy.sh
chown ubuntu:ubuntu /home/ubuntu/app/deploy.sh

echo "Running deployment script as ubuntu..."
sudo -u ubuntu /home/ubuntu/app/deploy.sh