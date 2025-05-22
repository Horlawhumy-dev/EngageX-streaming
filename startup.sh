#!/bin/bash

# This script is intended to be run on an Ubuntu server instance
# It installs Docker, Docker Compose, and other necessary packages
# and sets up the environment for a web application.
# It also logs all output to a file for debugging purposes.
# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" 1>&2
    exit 1
fi
# Log all output for debugging
exec > >(tee /var/log/user-data.log | logger -t user-data) 2>&1

# Update and upgrade all packages
apt update -y && apt upgrade -y

# Install Docker
apt install -y docker.io
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group (replace 'ubuntu' if using another user)
usermod -aG docker ubuntu

# Install Docker Compose (latest stable version)
DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)
curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python3, pip3 already installed with ubuntu minimal, but ensure latest
apt install -y python3 python3-pip
pip3 install --upgrade pip setuptools wheel

# Install Redis client only
apt install -y redis-tools

# Install PostgreSQL client libraries
apt install -y postgresql-client libpq-dev

# Install build tools
apt install -y build-essential cmake

# Install ffmpeg
apt install -y ffmpeg

# Install LibreOffice
apt install -y libreoffice

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
unzip /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

# add their vlaues to the environment variables
export OPENAI_API_KEY=
export POSTGRESQL_DATABASE_NAME=
export POSTGRESQL_USERNAME=
export POSTGRESQL_PASSWORD=
export POSTGRESQL_SERVER_NAME=
export DEEPGRAM_API_KEY=
export EMAIL_USE_TLS=True
export EMAIL_HOST_USER=
export EMAIL_HOST_PASSWORD=
export DEFAULT_FROM_EMAIL=
export ALLOWED_HOSTS="localhost,*"
export USE_S3=True
export AWS_STORAGE_BUCKET_NAME=
export AWS_S3_REGION_NAME=us-west-1
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AUTH_TOKEN_FOR_WEBSOCKET=
export STRIPE_SECRET_KEY=
export STRIPE_PUBLISHABLE_KEY=
export STRIPE_WEBHOOK_SECRET=
export INTUIT_VERIFIER_TOKEN=
export INTUIT_CLIENT_ID=
export INTUIT_CLIENT_SECRET=
export NEW_INTUIT_REDIRECT_URI=
export INTUIT_ENVIRONMENT=production
export VITE_ENGAGEX_PASS=
export REDIS_URL=

# Login to ECR
aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin 266735827053.dkr.ecr.us-west-1.amazonaws.com

# Set up application directory
mkdir -p /home/ubuntu/app
cd /home/ubuntu/app

# the repository is already cloned or add your ssh key to github if its new server and clone the repo
#so we just need to pull the latest changes 
git pull origin main
# Start the app
docker-compose pull
docker-compose up -d --remove-orphans

echo "Startup completed successfully!"
# Clean up apt cache
apt clean