# AWS Region
variable "aws_region" {
  description = "The AWS region"
  type        = string
  default     = "us-west-1"
}

# VPC ID
variable "vpc_id" {
  description = "The VPC ID"
  type        = string
}

variable "ecr_repository_name" {
  description = "The name of the ECR repository"
  type        = string
  
}

# Docker Image Tag
variable "image_tag" {
  description = "The tag for the Docker image"
  type        = string
  default     = "latest"
}

# ECR Repository Name
variable "ecr_repository" {
  description = "ECR repository for storing Docker images"
  type        = string
}

# PostgreSQL Configuration
variable "postgresql_database_name" {
  description = "PostgreSQL database name"
  type        = string
}

variable "postgresql_username" {
  description = "PostgreSQL username"
  type        = string
}

variable "postgresql_password" {
  description = "PostgreSQL password"
  type        = string
}

variable "postgresql_server_name" {
  description = "PostgreSQL server name"
  type        = string
}

# Redis URL
variable "redis_url" {
  description = "The Redis URL"
  type        = string
}

# OpenAI API Key
variable "openai_api_key" {
  description = "The OpenAI API key"
  type        = string
}

# Deepgram API Key
variable "deepgram_api_key" {
  description = "The Deepgram API key"
  type        = string
}

# AWS SES Configuration
variable "email_host" {
  description = "AWS SES email host"
  type        = string
}

variable "email_port" {
  description = "AWS SES email port"
  type        = number
  default     = 587
}

variable "email_use_tls" {
  description = "Enable AWS SES TLS"
  type        = bool
  default     = true
}

variable "email_host_user" {
  description = "AWS SES email host user"
  type        = string
}

variable "email_host_password" {
  description = "AWS SES email host password"
  type        = string
}

variable "default_from_email" {
  description = "The default email from address"
  type        = string
}

# AWS S3 Configuration
variable "use_s3" {
  description = "Whether to use AWS S3 for storage"
  type        = bool
  default     = true
}

variable "aws_storage_bucket_name" {
  description = "AWS S3 storage bucket name"
  type        = string
}

variable "aws_s3_region_name" {
  description = "The AWS S3 region"
  type        = string
}

variable "aws_access_key_id" {
  description = "AWS access key ID"
  type        = string
}

variable "aws_secret_access_key" {
  description = "AWS secret access key"
  type        = string
}

# Intuit API Configuration
variable "intuit_verifier_token" {
  description = "Intuit verifier token"
  type        = string
}

variable "intuit_client_id" {
  description = "Intuit client ID"
  type        = string
}

variable "intuit_client_secret" {
  description = "Intuit client secret"
  type        = string
}

variable "new_intuit_redirect_uri" {
  description = "New Intuit redirect URI"
  type        = string
}

variable "intuit_environment" {
  description = "The Intuit environment (e.g., production)"
  type        = string
}

# Stripe API Configuration
variable "stripe_secret_key" {
  description = "Stripe secret key"
  type        = string
}

variable "stripe_publishable_key" {
  description = "Stripe publishable key"
  type        = string
}

variable "stripe_webhook_secret" {
  description = "Stripe webhook secret"
  type        = string
}

# WebSocket Authentication Token
variable "auth_token_for_websocket" {
  description = "Authentication token for WebSocket connections"
  type        = string
}

# Port for PostgreSQL
variable "port" {
  description = "Port number for PostgreSQL"
  type        = number
  default     = 5432
}
