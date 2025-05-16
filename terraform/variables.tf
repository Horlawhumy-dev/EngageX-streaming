# core
variable "region" {
  description = "The AWS region to create resources in."
  default     = "us-west-1"
}

# networking
variable "public_subnet_1_cidr" {
  description = "CIDR Block for Public Subnet 1"
  default     = "10.0.1.0/24"
}
variable "public_subnet_2_cidr" {
  description = "CIDR Block for Public Subnet 2"
  default     = "10.0.2.0/24"
}
variable "private_subnet_1_cidr" {
  description = "CIDR Block for Private Subnet 1"
  default     = "10.0.3.0/24"
}
variable "private_subnet_2_cidr" {
  description = "CIDR Block for Private Subnet 2"
  default     = "10.0.4.0/24"
}
variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-1b", "us-west-1c"]
}

# load balancer
variable "health_check_path" {
  description = "Health check path for the default target group"
  default     = "/ping/"
}

# ECS cluster
variable "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  default     = "production"
}

# Docker images
variable "docker_image_url_django" {
  description = "Docker image for Django app in ECS"
  default     = "266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest"
}
variable "docker_image_url_nginx" {
  description = "Docker image for Nginx in ECS"
  default     = "266735827053.dkr.ecr.us-west-1.amazonaws.com/nginx:latest"
}

# Task scaling and sizing
variable "app_count" {
  description = "Number of Docker containers to run"
  default     = 1
}

variable "fargate_cpu" {
  description = "Amount of CPU units for Fargate task (e.g., 256 = 0.25 vCPU)"
  default     = "8192"
}

variable "fargate_memory" {
  description = "Amount of memory in MiB for Fargate task (e.g., 512 = 0.5GB)"
  default     = "16386"
}

variable "allowed_hosts" {
  description = "Comma-separated list of allowed hosts"
  default     = "localhost,*"
}

# Logging
variable "log_retention_in_days" {
  description = "CloudWatch Logs retention in days"
  default     = 30
}

# ECS Service Auto Scaling
variable "autoscale_min" {
  description = "Minimum number of ECS tasks"
  default     = 1
}
variable "autoscale_max" {
  description = "Maximum number of ECS tasks"
  default     = 4
}
variable "autoscale_desired" {
  description = "Desired number of ECS tasks at start"
  default     = 1
}

# # RDS database
# variable "rds_db_name" {
#   description = "RDS database name"
#   default     = "mydb"
# }
# variable "rds_username" {
#   description = "RDS database username"
#   default     = "foo"
# }
# variable "rds_password" {
#   description = "RDS database password"
#   sensitive   = true
# }
# variable "rds_instance_class" {
#   description = "RDS instance type"
#   default     = "db.t3.micro"
# }

# Domain & SSL
variable "certificate_arn" {
  description = "AWS Certificate Manager ARN for validated domain"
  default     = "YOUR ARN"
}

# Environment variables that correspond to Docker ENV variables
variable "postgresql_database_name" {
  description = "PostgreSQL database name"
  default     = ""
}

variable "postgresql_username" {
  description = "PostgreSQL username"
  default     = ""
}

variable "postgresql_password" {
  description = "PostgreSQL password"
  sensitive   = true
}

variable "postgresql_server_name" {
  description = "PostgreSQL hostname"
  default     = ""
}

variable "port" {
  description = "Port number to expose"
  default     = "5432"
}

variable "openai_api_key" {
  description = "OpenAI API key"
  default     = ""
  sensitive   = true
}

variable "deepgram_api_key" {
  description = "Deepgram API key"
  default     = ""
  sensitive   = true
}

variable "email_host" {
  description = "Email SMTP host"
  default     = ""
}

variable "email_port" {
  description = "Email SMTP port"
  default     = ""
}

variable "email_use_tls" {
  description = "Email use TLS (true/false)"
  default     = ""
}

variable "email_host_user" {
  description = "Email SMTP user"
  default     = ""
}

variable "email_host_password" {
  description = "Email SMTP password"
  sensitive   = true
}

variable "default_from_email" {
  description = "Default From email address"
  default     = ""
}

variable "use_s3" {
  description = "Use AWS S3 for storage (true/false)"
  default     = "True"
}

variable "aws_storage_bucket_name" {
  description = "AWS S3 bucket name"
  default     = ""
}

variable "aws_s3_region_name" {
  description = "AWS S3 region"
  default     = ""
}

variable "aws_access_key_id" {
  description = "AWS Access Key ID"
  default     = ""
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS Secret Access Key"
  default     = ""
  sensitive   = true
}

variable "auth_token_for_websocket" {
  description = "Auth token for WebSocket"
  default     = ""
}

variable "django_settings_module" {
  description = "Django settings module"
  default     = "EngageX_Streaming.settings"
}

variable "redis_url" {
  description = "Redis connection URL"
  default     = ""
}

variable "intuit_verifier_token" {
  description = "Intuit verifier token"
  default     = ""
}

variable "intuit_client_id" {
  description = "Intuit client ID"
  default     = ""
}

variable "intuit_client_secret" {
  description = "Intuit client secret"
  default     = ""
}

variable "new_intuit_redirect_uri" {
  description = "New Intuit redirect URI"
  default     = ""
}

variable "intuit_environment" {
  description = "Intuit environment (sandbox/production)"
  default     = "production"
}

variable "stripe_secret_key" {
  description = "Stripe secret key"
  default     = ""
  sensitive   = true
}

variable "stripe_publishable_key" {
  description = "Stripe publishable key"
  default     = ""
}

variable "stripe_webhook_secret" {
  description = "Stripe webhook secret"
  default     = ""
  sensitive   = true
}