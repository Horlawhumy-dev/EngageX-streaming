variable "AWS_REGION" {
  description = "The AWS region to deploy resources"
  type        = string
  default     = "us-west-1"  
}

variable "POSTGRESQL_DATABASE_NAME" {}
variable "POSTGRESQL_USERNAME" {}
variable "POSTGRESQL_SERVER_NAME" {}
variable "POSTGRESQL_PASSWORD" {}
variable "OPENAI_API_KEY" {}
variable "DEEPGRAM_API_KEY" {}
variable "EMAIL_HOST" {}
variable "EMAIL_HOST_USER" {}
variable "EMAIL_HOST_PASSWORD" {}
variable "DEFAULT_FROM_EMAIL" {}
variable "AWS_STORAGE_BUCKET_NAME" {}
variable "AWS_S3_REGION_NAME" {}
variable "AWS_ACCESS_KEY_ID" {}
variable "AWS_SECRET_ACCESS_KEY" {}
variable "INTUIT_VERIFIER_TOKEN" {}
variable "INTUIT_CLIENT_ID" {}
variable "INTUIT_CLIENT_SECRET" {}
variable "NEW_INTUIT_REDIRECT_URI" {}
variable "INTUIT_ENVIRONMENT" {}
variable "STRIPE_SECRET_KEY" {}
variable "STRIPE_PUBLISHABLE_KEY" {}
variable "STRIPE_WEBHOOK_SECRET" {}
variable "REDIS_URL" {
  default = "redis://redis:6379/0"
}