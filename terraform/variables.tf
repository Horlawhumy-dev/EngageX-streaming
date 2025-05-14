variable "aws_region" {
  description = "The AWS region to deploy resources"
  type        = string
  default     = "eu-north-1"  
}

variable "postgresql_database_name" {}
variable "postgresql_username" {}
variable "postgresql_server_name" {}
variable "postgresql_password" {}
variable "openai_api_key" {}
variable "deepgram_api_key" {}
variable "email_host" {}
variable "email_host_user" {}
variable "email_host_password" {}
variable "default_from_email" {}
variable "aws_storage_bucket_name" {}
variable "aws_s3_region_name" {}
variable "aws_access_key_id" {}
variable "aws_secret_access_key" {}
variable "intuit_verifier_token" {}
variable "intuit_client_id" {}
variable "intuit_client_secret" {}
variable "new_intuit_redirect_uri" {}
variable "intuit_environment" {}
variable "stripe_secret_key" {}
variable "stripe_publishable_key" {}
variable "stripe_webhook_secret" {}
variable "redis_url" {
  default = "redis://redis:6379/0"
}
