# # Output the name of the ECS Cluster
# output "ecs_cluster_name" {
#   description = "The name of the ECS cluster"
#   value       = aws_ecs_cluster.main.name
# }

# # Output the name of the ECS Service for the Django app
# output "ecs_service_name" {
#   description = "The name of the ECS service running the Django app"
#   value       = aws_ecs_service.django_service.name
# }

# # Output the name of the ECS Service for the Celery worker
# output "ecs_celery_service_name" {
#   description = "The name of the ECS service running the Celery worker"
#   value       = aws_ecs_service.celery_service.name
# }

# # Output the name of the ECS Service for the Flower monitoring tool
# output "ecs_flower_service_name" {
#   description = "The name of the ECS service running the Flower monitoring tool"
#   value       = aws_ecs_service.flower_service.name
# }

# # Output the URL for the ECR repository where the Docker image is stored
# output "ecr_repository_url" {
#   description = "The URL of the ECR repository"
#   value       = aws_ecr_repository.celery_app.repository_url
# }

# # Output the ECS task definition ARN for the Celery & Flower tasks
# output "celery_flower_task_definition_arn" {
#   description = "The ARN of the ECS task definition for Celery and Flower"
#   value       = aws_ecs_task_definition.celery_flower_task.arn
# }

# # Output the S3 Bucket Name
# output "aws_storage_bucket_name" {
#   description = "The name of the AWS S3 storage bucket"
#   value       = var.aws_storage_bucket_name
# }

# # Output the S3 Region Name
# output "aws_s3_region_name" {
#   description = "The AWS S3 region"
#   value       = var.aws_s3_region_name
# }

# # Output the URL of the Redis service
# output "redis_url" {
#   description = "The URL of the Redis service"
#   value       = var.redis_url
# }

# # Output the email host used for AWS SES
# output "email_host" {
#   description = "The email host used for AWS SES"
#   value       = var.email_host
# }

# Output the public DNS name of the Application Load Balancer
output "load_balancer_dns_name" {
  description = "The public DNS name of the ECS Application Load Balancer"
  value       = aws_lb.main.dns_name
}

# Optional: Link to Flower dashboard (assuming port 5555 is exposed in ALB target group)
output "flower_dashboard_url" {
  description = "URL to access the Celery Flower dashboard"
  value       = "http://${aws_lb.main.dns_name}:5555"
}