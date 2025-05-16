resource "aws_cloudwatch_log_group" "django-log-group" {
  name              = "/ecs/engagex_stream_app"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "celery-log-group" {
  name              = "/ecs/celery_worker"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "flower-log-group" {
  name              = "/ecs/flower"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_group" "nginx-log-group" {
  name              = "/ecs/nginx"
  retention_in_days = var.log_retention_in_days
}

resource "aws_cloudwatch_log_stream" "nginx-log-stream" {
  name           = "nginx-log-stream"
  log_group_name = aws_cloudwatch_log_group.nginx-log-group.name
}
