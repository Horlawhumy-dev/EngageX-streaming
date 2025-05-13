provider "aws" {
  region = var.aws_region
}

resource "aws_ecs_cluster" "main" {
  name = "celery-flower-cluster"
}

resource "aws_ecr_repository" "celery_app" {
  name = "celery-app-repo"
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecsTaskExecutionRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action    = "sts:AssumeRole"
        Effect    = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_ecs_task_definition" "celery_flower_task" {
  family                   = "celery-flower-task"
  task_role_arn            = aws_iam_role.ecs_task_execution_role.arn
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "1024"
  requires_compatibilities = ["FARGATE"]
  
  container_definitions = jsonencode([
    {
      name      = "celery"
      image     = "${aws_ecr_repository.celery_app.repository_url}:${var.image_tag}"
      essential = true
      environment = [
        { name = "REDIS_URL", value = var.redis_url }
      ]
      command = [
        "celery", "-A", "EngageX_Streaming", "worker", "--loglevel=info", "--pool=threads", "--broker=${var.redis_url}"
      ]
    },
    {
      name      = "flower"
      image     = "${aws_ecr_repository.celery_app.repository_url}:${var.image_tag}"
      essential = true
      environment = [
        { name = "REDIS_URL", value = var.redis_url }
      ]
      command = [
        "celery", "-A", "EngageX_Streaming", "flower", "--port=5555", "--broker=${var.redis_url}"
      ]
    }
  ])
}