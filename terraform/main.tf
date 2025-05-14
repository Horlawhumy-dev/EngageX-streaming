terraform {
  backend "s3" {
    bucket         = "engagex-user-content-1234"
    key            = "terraform.tfstate"
    region         = "us-west-1"
    encrypt        = true
  }
}

provider "aws" {
  region = "us-west-1"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
}

resource "aws_subnet" "subnet_a" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-west-1a"
}

resource "aws_subnet" "subnet_b" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "us-west-1b"
}

resource "aws_security_group" "ecs_sg" {
  name        = "ecs-sg"
  description = "Allow inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5555
    to_port     = 5555
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "main" {
  name               = "django-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs_sg.id]
  subnets            = [aws_subnet.subnet_a.id, aws_subnet.subnet_b.id]
}

resource "aws_lb_target_group" "django_tg" {
  name     = "django-tg"
  port     = 80
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
}

resource "aws_lb_listener" "django_listener" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.django_tg.arn
  }
}

# Create Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

# Create Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

# Associate Route Table with Subnet A
resource "aws_route_table_association" "subnet_a" {
  subnet_id      = aws_subnet.subnet_a.id
  route_table_id = aws_route_table.public.id
}

# Associate Route Table with Subnet B
resource "aws_route_table_association" "subnet_b" {
  subnet_id      = aws_subnet.subnet_b.id
  route_table_id = aws_route_table.public.id
}

resource "aws_ecs_cluster" "main" {
  name = "django-cluster"
}

resource "aws_cloudwatch_log_group" "django_log_group" {
  name = "/ecs/django"
}

resource "aws_cloudwatch_log_stream" "django_log_stream" {
  name           = "django-stream"
  log_group_name = aws_cloudwatch_log_group.django_log_group.name
}

resource "aws_ecs_task_definition" "django" {
  family                   = "django-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "4096"
  memory                   = "8192"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "django"
      image     = "266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest"
      essential = true
      portMappings = [{ containerPort = 80, hostPort = 80, protocol = "tcp" }]
      environment = [
        { name = "POSTGRESQL_DATABASE_NAME", value = var.POSTGRESQL_DATABASE_NAME },
        { name = "POSTGRESQL_USERNAME", value = var.POSTGRESQL_USERNAME },
        { name = "POSTGRESQL_SERVER_NAME", value = var.POSTGRESQL_SERVER_NAME },
        { name = "POSTGRESQL_PASSWORD", value = var.POSTGRESQL_PASSWORD },
        { name = "OPENAI_API_KEY", value = var.OPENAI_API_KEY },
        { name = "DEEPGRAM_API_KEY", value = var.DEEPGRAM_API_KEY },
        { name = "EMAIL_HOST", value = var.EMAIL_HOST },
        { name = "EMAIL_HOST_USER", value = var.EMAIL_HOST_USER },
        { name = "EMAIL_HOST_PASSWORD", value = var.EMAIL_HOST_PASSWORD },
        { name = "DEFAULT_FROM_EMAIL", value = var.DEFAULT_FROM_EMAIL },
        { name = "AWS_STORAGE_BUCKET_NAME", value = var.AWS_STORAGE_BUCKET_NAME },
        { name = "AWS_S3_REGION_NAME", value = var.AWS_S3_REGION_NAME },
        { name = "AWS_ACCESS_KEY_ID", value = var.AWS_ACCESS_KEY_ID },
        { name = "AWS_SECRET_ACCESS_KEY", value = var.AWS_SECRET_ACCESS_KEY },
        { name = "INTUIT_VERIFIER_TOKEN", value = var.INTUIT_VERIFIER_TOKEN },
        { name = "INTUIT_CLIENT_ID", value = var.INTUIT_CLIENT_ID },
        { name = "INTUIT_CLIENT_SECRET", value = var.INTUIT_CLIENT_SECRET },
        { name = "NEW_INTUIT_REDIRECT_URI", value = var.NEW_INTUIT_REDIRECT_URI },
        { name = "INTUIT_ENVIRONMENT", value = var.INTUIT_ENVIRONMENT },
        { name = "STRIPE_SECRET_KEY", value = var.STRIPE_SECRET_KEY },
        { name = "STRIPE_PUBLISHABLE_KEY", value = var.STRIPE_PUBLISHABLE_KEY },
        { name = "STRIPE_WEBHOOK_SECRET", value = var.STRIPE_WEBHOOK_SECRET },
        { name = "REDIS_URL", value = var.REDIS_URL },
        { name = "AWS_REGION", value = var.AWS_REGION }
      ],
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.django_log_group.name
          "awslogs-stream-prefix" = aws_cloudwatch_log_stream.django_log_stream.name
          "awslogs-region"        = "us-west-1"
          "awslogs-create-group"  = "true"
        }
      }
    },
    {
      name  = "flower"
      image = "mher/flower"
      portMappings = [{ containerPort = 5555, hostPort = 5555, protocol = "tcp" }]
      environment = [{ name = "CELERY_BROKER_URL", value = "redis://redis:6379/0" }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.django_log_group.name
          "awslogs-stream-prefix" = "flower-stream"
          "awslogs-region"        = "us-west-1"
          "awslogs-create-group"  = "true"
        }
      }
    },
    {
      name  = "redis"
      image = "redis"
      portMappings = [{ containerPort = 6379, hostPort = 6379, protocol = "tcp" }]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.django_log_group.name
          "awslogs-stream-prefix" = "redis-stream"
          "awslogs-region"        = "us-west-1"
          "awslogs-create-group"  = "true"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "django" {
  name            = "django-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.django.arn
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = [aws_subnet.subnet_a.id, aws_subnet.subnet_b.id]
    security_groups = [aws_security_group.ecs_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.django_tg.arn
    container_name   = "django"
    container_port   = 80
  }

  desired_count = 1
}

resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecsTaskExecutionRole-custom"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action    = "sts:AssumeRole",
        Effect    = "Allow",
        Principal = { Service = "ecs-tasks.amazonaws.com" }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Auto Scaling Configuration
resource "aws_appautoscaling_target" "django" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.django.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "scale_up" {
  name               = "scale-up"
  policy_type        = "StepScaling"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.django.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 300
    metric_aggregation_type = "Average"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }
}

resource "aws_appautoscaling_policy" "scale_down" {
  name               = "scale-down"
  policy_type        = "StepScaling"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.django.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 300
    metric_aggregation_type = "Average"

    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = -1
    }
  }
}


resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name = "cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods = 1
  metric_name = "CPUUtilization"
  namespace = "AWS/ECS"
  period = 60
  statistic = "Average"
  threshold = 80
  alarm_actions = [aws_appautoscaling_policy.scale_up.arn]
  dimensions = {
  ClusterName = aws_ecs_cluster.main.name
  ServiceName = aws_ecs_service.django.name
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name = "cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods = 1
  metric_name = "CPUUtilization"
  namespace = "AWS/ECS"
  period = 60
  statistic = "Average"
  threshold = 20
  alarm_actions = [aws_appautoscaling_policy.scale_down.arn]
  dimensions = {
  ClusterName = aws_ecs_cluster.main.name
  ServiceName = aws_ecs_service.django.name
  }
}
