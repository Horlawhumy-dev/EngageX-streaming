resource "aws_ecs_cluster" "production" {
  name = "${var.ecs_cluster_name}-cluster"
}

data "aws_s3_bucket_object" "env_file" {
  bucket = "enagexbackendenv"
  key    = "be.env"
}

locals {
  # Assuming env file lines like KEY=VALUE, convert to map
  env_lines = split("\n", trim(data.aws_s3_bucket_object.env_file.body))
  env_vars = [
    for line in local.env_lines : {
      name  = split("=", line)[0]
      value = split("=", line)[1]
    }
    if length(trim(line)) > 0 && !starts_with(line, "#")
  ]
}

data "template_file" "app" {
  template = file("templates/django_app.json.tpl")

  vars = {
  docker_image_url_django       = var.docker_image_url_django
  docker_image_url_nginx        = var.docker_image_url_nginx
  region                       = var.region

  postgresql_database_name     = var.postgresql_database_name
  postgresql_username          = var.postgresql_username
  postgresql_password          = var.postgresql_password
  postgresql_server_name       = var.postgresql_server_name
  port                        = "5432"

  openai_api_key              = var.openai_api_key
  deepgram_api_key            = var.deepgram_api_key

  email_host                  = var.email_host
  email_port                  = var.email_port
  email_use_tls               = var.email_use_tls
  email_host_user             = var.email_host_user
  email_host_password         = var.email_host_password
  default_from_email          = var.default_from_email

  use_s3                     = "True"
  aws_storage_bucket_name    = var.aws_storage_bucket_name
  aws_s3_region_name         = var.aws_s3_region_name

  aws_access_key_id          = var.aws_access_key_id
  aws_secret_access_key      = var.aws_secret_access_key
  auth_token_for_websocket   = var.auth_token_for_websocket
  django_settings_module     = "EngageX_Streaming.settings"
  redis_url                  = var.redis_url

  intuit_verifier_token      = var.intuit_verifier_token
  intuit_client_id           = var.intuit_client_id
  intuit_client_secret       = var.intuit_client_secret
  new_intuit_redirect_uri    = var.new_intuit_redirect_uri
  intuit_environment         = var.intuit_environment

  stripe_secret_key          = var.stripe_secret_key
  stripe_publishable_key     = var.stripe_publishable_key
  stripe_webhook_secret      = var.stripe_webhook_secret

  allowed_hosts              = var.allowed_hosts
}

}

resource "aws_ecs_task_definition" "app" {
  family                   = "engagex-stream-task"
  # depends_on               = []
  network_mode             = "awsvpc" # Required for Fargate
  requires_compatibilities = ["FARGATE"]
  cpu                      = "${var.fargate_cpu}"
  memory                   = "${var.fargate_memory}"
  execution_role_arn       = aws_iam_role.ecs-task-execution-role.arn
  task_role_arn            = aws_iam_role.ecs-task-execution-role.arn
  container_definitions    = data.template_file.app.rendered
  volume {
    name = "efs-volume"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.efs.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2049
      authorization_config {
        access_point_id = aws_efs_access_point.app_access_point.id
        iam             = "ENABLED"
      }
    }
  }
}

resource "aws_ecs_service" "production" {
  name            = "${var.ecs_cluster_name}-service"
  cluster         = aws_ecs_cluster.production.id
  task_definition = aws_ecs_task_definition.app.arn
  launch_type     = "FARGATE"
  desired_count   = var.app_count
  network_configuration {
    subnets          = [aws_subnet.public-subnet-1.id, aws_subnet.public-subnet-2.id]
    security_groups  = [aws_security_group.ecs-fargate.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_alb_target_group.default-target-group.arn
    container_name   = "nginx"
    container_port   = 80
  }
}

resource "aws_ecs_task_definition" "django_migration" {
  family                = "django-migration-task"
  network_mode          = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                   = var.fargate_cpu
  memory                = var.fargate_memory
  execution_role_arn    = aws_iam_role.ecs-task-execution-role.arn

  container_definitions = jsonencode([
    {
      name  = "django-migration-container"
      image = var.docker_image_url_django
      environment: local.env_vars,
      # Run migrations as the command
      command = ["python", "manage.py", "migrate"]

      # Add log configuration for CloudWatch
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/django-app"
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }

      # Other required configurations go here...
      portMappings = [
        {
          containerPort = 8000
        }
      ]
    }
  ])
}

resource "aws_efs_file_system" "efs" {
  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }
}

resource "aws_efs_access_point" "app_access_point" {
  file_system_id = aws_efs_file_system.efs.id
  posix_user {
    uid = 1000
    gid = 1000
  }
  root_directory {
    path = "/efs"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "755"
    }
  }
}

resource "aws_efs_mount_target" "efs_mount" {
  count           = length([aws_subnet.public-subnet-1.id, aws_subnet.public-subnet-2.id])
  file_system_id  = aws_efs_file_system.efs.id
  subnet_id       = [aws_subnet.public-subnet-1.id, aws_subnet.public-subnet-2.id][count.index]
  security_groups = [aws_security_group.efs_sg.id]
}

resource "aws_ecs_task_definition" "django_collectstatic" {
  family                = "django-collectstatic-task"
  network_mode          = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                   = var.fargate_cpu
  memory                = var.fargate_memory
  execution_role_arn    = aws_iam_role.ecs-task-execution-role.arn
  task_role_arn         = aws_iam_role.ecs-task-execution-role.arn
  volume {
    name = "efs-volume"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.efs.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2049
      authorization_config {
        access_point_id = aws_efs_access_point.app_access_point.id
        iam             = "ENABLED"
      }
    }
  }
  container_definitions = jsonencode([
    {
      name  = "django-collectstatic-container"
      image = var.docker_image_url_django
      linuxParameters = {
        user = "1000:1000"
      }
      environment = local.env_vars

      # Run collectstatic as the command
      command = ["python", "manage.py", "collectstatic", "--no-input", "-v", "3"]

      mountPoints = [
        {
          sourceVolume = "efs-volume",
          containerPath = "/efs/staticfiles/",
          readOnly = false
        }
      ]

      # Add log configuration for CloudWatch
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/django-app"
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }

      # Other required configurations go here...
      portMappings = [
        {
          containerPort = 8000
        }
      ]
    }
  ])
}
