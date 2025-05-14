[
  {
    name      = "django"
    image     = "266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest"
    essential = true
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
      ]
    command = ["/app/docker-entrypoint.sh"]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.django_log_group.name
        "awslogs-stream-prefix" = "django-stream"
        "awslogs-region"        = "us-west-1"
        "awslogs-create-group"  = "true"
      }
    }
  }

  {
    name      = "celery-worker"
    image     = "266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest"
    essential = false
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
      ]
    command = ["/app/docker-entrypoint.sh"]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.django_log_group.name
        "awslogs-stream-prefix" = "celery-stream"
        "awslogs-region"        = "us-west-1"
        "awslogs-create-group"  = "true"
      }
    }
  }

  # Flower container
  {
    name      = "flower"
    image     = "266735827053.dkr.ecr.us-west-1.amazonaws.com/engagex-streaming:latest"
    essential = false
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
      ]
    command = ["/app/docker-entrypoint.sh"]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.django_log_group.name
        "awslogs-stream-prefix" = "flower-stream"
        "awslogs-region"        = "us-west-1"
        "awslogs-create-group"  = "true"
      }
    }
  }

  {
    name      = "redis"
    image     = "redis:latest"
    essential = true
    portMappings = [
      {
        containerPort = 6379
        hostPort      = 6379
      }
    ]
  }

]