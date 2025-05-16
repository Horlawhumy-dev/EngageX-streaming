[
  {
    "name": "engagex_stream_app",
    "image": "${docker_image_url_django}",
    "essential": true,
    "cpu": 3072,
    "memory": 4096,
    "portMappings": [
      {
        "containerPort": 8000,
        "protocol": "tcp"
      }
    ],
    "command": ["gunicorn", "-w", "3", "-b", ":8000", "EngageX_Streaming.wsgi:application"],
    "environment": [
      { "name": "POSTGRESQL_DATABASE_NAME", "value": "${postgres_db_name}" },
      { "name": "POSTGRESQL_USERNAME", "value": "${postgres_username}" },
      { "name": "POSTGRESQL_PASSWORD", "value": "${postgres_password}" },
      { "name": "POSTGRESQL_SERVER_NAME", "value": "${postgres_hostname}" },
      { "name": "PORT", "value": "5432" },

      { "name": "OPENAI_API_KEY", "value": "${openai_api_key}" },
      { "name": "DEEPGRAM_API_KEY", "value": "${deepgram_api_key}" },

      { "name": "EMAIL_HOST", "value": "${email_host}" },
      { "name": "EMAIL_PORT", "value": "${email_port}" },
      { "name": "EMAIL_USE_TLS", "value": "${email_use_tls}" },
      { "name": "EMAIL_HOST_USER", "value": "${email_host_user}" },
      { "name": "EMAIL_HOST_PASSWORD", "value": "${email_host_password}" },
      { "name": "DEFAULT_FROM_EMAIL", "value": "${default_from_email}" },

      { "name": "USE_S3", "value": "True" },
      { "name": "AWS_STORAGE_BUCKET_NAME", "value": "${aws_storage_bucket_name}" },
      { "name": "AWS_S3_REGION_NAME", "value": "${aws_s3_region_name}" },

      { "name": "AWS_ACCESS_KEY_ID", "value": "${aws_access_key_id}" },
      { "name": "AWS_SECRET_ACCESS_KEY", "value": "${aws_secret_access_key}" },
      { "name": "AUTH_TOKEN_FOR_WEBSOCKET", "value": "${auth_token_for_websocket}" },
      { "name": "DJANGO_SETTINGS_MODULE", "value": "EngageX_Streaming.settings" },
      { "name": "REDIS_URL", "value": "${redis_url}" },

      { "name": "INTUIT_VERIFIER_TOKEN", "value": "${intuit_verifier_token}" },
      { "name": "INTUIT_CLIENT_ID", "value": "${intuit_client_id}" },
      { "name": "INTUIT_CLIENT_SECRET", "value": "${intuit_client_secret}" },
      { "name": "NEW_INTUIT_REDIRECT_URI", "value": "${new_intuit_redirect_uri}" },
      { "name": "INTUIT_ENVIRONMENT", "value": "${intuit_environment}" },

      { "name": "STRIPE_SECRET_KEY", "value": "${stripe_secret_key}" },
      { "name": "STRIPE_PUBLISHABLE_KEY", "value": "${stripe_publishable_key}" },
      { "name": "STRIPE_WEBHOOK_SECRET", "value": "${stripe_webhook_secret}" },

      { "name": "ALLOWED_HOSTS", "value": "${allowed_hosts}" }
    ],
    "mountPoints": [
      {
        "containerPath": "/efs/staticfiles/",
        "sourceVolume": "efs-volume",
        "readOnly": false
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/engagex_stream_app",
        "awslogs-region": "${region}",
        "awslogs-stream-prefix": "engagex_stream_app-log-stream"
      }
    }
  },
  {
    "name": "nginx",
    "image": "${docker_image_url_nginx}",
    "essential": true,
    "cpu": 1024,
    "memory": 2048,
    "portMappings": [
      {
        "containerPort": 80,
        "protocol": "tcp"
      }
    ],
    "mountPoints": [
      {
        "containerPath": "/efs/staticfiles/",
        "sourceVolume": "efs-volume",
        "readOnly": false
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/nginx",
        "awslogs-region": "${region}",
        "awslogs-stream-prefix": "nginx-log-stream"
      }
    }
  },
  {
    "name": "celery_worker",
    "image": "${docker_image_url_django}",
    "essential": true,
    "cpu": 3072,
    "memory": 8192,
    "command": ["celery", "-A", "EngageX_Streaming", "worker", "--loglevel=info", "--pool=threads"],
    "environment": [
      { "name": "POSTGRESQL_DATABASE_NAME", "value": "${postgres_db_name}" },
      { "name": "POSTGRESQL_USERNAME", "value": "${postgres_username}" },
      { "name": "POSTGRESQL_PASSWORD", "value": "${postgres_password}" },
      { "name": "POSTGRESQL_SERVER_NAME", "value": "${postgres_hostname}" },
      { "name": "PORT", "value": "5432" },

      { "name": "OPENAI_API_KEY", "value": "${openai_api_key}" },
      { "name": "DEEPGRAM_API_KEY", "value": "${deepgram_api_key}" },

      { "name": "EMAIL_HOST", "value": "${email_host}" },
      { "name": "EMAIL_PORT", "value": "${email_port}" },
      { "name": "EMAIL_USE_TLS", "value": "${email_use_tls}" },
      { "name": "EMAIL_HOST_USER", "value": "${email_host_user}" },
      { "name": "EMAIL_HOST_PASSWORD", "value": "${email_host_password}" },
      { "name": "DEFAULT_FROM_EMAIL", "value": "${default_from_email}" },

      { "name": "USE_S3", "value": "True" },
      { "name": "AWS_STORAGE_BUCKET_NAME", "value": "${aws_storage_bucket_name}" },
      { "name": "AWS_S3_REGION_NAME", "value": "${aws_s3_region_name}" },

      { "name": "AWS_ACCESS_KEY_ID", "value": "${aws_access_key_id}" },
      { "name": "AWS_SECRET_ACCESS_KEY", "value": "${aws_secret_access_key}" },
      { "name": "AUTH_TOKEN_FOR_WEBSOCKET", "value": "${auth_token_for_websocket}" },
      { "name": "DJANGO_SETTINGS_MODULE", "value": "EngageX_Streaming.settings" },
      { "name": "REDIS_URL", "value": "${redis_url}" },

      { "name": "INTUIT_VERIFIER_TOKEN", "value": "${intuit_verifier_token}" },
      { "name": "INTUIT_CLIENT_ID", "value": "${intuit_client_id}" },
      { "name": "INTUIT_CLIENT_SECRET", "value": "${intuit_client_secret}" },
      { "name": "NEW_INTUIT_REDIRECT_URI", "value": "${new_intuit_redirect_uri}" },
      { "name": "INTUIT_ENVIRONMENT", "value": "${intuit_environment}" },

      { "name": "STRIPE_SECRET_KEY", "value": "${stripe_secret_key}" },
      { "name": "STRIPE_PUBLISHABLE_KEY", "value": "${stripe_publishable_key}" },
      { "name": "STRIPE_WEBHOOK_SECRET", "value": "${stripe_webhook_secret}" },

      { "name": "ALLOWED_HOSTS", "value": "${allowed_hosts}" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/celery_worker",
        "awslogs-region": "${region}",
        "awslogs-stream-prefix": "celery-worker-log-stream"
      }
    }
  },
  {
    "name": "flower",
    "image": "${docker_image_url_django}",
    "essential": false,
    "cpu": 1024,
    "memory": 1024,
    "portMappings": [
      {
        "containerPort": 5555,
        "protocol": "tcp"
      }
    ],
    "command": ["celery", "-A", "EngageX_Streaming", "flower", "--port=5555"],
    "environment": [
      { "name": "REDIS_URL", "value": "${redis_url}" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/flower",
        "awslogs-region": "${region}",
        "awslogs-stream-prefix": "flower-log-stream"
      }
    }
  }
]