[
  {
    "name": "django",
    "image": "266735827053.dkr.ecr.eu-north-1.amazonaws.com/practice-session:latest",
    "essential": true,
    "cpu": 2048,  // 2 vCPUs
    "memory": 4096,  // 4 GB of memory
    "portMappings": [
      {
        "containerPort": 8000,
        "hostPort": 8000,
        "protocol": "tcp"
      }
    ],
    "environment": [
      {
        "name": "DJANGO_SETTINGS_MODULE",
        "value": "EngageX_Streaming.settings"
      },
      {
        "name": "CELERY_BROKER_URL",
        "value": "redis://localhost:6379/0"
      },
      {
        "name": "CELERY_RESULT_BACKEND",
        "value": "redis://localhost:6379/0"
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/${ecs_service_name}",
        "awslogs-region": "${region}",
        "awslogs-stream-prefix": "django"
      }
    },
    "command": [
      "/bin/bash",
      "-c",
      "python manage.py runserver 0.0.0.0:8000 & celery -A EngageX_Streaming worker --loglevel=info --pool=prefork & celery -A EngageX_Streaming flower --loglevel=info & wait -n"
    ],
    "dependsOn": [
      {
        "containerName": "redis",
        "condition": "START"
      }
    ]
  },
  {
    "name": "redis",
    "image": "redis",
    "essential": false,
    "memory": 1024,  // 1 GB
    "cpu": 256,  // 1/2 vCPU
    "portMappings": [
      {
        "containerPort": 6379,
        "hostPort": 6379,
        "protocol": "tcp"
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/${ecs_service_name}",
        "awslogs-region": "${region}",
        "awslogs-stream-prefix": "redis"
      }
    }
  }
]