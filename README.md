celery -A EngageX_Streaming worker --loglevel=info --pool=threads
celery --app EngageX_Streaming flower --loglevel=info
REDIS_URL=redis://localhost:6379/0  # Change to ElastiCache endpoint in prod