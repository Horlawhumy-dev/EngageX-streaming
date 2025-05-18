celery -A EngageX_Streaming worker --loglevel=info --pool=threads
celery -A EngageX_Streaming flower --loglevel=info
export REDIS_URL=redis://localhost:6379/0
chmod 400 engagex-streaming-key.pem
ssh -i "engagex-streaming-key.pem" ec2-user@ec2-54-153-6-254.us-west-1.compute.amazonaws.com
docker ps
docker-compose logs -f django
docker-compose logs -f celery
docker-compose logs -f celery-flower