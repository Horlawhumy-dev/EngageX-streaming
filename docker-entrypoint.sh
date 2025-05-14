#!/bin/bash

# Start Django server
python manage.py runserver 0.0.0.0:8000 &

# Start Celery Worker
celery -A EngageX_Streaming worker --loglevel=info --pool=threads &

# Start Celery Beat
celery -A EngageX_Streaming flower --loglevel=info &

# Wait for any of the background processes to exit
wait -n
