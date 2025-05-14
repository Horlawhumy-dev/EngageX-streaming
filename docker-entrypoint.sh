# #!/bin/bash

# # Start Django server
# python manage.py runserver 0.0.0.0:8000 &

# # Start Celery Worker
# celery -A EngageX_Streaming worker --loglevel=info --pool=threads &

# # Start Celery Beat
# celery -A EngageX_Streaming flower --loglevel=info &

# # Wait for any of the background processes to exit
# wait -n


#!/bin/bash
python manage.py migrate
python manage.py collectstatic --noinput
gunicorn EngageX_Streaming.wsgi:application --bind 0.0.0.0:80

# Start Celery worker
celery -A EngageX_Streaming worker --loglevel=info --pool=threads

# Start Flower for monitoring Celery tasks
celery -A EngageX_Streaming flower --loglevel=info
