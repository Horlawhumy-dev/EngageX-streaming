#!/bin/bash
docker exec -it app-celery-1 celery -A EngageX_Streaming inspect ping