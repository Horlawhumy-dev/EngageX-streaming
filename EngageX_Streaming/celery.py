from __future__ import absolute_import, unicode_literals
import os
import ssl
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "EngageX_Streaming.settings")

app = Celery(
    "EngageX_Streaming",
    # broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    # redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
)
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.update(
    BROKER_URL=os.getenv("REDIS_URL"),
    CELERY_RESULT_BACKEND=os.getenv("REDIS_URL"),
    CELERY_ACCEPT_CONTENT=["application/json"],
    CELERY_TASK_SERIALIZER="json",
    CELERY_RESULT_SERIALIZER="json",
)
app.autodiscover_tasks()