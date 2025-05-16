from __future__ import absolute_import, unicode_literals
import os
import ssl
from celery import Celery
import ssl
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "EngageX_Streaming.settings")

app = Celery("EngageX_Streaming",
    broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )

app.config_from_object("django.conf:settings", namespace="CELERY")

app.conf.update(
    broker_url=os.getenv("REDIS_URL"),
    result_backend=os.getenv("REDIS_URL"),
    accept_content=["application/json"],
    task_serializer="json",
    result_serializer="json",
)

app.autodiscover_tasks()