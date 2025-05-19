# tasks.py
from celery import shared_task
from .sentiment_analysis import (
    transcribe_audio,
    analyze_results,
    ai_audience_question,
)
import numpy as np

def convert_numpy_types(obj):
    #celery will not serialize numpy types
    # so we convert them to native python types
    # before returning
    # this is a workaround for the issue
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

@shared_task(name="analyze_results_task", queue="celery", bind=True)
def analyze_results_task(transcript, video_path, audio_path):
    result = analyze_results(transcript, video_path, audio_path)
    return convert_numpy_types(result)


@shared_task(name="transcribe_audio_task", queue="cpu_queue", bind=True)
def transcribe_audio_task(audio_path):
    return transcribe_audio(audio_path)

@shared_task(name="ai_audience_question_task", queue="cpu_queue", bind=True)
def ai_audience_question_task(transcript):
    return ai_audience_question(transcript)