# tasks.py
from celery import shared_task
from .sentiment_analysis import (
    transcribe_audio,
    analyze_results,
    ai_audience_question,
)

@shared_task
def transcribe_audio_task(audio_path):
    return transcribe_audio(audio_path)

@shared_task
def analyze_results_task(transcript, video_path, audio_path):
    return analyze_results(transcript, video_path, audio_path)

@shared_task
def ai_audience_question_task(transcript):
    return ai_audience_question(transcript)