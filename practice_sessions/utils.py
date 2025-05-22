#################################################################################################################
# This version uses the original mechanism but now runs the database querying and s3 upload in the background
#################################################################################################################

# Cleanest working version 6-8 secs sentiment analysis
# Fixed the order of the first three chunks
# S3 video concatenation
# Added AI Audience Question functionality triggered at intervals and controlled by a toggle
# Updated S3 bucket structure to include user ID
# Fixed SynchronousOnlyOperation error in connect by accessing user within sync_to_async
# Fixed S3 key extraction for compilation with new URL format
# Handled CancelledError in background task

import asyncio
import platform

# Set the event loop policy for Windows if necessary
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


import json
import os
import asyncio
import tempfile
import concurrent.futures
import subprocess
import boto3
import openai
import django
import time
import traceback
import random # Import random for selecting variations
import numpy as np # Import numpy to handle potential numpy types
from urllib.parse import urlparse # Import urlparse for S3 URL parsing

from base64 import b64decode
from datetime import timedelta
from asgiref.sync import async_to_sync
from celery import shared_task
from channels.generic.websocket import AsyncWebsocketConsumer
# Import database_sync_to_async for handling synchronous database operations in async context
from channels.db import database_sync_to_async

# Assuming these are in a local file sentiment_analysis.py
# transcribe_audio now needs to handle a single audio file (used in process_media_chunk)
# analyze_results now receives a concatenated transcript and the combined audio path (like the original)
# Import the ai_audience_question function
from .sentiment_analysis import analyze_results, transcribe_audio, ai_audience_question

from practice_sessions.models import PracticeSession, SessionChunk, ChunkSentimentAnalysis
from practice_sessions.serializers import SessionChunkSerializer, ChunkSentimentAnalysisSerializer # PracticeSessionSerializer might not be directly needed here
from django.contrib.auth import get_user_model # Import to get the User model
from botocore.config import Config

# Configure a larger connection pool
s3_config = Config(
    max_pool_connections=50
)

User = get_user_model() # Get the active user model

# Ensure Django settings are configured
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "EngageX_Streaming.settings")
django.setup()

# Initialize OpenAI client
# Ensure OPENAI_API_KEY is set in your environment
openai.api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI() if openai.api_key else None  # Initialize client only if API key is available

# Initialize S3 client
# Ensure AWS_REGION is set in your environment or settings
# Create the S3 client with the config applied
s3 = boto3.client(
    "s3",
    region_name=os.environ.get('AWS_REGION'),
    config=s3_config
)
BUCKET_NAME = "engagex-user-content-1234" # Replace with your actual S3 bucket name
BASE_FOLDER = "user-videos/" # Base folder in S3 bucket
TEMP_MEDIA_ROOT = tempfile.gettempdir() # Use system's temporary directory
EMOTION_STATIC_FOLDER = "static-videos"  # Top-level folder for static emotion videos

# Define the rooms the user can choose from. Used for validation.
POSSIBLE_ROOMS = ['conference_room', 'board_room_1', 'board_room_2', 'pitch_studio']

# Assume a fixed number of variations for each emotion video (1.mp4 to 5.mp4)
NUMBER_OF_VARIATIONS = 5

# Define the window size for analysis (number of chunks)
ANALYSIS_WINDOW_SIZE = 4  # Keeping the reduced window size from the previous test

# Define the interval for generating AI questions (in terms of number of analysis windows)
QUESTION_INTERVAL_WINDOWS = 5

    
@database_sync_to_async
def update_session_with_video_url(session_id, video_url):
    """Updates the PracticeSession with the final compiled video URL."""
    try:
        session = PracticeSession.objects.get(id=session_id)
        session.compiled_video_url = video_url  # Assuming you have a field named compiled_video_url
        session.save(update_fields=['compiled_video_url'])
        print(f"WS: Updated session {session_id} with compiled video URL: {video_url}")
        # You might want to send a WebSocket message to the frontend here if the connection is still open
    except PracticeSession.DoesNotExist:
        print(f"WS: PracticeSession with id {session_id} not found during video URL update.")
    except Exception as e:
        print(f"WS: Error updating session {session_id} with compiled video URL: {e}")
        traceback.print_exc()

@database_sync_to_async
def get_session_chunk_urls(session_id):
    """Retrieves S3 URLs for all chunks of a session."""
    try:
        # Order by chunk_number to ensure correct compilation order
        chunks = SessionChunk.objects.filter(session__id=session_id).order_by('chunk_number')
        # Directly use chunk.video_file as it's already the S3 URL string
        chunk_urls = [chunk.video_file for chunk in chunks if chunk.video_file]
        print(f"WS: Retrieved {len(chunk_urls)} chunk URLs for session {session_id}")
        return chunk_urls
    except Exception as e:
        print(f"WS: Error retrieving chunk URLs for session {session_id}: {e}")
        traceback.print_exc()
        return []