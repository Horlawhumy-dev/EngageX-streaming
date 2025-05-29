# tasks.py
import re
from celery import shared_task
from .sentiment_analysis import (
    transcribe_audio,
    analyze_results,
    ai_audience_question,
)
import numpy as np


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
from .sentiment_analysis import analyze_results, transcribe_audio, ai_audience_question

from practice_sessions.models import PracticeSession, SessionChunk, ChunkSentimentAnalysis
from practice_sessions.serializers import SessionChunkSerializer, ChunkSentimentAnalysisSerializer # PracticeSessionSerializer might not be directly needed here
from django.contrib.auth import get_user_model # Import to get the User model
from botocore.config import Config
from .utils import get_session_chunk_urls, update_session_with_video_url

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

@shared_task
def analyze_results_task(transcript, video_path, audio_path):
    result = analyze_results(transcript, video_path, audio_path)
    return convert_numpy_types(result)


@shared_task
def transcribe_audio_task(audio_path):
    return transcribe_audio(audio_path)

@shared_task
def ai_audience_question_task(transcript):
    return ai_audience_question(transcript)


# @shared_task
# def compile_session_video_task(session_id, user_id):
#     async def _compile():
#         """Background task to compile all chunks for a session."""
#         print(f"WS: Starting video compilation for session {session_id} in background task.")
#         temp_file_paths = []
#         try:
#             chunk_urls = await get_session_chunk_urls(session_id)
#             if not chunk_urls:
#                 print(f"WS: No chunk URLs found for session {session_id}. Skipping compilation.")
#                 return

#             print(f"WS: Downloading {len(chunk_urls)} chunks for session {session_id}.")
#             downloaded_chunk_paths = []
#             for i, url in enumerate(chunk_urls):
#                 try:
#                     parsed_url = urlparse(url)
#                     hostname_parts = parsed_url.hostname.split('.') if parsed_url.hostname else []
#                     extracted_bucket_name = hostname_parts[0] if hostname_parts else None
#                     key_path = parsed_url.path.lstrip('/') if parsed_url.path else None
#                     if extracted_bucket_name == BUCKET_NAME and key_path:
#                         s3_key = key_path
#                         print(f"WS: Extracted S3 key from URL {url}: {s3_key}")
#                     else:
#                         print(f"WS: Could not extract S3 key or bucket name from URL: {url}. Skipping.")
#                         continue
#                 except Exception as e:
#                     print(f"WS: Error parsing URL {url}: {e}. Skipping.")
#                     continue

#                 temp_input_path = os.path.join(TEMP_MEDIA_ROOT, f"{session_id}_chunk_{i}.webm")
#                 temp_file_paths.append(temp_input_path)
#                 try:
#                     await asyncio.to_thread(s3.download_file, BUCKET_NAME, s3_key, temp_input_path)
#                     downloaded_chunk_paths.append(temp_input_path)
#                     print(f"WS: Downloaded chunk {i+1}/{len(chunk_urls)} to {temp_input_path}")
#                 except Exception as e:
#                     print(f"WS: Error downloading chunk {i+1}: {e}")
#                     continue

#             if not downloaded_chunk_paths:
#                 print(f"WS: No chunks were successfully downloaded for session {session_id}.")
#                 return

#             # === NEW STEP: Convert to MP4 ===
#             converted_mp4_paths = []
#             for i, input_path in enumerate(downloaded_chunk_paths):
#                 mp4_path = input_path.replace(".webm", "_converted.mp4")
#                 temp_file_paths.append(mp4_path)
#                 command = [
#                     "ffmpeg", "-y", "-i", input_path,
#                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
#                     "-c:a", "aac",
#                     mp4_path
#                 ]
#                 print(f"WS: Converting chunk {i+1} to MP4...")
#                 process = await asyncio.to_thread(subprocess.run, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#                 if process.returncode == 0:
#                     converted_mp4_paths.append(mp4_path)
#                     print(f"WS: Converted to {mp4_path}")
#                 else:
#                     print(f"WS: Conversion failed for {input_path}: {process.stderr.decode()}")

#             if not converted_mp4_paths:
#                 print(f"WS: No converted MP4 files to compile. Skipping.")
#                 return

#             # === Generate concat list file ===
#             list_file_path = os.path.join(TEMP_MEDIA_ROOT, f"{session_id}_concat_list.txt")
#             temp_file_paths.append(list_file_path)
#             with open(list_file_path, 'w') as f:
#                 for path in sorted(converted_mp4_paths, key=lambda p: int(re.search(r"_(\d+)", p).group(1))):
#                     f.write(f"file '{path.replace(os.sep, '/')}'\n")
#             print(f"WS: Created concat list file: {list_file_path}")

#             # === Concatenate with FFmpeg ===
#             compiled_video_filename = f"{session_id}_compiled.mp4"
#             compiled_video_path = os.path.join(TEMP_MEDIA_ROOT, compiled_video_filename)
#             temp_file_paths.append(compiled_video_path)
#             ffmpeg_command = [
#                 "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path,
#                 "-c", "copy", compiled_video_path
#             ]
#             print(f"WS: Running FFmpeg compilation command: {' '.join(ffmpeg_command)}")
#             process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             stdout, stderr = await asyncio.to_thread(process.communicate)
#             returncode = await asyncio.to_thread(lambda p: p.returncode, process)

#             if returncode != 0:
#                 print(f"WS: FFmpeg compilation error (code {returncode}) for session {session_id}: {stderr.decode()}")
#                 return
#             else:
#                 print(f"WS: Video compiled successfully to: {compiled_video_path}")

#             # === Upload to S3 ===
#             print(f"WS: Uploading compiled video to S3 for session {session_id}.")
#             if not user_id:
#                 print(f"WS: Error: User ID not available. Cannot upload.")
#                 return
#             compiled_s3_key = f"{BASE_FOLDER}{user_id}/{session_id}/{compiled_video_filename}"
#             await asyncio.to_thread(s3.upload_file, compiled_video_path, BUCKET_NAME, compiled_s3_key)

#             region_name = os.environ.get('AWS_S3_REGION_NAME', os.environ.get('AWS_REGION', 'us-east-1'))
#             compiled_s3_url = f"https://{BUCKET_NAME}.s3.{region_name}.amazonaws.com/{compiled_s3_key}"

#             if compiled_s3_url:
#                 await update_session_with_video_url(session_id, compiled_s3_url)
#                 print(f"WS: Compilation and upload complete: {compiled_s3_url}")
#             else:
#                 print("WS: Failed to construct S3 URL")

#         except Exception as e:
#             print(f"WS: An error occurred during video compilation for session {session_id}: {e}")
#             traceback.print_exc()
#         finally:
#             print(f"WS: Cleaning up temporary files for session {session_id}.")
#             for file_path in temp_file_paths:
#                 if os.path.exists(file_path):
#                     try:
#                         os.remove(file_path)
#                         print(f"WS: Removed temporary file: {file_path}")
#                     except Exception as e:
#                         print(f"WS: Error removing file {file_path}: {e}")
    
#     async_to_sync(_compile)()


@shared_task(bind=True)
def compile_session_video_task(self, session_id, user_id):
    async def _compile():
        """
        Background task to compile all video chunks for a session.
        Sets PracticeSession.compiled_video_url upon success, or None on failure.
        """
        print(f"Starting video compilation for session {session_id} (Celery Task ID: {self.request.id}).")
        temp_file_paths = []
        try:
            # 1. Fetch chunk URLs
            chunk_urls = await get_session_chunk_urls(session_id)
            if not chunk_urls:
                print(f"No chunk URLs found for session {session_id}. Skipping compilation.")
                await update_session_with_video_url(session_id, None) # Mark as failed/not compilable
                return

            # 2. Download chunks from S3
            print(f"Downloading {len(chunk_urls)} chunks for session {session_id}.")
            downloaded_chunk_paths = []
            for i, url in enumerate(chunk_urls):
                try:
                    parsed_url = urlparse(url)
                    hostname_parts = parsed_url.hostname.split('.') if parsed_url.hostname else []
                    extracted_bucket_name = hostname_parts[0] if hostname_parts else None
                    key_path = parsed_url.path.lstrip('/') if parsed_url.path else None

                    # Validate bucket name and key path for security and correctness
                    if extracted_bucket_name == BUCKET_NAME and key_path:
                        s3_key = key_path
                    else:
                        print(f"Could not extract valid S3 key or bucket name from URL: {url}. Skipping chunk.")
                        continue

                except Exception as e:
                    print(f"Error parsing URL {url}: {e}. Skipping chunk.", exc_info=True)
                    continue

                temp_input_path = os.path.join(TEMP_MEDIA_ROOT, f"{session_id}_chunk_{i}.webm")
                temp_file_paths.append(temp_input_path) # Add to cleanup list early
                try:
                    await asyncio.to_thread(s3.download_file, BUCKET_NAME, s3_key, temp_input_path)
                    downloaded_chunk_paths.append(temp_input_path)
                    print(f"Downloaded chunk {i+1}/{len(chunk_urls)} to {temp_input_path}")
                except Exception as e:
                    print(f"Error downloading chunk {i+1} from {s3_key}: {e}", exc_info=True)
                    continue # Try next chunk

            if not downloaded_chunk_paths:
                print(f"No chunks were successfully downloaded for session {session_id}.")
                await update_session_with_video_url(session_id, None)
                return

            # 3. Convert WebM chunks to MP4 (if necessary)
            converted_mp4_paths = []
            for i, input_path in enumerate(downloaded_chunk_paths):
                mp4_path = input_path.replace(".webm", "_converted.mp4")
                temp_file_paths.append(mp4_path)
                command = [
                    "ffmpeg", "-y", "-i", input_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23", # Video codec settings
                    "-c:a", "aac", # Audio codec settings
                    mp4_path
                ]
                print(f"Converting chunk {i+1} to MP4: {input_path} -> {mp4_path}")
                process = await asyncio.to_thread(subprocess.run, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode == 0:
                    converted_mp4_paths.append(mp4_path)
                    print(f"Successfully converted chunk to {mp4_path}")
                else:
                    print(f"FFmpeg conversion failed for {input_path} (code {process.returncode}): {process.stderr.decode()}", exc_info=True)

            if not converted_mp4_paths:
                print(f"No converted MP4 files available for session {session_id}. Cannot compile.")
                await update_session_with_video_url(session_id, None)
                return

            # 4. Generate FFmpeg concat list file
            list_file_path = os.path.join(TEMP_MEDIA_ROOT, f"{session_id}_concat_list.txt")
            temp_file_paths.append(list_file_path)
            with open(list_file_path, 'w') as f:
                # Ensure chunks are sorted by number (assuming _X in filename)
                # Use forward slashes for FFmpeg compatibility across OS
                for path in sorted(converted_mp4_paths, key=lambda p: int(re.search(r"_(\d+)", p).group(1))):
                    f.write(f"file '{path.replace(os.sep, '/')}'\n")
            print(f"Created concat list file: {list_file_path}")

            # 5. Concatenate videos with FFmpeg
            compiled_video_filename = f"{session_id}_compiled.mp4"
            compiled_video_path = os.path.join(TEMP_MEDIA_ROOT, compiled_video_filename)
            temp_file_paths.append(compiled_video_path)
            ffmpeg_command = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path,
                "-c", "copy", compiled_video_path # Copy stream without re-encoding if possible
            ]
            print(f"Running FFmpeg concatenation command: {' '.join(ffmpeg_command)}")
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await asyncio.to_thread(process.communicate)
            returncode = process.returncode

            if returncode != 0:
                print(f"FFmpeg concatenation error (code {returncode}) for session {session_id}: {stderr.decode()}", exc_info=True)
                await update_session_with_video_url(session_id, None) # Mark as failed
                return
            else:
                print(f"Video compiled successfully to: {compiled_video_path}")

            # 6. Upload compiled video to S3
            if not user_id:
                print(f"User ID not available for session {session_id}. Cannot upload compiled video to S3.")
                await update_session_with_video_url(session_id, None)
                return

            compiled_s3_key = f"{BASE_FOLDER}{user_id}/{session_id}/{compiled_video_filename}"
            print(f"Uploading compiled video {compiled_video_path} to S3 bucket {BUCKET_NAME} with key {compiled_s3_key}.")
            await asyncio.to_thread(s3.upload_file, compiled_video_path, BUCKET_NAME, compiled_s3_key)

            # Construct final S3 URL
            region_name = os.environ.get('AWS_S3_REGION_NAME', os.environ.get('AWS_REGION', 'us-east-1'))
            compiled_s3_url = f"https://{BUCKET_NAME}.s3.{region_name}.amazonaws.com/{compiled_s3_key}"

            if compiled_s3_url:
                await update_session_with_video_url(session_id, compiled_s3_url)
                print(f"Compilation and upload complete for session {session_id}. URL: {compiled_s3_url}")
            else:
                print("Failed to construct S3 URL after upload. Setting video URL to None.")
                await update_session_with_video_url(session_id, None)

        except PracticeSession.DoesNotExist:
            print(f"PracticeSession with ID {session_id} not found during compilation task. Task cannot proceed.")
            # No update needed if session doesn't exist, as there's no object to update
        except Exception as e:
            print(f"An unexpected error occurred during video compilation for session {session_id}: {e}", exc_info=True)
            await update_session_with_video_url(session_id, None) # Mark as failed on any unhandled exception
        finally:
            # 7. Clean up temporary files
            print(f"Cleaning up temporary files for session {session_id}.")
            for file_path in temp_file_paths:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Removed temporary file: {file_path}")
                    except OSError as e:
                        print(f"Error removing temporary file {file_path}: {e}", exc_info=True)
                else:
                    print(f"Temporary file not found during cleanup: {file_path}")

    # Run the async compilation logic using async_to_sync for Celery
    async_to_sync(_compile)()