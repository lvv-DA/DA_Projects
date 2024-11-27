import os
import re
import csv
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Get API Key from environment
API_KEY = os.getenv('YOUTUBE_API_KEY')

# Rest of the script remains the same...
