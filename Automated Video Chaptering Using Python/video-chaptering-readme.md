# Video Chaptering Project

## Overview

This Python project implements an innovative video chaptering solution using Natural Language Processing (NLP) and machine learning techniques. The application automatically segments YouTube video transcripts into coherent chapters, enhancing user navigation and content understanding.

## Features

- Extract video transcripts from YouTube
- Perform topic modeling using Non-negative Matrix Factorization (NMF)
- Automatically identify logical chapter breaks
- Generate meaningful chapter titles
- Save chapter information for easy reference

## Prerequisites

### Software Requirements
- Python 3.8+
- pip package manager

### Required Libraries
- pandas
- numpy
- matplotlib
- scikit-learn
- google-api-python-client
- youtube-transcript-api

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-chaptering.git
cd video-chaptering
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

### YouTube Data API Setup
1. Go to Google Cloud Console
2. Create a new project
3. Enable YouTube Data API v3
4. Generate an API key
5. Replace `'Your API Key'` in the script with your actual API key

## Usage

### Running the Script
```bash
python video_chaptering.py
```

When prompted, enter a YouTube video URL. The script will:
- Extract the video transcript
- Perform topic modeling
- Identify chapter breaks
- Generate chapter names
- Save results to a CSV file

### Example Output
```
Enter the YouTube video link: https://youtu.be/example_video
Transcript saved to video_id_transcript.csv

Final Chapter Points with Names:
00:00:01 - Chapter 1: initial topic
00:05:02 - Chapter 2: middle topic
00:10:03 - Chapter 3: final topic
```

## Technical Details

### Methodology

1. **Transcript Extraction**
   - Uses YouTube Transcript API
   - Retrieves text segments with timestamps

2. **Topic Modeling**
   - Applies Non-negative Matrix Factorization (NMF)
   - Identifies 10 primary topics
   - Assigns dominant topics to text segments

3. **Chapter Segmentation**
   - Detects topic transitions
   - Consolidates breaks with configurable thresholds
   - Generates chapter names using TF-IDF

### Customization

- Adjust `n_topics` to change topic complexity
- Modify `threshold` to control chapter granularity
- Experiment with vectorization parameters

## Limitations

- Requires videos with available transcripts
- Chapter quality depends on transcript accuracy
- Works best with longer, structured content







