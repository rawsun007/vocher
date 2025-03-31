import argparse
import os
import re
import tempfile
import cv2
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor
from google.cloud import vision
from google.cloud.vision import types
import io
import requests

# Amazon voucher code pattern (adjust as needed for different platforms)
CODE_PATTERN = r'\b[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b'

# Initialize Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def validate_url(url: str) -> bool:
    """Validate YouTube URL format"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return re.match(youtube_regex, url) is not None

def download_video(url: str) -> str:
    """Download YouTube video using yt-dlp"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'outtmpl': os.path.join(tmpdir, 'video.%(ext)s'),
            'format': 'bestaudio+besteffmpeg/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'noplaylist': True,
            'quiet': True,
        }
        
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info)
        except Exception as e:
            raise RuntimeError(f"Video download failed: {str(e)}")

def extract_text_from_image(image_content):
    """Extract text from an image using Google Cloud Vision API"""
    try:
        image = types.Image(content=image_content)
        response = client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"API Error: {response.error.message}")
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        raise RuntimeError(f"Error extracting text from image: {str(e)}")

def process_frame(frame):
    """Process a single frame for text using Google Cloud Vision API"""
    # Encode the frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        raise ValueError("Failed to encode frame as JPEG")
    img_bytes = buffer.tobytes()
    
    # Use Google Cloud Vision API for OCR
    text = extract_text_from_image(img_bytes)
    return re.findall(CODE_PATTERN, text)

def extract_codes(video_path: str) -> set:
    """Extract frames from the video and analyze them for voucher codes"""
    codes = set()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Invalid video frame rate")

    frame_interval = int(fps)  # Process 1 frame per second
    
    with ThreadPoolExecutor() as executor:
        futures = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                futures.append(executor.submit(process_frame, frame))
                
            frame_count += 1
        
        for future in futures:
            try:
                codes.update(future.result())
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
    
    cap.release()
    return codes

def main():
    parser = argparse.ArgumentParser(description='YouTube Voucher Code Extractor using Cloud OCR')
    parser.add_argument('url', help='YouTube video URL')
    args = parser.parse_args()

    if not validate_url(args.url):
        print("Error: Invalid YouTube URL")
        return

    try:
        print("⏳ Downloading video...")
        video_path = download_video(args.url)
        
        print("🔍 Analyzing video frames using Cloud OCR...")
        codes = extract_codes(video_path)
        
        if codes:
            print("\n🎉 Found potential voucher codes:")
            for code in codes:
                print(f"👉 {code}")
        else:
            print("\n❌ No valid codes found in the video")
            
    except Exception as e:
        print(f"\n🚨 Error: {str(e)}")
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    main()
