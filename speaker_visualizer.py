#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Speaker Visualization Tool

This script:
1. Takes an audio file as input
2. Uses AssemblyAI to create a transcript with speaker diarization and timestamps
3. Processes the transcript to identify when each speaker is talking
4. Creates a split-screen video visualization showing which speaker is active 
5. Visualizes audio levels with volume indicators for each speaker
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
import assemblyai as aai
# Update imports for MoviePy 2.0
from moviepy import (
    VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip,
    concatenate_videoclips, clips_array, VideoClip, TextClip
)
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont  # Add PIL imports for better text rendering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SpeakerSegment:
    """Represents a segment where a specific speaker is talking"""
    def __init__(self, speaker: str, start: float, end: float, text: str):
        self.speaker = speaker
        self.start = start
        self.end = end
        self.duration = end - start
        self.text = text

    def __str__(self):
        return f"Speaker {self.speaker}: {self.start:.2f}s - {self.end:.2f}s ({self.duration:.2f}s) | {self.text}"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create a split-screen video visualization for speaker diarization')
    parser.add_argument('audio_file', help='Path to the input audio file')
    parser.add_argument('--swap-speaker', action='store_true', help='Swap speaker A to be Taylor instead of Ben')
    parser.add_argument('--preview', action='store_true', help='Only render the first 10 seconds for preview')
    
    return parser.parse_args()

def hex_to_rgb(hex_color):
    """Convert hex color code to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def transcribe_audio(audio_path: str, api_key: str) -> List[SpeakerSegment]:
    """
    Transcribe the audio using AssemblyAI with speaker diarization.
    If a transcript file already exists, load it instead of re-transcribing.
    
    Args:
        audio_path: Path to the audio file
        api_key: AssemblyAI API key
    
    Returns:
        List of SpeakerSegment objects
    """
    # Create transcript file path
    audio_path_obj = Path(audio_path)
    transcript_path = audio_path_obj.with_suffix('.transcript.json')
    
    # Check if transcript file exists
    if transcript_path.exists():
        logger.info(f"Loading existing transcript from {transcript_path}")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Reconstruct segments from saved data
            segments = []
            for seg_data in transcript_data:
                segments.append(SpeakerSegment(
                    speaker=seg_data['speaker'],
                    start=seg_data['start'],
                    end=seg_data['end'],
                    text=seg_data['text']
                ))
            logger.info(f"Loaded {len(segments)} segments from existing transcript")
            return segments
        except Exception as e:
            logger.warning(f"Failed to load existing transcript: {e}. Will re-transcribe.")
    
    logger.info(f"Transcribing audio: {audio_path}")
    
    # Initialize AssemblyAI
    aai.settings.api_key = api_key
    
    # Configure transcription with speaker diarization
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        punctuate=True,
        format_text=True
    )
    
    # Create the transcriber
    transcriber = aai.Transcriber()
    
    # Start the transcription
    logger.info("Starting transcription with speaker diarization...")
    transcript = transcriber.transcribe(audio_path, config)
    
    # Check for errors
    if transcript.status == aai.TranscriptStatus.error:
        logger.error(f"Transcription failed: {transcript.error}")
        sys.exit(1)
    
    # Process the transcript to extract speaker segments
    segments = []
    
    # Process each utterance (segment of speech by a single speaker)
    for utterance in transcript.utterances:
        segment = SpeakerSegment(
            speaker=utterance.speaker,
            start=utterance.start / 1000,  # Convert ms to seconds
            end=utterance.end / 1000,      # Convert ms to seconds
            text=utterance.text
        )
        segments.append(segment)
        logger.debug(str(segment))
    
    # Save transcript to file
    logger.info(f"Saving transcript to {transcript_path}")
    transcript_data = [
        {
            'speaker': seg.speaker,
            'start': seg.start,
            'end': seg.end,
            'text': seg.text
        }
        for seg in segments
    ]
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments

def get_speaker_mapping(swap_speaker: bool) -> Dict[str, Dict[str, Any]]:
    """
    Get the mapping between transcript speakers and visual representation.
    
    Args:
        swap_speaker: Whether to swap the default speaker mapping
    
    Returns:
        Dictionary mapping transcript speakers to visual properties
    """
    # Define speaker colors
    ben_color = '#e74c3c'  # Orange/Red
    taylor_color = '#3498db'  # Blue
    
    # Define the mapping
    if swap_speaker:
        # Speaker A -> Taylor, Speaker B -> Ben
        return {
            "A": {
                "name": "Taylor",
                "color": taylor_color,
                "position": "right"
            },
            "B": {
                "name": "Ben",
                "color": ben_color,
                "position": "left"
            }
        }
    else:
        # Speaker A -> Ben, Speaker B -> Taylor
        return {
            "A": {
                "name": "Ben",
                "color": ben_color,
                "position": "left"
            },
            "B": {
                "name": "Taylor",
                "color": taylor_color,
                "position": "right"
            }
        }

def create_audio_visualization(
    segments: List[SpeakerSegment],
    audio_path: str,
    output_path: str,
    swap_speaker: bool = False,
    preview: bool = False
):
    """
    Create a split-screen visualization showing which speaker is active,
    with waveform visualization.
    
    Args:
        segments: List of SpeakerSegment objects
        audio_path: Path to the input audio file
        output_path: Path to save the output video
        swap_speaker: Whether to swap the default speaker mapping
        preview: If True, only render the first 10 seconds
    """
    logger.info("Creating speaker visualization video with waveform effects")
    
    # Fixed parameters
    width = 1920
    height = 1080
    fps = 24
    
    # Get speaker mapping
    speaker_mapping = get_speaker_mapping(swap_speaker)
    
    # Define visual properties based on position (left and right)
    left_speaker = next(s for s in speaker_mapping.values() if s["position"] == "left")
    right_speaker = next(s for s in speaker_mapping.values() if s["position"] == "right")
    
    # Extract colors
    left_color = left_speaker["color"]
    right_color = right_speaker["color"]
    
    # Convert hex colors to RGB
    left_rgb = hex_to_rgb(left_color)
    right_rgb = hex_to_rgb(right_color)
    
    # Create half-width for each speaker's side
    half_width = width // 2
    
    # Load audio to extract volume data
    audio_clip = AudioFileClip(audio_path)
    
    # Get the duration from the segments
    max_end_time = max(segment.end for segment in segments) if segments else audio_clip.duration
    
    # Apply preview limit if requested
    if preview:
        preview_duration = 10.0  # 10 seconds
        max_end_time = min(max_end_time, preview_duration)
        logger.info(f"Preview mode: rendering only the first {preview_duration} seconds")
    
    # Trim audio to match the max duration from segments
    audio_clip = audio_clip.subclipped(0, max_end_time)
    
    # Filter segments to only include those within the preview range if in preview mode
    if preview:
        segments = [s for s in segments if s.start < max_end_time]
        
    # Precompute volume data at regular intervals
    logger.info("Extracting volume data from audio")
    sample_rate = audio_clip.fps
    n_samples = int(audio_clip.duration * fps)
    volume_data = []
    
    # Analyze audio volume in small chunks
    chunk_duration = 1/fps  # Duration for each frame
    for i in tqdm(range(n_samples), desc="Analyzing audio volume"):
        t = i / fps
        if t >= audio_clip.duration:
            break
        
        # Get audio chunk
        chunk_start = t
        chunk_end = min(t + chunk_duration, audio_clip.duration)
        chunk = audio_clip.subclipped(chunk_start, chunk_end)
        
        # Get audio array
        audio_array = chunk.to_soundarray()
        
        # Calculate volume (RMS of the audio signal)
        volume = np.sqrt(np.mean(np.square(audio_array)))
        volume_data.append(volume)
    
    # Normalize volume data
    if volume_data:
        max_volume = max(volume_data)
        min_volume = min(volume_data)
        volume_range = max_volume - min_volume
        
        if volume_range > 0:
            volume_data = [(vol - min_volume) / volume_range for vol in volume_data]
        else:
            volume_data = [0.5 for _ in volume_data]
    
    # Try to load a modern font - use a system font if available, otherwise fall back to default
    try:
        # Path to a nice modern font - adjust this path as needed for your system
        font_path = "/System/Library/Fonts/Helvetica.ttc"  # macOS path
        if not os.path.exists(font_path):
            # Try Windows path
            font_path = "C:/Windows/Fonts/segoeui.ttf"
        if not os.path.exists(font_path):
            # Try common Linux path
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        
        # Load the font with PIL
        title_font = ImageFont.truetype(font_path, 48)  # Increased from 32 to 48 for speaker names
        subtitle_font = ImageFont.truetype(font_path, 36)  # Increased from 26 to 36 for subtitles
        time_font = ImageFont.truetype(font_path, 18)  # For time display
        
        logger.info(f"Using font: {font_path}")
    except Exception as e:
        # Fallback to default font if custom font not available
        logger.warning(f"Could not load custom font, using default: {e}")
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        time_font = ImageFont.load_default()
    
    # Create a function to generate each frame of the output video
    def make_frame(t):
        # Create a base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill the left and right halves with baseline colors (darker versions)
        left_color_darker = tuple(int(c * 0.5) for c in left_rgb)
        right_color_darker = tuple(int(c * 0.5) for c in right_rgb)
        
        # Convert RGB to BGR for OpenCV
        left_color_bgr = left_color_darker[::-1]
        right_color_bgr = right_color_darker[::-1]
        
        # Fill the frame halves with base colors
        frame[:, :half_width] = left_color_bgr
        frame[:, half_width:] = right_color_bgr
        
        # Find which speaker is active at time t
        active_speaker = None
        current_text = ""
        for segment in segments:
            if segment.start <= t < segment.end:
                active_speaker = segment.speaker
                current_text = segment.text
                break
        
        # Get the current volume level
        frame_idx = min(int(t * fps), len(volume_data) - 1) if volume_data else 0
        volume = volume_data[frame_idx] if frame_idx >= 0 and frame_idx < len(volume_data) else 0
        
        # Calculate base circle parameters
        max_radius = min(half_width // 3, height // 4)  # Smaller maximum size
        base_radius = max(int(max_radius * 0.6), 40)  # Larger minimum base radius (60% of max)
        
        # Map active speaker to left/right position
        active_position = speaker_mapping.get(active_speaker, {}).get("position", None)
        
        # Set radius for each speaker
        left_radius = base_radius
        right_radius = base_radius
        
        if active_position == "left":
            left_radius = base_radius + int(max_radius * volume * 0.4)
        elif active_position == "right":
            right_radius = base_radius + int(max_radius * volume * 0.4)
        
        # Define circle centers for both speakers
        center_y = height // 2
        center_x_left = half_width // 2
        center_x_right = half_width + half_width // 2
        
        # Draw circles for left speaker
        for r in range(left_radius, 0, -5):
            # Create color gradient effect
            ratio = r / left_radius
            
            # Base color with slight variations for inner circles
            r_val = int(left_rgb[0] * (0.5 + 0.5 * ratio))
            g_val = int(left_rgb[1] * (0.5 + 0.5 * ratio))
            b_val = int(left_rgb[2] * (0.5 + 0.5 * ratio))
            
            # Add wave pattern effect inside
            if r % 10 < 5:
                # Alternate pattern for visual interest
                r_val = min(255, int(r_val * 1.2))
                g_val = min(255, int(g_val * 1.2))
                b_val = min(255, int(b_val * 1.2))
            
            # Darken the inactive speaker's circle
            if active_position != "left":
                r_val = int(r_val * 0.7)
                g_val = int(g_val * 0.7)
                b_val = int(b_val * 0.7)
            
            # BGR for OpenCV
            circle_color = (b_val, g_val, r_val)
            
            # Draw concentric circles with decreasing radius
            cv2.circle(
                frame,
                (center_x_left, center_y),
                r,
                circle_color,
                2 if r == left_radius else -1  # Outline for outer circle, filled for inner
            )
        
        # Draw circles for right speaker
        for r in range(right_radius, 0, -5):
            # Create color gradient effect
            ratio = r / right_radius
            
            # Base color with slight variations for inner circles
            r_val = int(right_rgb[0] * (0.5 + 0.5 * ratio))
            g_val = int(right_rgb[1] * (0.5 + 0.5 * ratio))
            b_val = int(right_rgb[2] * (0.5 + 0.5 * ratio))
            
            # Add wave pattern effect inside
            if r % 10 < 5:
                # Alternate pattern for visual interest
                r_val = min(255, int(r_val * 1.2))
                g_val = min(255, int(g_val * 1.2))
                b_val = min(255, int(b_val * 1.2))
            
            # Darken the inactive speaker's circle
            if active_position != "right":
                r_val = int(r_val * 0.7)
                g_val = int(g_val * 0.7)
                b_val = int(b_val * 0.7)
            
            # BGR for OpenCV
            circle_color = (b_val, g_val, r_val)
            
            # Draw concentric circles with decreasing radius
            cv2.circle(
                frame,
                (center_x_right, center_y),
                r,
                circle_color,
                2 if r == right_radius else -1  # Outline for outer circle, filled for inner
            )
        
        # Add a pulse effect outer ring only for active speaker
        if active_position == "left":
            # Highlight the left side
            frame[:, :half_width] = left_rgb[::-1]  # Convert RGB to BGR
            
            # Add a pulse effect outer ring
            pulse_size = int(left_radius * (1.0 + 0.1 * np.sin(t * 8)))
            cv2.circle(
                frame,
                (center_x_left, center_y),
                pulse_size,
                (255, 255, 255),
                2
            )
            
            # Add a highlight border for the side
            cv2.rectangle(
                frame,
                (0, 0),
                (half_width, height),
                (255, 255, 255),  # White highlight
                3  # Border thickness
            )
            
        elif active_position == "right":
            # Highlight the right side
            frame[:, half_width:] = right_rgb[::-1]  # Convert RGB to BGR
            
            # Add a pulse effect outer ring
            pulse_size = int(right_radius * (1.0 + 0.1 * np.sin(t * 8)))
            cv2.circle(
                frame,
                (center_x_right, center_y),
                pulse_size,
                (255, 255, 255),
                2
            )
            
            # Add a highlight border for the side
            cv2.rectangle(
                frame,
                (half_width, 0),
                (width, height),
                (255, 255, 255),  # White highlight
                3  # Border thickness
            )
        
        # Convert OpenCV BGR frame to PIL Image for better text rendering
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Add speaker labels with anti-aliased text - moved down closer to middle
        name_y_position = center_y - 300  # Increased distance from center (was -180)
        draw.text((half_width // 2 - 80, name_y_position), left_speaker["name"], fill=(255, 255, 255), font=title_font)
        draw.text((half_width + half_width // 2 - 80, name_y_position), right_speaker["name"], fill=(255, 255, 255), font=title_font)
        
        # Add current text at the bottom of the frame with modern styling
        if current_text:
            # Limit text length for display
            if len(current_text) > 60:
                current_text = current_text[:57] + "..."
            
            # Calculate text size for centering
            text_bbox = draw.textbbox((0, 0), current_text, font=subtitle_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = (width - text_width) // 2
            text_y = center_y + 300  # Increased distance from center (was +180)
            
            # Draw semi-transparent background for subtitle
            # Create a rounded rectangle shape for the background
            background_color = (0, 0, 0, 180)  # Semi-transparent black
            background_padding = 20  # Increased padding for larger text
            
            # Draw rounded rectangle with alpha
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Draw rounded rectangle for subtitle background
            overlay_draw.rounded_rectangle(
                (
                    text_x - background_padding,
                    text_y - background_padding,
                    text_x + text_width + background_padding,
                    text_y + text_height + background_padding
                ),
                radius=10,  # Rounded corners
                fill=background_color
            )
            
            # Composite the overlay onto the main image
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(pil_img)
            
            # Draw text with anti-aliasing
            draw.text((text_x, text_y), current_text, fill=(255, 255, 255), font=subtitle_font)
        
        # Add time counter
        time_str = f"Time: {int(t // 60):02d}:{int(t % 60):02d}"
        draw.text((width - 150, height - 30), time_str, fill=(255, 255, 255), font=time_font)
        
        # Convert back to OpenCV format for output
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Create the video clip
    logger.info("Creating video clip")
    video_clip = VideoClip(make_frame, duration=max_end_time)
    
    # Add audio from the original audio file
    logger.info("Adding audio to the video")
    final_clip = video_clip.with_audio(audio_clip)
    
    # Write output file
    logger.info(f"Writing output video to {output_path}")
    final_clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    # Close all clips to release resources
    audio_clip.close()
    final_clip.close()
    
    logger.info("Video creation complete")

def main():
    """Main function to process audio and create visualization video"""
    # Parse command line arguments
    args = parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        logger.error("AssemblyAI API key not found in environment. Please set the ASSEMBLYAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Generate output path based on input file
    audio_path = Path(args.audio_file)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_suffix = f'.viz_{timestamp}'
    if args.preview:
        output_suffix += '_preview'
    output_path = str(audio_path.with_suffix(f'{output_suffix}.mp4'))
    logger.info(f"Output will be saved to: {output_path}")
    
    try:
        # Step 1: Transcribe audio with speaker diarization
        segments = transcribe_audio(args.audio_file, api_key)
        
        if not segments:
            logger.error("No speaker segments found in the transcript")
            sys.exit(1)
        
        # Print speaker segments for debugging
        logger.info(f"Found {len(segments)} speaker segments:")
        for i, segment in enumerate(segments[:5], 1):  # Print first 5 segments
            logger.info(f"  {i}. {segment}")
        if len(segments) > 5:
            logger.info(f"  ... and {len(segments) - 5} more segments")
        
        # Step 2: Create audio visualization with optional speaker swap
        create_audio_visualization(
            segments,
            args.audio_file,
            output_path,
            swap_speaker=args.swap_speaker,
            preview=args.preview
        )
        
        logger.info(f"Speaker visualization video created successfully: {output_path}")
        print(f"\nOutput video: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 