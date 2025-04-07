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

def create_audio_visualization(
    segments: List[SpeakerSegment],
    audio_path: str,
    output_path: str
):
    """
    Create a split-screen visualization showing which speaker is active,
    with waveform visualization.
    
    Args:
        segments: List of SpeakerSegment objects
        audio_path: Path to the input audio file
        output_path: Path to save the output video
    """
    logger.info("Creating speaker visualization video with waveform effects")
    
    # Fixed parameters
    width = 1920
    height = 1080
    fps = 24
    speaker1_color = '#3498db'  # Blue
    speaker2_color = '#e74c3c'  # Red
    
    # Convert hex colors to RGB
    speaker1_rgb = hex_to_rgb(speaker1_color)
    speaker2_rgb = hex_to_rgb(speaker2_color)
    
    # Create half-width for each speaker's side
    half_width = width // 2
    
    # Load audio to extract volume data
    audio_clip = AudioFileClip(audio_path)
    
    # Get the duration from the segments
    max_end_time = max(segment.end for segment in segments) if segments else audio_clip.duration
    
    # Trim audio to match the max duration from segments
    audio_clip = audio_clip.subclipped(0, max_end_time)
    
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
    
    # Create a function to generate each frame of the output video
    def make_frame(t):
        # Create a base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill the left and right halves with baseline colors (darker versions)
        left_color = tuple(int(c * 0.5) for c in speaker1_rgb)   # Speaker A - darker
        right_color = tuple(int(c * 0.5) for c in speaker2_rgb)  # Speaker B - darker
        
        # Convert RGB to BGR for OpenCV
        left_color_bgr = left_color[::-1]
        right_color_bgr = right_color[::-1]
        
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
        
        # Set radius for each speaker
        # Speaker A only changes size when they are the active speaker
        if active_speaker == "A":
            volume_radius_a = base_radius + int(max_radius * volume * 0.4)
        else:
            volume_radius_a = base_radius  # Constant size when not speaking
            
        # Speaker B only changes size when they are the active speaker
        if active_speaker == "B":
            volume_radius_b = base_radius + int(max_radius * volume * 0.4)
        else:
            volume_radius_b = base_radius  # Constant size when not speaking
        
        # Define circle centers for both speakers
        center_y = height // 2
        center_x_a = half_width // 2
        center_x_b = half_width + half_width // 2
        
        # Draw circles for Speaker A (always present)
        # Create gradient pattern inside the circle for Speaker A
        for r in range(volume_radius_a, 0, -5):
            # Create color gradient effect
            ratio = r / volume_radius_a
            
            # Base color with slight variations for inner circles
            r_val = int(speaker1_rgb[0] * (0.5 + 0.5 * ratio))
            g_val = int(speaker1_rgb[1] * (0.5 + 0.5 * ratio))
            b_val = int(speaker1_rgb[2] * (0.5 + 0.5 * ratio))
            
            # Add wave pattern effect inside
            if r % 10 < 5:
                # Alternate pattern for visual interest
                r_val = min(255, int(r_val * 1.2))
                g_val = min(255, int(g_val * 1.2))
                b_val = min(255, int(b_val * 1.2))
            
            # Darken the inactive speaker's circle
            if active_speaker != "A":
                r_val = int(r_val * 0.7)
                g_val = int(g_val * 0.7)
                b_val = int(b_val * 0.7)
            
            # BGR for OpenCV
            circle_color = (b_val, g_val, r_val)
            
            # Draw concentric circles with decreasing radius
            cv2.circle(
                frame,
                (center_x_a, center_y),
                r,
                circle_color,
                2 if r == volume_radius_a else -1  # Outline for outer circle, filled for inner
            )
        
        # Draw circles for Speaker B (always present)
        # Create gradient pattern inside the circle for Speaker B
        for r in range(volume_radius_b, 0, -5):
            # Create color gradient effect
            ratio = r / volume_radius_b
            
            # Base color with slight variations for inner circles
            r_val = int(speaker2_rgb[0] * (0.5 + 0.5 * ratio))
            g_val = int(speaker2_rgb[1] * (0.5 + 0.5 * ratio))
            b_val = int(speaker2_rgb[2] * (0.5 + 0.5 * ratio))
            
            # Add wave pattern effect inside
            if r % 10 < 5:
                # Alternate pattern for visual interest
                r_val = min(255, int(r_val * 1.2))
                g_val = min(255, int(g_val * 1.2))
                b_val = min(255, int(b_val * 1.2))
            
            # Darken the inactive speaker's circle
            if active_speaker != "B":
                r_val = int(r_val * 0.7)
                g_val = int(g_val * 0.7)
                b_val = int(b_val * 0.7)
            
            # BGR for OpenCV
            circle_color = (b_val, g_val, r_val)
            
            # Draw concentric circles with decreasing radius
            cv2.circle(
                frame,
                (center_x_b, center_y),
                r,
                circle_color,
                2 if r == volume_radius_b else -1  # Outline for outer circle, filled for inner
            )
        
        # Add a pulse effect outer ring only for active speaker
        if active_speaker == "A":
            # Highlight the left side (Speaker A)
            # Convert RGB to BGR for OpenCV
            active_color_bgr = speaker1_rgb[::-1]
            
            # Create brighter version for active speaker
            frame[:, :half_width] = active_color_bgr
            
            # Add a pulse effect outer ring
            pulse_size = int(volume_radius_a * (1.0 + 0.1 * np.sin(t * 8)))
            cv2.circle(
                frame,
                (center_x_a, center_y),
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
            
        elif active_speaker == "B":
            # Highlight the right side (Speaker B)
            # Convert RGB to BGR for OpenCV
            active_color_bgr = speaker2_rgb[::-1]
            
            # Create brighter version for active speaker
            frame[:, half_width:] = active_color_bgr
            
            # Add a pulse effect outer ring
            pulse_size = int(volume_radius_b * (1.0 + 0.1 * np.sin(t * 8)))
            cv2.circle(
                frame,
                (center_x_b, center_y),
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
        
        # Add speaker labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Ben", (half_width // 2 - 60, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Taylor", (half_width + half_width // 2 - 60, 50), font, 1, (255, 255, 255), 2)
        
        # Add current text at the bottom of the frame
        if current_text:
            # Limit text length for display
            if len(current_text) > 50:
                current_text = current_text[:47] + "..."
            
            # Add a dark background for the text
            text_size = cv2.getTextSize(current_text, font, 0.8, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (text_x - 10, text_y - 30),
                (text_x + text_size[0] + 10, text_y + 10),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(frame, current_text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
        
        # Add time counter
        time_str = f"Time: {int(t // 60):02d}:{int(t % 60):02d}"
        cv2.putText(frame, time_str, (width - 150, height - 20), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
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
    output_path = str(audio_path.with_suffix(f'.viz_{timestamp}.mp4'))
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
        
        # Step 2: Create audio visualization
        create_audio_visualization(
            segments,
            args.audio_file,
            output_path
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