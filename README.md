# Speaker Visualization Tool

This tool creates a split-screen visualization showing which speaker is talking at any given time, based on speaker diarization from an audio file. It generates waveform visualizations for each speaker.

## Features

- Transcribes audio using AssemblyAI's speaker diarization
- Creates a split-screen visualization with each speaker on one side
- Displays waveform visualization that reacts to the volume of speech
- Shows the current transcript text at the bottom of the screen
- Visually highlights which speaker is currently talking
- Caches transcription results to avoid re-transcribing

## Requirements

- Python 3.7 or higher
- Install dependencies with: `pip install -r requirements.txt`
- AssemblyAI API key (sign up at [assemblyai.com](https://www.assemblyai.com/))

## Setup

Set your AssemblyAI API key as an environment variable:

```bash
# For Linux/macOS
export ASSEMBLYAI_API_KEY="your-api-key-here"

# For Windows (Command Prompt)
set ASSEMBLYAI_API_KEY=your-api-key-here

# For Windows (PowerShell)
$env:ASSEMBLYAI_API_KEY="your-api-key-here"
```

## Usage

Run the script providing the path to your audio file:

```bash
python speaker_visualizer.py your_audio_file.mp3
```

The output video will be saved in the same location as the input file with a timestamped suffix (e.g., `your_audio_file.viz_20250101_120000.mp4`).

### Example

```bash
python speaker_visualizer.py interview.mp3
```

This will create `interview.viz_TIMESTAMP.mp4` in the same directory.

## How It Works

1. The audio file is transcribed using AssemblyAI's speaker diarization API
2. The transcript is processed to identify which speaker is talking at what time
3. The script analyzes the audio to extract volume data for waveform visualization
4. A split-screen visualization is created with:
   - Left side for Speaker A (blue)
   - Right side for Speaker B (red)
5. When a speaker is talking:
   - Their side is brightened and highlighted 
   - A waveform visualization appears showing the volume level
   - The current text is displayed at the bottom of the screen

## License

MIT