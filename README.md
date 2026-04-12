# Automated VOD-to-Shorts Clipper

## Alpha 0.7.2 Version
### Expect some issues, if you're a developer feel free to make a pull request

An intelligent Python application that automatically processes long-form VODs (2-16 hours), transcribes them using Whisper, detects "clippable" moments based on keywords, and generates short-form video clips optimized for platforms like YouTube Shorts, TikTok, and Instagram Reels.

## Features

### Core Features
- **Automatic Transcription**: OpenAI Whisper or Faster-Whisper for accurate timestamped transcription
- **Smart Detection**: Customizable keyword/phrase matching for highlight moments
- **Intelligent Buffering**: Configurable buffers (10-15s) before and after detected segments
- **Clip Generation**: High-quality video clips with FFmpeg
- **Vertical Format**: Auto-converts to vertical (9:16) format for Shorts platforms
- **Batch Processing**: Process multiple videos automatically

### Multimodal Enhancement
- **Energy Spike Detection**: RMS energy analysis to identify loud/exciting moments
- **Voice Activity Detection (VAD)**: Silero VAD for speech validation - ensures clips contain speech
- **Sentiment Analysis**: Text-based sentiment scoring for detected segments
- **Speech-Aware Buffering**: No cutting mid-sentence - intelligently finds sentence boundaries
- **Zone-Based Overlay Composition**: VTuber face cam overlay in configurable positions

### Subtitle Features
- **Auto-Subtitles**: Burn subtitles into clips using FFmpeg
- **Keyword Highlighting**: Emphasizes detected trigger words in subtitles

## Requirements

### System Requirements
- Python 3.10+ (Python 3.14 tested)
- FFmpeg 7.1+ (must be in PATH)
- Minimum 8GB RAM (16GB+ recommended for long VODs)
- GPU with CUDA support (optional, for faster transcription)

### Quick Install
Run `install.bat` - this will install Python and FFmpeg, set up the virtual environment, and install dependencies automatically.

### First Time Setup
1. Double-click `install.bat`
2. If prompted, close and reopen your terminal, then run `install.bat` again
3. Wait for "INSTALLATION COMPLETE!"
4. Double-click `run.bat` to start clipping

### Processing Videos
1. Put your video files in the `input\` folder
2. Edit keywords in `config\config.yaml` if needed
3. Double-click `run.bat`
4. Find your clips in `output\clips\`

### Troubleshooting Setup
- **"Python not found"** - Close terminal, reopen, run `install.bat` again
- **"FFmpeg not found"** - Close terminal, reopen, run `install.bat` again

### Manual Install (optional)
```bash
# Install Python 3.10+ from python.org
# Install FFmpeg from ffmpeg.org and add to PATH
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python vod_clipper.py --batch
```

## Configuration

Edit `config/config.yaml` to customize:

### Whisper Settings
```yaml
whisper:
  model: "base"  # tiny, base, small, medium, large, large-v3
  device: "cpu"  # cpu or cuda
```

### Detection Keywords
```yaml
detection:
  keywords:
    - "no way"
    - "let's go"
    - "what just happened"
    - "oh my god"
    - "unbelievable"
```

### Multimodal Detection
```yaml
multimodal:
  enabled: true
  energy_weight: 0.3
  vad_weight: 0.3
  sentiment_weight: 0.2
  keyword_weight: 0.2
```

### VAD Settings
```yaml
audio_analysis:
  min_speech_duration: 0.3  # seconds
```

### Buffering
```yaml
buffering:
  pre_buffer: 12.0
  post_buffer: 12.0
  speech_aware: true  # Don't cut mid-sentence
```

### Zone Overlay (VTuber face cam)
```yaml
zones:
  enabled: true
  face_cam:
    position: "right_bottom"
    extract_resolution: "512x512"
```

### Subtitles
```yaml
subtitles:
  enabled: true
  highlight_keywords: true
  font: "Arial"
  font_size: 48
```

## Usage

### Basic Usage
```bash
# Process a single video
python vod_clipper.py --video "path/to/video.mp4"

# Process all videos in input directory
python vod_clipper.py --batch

# Interactive mode
python vod_clipper.py
```

### Command Line Options
```
--video PATH     Process specific video file
--batch         Process all videos in input directory
--config PATH   Custom config file
--input-dir     Custom input directory
--help         Show help
```

### Workflow
1. Place VOD files in `input/` directory
2. Edit keywords in `config/config.yaml`
3. Run `python vod_clipper.py --batch`
4. Generated clips appear in `output/clips/`
5. Check logs in `output/logs/`

## Project Structure

```
aimi-vod/
├── vod_clipper.py          # Main application entry point
├── install.bat            # First-time setup script
├── run.bat                # Run the application
├── requirements.txt       # Python dependencies
├── config/
│   └── config.yaml       # Configuration file
├── modules/
│   ├── __init__.py
│   ├── input_handler.py   # Video validation & metadata
│   ├── audio_extractor.py # Audio extraction
│   ├── transcriber.py     # Whisper transcription
│   ├── detector.py       # Multimodal segment detection
│   ├── clip_generator.py # Clip generation with subtitles
│   ├── audio_analyzer.py # Energy, VAD, sentiment analysis
│   ├── zone_composer.py  # Zone-based overlay composition
│   └── subtitle_generator.py # SRT generation with highlighting
├── input/                 # Place VOD files here
├── output/
│   ├── clips/           # Generated video clips
│   └── logs/            # Processing logs
├── temp/                  # Temporary files
└── README.md
```

## How It Works

1. **Input Handling**: Validates video file, extracts metadata
2. **Audio Extraction**: High-quality mono audio for Whisper
3. **Audio Analysis**: Energy spikes, VAD, sentiment scoring
4. **Transcription**: Whisper creates timestamped transcript
5. **Detection**: Multimodal scoring (energy + VAD + sentiment + keywords)
6. **Buffering**: Speech-aware buffers around detected segments
7. **Zone Extraction**: (Optional) VTuber face cam from configured zones
8. **Subtitle Generation**: SRT with keyword highlighting
9. **Clip Generation**: FFmpeg extracts clips, burns subtitles, applies overlays

## Output

**Input**: `gaming_stream.mp4` (4 hour VOD)

**Output** (`output/clips/`):
- `gaming_stream_clip_1_00h05m23s_00h05m47s.mp4` (24s clip)
- `gaming_stream_clip_2_00h08m55s_00h09m18s.mp4` (23s clip)

**Logs** (`output/logs/`):
- `vod_clipper_20250625_143022.log`
- `summary_gaming_stream.json`

## Troubleshooting

### FFmpeg not found
Install FFmpeg and ensure it's in PATH:
```bash
ffmpeg -version
```

### No clips generated
- Check keywords in config
- Verify transcription quality
- Review logs in `output/logs/`

### Memory issues
- Use smaller Whisper model (tiny/base)
- Set `device: "cuda"` if using GPU

## Performance Tips

1. **GPU Acceleration**: Set `device: "cuda"` in config for NVIDIA GPU
2. **Faster-Whisper**: Use `faster-whisper` model for better performance
3. **Model Size**: Smaller models (tiny/base) process faster
4. **Clean Temp**: Enable `cleanup_temp: true` to save disk space

## License

MIT License
