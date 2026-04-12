"""
Audio Extractor Module
Extracts audio from video files for transcription
"""

import os
import subprocess
import ffmpeg
from pathlib import Path
from typing import Dict, Any
import logging


class AudioExtractor:
    """Extracts audio from video files optimized for Whisper transcription"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.audio_config = config.get("audio", {})

    def extract_audio(self, video_path: str, video_info: Dict[str, Any]) -> str:
        """Extract audio from video file and save as WAV"""
        self.logger.info(f"Extracting audio from: {video_path}")

        # Create output path for audio file
        temp_dir = Path(self.config["paths"]["temp_dir"])
        temp_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        audio_path = temp_dir / f"{video_name}_audio.wav"

        try:
            # Use ffmpeg-python for audio extraction
            self._extract_with_ffmpeg_python(video_path, str(audio_path))

            # Verify the extracted audio file
            if self._verify_audio_file(str(audio_path)):
                self.logger.info(f"Audio extracted successfully: {audio_path}")
                return str(audio_path)
            else:
                raise RuntimeError(f"Audio extraction failed for {video_path}")

        except ffmpeg.Error as e:
            self.logger.warning(f"FFmpeg-python failed, trying subprocess: {e}")
            # Fallback to subprocess
            return self._extract_with_subprocess(video_path, str(audio_path))
        except Exception as e:
            self.logger.error(f"Audio extraction error: {e}")
            raise RuntimeError(f"Failed to extract audio from {video_path}")

    def _extract_with_ffmpeg_python(self, video_path: str, output_path: str):
        """Extract audio using ffmpeg-python library"""
        sample_rate = self.audio_config.get("sample_rate", 16000)
        channels = self.audio_config.get("channels", 1)
        codec = self.audio_config.get("codec", "pcm_s16le")

        # Build ffmpeg command
        stream = ffmpeg.input(video_path)

        # Extract audio with optimal settings for Whisper
        audio = stream.audio

        # Apply filters for optimal Whisper input
        audio = ffmpeg.filter(audio, "aresample", sample_rate)
        audio = ffmpeg.filter(audio, "aformat", sample_rates=sample_rate)

        # Output configuration
        output = ffmpeg.output(
            audio,
            output_path,
            acodec=codec,
            ac=channels,  # Mono channel
            ar=sample_rate,
            f="wav",
        )

        # Run ffmpeg
        ffmpeg.run(
            output, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )

    def _extract_with_subprocess(self, video_path: str, output_path: str) -> str:
        """Extract audio using ffmpeg subprocess (fallback)"""
        sample_rate = self.audio_config.get("sample_rate", 16000)
        channels = self.audio_config.get("channels", 1)

        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM 16-bit
            "-ar",
            str(sample_rate),  # Sample rate
            "-ac",
            str(channels),  # Mono channel
            "-y",  # Overwrite output
            output_path,
        ]

        self.logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(
                    f"FFmpeg failed with return code {result.returncode}"
                )

            return output_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg subprocess error: {e}")
            raise RuntimeError(f"Audio extraction failed: {e.stderr}")

    def _verify_audio_file(self, audio_path: str) -> bool:
        """Verify that the extracted audio file is valid"""
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not created: {audio_path}")
            return False

        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            self.logger.error(f"Audio file is empty: {audio_path}")
            return False

        # Check if file is a valid WAV file
        try:
            # Simple WAV header check
            with open(audio_path, "rb") as f:
                header = f.read(12)
                if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                    self.logger.warning(
                        f"File doesn't appear to be a valid WAV: {audio_path}"
                    )
                    # Still return True as ffmpeg might produce valid audio with different headers
        except Exception as e:
            self.logger.warning(f"Error checking WAV header: {e}")

        # Try to get audio info using ffprobe
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )

            if audio_stream:
                self.logger.debug(
                    f"Audio info: {audio_stream.get('codec_name')}, "
                    f"sample_rate: {audio_stream.get('sample_rate')}, "
                    f"channels: {audio_stream.get('channels')}"
                )
                return True
            else:
                self.logger.error(f"No audio stream found in: {audio_path}")
                return False

        except ffmpeg.Error as e:
            self.logger.warning(f"Could not probe audio file: {e}")
            # File might still be valid, use file size as indicator
            return file_size > 1024  # At least 1KB

    def cleanup_audio_file(self, audio_path: str):
        """Clean up extracted audio file"""
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                self.logger.debug(f"Cleaned up audio file: {audio_path}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up audio file {audio_path}: {e}")
