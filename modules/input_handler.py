"""
Input Handler Module
Handles video file validation and metadata extraction
"""

import os
import subprocess
import json
import ffmpeg
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class InputHandler:
    """Handles video input validation and metadata extraction"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.supported_formats = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv"}

    def validate_and_analyze(self, video_path: str) -> Dict[str, Any]:
        """Validate video file and extract metadata"""
        self.logger.info(f"Validating video file: {video_path}")

        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check file format
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in self.supported_formats:
            self.logger.warning(f"Unsupported file format: {file_ext}")

        # Extract metadata
        metadata = self._extract_metadata(video_path)

        # Validate video duration (2-16 hours as per requirements)
        duration = metadata.get("duration", 0)
        if duration < 2 * 3600:  # Less than 2 hours
            self.logger.warning(
                f"Video duration ({duration / 3600:.1f}h) is less than recommended 2 hours"
            )
        elif duration > 16 * 3600:  # More than 16 hours
            self.logger.warning(
                f"Video duration ({duration / 3600:.1f}h) exceeds recommended 16 hours"
            )

        return {
            "file_path": video_path,
            "file_name": Path(video_path).name,
            "file_size": os.path.getsize(video_path),
            "duration": duration,
            "resolution": metadata.get("resolution", "Unknown"),
            "frame_rate": metadata.get("frame_rate", 0),
            "video_codec": metadata.get("video_codec", "Unknown"),
            "audio_codec": metadata.get("audio_codec", "Unknown"),
            "valid": True,
        }

    def _extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffmpeg"""
        try:
            # Use ffmpeg.probe to get metadata
            probe = ffmpeg.probe(video_path)

            # Get video stream
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            # Get audio stream
            audio_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )

            metadata = {
                "duration": float(probe["format"]["duration"]),
                "size": int(probe["format"]["size"]),
                "bit_rate": int(probe["format"]["bit_rate"])
                if "bit_rate" in probe["format"]
                else 0,
            }

            if video_stream:
                metadata.update(
                    {
                        "resolution": f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}",
                        "frame_rate": self._parse_frame_rate(
                            video_stream.get("r_frame_rate", "0/0")
                        ),
                        "video_codec": video_stream.get("codec_name", "Unknown"),
                        "pixel_format": video_stream.get("pix_fmt", "Unknown"),
                        "has_b_frames": video_stream.get("has_b_frames", 0),
                    }
                )

            if audio_stream:
                metadata.update(
                    {
                        "audio_codec": audio_stream.get("codec_name", "Unknown"),
                        "audio_channels": audio_stream.get("channels", 0),
                        "audio_sample_rate": audio_stream.get("sample_rate", 0),
                    }
                )

            return metadata

        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg probe error: {e}")
            # Fallback to using ffprobe directly
            return self._extract_metadata_fallback(video_path)
        except Exception as e:
            self.logger.error(f"Metadata extraction error: {e}")
            return {
                "duration": 0,
                "resolution": "Unknown",
                "frame_rate": 0,
                "video_codec": "Unknown",
                "audio_codec": "Unknown",
            }

    def _extract_metadata_fallback(self, video_path: str) -> Dict[str, Any]:
        """Fallback metadata extraction using ffprobe directly"""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)

            return self._extract_metadata(video_path)  # Reuse the parsing logic

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFprobe subprocess error: {e}")
            raise RuntimeError(f"Failed to extract metadata from {video_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            raise RuntimeError(f"Invalid metadata from {video_path}")

    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """Parse frame rate string (e.g., '30000/1001' to 29.97)"""
        try:
            if "/" in frame_rate_str:
                num, den = frame_rate_str.split("/")
                return float(num) / float(den)
            else:
                return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe["format"]["duration"])
        except (ffmpeg.Error, KeyError):
            # Fallback method
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            try:
                return float(result.stdout.strip())
            except ValueError:
                return 0.0
