"""
Subtitle Generator Module
Generates burn-in subtitles with keyword highlighting
"""

import os
import re
import json
import subprocess
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path


class SubtitleGenerator:
    """Generates subtitles with optional keyword highlighting"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.subtitle_config = config.get("subtitles", {})
        self.detection_config = config.get("detection", {})
        self.keywords = self.detection_config.get("keywords", [])

    def is_enabled(self) -> bool:
        """Check if subtitles are enabled"""
        return self.subtitle_config.get("enabled", True)

    def generate_subtitle_file(
        self,
        transcription: List[Dict[str, Any]],
        output_path: str,
        clip_start_time: float = 0,
        clip_end_time: Optional[float] = None,
    ) -> Optional[str]:
        """Generate SRT subtitle file from transcription"""
        if not transcription:
            return None

        try:
            srt_path = output_path.replace(".mp4", ".srt")

            with open(srt_path, "w", encoding="utf-8") as f:
                subtitle_index = 1

                for segment in transcription:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "").strip()

                    if not text:
                        continue

                    # Filter to only include segments within clip time range
                    if clip_end_time is not None and start > clip_end_time:
                        break
                    if end < clip_start_time:
                        continue

                    # Adjust timestamps relative to clip start
                    adj_start = max(0, start - clip_start_time)
                    adj_end = end - clip_start_time

                    # Format: index
                    f.write(f"{subtitle_index}\n")

                    # Format: HH:MM:SS,mmm --> HH:MM:SS,mmm
                    f.write(
                        f"{self._format_time(adj_start)} --> {self._format_time(adj_end)}\n"
                    )

                    # Format: text (with optional highlighting)
                    if self.subtitle_config.get("keyword_highlight", True):
                        text = self._highlight_keywords(text)

                    f.write(f"{text}\n\n")

                    subtitle_index += 1

            self.logger.info(f"Generated subtitle file: {srt_path}")
            return srt_path

        except Exception as e:
            self.logger.error(f"Subtitle generation error: {e}")
            return None

    def _format_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _highlight_keywords(self, text: str) -> str:
        """Add highlighting markers around detected keywords"""
        if not self.keywords:
            return text

        highlighted = text
        highlight_color = self.subtitle_config.get("highlight_color", "#FF6B6B")

        # Simple approach: uppercase the text with keywords for visual distinction
        # In a full implementation, you'd use ASS styles or HTML markup
        # For SRT, we'll use text casing to indicate highlighted words

        for keyword in self.keywords:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)

            # Replace with uppercase version to make keywords stand out
            def upper_match(match):
                return match.group().upper()

            highlighted = pattern.sub(upper_match, highlighted)

        return highlighted

    def create_ass_styles(self) -> str:
        """Create ASS style definitions for advanced highlighting"""
        font = self.subtitle_config.get("font", "Arial")
        font_size = self.subtitle_config.get("font_size", 48)
        font_color = self.subtitle_config.get("font_color", "white")
        position = self.subtitle_config.get("position", "bottom")
        highlight_color = self.subtitle_config.get("highlight_color", "#FF6B6B")
        has_bg = self.subtitle_config.get("background", True)
        bg_color = self.subtitle_config.get("background_color", "#00000080")

        # Convert colors to ASS format (AABBGGRR)
        font_color_ass = self._convert_color_to_ass(font_color)
        highlight_color_ass = self._convert_color_to_ass(highlight_color)
        bg_color_ass = self._convert_color_to_ass(bg_color) if has_bg else "00000000"

        # Position: 0=top, 2=bottom (ASS coordinates)
        margin_v = 30 if position == "bottom" else 120

        styles = f"""[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{font_color_ass},00000000,{font_color_ass},{bg_color_ass},0,0,0,0,100,100,0,0,1,2,2,{self._get_ass_alignment(position)},10,10,{margin_v},1
Style: Highlighted,{font},{font_size},{highlight_color_ass},00000000,{highlight_color_ass},{bg_color_ass},1,0,0,0,100,100,0,0,1,2,2,{self._get_ass_alignment(position)},10,10,{margin_v},1
"""
        return styles

    def _convert_color_to_ass(self, color: str) -> str:
        """Convert HTML color to ASS format (AABBGGRR)"""
        if color.startswith("#"):
            color = color[1:]

        if len(color) == 6:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            return f"00{b:02x}{g:02x}{r:02x}"

        return "00FFFFFF"  # Default white

    def _get_ass_alignment(self, position: str) -> int:
        """Get ASS alignment value (2=bottom center, 8=top center, etc.)"""
        mapping = {
            "top": 8,
            "top_center": 8,
            "bottom": 2,
            "bottom_center": 2,
            "center": 5,
        }
        return mapping.get(position, 2)

    def burn_subtitles(
        self, video_path: str, subtitle_path: str, output_path: str
    ) -> bool:
        """Burn subtitles into video using FFmpeg"""
        if not os.path.exists(video_path):
            self.logger.error(f"Video not found: {video_path}")
            return False

        if not os.path.exists(subtitle_path):
            self.logger.error(f"Subtitle file not found: {subtitle_path}")
            return False

        try:
            # Copy SRT to output directory with simple name to avoid path issues
            absolute_video_path = os.path.abspath(video_path)
            absolute_output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(absolute_output_path)
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(output_path).stem)
            simple_srt_name = f"{safe_stem}_subtitles.srt"
            simple_srt_path = os.path.join(output_dir, simple_srt_name)

            # Copy SRT to simple path
            import shutil

            shutil.copy2(os.path.abspath(subtitle_path), simple_srt_path)

            cmd = [
                "ffmpeg",
                "-i",
                absolute_video_path,
                "-vf",
                f"subtitles={simple_srt_name}",
                "-c:a",
                "copy",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-y",
                absolute_output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)

            if result.returncode == 0:
                self.logger.info(f"Burned subtitles into: {output_path}")
                return True
            else:
                self.logger.error(f"Subtitle burning failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Error burning subtitles: {e}")
            return False
        finally:
            if "simple_srt_path" in locals() and os.path.exists(simple_srt_path):
                try:
                    os.remove(simple_srt_path)
                except OSError:
                    pass

    def generate_subtitles_for_clip(
        self,
        video_path: str,
        transcription: List[Dict[str, Any]],
        clip_info: Dict[str, Any],
        temp_dir: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate subtitles and optionally burn them into the clip

        Returns:
            Tuple of (srt_path, burned_video_path) - either may be None
        """
        if not self.is_enabled():
            return None, None

        clip_start = clip_info.get("start_time", 0)
        clip_end = clip_info.get("end_time", 0)

        # Generate SRT
        video_name = Path(video_path).stem
        clip_id = clip_info.get("clip_id", 0)
        srt_path = os.path.join(temp_dir, f"{video_name}_clip_{clip_id}_subtitles.srt")

        srt_path = self.generate_subtitle_file(
            transcription, srt_path, clip_start_time=clip_start, clip_end_time=clip_end
        )

        if srt_path is None:
            return None, None

        return srt_path, None  # Actual burning will be done by clip_generator

    def create_segment_subtitles(
        self, transcription: List[Dict[str, Any]], start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """Extract subtitle segments for a specific time range"""
        segments = []

        for segment in transcription:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)

            # Check if segment overlaps with time range
            if seg_start < end_time and seg_end > start_time:
                segments.append(
                    {
                        "start": max(start_time, seg_start),
                        "end": min(end_time, seg_end),
                        "text": segment.get("text", ""),
                        "words": segment.get("words", []),
                    }
                )

        return segments
