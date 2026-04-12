"""
Zone Composer Module
Handles zone-based extraction and composition of video elements
"""

import os
import subprocess
import ffmpeg
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path


class ZoneComposer:
    """Handles zone-based extraction and composition"""

    # Position mappings for named positions
    POSITION_MAPPINGS = {
        "right_bottom": {"x": "W-tw", "y": "H-th"},
        "left_bottom": {"x": "0", "y": "H-th"},
        "right_top": {"x": "W-tw", "y": "0"},
        "left_top": {"x": "0", "y": "0"},
        "center": {"x": "(W-tw)/2", "y": "(H-th)/2"},
        "top_center": {"x": "(W-tw)/2", "y": "0"},
        "bottom_center": {"x": "(W-tw)/2", "y": "H-th"},
    }

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.zone_config = config.get("zones", {})
        self.extraction_config = config.get("extraction", {})

    def is_enabled(self) -> bool:
        """Check if zone composition is enabled"""
        return self.zone_config.get("enabled", False)

    def get_zone_definition(self, zone_name: str) -> Optional[Dict[str, Any]]:
        """Get zone configuration by name"""
        if zone_name in self.zone_config:
            zone = self.zone_config[zone_name]
            if zone.get("extract_enabled", True):
                return zone
        return None

    def get_video_resolution(self, video_path: str) -> Tuple[int, int]:
        """Get video resolution using ffprobe"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"), None
            )
            if video_stream:
                return int(video_stream["width"]), int(video_stream["height"])
        except Exception as e:
            self.logger.warning(f"Could not get video resolution: {e}")

        return 1920, 1080  # Default resolution

    def calculate_zone_crop(
        self,
        zone_name: str,
        video_resolution: Tuple[int, int],
        zone_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, int, int]:
        """Calculate crop parameters for a zone"""
        zone = self.get_zone_definition(zone_name)
        if zone is None:
            # Return center crop if zone not defined
            return self._center_crop_params(
                video_resolution, zone_resolution or video_resolution
            )

        video_w, video_h = video_resolution
        position_name = zone.get("position", "right_bottom")
        extract_res = zone.get("extract_resolution", "512x512")

        # Parse extract resolution
        if isinstance(extract_res, str) and "x" in extract_res:
            crop_w, crop_h = map(int, extract_res.split("x"))
        else:
            crop_w, crop_h = 512, 512  # Default

        # Calculate position
        if position_name in self.POSITION_MAPPINGS:
            pos = self.POSITION_MAPPINGS[position_name]
            x_expr = pos["x"].replace("W", str(video_w)).replace("H", str(video_h))
            y_expr = (
                pos["y"]
                .replace("W", str(video_w))
                .replace("H", str(video_h))
                .replace("tw", str(crop_w))
                .replace("th", str(crop_h))
            )

            # Evaluate expressions safely
            x = self._eval_expr(x_expr, video_w - crop_w)
            y = self._eval_expr(y_expr, video_h - crop_h)
        else:
            # Default to right bottom
            x = video_w - crop_w
            y = video_h - crop_h

        # Ensure bounds
        x = max(0, min(x, video_w - crop_w))
        y = max(0, min(y, video_h - crop_h))

        return x, y, crop_w, crop_h

    def _eval_expr(self, expr: str, default: int) -> int:
        """Safely evaluate a simple expression"""
        try:
            # Only allow basic math operations with numbers
            allowed = set("0123456789+-*/() ")
            if all(c in allowed for c in expr):
                result = eval(expr)
                return int(result)
        except:
            pass
        return default

    def _center_crop_params(
        self, video_res: Tuple[int, int], target_res: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Calculate center crop parameters"""
        video_w, video_h = video_res
        target_w, target_h = target_res

        # Calculate crop to make video match target aspect ratio
        video_aspect = video_w / video_h
        target_aspect = target_w / target_h

        if video_aspect > target_aspect:
            # Video is wider, crop sides
            crop_w = int(video_h * target_aspect)
            crop_h = video_h
            x = (video_w - crop_w) // 2
            y = 0
        else:
            # Video is taller, crop top/bottom
            crop_w = video_w
            crop_h = int(video_w / target_aspect)
            x = 0
            y = (video_h - crop_h) // 2

        return x, y, crop_w, crop_h

    def extract_zone_video(
        self, video_path: str, zone_name: str, output_path: str
    ) -> bool:
        """Extract a zone from video using FFmpeg"""
        try:
            video_w, video_h = self.get_video_resolution(video_path)
            x, y, crop_w, crop_h = self.calculate_zone_crop(
                zone_name, (video_w, video_h)
            )

            stream = ffmpeg.input(video_path)
            cropped = ffmpeg.filter(stream, "crop", crop_w, crop_h, x, y)
            output = ffmpeg.output(
                cropped, output_path, vcodec="libx264", acodec="copy", f="mp4"
            )
            ffmpeg.run(
                output, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )

            self.logger.info(
                f"Extracted zone '{zone_name}': ({x},{y}) {crop_w}x{crop_h} -> {output_path}"
            )
            return True

        except ffmpeg.Error as e:
            self.logger.error(f"Zone extraction failed: {e}")
            return False

    def compose_with_zones(
        self,
        video_path: str,
        zone_clips: List[Dict[str, Any]],
        output_path: str,
        composition_layout: str = "vtuber_top_game_full",
        target_resolution: str = "1080x1920",
    ) -> bool:
        """Compose final video with zone overlays"""
        try:
            width, height = map(int, target_resolution.split("x"))

            if composition_layout == "vtuber_top_game_full":
                return self._compose_vtuber_top_game(
                    video_path, zone_clips, output_path, width, height
                )
            elif composition_layout == "vtuber_pip":
                return self._compose_pip(
                    video_path, zone_clips, output_path, width, height
                )
            else:
                self.logger.warning(
                    f"Unknown layout '{composition_layout}', using simple vertical crop"
                )
                return self._simple_vertical_crop(
                    video_path, output_path, width, height
                )

        except Exception as e:
            self.logger.error(f"Composition failed: {e}", exc_info=True)
            return False

    def _compose_vtuber_top_game(
        self,
        video_path: str,
        zone_clips: List[Dict[str, Any]],
        output_path: str,
        width: int,
        height: int,
    ) -> bool:
        """Compose: VTuber at top, game view fills rest"""
        try:
            # Parse zone configs
            face_zone = self.zone_config.get("face", {})
            overlay_pos = face_zone.get("overlay_position", "top_center")
            overlay_scale = face_zone.get("overlay_scale", 1.5)
            overlay_opacity = face_zone.get("overlay_opacity", 1.0)

            # Get face overlay dimensions
            extract_res = face_zone.get("extract_resolution", "512x512")
            if isinstance(extract_res, str) and "x" in extract_res:
                face_w, face_h = map(int, extract_res.split("x"))
            else:
                face_w, face_h = 512, 512

            # Scale face for overlay
            overlay_w = int(face_w * overlay_scale)
            overlay_h = int(face_h * overlay_scale)

            # Position overlay
            if overlay_pos in self.POSITION_MAPPINGS:
                pos = self.POSITION_MAPPINGS[overlay_pos]
                x_expr = pos["x"].replace("W", str(width)).replace("H", str(height))
                y_expr = pos["y"].replace("W", str(width)).replace("H", str(height))
                y_expr = y_expr.replace("tw", str(overlay_w)).replace(
                    "th", str(overlay_h)
                )

                overlay_x = self._eval_expr(x_expr, (width - overlay_w) // 2)
                overlay_y = self._eval_expr(y_expr, 0)
            else:
                overlay_x = (width - overlay_w) // 2
                overlay_y = 0

            # Build FFmpeg filter graph
            # Main video: crop center to 9:16, scale to target
            main_stream = ffmpeg.input(video_path)

            # Calculate center crop for main video
            main_crop_w, main_crop_h = self._calc_9_16_crop((1920, 1080))
            main_cropped = ffmpeg.filter(
                main_stream,
                "crop",
                main_crop_w,
                main_crop_h,
                (1920 - main_crop_w) // 2,
                (1080 - main_crop_h) // 2,
            )
            main_scaled = ffmpeg.filter(main_cropped, "scale", width, height)

            # If we have a face zone video, overlay it
            if zone_clips:
                face_clip = next(
                    (c for c in zone_clips if c.get("zone") == "face"), None
                )
                if face_clip and os.path.exists(face_clip["path"]):
                    face_stream = ffmpeg.input(face_clip["path"])
                    face_scaled = ffmpeg.filter(
                        face_stream, "scale", overlay_w, overlay_h
                    )

                    # Overlay face on top of main
                    overlaid = ffmpeg.filter(
                        main_scaled, "overlay", overlay_x, overlay_y
                    )

                    output = ffmpeg.output(
                        overlaid,
                        output_path,
                        vcodec="libx264",
                        acodec="aac",
                        preset="fast",
                        crf=23,
                    )
                else:
                    output = ffmpeg.output(
                        main_scaled,
                        output_path,
                        vcodec="libx264",
                        acodec="aac",
                        preset="fast",
                        crf=23,
                    )
            else:
                output = ffmpeg.output(
                    main_scaled,
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    preset="fast",
                    crf=23,
                )

            ffmpeg.run(
                output, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )

            self.logger.info(
                f"Composed video with vtuber_top_game_full layout: {output_path}"
            )
            return True

        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg composition error: {e}")
            return False

    def _compose_pip(
        self,
        video_path: str,
        zone_clips: List[Dict[str, Any]],
        output_path: str,
        width: int,
        height: int,
    ) -> bool:
        """Compose: Picture-in-picture with VTuber in corner"""
        try:
            pip_config = self.zone_config.get("face", {})
            pip_scale = pip_config.get("overlay_scale", 0.3)
            pip_opacity = pip_config.get("overlay_opacity", 0.9)

            # Main video scaled to 70% with padding
            main_stream = ffmpeg.input(video_path)
            main_scaled = ffmpeg.filter(main_stream, "scale", width, height)

            if zone_clips:
                face_clip = next(
                    (c for c in zone_clips if c.get("zone") == "face"), None
                )
                if face_clip and os.path.exists(face_clip["path"]):
                    face_stream = ffmpeg.input(face_clip["path"])
                    face_scaled = ffmpeg.filter(
                        face_stream,
                        "scale",
                        int(width * pip_scale),
                        int(height * pip_scale),
                    )

                    # PIP in bottom-right corner
                    pip_x = width - int(width * pip_scale) - 20
                    pip_y = height - int(height * pip_scale) - 20

                    overlaid = ffmpeg.filter(main_scaled, "overlay", pip_x, pip_y)
                    output = ffmpeg.output(
                        overlaid,
                        output_path,
                        vcodec="libx264",
                        acodec="aac",
                        preset="fast",
                        crf=23,
                    )
                else:
                    output = ffmpeg.output(
                        main_scaled,
                        output_path,
                        vcodec="libx264",
                        acodec="aac",
                        preset="fast",
                        crf=23,
                    )
            else:
                output = ffmpeg.output(
                    main_scaled,
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    preset="fast",
                    crf=23,
                )

            ffmpeg.run(
                output, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )

            self.logger.info(f"Composed video with PIP layout: {output_path}")
            return True

        except ffmpeg.Error as e:
            self.logger.error(f"FFmpeg PIP composition error: {e}")
            return False

    def _simple_vertical_crop(
        self, video_path: str, output_path: str, width: int, height: int
    ) -> bool:
        """Simple vertical crop without zones"""
        try:
            main_stream = ffmpeg.input(video_path)
            main_crop_w, main_crop_h = self._calc_9_16_crop((1920, 1080))
            main_cropped = ffmpeg.filter(
                main_stream,
                "crop",
                main_crop_w,
                main_crop_h,
                (1920 - main_crop_w) // 2,
                (1080 - main_crop_h) // 2,
            )
            main_scaled = ffmpeg.filter(main_cropped, "scale", width, height)
            output = ffmpeg.output(
                main_scaled,
                output_path,
                vcodec="libx264",
                acodec="aac",
                preset="fast",
                crf=23,
            )
            ffmpeg.run(
                output, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )
            return True
        except ffmpeg.Error as e:
            self.logger.error(f"Simple crop error: {e}")
            return False

    def _calc_9_16_crop(self, video_res: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate center crop for 9:16 aspect ratio"""
        video_w, video_h = video_res
        target_ratio = 9 / 16

        video_ratio = video_w / video_h

        if video_ratio > target_ratio:
            # Video is wider, crop width
            crop_h = video_h
            crop_w = int(video_h * target_ratio)
        else:
            # Video is taller, crop height
            crop_w = video_w
            crop_h = int(video_w / target_ratio)

        return crop_w, crop_h

    def extract_zone_for_clip(
        self,
        video_path: str,
        zone_name: str,
        start_time: float,
        duration: float,
        temp_dir: str,
    ) -> Optional[str]:
        """Extract a zone segment for a specific clip"""
        try:
            zone = self.get_zone_definition(zone_name)
            if zone is None:
                return None

            # Generate output path
            video_name = Path(video_path).stem
            output_path = (
                Path(temp_dir) / f"zone_{zone_name}_{video_name}_{start_time:.1f}.mp4"
            )

            # Calculate crop for this zone
            video_w, video_h = self.get_video_resolution(video_path)
            x, y, crop_w, crop_h = self.calculate_zone_crop(
                zone_name, (video_w, video_h)
            )

            # Build FFmpeg command for extraction with segment
            cmd = [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-t",
                str(duration),
                "-vf",
                f"crop={crop_w}:{crop_h}:{x}:{y}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and output_path.exists():
                return str(output_path)
            else:
                self.logger.error(f"Zone extraction failed: {result.stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Zone extraction error: {e}")
            return None
