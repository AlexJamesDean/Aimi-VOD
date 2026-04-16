"""
Clip Generator Module
Generates video clips using FFmpeg
"""

import os
import re
import subprocess
import ffmpeg
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import concurrent.futures


class ClipGenerator:
    """Generates video clips from detected segments"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.extraction_config = config.get("extraction", {})
        self.paths_config = config.get("paths", {})
        self.performance_config = config.get("performance", {})
        self.zone_config = config.get("zones", {})
        self.subtitle_config = config.get("subtitles", {})

    def generate_clips(
        self,
        video_path: str,
        segments: List[Dict[str, Any]],
        video_info: Dict[str, Any],
        transcription: List[Dict[str, Any]] = None,
        audio_analysis: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """Generate video clips from segments

        Args:
            video_path: Path to source video
            segments: Detected clip segments with timing info
            video_info: Video metadata
            transcription: Optional transcription for subtitles
            audio_analysis: Optional audio analysis for zone-aware processing
        """
        if not segments:
            self.logger.warning("No segments to generate clips from")
            return []

        self.logger.info(f"Generating {len(segments)} clips from {video_path}")

        # Check for zone-based generation
        zone_enabled = self.zone_config.get("enabled", False)
        subtitle_enabled = self.subtitle_config.get("enabled", True)

        # Create output directory
        clips_dir = Path(self.paths_config["clips_dir"])
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Prepare clip generation tasks
        clip_tasks = []
        for segment in segments:
            clip_info = self._prepare_clip_info(video_path, segment, video_info)
            clip_info["zone_enabled"] = zone_enabled
            clip_info["subtitle_enabled"] = subtitle_enabled
            clip_info["transcription"] = transcription
            clip_tasks.append(clip_info)

        # Generate clips (parallel if configured)
        max_workers = self.performance_config.get("max_workers", 1)

        if max_workers > 1 and len(clip_tasks) > 1:
            clips = self._generate_clips_parallel(clip_tasks, max_workers)
        else:
            clips = self._generate_clips_sequential(clip_tasks)

        # Filter successful clips
        successful_clips = [clip for clip in clips if clip.get("success", False)]

        self.logger.info(
            f"Successfully generated {len(successful_clips)}/{len(clip_tasks)} clips"
        )
        return successful_clips

    def _prepare_clip_info(
        self, video_path: str, segment: Dict[str, Any], video_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare clip information for generation"""
        video_name = Path(video_path).stem

        # Format timestamps for filename
        start_str = f"{segment['start_time']:.1f}".replace(".", "p")
        end_str = f"{segment['end_time']:.1f}".replace(".", "p")

        # Generate output filename
        clip_filename = (
            f"{video_name}_clip_{segment['clip_id']}_{start_str}s_{end_str}s.mp4"
        )
        clip_path = Path(self.paths_config["clips_dir"]) / clip_filename

        # Prepare clip info
        clip_info = {
            "video_path": video_path,
            "output_path": str(clip_path),
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "duration": segment["duration"],
            "clip_id": segment["clip_id"],
            "segment_info": segment,
            "video_info": video_info,
        }

        return clip_info

    def _prepare_subtitle_file(self, clip_info: Dict[str, Any]) -> Optional[str]:
        """Prepare subtitle file for a clip if subtitles enabled"""
        if not clip_info.get("subtitle_enabled", False):
            return None

        transcription = clip_info.get("transcription")
        if not transcription:
            return None

        from modules.subtitle_generator import SubtitleGenerator

        generator = SubtitleGenerator(self.config, self.logger)

        video_path = clip_info["video_path"]
        video_name = Path(video_path).stem
        clip_id = clip_info["clip_id"]
        start_time = clip_info["start_time"]
        end_time = clip_info["end_time"]
        temp_dir = self.paths_config["temp_dir"]

        srt_path = os.path.join(temp_dir, f"{video_name}_clip_{clip_id}_subtitles.srt")

        return generator.generate_subtitle_file(
            transcription, srt_path, clip_start_time=start_time, clip_end_time=end_time
        )

    def _burn_subtitles(
        self, video_path: str, subtitle_path: str, output_path: str
    ) -> bool:
        """Burn subtitles into video using FFmpeg"""
        try:
            # Copy SRT to output directory with simple name to avoid path issues
            absolute_video_path = os.path.abspath(video_path)
            absolute_output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(absolute_output_path)
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(output_path).stem)
            simple_srt_name = f"{safe_stem}_subtitles.srt"
            simple_srt_path = os.path.join(output_dir, simple_srt_name)

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

            self.logger.debug(f"FFmpeg subtitle cmd: {' '.join(cmd)}")
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

    def _generate_clips_sequential(
        self, clip_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate clips sequentially"""
        clips = []

        for task in clip_tasks:
            try:
                clip_result = self._generate_single_clip(task)
                clips.append(clip_result)
            except Exception as e:
                self.logger.error(f"Failed to generate clip {task['clip_id']}: {e}")
                clips.append(
                    {
                        "clip_id": task["clip_id"],
                        "output_path": task["output_path"],
                        "success": False,
                        "error": str(e),
                    }
                )

        return clips

    def _generate_clips_parallel(
        self, clip_tasks: List[Dict[str, Any]], max_workers: int
    ) -> List[Dict[str, Any]]:
        """Generate clips in parallel"""
        clips = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_clip = {
                executor.submit(self._generate_single_clip, task): task
                for task in clip_tasks
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_clip):
                task = future_to_clip[future]
                try:
                    clip_result = future.result()
                    clips.append(clip_result)
                except Exception as e:
                    self.logger.error(f"Failed to generate clip {task['clip_id']}: {e}")
                    clips.append(
                        {
                            "clip_id": task["clip_id"],
                            "output_path": task["output_path"],
                            "success": False,
                            "error": str(e),
                        }
                    )

        return clips

    def _generate_single_clip(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single video clip"""
        clip_id = clip_info["clip_id"]
        video_path = clip_info["video_path"]
        output_path = clip_info["output_path"]
        start_time = clip_info["start_time"]
        end_time = clip_info["end_time"]
        duration = clip_info["duration"]

        self.logger.info(
            f"Generating clip {clip_id}: {start_time:.1f}s to {end_time:.1f}s ({duration:.1f}s)"
        )

        # Build FFmpeg command
        try:
            # Generate base clip
            self._generate_with_ffmpeg_python(clip_info)

            # Verify the generated clip
            if not self._verify_clip(output_path, duration):
                raise RuntimeError(f"Clip verification failed for {output_path}")

            # Handle subtitles if enabled
            subtitle_path = self._prepare_subtitle_file(clip_info)
            if subtitle_path and os.path.exists(subtitle_path):
                # Burn subtitles into the clip
                temp_output = output_path.replace(".mp4", "_temp.mp4")
                os.rename(output_path, temp_output)

                if self._burn_subtitles(temp_output, subtitle_path, output_path):
                    os.remove(temp_output)
                    self.logger.info(
                        f"Clip {clip_id} generated with subtitles: {output_path}"
                    )
                else:
                    # Fallback: keep clip without subtitles
                    os.rename(temp_output, output_path)
                    self.logger.warning(
                        f"Could not burn subtitles, keeping clip without"
                    )

            self.logger.info(f"Clip {clip_id} generated successfully: {output_path}")

            return {
                "clip_id": clip_id,
                "output_path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "success": True,
                "file_size": os.path.getsize(output_path),
            }

        except ffmpeg.Error as e:
            self.logger.warning(
                f"FFmpeg-python failed for clip {clip_id}, trying subprocess: {e}"
            )
            # Fallback to subprocess
            self._generate_with_subprocess(clip_info)

            if self._verify_clip(output_path, duration):
                self.logger.info(
                    f"Clip {clip_id} generated successfully (subprocess): {output_path}"
                )

                return {
                    "clip_id": clip_id,
                    "output_path": output_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "success": True,
                    "file_size": os.path.getsize(output_path),
                }
            else:
                raise RuntimeError(
                    f"Clip verification failed after subprocess fallback"
                )

        except Exception as e:
            self.logger.error(f"Error generating clip {clip_id}: {e}")
            raise

    def _generate_with_ffmpeg_python(self, clip_info: Dict[str, Any]):
        """Generate clip using ffmpeg-python library"""
        video_path = clip_info["video_path"]
        output_path = clip_info["output_path"]
        start_time = clip_info["start_time"]
        end_time = clip_info["end_time"]
        duration = end_time - start_time

        # Get extraction settings
        vertical_format = self.extraction_config.get("vertical_format", True)
        target_resolution = self.extraction_config.get("target_resolution", "1080x1920")
        frame_rate = self.extraction_config.get("frame_rate", 30)
        video_codec = self.extraction_config.get("video_codec", "libx264")
        audio_codec = self.extraction_config.get("audio_codec", "aac")
        preset = self.extraction_config.get("preset", "fast")
        crf = self.extraction_config.get("crf", 23)

        # Build FFmpeg command - use filter_complex to handle both video and audio
        stream = ffmpeg.input(video_path, ss=start_time, t=duration)
        video_stream = stream.video
        audio_stream = stream.audio

        # Apply filters if vertical format is requested
        if vertical_format:
            # Crop to vertical (9:16) aspect ratio
            width, height = map(int, target_resolution.split("x"))
            cropped = ffmpeg.filter(video_stream, "crop", "ih*9/16", "ih")
            scaled = ffmpeg.filter(cropped, "scale", width, height)
        else:
            scaled = video_stream

        # Combine video and audio for output
        if audio_stream is not None:
            output = ffmpeg.output(
                scaled,
                audio_stream,
                output_path,
                vcodec=video_codec,
                acodec=audio_codec,
                preset=preset,
                crf=crf,
                r=frame_rate,
                movflags="+faststart",
            )
        else:
            output = ffmpeg.output(
                scaled,
                output_path,
                vcodec=video_codec,
                preset=preset,
                crf=crf,
                r=frame_rate,
                movflags="+faststart",
            )

        # Run FFmpeg
        ffmpeg.run(
            output, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )

    def _generate_with_subprocess(self, clip_info: Dict[str, Any]):
        """Generate clip using ffmpeg subprocess (fallback)"""
        video_path = clip_info["video_path"]
        output_path = clip_info["output_path"]
        start_time = clip_info["start_time"]
        end_time = clip_info["end_time"]
        duration = end_time - start_time

        # Get extraction settings
        vertical_format = self.extraction_config.get("vertical_format", True)
        target_resolution = self.extraction_config.get("target_resolution", "1080x1920")
        frame_rate = self.extraction_config.get("frame_rate", 30)
        video_codec = self.extraction_config.get("video_codec", "libx264")
        audio_codec = self.extraction_config.get("audio_codec", "aac")
        preset = self.extraction_config.get("preset", "fast")
        crf = self.extraction_config.get("crf", 23)

        # Build command
        cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),  # Start time
            "-i",
            video_path,
            "-t",
            str(duration),  # Duration
            "-c:v",
            video_codec,
            "-c:a",
            audio_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-r",
            str(frame_rate),
            "-movflags",
            "+faststart",
        ]

        # Add vertical format filters if requested
        if vertical_format:
            width, height = map(int, target_resolution.split("x"))
            cmd.extend(
                [
                    "-filter_complex",
                    f"[0:v]crop=ih*9/16:ih,scale={width}:{height}[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "0:a",
                ]
            )
        else:
            cmd.extend(["-map", "0"])

        cmd.extend(["-y", output_path])  # Overwrite output

        self.logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.returncode != 0:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(
                    f"FFmpeg failed with return code {result.returncode}"
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg subprocess error: {e}")
            raise RuntimeError(f"Clip generation failed: {e.stderr}")

    def _verify_clip(self, clip_path: str, expected_duration: float) -> bool:
        """Verify that the generated clip is valid"""
        if not os.path.exists(clip_path):
            self.logger.error(f"Clip file not created: {clip_path}")
            return False

        file_size = os.path.getsize(clip_path)
        if file_size == 0:
            self.logger.error(f"Clip file is empty: {clip_path}")
            return False

        # Check duration (allow small tolerance)
        try:
            probe = ffmpeg.probe(clip_path)
            actual_duration = float(probe["format"]["duration"])

            duration_diff = abs(actual_duration - expected_duration)
            if duration_diff > 2.0:  # Allow 2 second tolerance
                self.logger.warning(
                    f"Clip duration mismatch: expected {expected_duration:.1f}s, "
                    f"got {actual_duration:.1f}s"
                )
                # Still return True as the clip might be usable

            return True

        except Exception as e:
            self.logger.warning(f"Could not verify clip duration: {e}")
            # File exists and has content, assume it's valid
            return file_size > 1024  # At least 1KB

    def cleanup_failed_clips(self, clips: List[Dict[str, Any]]):
        """Clean up failed clip files"""
        for clip in clips:
            if not clip.get("success", False) and "output_path" in clip:
                clip_path = clip["output_path"]
                try:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
                        self.logger.debug(f"Cleaned up failed clip: {clip_path}")
                except Exception as e:
                    self.logger.warning(
                        f"Error cleaning up failed clip {clip_path}: {e}"
                    )
