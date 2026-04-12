#!/usr/bin/env python3
"""
Automated VOD-to-Shorts Clipper (Enhanced)
Main application entry point with multimodal detection
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.input_handler import InputHandler
from modules.audio_extractor import AudioExtractor
from modules.transcriber import Transcriber
from modules.audio_analyzer import AudioAnalyzer
from modules.detector import SegmentDetector
from modules.clip_generator import ClipGenerator


class VodClipper:
    """Main VOD-to-Shorts Clipper application (Enhanced)"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the VOD Clipper with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()

        # Initialize components
        self.input_handler = InputHandler(self.config, self.logger)
        self.audio_extractor = AudioExtractor(self.config, self.logger)
        self.transcriber = Transcriber(self.config, self.logger)
        self.audio_analyzer = AudioAnalyzer(self.config, self.logger)
        self.detector = SegmentDetector(self.config, self.logger)
        self.clip_generator = ClipGenerator(self.config, self.logger)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self._validate_config(config)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            print("Please ensure config/config.yaml exists.")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure"""
        required_sections = [
            "whisper",
            "audio",
            "detection",
            "buffering",
            "extraction",
            "paths",
            "performance",
            "logging",
        ]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration"""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO").upper())

        logger = logging.getLogger("vod_clipper")
        logger.setLevel(log_level)

        # Remove existing handlers
        logger.handlers.clear()

        # Console handler
        if log_config.get("console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_config.get("file_output", True):
            logs_dir = Path(self.config["paths"]["logs_dir"])
            logs_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = log_config.get("log_filename", "vod_clipper_{timestamp}.log")
            log_filename = log_filename.format(timestamp=timestamp)

            log_file = logs_dir / log_filename
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def process_vod(self, video_path: str) -> Dict[str, Any]:
        """Process a VOD file and generate clips (Enhanced with multimodal detection)"""
        self.logger.info(f"Starting processing of: {video_path}")

        results = {
            "input_file": video_path,
            "clips_generated": 0,
            "clips": [],
            "errors": [],
            "processing_time": None,
        }

        start_time = datetime.now()

        try:
            # Step 1: Validate and analyze input video
            self.logger.info("Step 1: Validating input video...")
            video_info = self.input_handler.validate_and_analyze(video_path)
            results["video_info"] = video_info
            self.logger.info(f"Video info: {video_info}")

            # Step 2: Extract audio
            self.logger.info("Step 2: Extracting audio...")
            audio_path = self.audio_extractor.extract_audio(video_path, video_info)
            results["audio_path"] = audio_path

            # Step 3: Transcribe audio with word-level timestamps
            self.logger.info("Step 3: Transcribing audio...")
            transcription = self.transcriber.transcribe(audio_path)
            results["transcription"] = transcription

            # Save transcription for debugging
            transcription_file = (
                Path(self.config["paths"]["temp_dir"])
                / f"transcription_{Path(video_path).stem}.json"
            )
            with open(transcription_file, "w", encoding="utf-8") as f:
                json.dump(transcription, f, indent=2)

            # Step 3.5: Audio analysis (energy, VAD, sentiment) - NEW
            audio_analysis_enabled = self.config.get("audio_analysis", {}).get(
                "enabled", True
            )
            audio_analysis = None

            if audio_analysis_enabled:
                self.logger.info(
                    "Step 3.5: Analyzing audio (energy, VAD, sentiment)..."
                )
                try:
                    audio_analysis = self.audio_analyzer.analyze_audio(audio_path)
                    self.logger.info(
                        f"Audio analysis: {len(audio_analysis.get('energy_spikes', []))} energy spikes, "
                        f"{len(audio_analysis.get('speech_segments', []))} speech segments"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Audio analysis failed (continuing without): {e}"
                    )
                    audio_analysis = None

            # Step 4: Detect clippable segments with multimodal scoring - ENHANCED
            self.logger.info("Step 4: Detecting clippable segments (multimodal)...")
            segments = self.detector.detect_segments(transcription, audio_analysis)
            results["detected_segments"] = segments
            self.logger.info(f"Detected {len(segments)} segments")

            # Step 5: Apply buffering (speech-aware if enabled) - ENHANCED
            self.logger.info("Step 5: Applying buffering logic...")

            speech_aware_enabled = self.config.get("speech_aware", {}).get(
                "enabled", True
            )

            if speech_aware_enabled and transcription:
                buffered_segments = self.detector.apply_speech_aware_buffering(
                    segments, transcription, video_info["duration"], audio_analysis
                )
            else:
                buffered_segments = self.detector.apply_buffering(
                    segments, video_info["duration"]
                )

            results["buffered_segments"] = buffered_segments

            # Step 6: Generate clips with subtitles
            self.logger.info("Step 6: Generating clips...")
            clips = self.clip_generator.generate_clips(
                video_path,
                buffered_segments,
                video_info,
                transcription=transcription,
                audio_analysis=audio_analysis,
            )
            results["clips"] = clips
            results["clips_generated"] = len(clips)

            # Step 7: Clean up temporary files
            if self.config["performance"]["cleanup_temp"]:
                self.logger.info("Step 7: Cleaning up temporary files...")
                self._cleanup_temp_files(audio_path, transcription_file)

        except Exception as e:
            self.logger.error(f"Error processing VOD: {e}", exc_info=True)
            results["errors"].append(str(e))

        # Calculate processing time
        end_time = datetime.now()
        results["processing_time"] = str(end_time - start_time)

        # Save results summary
        self._save_results_summary(results, video_path)

        self.logger.info(
            f"Processing completed. Generated {results['clips_generated']} clips."
        )
        self.logger.info(f"Total processing time: {results['processing_time']}")

        return results

    def _cleanup_temp_files(self, audio_path: str, transcription_file: Path):
        """Clean up temporary files"""
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                self.logger.debug(f"Removed audio file: {audio_path}")

            if transcription_file.exists():
                transcription_file.unlink()
                self.logger.debug(f"Removed transcription file: {transcription_file}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up temporary files: {e}")

    def _save_results_summary(self, results: Dict[str, Any], video_path: str):
        """Save processing results summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "input_file": results["input_file"],
            "clips_generated": results["clips_generated"],
            "processing_time": results["processing_time"],
            "video_info": results.get("video_info", {}),
            "detected_segments": len(results.get("detected_segments", [])),
            "clips": results.get("clips", []),
            "errors": results.get("errors", []),
        }

        summary_file = (
            Path(self.config["paths"]["logs_dir"])
            / f"summary_{Path(video_path).stem}.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Results summary saved to: {summary_file}")

    def batch_process(self, input_dir: Optional[str] = None):
        """Process all videos in the input directory"""
        if input_dir is None:
            input_dir = self.config["paths"]["input_dir"]

        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory not found: {input_dir}")
            return

        video_files = list(input_path.glob("*.*"))
        video_extensions = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv"}
        video_files = [f for f in video_files if f.suffix.lower() in video_extensions]

        if not video_files:
            self.logger.warning(f"No video files found in: {input_dir}")
            return

        self.logger.info(f"Found {len(video_files)} video files to process")

        for video_file in video_files:
            self.logger.info(f"Processing: {video_file.name}")
            try:
                self.process_vod(str(video_file))
            except Exception as e:
                self.logger.error(f"Failed to process {video_file.name}: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Automated VOD-to-Shorts Clipper")
    parser.add_argument("--video", type=str, help="Path to video file to process")
    parser.add_argument(
        "--batch", action="store_true", help="Process all videos in input directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input-dir", type=str, help="Custom input directory for batch processing"
    )

    args = parser.parse_args()

    # Initialize the clipper
    clipper = VodClipper(args.config)

    if args.video:
        # Process single video
        clipper.process_vod(args.video)
    elif args.batch:
        # Batch process
        clipper.batch_process(args.input_dir)
    else:
        # Interactive mode
        print("VOD-to-Shorts Clipper (Enhanced)")
        print("=" * 50)

        video_path = input(
            "Enter path to video file (or press Enter to process all in input directory): "
        ).strip()

        if video_path:
            if os.path.exists(video_path):
                clipper.process_vod(video_path)
            else:
                print(f"File not found: {video_path}")
        else:
            clipper.batch_process()


if __name__ == "__main__":
    main()
