"""
Transcriber Module
Handles audio transcription using Whisper
"""

import os
import warnings
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("OpenAI Whisper not available, attempting to use faster-whisper")

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class Transcriber:
    """Handles audio transcription using Whisper or Faster-Whisper"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.whisper_config = config.get("whisper", {})
        self.model = None
        self.model_loaded = False

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio file and return timestamped segments"""
        self.logger.info(f"Transcribing audio: {audio_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load model if not already loaded
        if not self.model_loaded:
            self._load_model()

        try:
            # Transcribe based on available library
            if FASTER_WHISPER_AVAILABLE and self.model is not None:
                segments = self._transcribe_faster_whisper(audio_path)
            elif WHISPER_AVAILABLE and self.model is not None:
                segments = self._transcribe_openai_whisper(audio_path)
            else:
                raise RuntimeError(
                    "No Whisper implementation available. Install whisper or faster-whisper"
                )

            # Format segments as required
            formatted_segments = self._format_segments(segments)

            self.logger.info(
                f"Transcription complete. Found {len(formatted_segments)} segments"
            )
            return formatted_segments

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to transcribe audio: {e}")

    def _load_model(self):
        """Load Whisper model based on configuration"""
        model_name = self.whisper_config.get("model", "base")
        device = self.whisper_config.get("device", "cpu")
        compute_type = self.whisper_config.get("compute_type", "float32")

        self.logger.info(f"Loading Whisper model: {model_name} on {device}")

        # Try faster-whisper first (more efficient for long audio)
        if FASTER_WHISPER_AVAILABLE:
            try:
                self.model = WhisperModel(
                    model_name, device=device, compute_type=compute_type
                )
                self.logger.info(f"Loaded faster-whisper model: {model_name}")
                self.model_loaded = True
                return
            except Exception as e:
                self.logger.warning(f"Failed to load faster-whisper: {e}")

        # Fallback to OpenAI Whisper
        if WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model(model_name, device=device)
                self.logger.info(f"Loaded OpenAI Whisper model: {model_name}")
                self.model_loaded = True
                return
            except Exception as e:
                self.logger.error(f"Failed to load OpenAI Whisper: {e}")

        raise RuntimeError(
            "Could not load any Whisper model. Please install whisper or faster-whisper"
        )

    def _transcribe_faster_whisper(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe using faster-whisper"""
        self.logger.debug("Using faster-whisper for transcription")

        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            best_of=5,
            word_timestamps=True,
            vad_filter=True,  # Voice Activity Detection filter
        )

        self.logger.debug(
            f"Detected language: {info.language}, probability: {info.language_probability}"
        )

        # Convert segments to list
        segments_list = []
        for segment in segments:
            segments_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in segment.words
                    ]
                    if segment.words
                    else [],
                }
            )

        return segments_list

    def _transcribe_openai_whisper(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper"""
        self.logger.debug("Using OpenAI Whisper for transcription")

        # Load audio
        audio = whisper.load_audio(audio_path)

        # Pad or trim audio to 30 seconds
        audio = whisper.pad_or_trim(audio)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Detect language
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        self.logger.debug(f"Detected language: {detected_lang}")

        # Decode options
        options = whisper.DecodingOptions(
            fp16=False,  # Use float32 for CPU
            language=detected_lang,
        )

        # Transcribe
        result = whisper.transcribe(self.model, audio_path, word_timestamps=True)

        # Format segments
        segments_list = []
        for segment in result["segments"]:
            segments_list.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", []),
                }
            )

        return segments_list

    def _format_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format segments into the required structure"""
        formatted = []

        for segment in segments:
            formatted_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }

            # Add word timestamps if available
            if "words" in segment and segment["words"]:
                formatted_segment["words"] = segment["words"]

            formatted.append(formatted_segment)

        return formatted

    def save_transcription(self, segments: List[Dict[str, Any]], output_path: str):
        """Save transcription to JSON file"""
        import json

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Transcription saved to: {output_path}")

    def chunk_transcribe(
        self, audio_path: str, chunk_duration: int = 600
    ) -> List[Dict[str, Any]]:
        """Transcribe long audio in chunks to manage memory"""
        self.logger.info(f"Chunk transcription for long audio: {audio_path}")

        # This is a simplified implementation
        # In a production system, you would:
        # 1. Split audio into chunks
        # 2. Transcribe each chunk
        # 3. Merge results with proper timestamp adjustment

        self.logger.warning(
            "Chunk transcription not fully implemented. Using normal transcription."
        )
        return self.transcribe(audio_path)
