"""
Audio Analyzer Module
Analyzes audio for energy spikes, voice activity detection, and sentiment
"""

import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False


class AudioAnalyzer:
    """Analyzes audio for energy, VAD, and sentiment scoring"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.audio_config = config.get("audio", {})
        self.analysis_config = config.get("audio_analysis", {})
        self.vad_model = None
        self.vad_utils = None

        # Analysis parameters
        self.energy_threshold = self.analysis_config.get("energy_threshold", 0.1)
        self.min_speech_duration = self.analysis_config.get("min_speech_duration", 0.3)

    def load_vad_model(self):
        """Load Silero VAD model using the official API"""
        if not SILERO_AVAILABLE:
            self.logger.warning("Silero VAD not available (torch/silero-vad missing)")
            return False

        try:
            torch.set_num_threads(1)

            # Use the official Silero VAD API from PyPI docs
            model = load_silero_vad()
            self.vad_model = model

            # get_speech_timestamps is the primary function we need
            self.logger.info("Silero VAD model loaded successfully")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load Silero VAD: {e}")
            return False

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Perform complete audio analysis"""
        self.logger.info(f"Analyzing audio: {audio_path}")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        results = {
            "energy_timeline": [],
            "vad_timeline": [],
            "sentiment_timeline": [],
            "speech_segments": [],
            "energy_spikes": [],
            "has_speech": False,
        }

        try:
            # Load audio data
            audio_data, sample_rate = self._load_audio(audio_path)

            # 1. Calculate RMS energy timeline
            results["energy_timeline"] = self._calculate_energy_timeline(
                audio_data, sample_rate
            )

            # 2. Detect energy spikes
            results["energy_spikes"] = self._detect_energy_spikes(
                results["energy_timeline"]
            )

            # 3. Load VAD and detect speech
            if self.vad_model is None:
                self.load_vad_model()

            if self.vad_model is not None:
                results["speech_segments"] = self._detect_speech_vad(
                    audio_path, sample_rate
                )
                results["has_speech"] = len(results["speech_segments"]) > 0
                results["vad_timeline"] = self._create_vad_timeline(
                    results["speech_segments"], results["energy_timeline"]
                )

            # 4. Calculate sentiment
            results["sentiment_timeline"] = self._calculate_sentiment(
                results["energy_timeline"], results.get("vad_timeline", [])
            )

            self.logger.info(
                f"Audio analysis complete: {len(results['energy_spikes'])} energy spikes, "
                f"{len(results['speech_segments'])} speech segments"
            )

        except Exception as e:
            self.logger.error(f"Audio analysis error: {e}", exc_info=True)

        return results

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file as numpy array"""
        try:
            # Try soundfile first (most reliable cross-platform)
            import soundfile as sf

            audio_data, sample_rate = sf.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            return audio_data, sample_rate
        except Exception:
            try:
                # Fallback: pydub
                from pydub import AudioSegment

                audio_segment = AudioSegment.from_file(audio_path)
                audio_data = np.array(
                    audio_segment.get_array_of_samples(), dtype=np.float32
                )
                if audio_segment.channels > 1:
                    audio_data = audio_data.reshape((-1, audio_segment.channels)).mean(
                        axis=1
                    )
                sample_rate = audio_segment.frame_rate
                return audio_data, sample_rate
            except Exception as e:
                self.logger.error(f"Error loading audio: {e}")
                raise

    def _load_wav_simple(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Simple WAV loader fallback"""
        import wave

        with wave.open(audio_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(-1)
            audio_data = (
                np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            )

        return audio_data, sample_rate

    def _calculate_energy_timeline(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Calculate RMS energy over time windows"""
        window_duration = 0.1  # 100ms windows
        window_size = int(sample_rate * window_duration)

        energy_timeline = []

        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i : i + window_size]
            rms = np.sqrt(np.mean(window**2))

            time_start = i / sample_rate
            time_end = (i + window_size) / sample_rate

            energy_timeline.append(
                {
                    "start": time_start,
                    "end": time_end,
                    "rms": float(rms),
                    "normalized": float(rms * 10),  # Scale for easier thresholding
                }
            )

        return energy_timeline

    def _detect_energy_spikes(
        self, energy_timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect moments of high audio energy"""
        spikes = []

        if not energy_timeline:
            return spikes

        # Calculate dynamic threshold based on mean
        rms_values = [e["rms"] for e in energy_timeline]
        mean_rms = np.mean(rms_values)
        std_rms = np.std(rms_values)

        # Spike threshold: mean + 2*std or fixed threshold, whichever is higher
        threshold = max(self.energy_threshold, mean_rms + 2 * std_rms)

        consecutive_high = 0
        spike_start = None

        for i, entry in enumerate(energy_timeline):
            if entry["rms"] > threshold:
                if spike_start is None:
                    spike_start = entry["start"]
                consecutive_high += 1
            else:
                if consecutive_high >= 3:  # At least 300ms of high energy
                    spike_end = energy_timeline[i - 1]["end"]
                    spikes.append(
                        {
                            "start": spike_start,
                            "end": spike_end,
                            "duration": spike_end - spike_start,
                            "peak_rms": max(rms_values[i - consecutive_high : i]),
                        }
                    )
                consecutive_high = 0
                spike_start = None

        # Handle spike at end of file
        if consecutive_high >= 3:
            spike_end = energy_timeline[-1]["end"]
            spikes.append(
                {
                    "start": spike_start,
                    "end": spike_end,
                    "duration": spike_end - spike_start,
                    "peak_rms": max(rms_values[-consecutive_high:]),
                }
            )

        self.logger.debug(
            f"Detected {len(spikes)} energy spikes above threshold {threshold:.3f}"
        )
        return spikes

    def _detect_speech_vad(
        self, audio_path: str, sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Detect speech using Silero VAD"""
        if self.vad_model is None:
            return []

        try:
            # Use soundfile to load audio (avoids torchaudio/torchcodec dependency issues)
            import soundfile as sf

            audio_data, sr = sf.read(audio_path, dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to 16 kHz if needed without pulling in torchaudio.
            if sr != 16000:
                target_length = max(1, int(round(len(audio_data) * 16000 / sr)))
                source_positions = np.linspace(
                    0, len(audio_data) - 1, num=len(audio_data)
                )
                target_positions = np.linspace(
                    0, len(audio_data) - 1, num=target_length
                )
                audio_data = np.interp(
                    target_positions, source_positions, audio_data
                ).astype(np.float32)

            audio_tensor = torch.from_numpy(audio_data.copy())

            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=16000,
                return_seconds=True,
                min_speech_duration_ms=max(1, int(self.min_speech_duration * 1000)),
            )

            speech_segments = []
            for ts in speech_timestamps:
                speech_segments.append(
                    {
                        "start": ts["start"],
                        "end": ts["end"],
                        "duration": ts["end"] - ts["start"],
                    }
                )

            return speech_segments
        except Exception as e:
            self.logger.warning(f"VAD detection error: {e}")
            return []

    def _create_vad_timeline(
        self,
        speech_segments: List[Dict[str, Any]],
        energy_timeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Create VAD timeline aligned with energy timeline"""
        if not speech_segments or not energy_timeline:
            return energy_timeline

        vad_timeline = []

        for entry in energy_timeline:
            time = (entry["start"] + entry["end"]) / 2
            has_speech = any(
                seg["start"] <= time <= seg["end"] for seg in speech_segments
            )

            vad_entry = entry.copy()
            vad_entry["has_speech"] = has_speech
            vad_timeline.append(vad_entry)

        return vad_timeline

    def _calculate_sentiment(
        self, energy_timeline: List[Dict[str, Any]], vad_timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate sentiment/excitement score over time"""
        if not energy_timeline:
            return []

        window_config = self.config.get("audio_analysis", {}).get(
            "sentiment_window", 5.0
        )
        window_samples = max(
            1, int(window_config / 0.1)
        )  # Convert to number of 100ms windows

        sentiment_timeline = []

        # Calculate baseline energy
        all_rms = [e["rms"] for e in energy_timeline]
        baseline_rms = np.percentile(all_rms, 25)  # Use 25th percentile as baseline

        for i, entry in enumerate(energy_timeline):
            # Get window of energy values
            window_start = max(0, i - window_samples // 2)
            window_end = min(len(energy_timeline), i + window_samples // 2 + 1)
            window_rms = [
                energy_timeline[j]["rms"] for j in range(window_start, window_end)
            ]

            # Calculate excitement score relative to baseline
            window_mean = np.mean(window_rms)
            relative_energy = window_mean / (baseline_rms + 0.001)

            # Check VAD status if available
            has_speech = entry.get("has_speech", True) if vad_timeline else True

            # Excitement score: normalized relative energy
            # If speech is present and energy is high, score is higher
            excitement = min(relative_energy / 3.0, 1.0)  # Normalize to 0-1

            sentiment_timeline.append(
                {
                    "start": entry["start"],
                    "end": entry["end"],
                    "excitement": float(excitement),
                    "relative_energy": float(relative_energy),
                    "has_speech": has_speech,
                }
            )

        return sentiment_timeline

    def get_excitement_score(
        self,
        start_time: float,
        end_time: float,
        sentiment_timeline: List[Dict[str, Any]],
    ) -> float:
        """Get average excitement score for a time range"""
        relevant = [
            s
            for s in sentiment_timeline
            if s["start"] >= start_time and s["end"] <= end_time
        ]

        if not relevant:
            # Fallback: get any overlapping entries
            relevant = [
                s
                for s in sentiment_timeline
                if s["start"] < end_time and s["end"] > start_time
            ]

        if not relevant:
            return 0.0

        return float(np.mean([s["excitement"] for s in relevant]))

    def is_speech_present(
        self, start_time: float, end_time: float, speech_segments: List[Dict[str, Any]]
    ) -> bool:
        """Check if speech is present in a time range"""
        return any(
            seg["start"] <= end_time and seg["end"] >= start_time
            for seg in speech_segments
        )

    def get_sentence_boundaries(
        self, transcription: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract natural sentence boundaries from transcription"""
        if not transcription:
            return []

        boundaries = []

        for segment in transcription:
            text = segment.get("text", "")
            words = segment.get("words", [])

            # If we have word timestamps, use them for precise boundaries
            if words and len(words) > 0:
                # Find sentence ends based on punctuation
                for i, word_info in enumerate(words):
                    word = word_info.get("word", "")
                    if any(p in word for p in ".!?"):
                        boundaries.append(
                            {
                                "start": word_info.get("start", 0),
                                "end": word_info.get("end", 0),
                                "type": "sentence_end",
                            }
                        )
            else:
                # Fallback: use segment boundaries
                boundaries.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "type": "segment_boundary",
                    }
                )

        return boundaries

    def find_natural_end_point(
        self, start_time: float, target_end: float, transcription: List[Dict[str, Any]]
    ) -> float:
        """Find the nearest natural sentence end after target_end"""
        boundaries = self.get_sentence_boundaries(transcription)

        # Find first boundary after target_end
        for boundary in boundaries:
            if boundary["start"] >= target_end:
                return boundary["end"]

        # No boundary found, return target
        return target_end
