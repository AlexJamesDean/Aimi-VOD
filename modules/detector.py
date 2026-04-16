"""
Segment Detector Module (Enhanced)
Detects clippable segments based on multimodal scoring:
- Keyword matching
- Audio energy spikes
- Voice Activity Detection (speech presence)
- Sentiment analysis
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import logging
import numpy as np


class SegmentDetector:
    """Detects clippable segments using multimodal scoring"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.detection_config = config.get("detection", {})
        self.buffering_config = config.get("buffering", {})
        self.speech_config = config.get("speech_aware", {})
        self.multimodal_weights = self.detection_config.get("multimodal", {})

        # Load keywords
        self.keywords = self.detection_config.get("keywords", [])
        self.case_sensitive = self.detection_config.get("case_sensitive", False)
        self.min_confidence = self.detection_config.get("min_confidence", 0.5)
        self.min_combined_score = self.multimodal_weights.get("min_combined_score", 0.4)

        # Compile keyword patterns
        self._compile_keyword_patterns()

    def _compile_keyword_patterns(self):
        """Compile keyword patterns for efficient matching"""
        self.keyword_patterns = []

        for keyword in self.keywords:
            if not self.case_sensitive:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(keyword))
            self.keyword_patterns.append(pattern)

        self.logger.debug(f"Compiled {len(self.keyword_patterns)} keyword patterns")

    def detect_segments(
        self,
        transcription: List[Dict[str, Any]],
        audio_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Detect clippable segments with multimodal scoring

        Args:
            transcription: Whisper transcription with timestamps
            audio_analysis: Optional audio analysis results (energy, VAD, sentiment)
        """
        self.logger.info(
            f"Detecting clippable segments in {len(transcription)} transcript segments"
        )

        # Extract weights
        energy_weight = self.multimodal_weights.get("energy_weight", 0.25)
        speech_weight = self.multimodal_weights.get("speech_weight", 0.25)
        keyword_weight = self.multimodal_weights.get("keyword_weight", 0.30)
        sentiment_weight = self.multimodal_weights.get("sentiment_weight", 0.20)

        detected_segments = []
        energy_timeline = (
            audio_analysis.get("energy_timeline", []) if audio_analysis else []
        )
        sentiment_timeline = (
            audio_analysis.get("sentiment_timeline", []) if audio_analysis else []
        )
        speech_segments = (
            audio_analysis.get("speech_segments", []) if audio_analysis else []
        )
        energy_spikes = (
            audio_analysis.get("energy_spikes", []) if audio_analysis else []
        )

        # Use a sliding window so keywords that span two Whisper segments
        # still get matched, and short segments can be scored with neighbors' context.
        window_size = self.detection_config.get("window_size", 2)
        non_empty = [s for s in transcription if s.get("text", "").strip()]

        for i, segment in enumerate(non_empty):
            window = non_empty[i : i + window_size]
            text = " ".join(s.get("text", "") for s in window)
            start = window[0].get("start", 0)
            end = window[-1].get("end", 0)

            # Score components
            keyword_score = self._score_keywords(text)
            energy_score = self._score_energy_in_range(start, end, energy_spikes)
            speech_score = self._score_speech_presence(start, end, speech_segments)
            sentiment_score = self._score_sentiment_in_range(
                start, end, sentiment_timeline
            )

            # Combined weighted score
            combined_score = (
                keyword_score * keyword_weight
                + energy_score * energy_weight
                + speech_score * speech_weight
                + sentiment_score * sentiment_weight
            )

            matches = self._find_keyword_matches(text)

            # Accept if the combined multimodal score is strong enough, even
            # without a keyword match — a loud reaction is still clip-worthy.
            if combined_score >= self.min_combined_score:
                detected_segment = {
                    "start": start,
                    "end": end,
                    "trigger_text": text,
                    "keywords_found": matches,
                    "segment_text": text,
                    "confidence": combined_score,
                    "scores": {
                        "keyword": keyword_score,
                        "energy": energy_score,
                        "speech": speech_score,
                        "sentiment": sentiment_score,
                        "combined": combined_score,
                    },
                }
                detected_segments.append(detected_segment)

                self.logger.debug(
                    f"Segment at {start:.2f}-{end:.2f}s: "
                    f"combined={combined_score:.2f} "
                    f"(kw={keyword_score:.2f}, en={energy_score:.2f}, "
                    f"sp={speech_score:.2f}, se={sentiment_score:.2f})"
                )

        # Deduplicate overlapping sliding-window hits, keeping the highest-scoring.
        detected_segments = self._dedupe_overlapping(detected_segments)

        self.logger.info(f"Detected {len(detected_segments)} clippable segments")

        return detected_segments

    def _dedupe_overlapping(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collapse overlapping windowed detections, keeping the best-scored one."""
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: s["start"])
        kept: List[Dict[str, Any]] = [sorted_segs[0]]
        for seg in sorted_segs[1:]:
            last = kept[-1]
            if seg["start"] < last["end"]:
                if seg["confidence"] > last["confidence"]:
                    kept[-1] = seg
            else:
                kept.append(seg)
        return kept

    def _find_keyword_matches(self, text: str) -> List[str]:
        """Find keyword matches in text"""
        matches = []
        for i, pattern in enumerate(self.keyword_patterns):
            if pattern.search(text):
                matches.append(self.keywords[i])
        return matches

    def _score_keywords(self, text: str) -> float:
        """Score based on keyword density"""
        matches = self._find_keyword_matches(text)
        if not matches:
            return 0.0

        word_count = len(text.split())
        keyword_density = len(matches) / max(word_count, 1)

        # Boost for multiple keywords
        base_score = min(1.0, len(matches) * 0.3)

        # Boost for excitement indicators
        excitement = 0.0
        for char in ["!", "?", "..."]:
            if char in text:
                excitement += 0.1

        # Boost for short punchy phrases
        if 3 <= word_count <= 15:
            excitement += 0.2

        return min(1.0, base_score + excitement)

    def _score_energy_in_range(
        self, start: float, end: float, energy_spikes: List[Dict[str, Any]]
    ) -> float:
        """Score based on audio energy in time range"""
        if not energy_spikes:
            return 0.5  # Neutral if no data

        overlapping = [
            sp for sp in energy_spikes if sp["start"] < end and sp["end"] > start
        ]

        if not overlapping:
            return 0.2

        # Higher score for sustained energy spikes
        total_duration = sum(sp["end"] - sp["start"] for sp in overlapping)
        segment_duration = end - start
        coverage = total_duration / max(segment_duration, 1)

        # Peak RMS bonus
        peak_rms = max(sp.get("peak_rms", 0) for sp in overlapping)
        rms_bonus = min(peak_rms / 0.5, 1.0) * 0.3

        return min(1.0, coverage * 0.7 + rms_bonus)

    def _score_speech_presence(
        self, start: float, end: float, speech_segments: List[Dict[str, Any]]
    ) -> float:
        """Score based on speech presence (VAD)"""
        if not speech_segments:
            return 0.5  # Neutral if no VAD data

        overlapping = [
            sp for sp in speech_segments if sp["start"] < end and sp["end"] > start
        ]

        if not overlapping:
            return 0.1  # Low score for no speech

        total_speech = sum(sp["end"] - sp["start"] for sp in overlapping)
        segment_duration = end - start
        speech_ratio = total_speech / max(segment_duration, 1)

        # Higher score for sustained speech
        return min(1.0, speech_ratio * 0.8 + 0.2)

    def _score_sentiment_in_range(
        self, start: float, end: float, sentiment_timeline: List[Dict[str, Any]]
    ) -> float:
        """Score based on excitement/sentiment in range"""
        if not sentiment_timeline:
            return 0.5  # Neutral if no data

        relevant = [
            s for s in sentiment_timeline if s["start"] >= start and s["end"] <= end
        ]

        if not relevant:
            relevant = [
                s for s in sentiment_timeline if s["start"] < end and s["end"] > start
            ]

        if not relevant:
            return 0.3

        avg_excitement = np.mean([s["excitement"] for s in relevant])
        max_excitement = max(s["excitement"] for s in relevant)

        # Blend of average and peak excitement
        return min(1.0, avg_excitement * 0.6 + max_excitement * 0.4)

    def apply_speech_aware_buffering(
        self,
        segments: List[Dict[str, Any]],
        transcription: List[Dict[str, Any]],
        video_duration: float,
        audio_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply buffering that respects speech boundaries

        This extends buffers to natural sentence breaks instead of cutting mid-sentence.
        """
        if not segments:
            return []

        pre_buffer = self.buffering_config.get("pre_buffer", 4.0)
        post_buffer = self.buffering_config.get("post_buffer", 4.0)
        merge_gap = self.buffering_config.get("merge_gap", 1.5)

        speech_enabled = self.speech_config.get("enabled", True)
        respect_boundaries = self.speech_config.get("respect_sentence_boundaries", True)

        buffered_segments = []

        for segment in segments:
            original_start = segment["start"]
            original_end = segment["end"]

            buffered_start = max(0, original_start - pre_buffer)
            buffered_end = min(video_duration, original_end + post_buffer)

            # Adjust to sentence boundaries if enabled
            if speech_enabled and respect_boundaries and transcription:
                # Find sentence boundaries in the buffer range
                boundaries = self._find_sentence_boundaries_in_range(
                    buffered_start, buffered_end, transcription
                )

                # Extend start to previous sentence boundary
                prev_boundary = self._find_previous_boundary(buffered_start, boundaries)
                if prev_boundary is not None:
                    buffered_start = prev_boundary

                # Extend end to next sentence boundary
                next_boundary = self._find_next_boundary(buffered_end, boundaries)
                if next_boundary is not None:
                    buffered_end = next_boundary

            buffered_segment = segment.copy()
            buffered_segment.update(
                {
                    "buffered_start": buffered_start,
                    "buffered_end": buffered_end,
                    "original_start": original_start,
                    "original_end": original_end,
                }
            )
            buffered_segments.append(buffered_segment)

        # Merge overlapping
        merged = self._merge_segments(buffered_segments, merge_gap=merge_gap)

        # Finalize clips with correct key names for clip_generator
        final_clips = []
        for clip in merged:
            # Defensive: ensure buffered_start/end exist
            buffered_start = clip.get("buffered_start", clip.get("original_start", 0))
            buffered_end = clip.get("buffered_end", clip.get("original_end", 0))
            final_clips.append(
                {
                    "clip_id": len(final_clips) + 1,
                    "start_time": buffered_start,
                    "end_time": buffered_end,
                    "duration": buffered_end - buffered_start,
                    "original_segment_start": clip.get(
                        "original_start", clip.get("start", 0)
                    ),
                    "original_segment_end": clip.get(
                        "original_end", clip.get("end", 0)
                    ),
                    "keywords": clip.get("keywords_found", []),
                    "trigger_text": clip.get("trigger_text", ""),
                    "confidence": clip.get("confidence", 0.5),
                }
            )

        return final_clips

    def _find_sentence_boundaries_in_range(
        self, start: float, end: float, transcription: List[Dict[str, Any]]
    ) -> List[float]:
        """Find sentence boundary timestamps within a range"""
        boundaries = []
        sentence_end_chars = {".", "!", "?"}

        for segment in transcription:
            if segment["start"] > end or segment["end"] < start:
                continue

            words = segment.get("words", [])
            if words:
                for word_info in words:
                    word = word_info.get("word", "")
                    if any(c in word for c in sentence_end_chars):
                        boundaries.append(word_info.get("start", 0))
            else:
                # Use segment boundary if no word timestamps
                boundaries.append(segment["end"])

        return sorted(set(boundaries))

    def _find_previous_boundary(
        self, time: float, boundaries: List[float]
    ) -> Optional[float]:
        """Find the nearest boundary before or at time"""
        before = [b for b in boundaries if b <= time]
        return max(before) if before else None

    def _find_next_boundary(
        self, time: float, boundaries: List[float]
    ) -> Optional[float]:
        """Find the nearest boundary after time"""
        after = [b for b in boundaries if b >= time]
        return min(after) if after else None

    def apply_buffering(
        self, segments: List[Dict[str, Any]], video_duration: float
    ) -> List[Dict[str, Any]]:
        """Apply buffering to segments (standard, non-speech-aware)"""
        if not segments:
            return []

        pre_buffer = self.buffering_config.get("pre_buffer", 4.0)
        post_buffer = self.buffering_config.get("post_buffer", 4.0)
        merge_gap = self.buffering_config.get("merge_gap", 1.5)

        # Apply buffers
        buffered = []
        for segment in segments:
            buffered_start = max(0, segment["start"] - pre_buffer)
            buffered_end = min(video_duration, segment["end"] + post_buffer)

            buffered_segment = segment.copy()
            buffered_segment.update(
                {
                    "buffered_start": buffered_start,
                    "buffered_end": buffered_end,
                    "original_start": segment["start"],
                    "original_end": segment["end"],
                }
            )
            buffered.append(buffered_segment)

        # Merge overlapping
        merged = self._merge_segments(buffered, merge_gap)

        # Finalize clips with correct key names for clip_generator
        final_clips = []
        for clip in merged:
            final_clips.append(
                {
                    "clip_id": len(final_clips) + 1,
                    "start_time": clip["buffered_start"],
                    "end_time": clip["buffered_end"],
                    "duration": clip["buffered_end"] - clip["buffered_start"],
                    "original_segment_start": clip["original_start"],
                    "original_segment_end": clip["original_end"],
                    "keywords": clip.get("keywords_found", []),
                    "trigger_text": clip.get("trigger_text", ""),
                    "confidence": clip.get("confidence", 0.5),
                }
            )

        self.logger.info(f"Buffered and merged to {len(final_clips)} clips")
        return final_clips

    def _merge_segments(
        self, segments: List[Dict[str, Any]], merge_gap: float
    ) -> List[Dict[str, Any]]:
        """Merge overlapping or closely spaced segments"""
        if not segments:
            return []

        sorted_segments = sorted(segments, key=lambda x: x["buffered_start"])

        merged = [sorted_segments[0].copy()]

        for segment in sorted_segments[1:]:
            if segment["buffered_start"] <= merged[-1]["buffered_end"] + merge_gap:
                merged[-1]["buffered_end"] = max(
                    merged[-1]["buffered_end"], segment["buffered_end"]
                )
                merged[-1]["original_end"] = max(
                    merged[-1]["original_end"], segment["original_end"]
                )
                merged[-1]["keywords_found"] = list(
                    set(
                        merged[-1].get("keywords_found", [])
                        + segment.get("keywords_found", [])
                    )
                )
                merged[-1]["trigger_text"] = (
                    f"{merged[-1].get('trigger_text', '')} | {segment.get('trigger_text', '')}"
                )
                current_conf = merged[-1].get("confidence", 0.5)
                segment_conf = segment.get("confidence", 0.5)
                merged[-1]["confidence"] = (current_conf + segment_conf) / 2
            else:
                merged.append(segment.copy())

        return merged

    def optimize_segments(
        self, segments: List[Dict[str, Any]], target_duration: float = 60.0
    ) -> List[Dict[str, Any]]:
        """Optimize segments to target duration"""
        optimized = []

        for segment in segments:
            duration = segment["end_time"] - segment["start_time"]

            if duration <= target_duration:
                optimized.append(segment)
            else:
                best_subsegment = self._find_best_subsegment(segment, target_duration)
                if best_subsegment:
                    optimized.append(best_subsegment)
                else:
                    optimized.append(segment)

        return optimized

    def _find_best_subsegment(
        self, segment: Dict[str, Any], target_duration: float
    ) -> Optional[Dict[str, Any]]:
        """Find best subsegment within a long segment"""
        segment_center = (
            segment["original_segment_start"] + segment["original_segment_end"]
        ) / 2
        sub_start = max(segment["start_time"], segment_center - target_duration / 2)
        sub_end = min(segment["end_time"], segment_center + target_duration / 2)

        if sub_end - sub_start < target_duration:
            if sub_start == segment["start_time"]:
                sub_end = min(segment["end_time"], sub_start + target_duration)
            else:
                sub_start = max(segment["start_time"], sub_end - target_duration)

        if sub_end - sub_start >= 10:
            subsegment = segment.copy()
            subsegment["start_time"] = sub_start
            subsegment["end_time"] = sub_end
            subsegment["duration"] = sub_end - sub_start
            return subsegment

        return None
