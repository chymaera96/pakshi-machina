from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from .config import RuntimeConfig


@dataclass
class Segment:
    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    waveform: np.ndarray


@dataclass
class SegmentedPhrase:
    phrase_id: int
    started_at_seconds: float
    ended_at_seconds: float
    reason: str
    waveform: np.ndarray
    segments: List[Segment]


def split_into_segments(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    segment_seconds: float,
) -> List[Segment]:
    segment_samples = int(round(segment_seconds * sample_rate))
    if segment_samples <= 0:
        raise ValueError("segment_seconds must resolve to at least one sample")

    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return []

    segments: List[Segment] = []
    for index, start in enumerate(range(0, wav.size, segment_samples)):
        end = min(start + segment_samples, wav.size)
        chunk = wav[start:end]
        if chunk.size < segment_samples:
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[: chunk.size] = chunk
            chunk = padded
        segments.append(
            Segment(
                index=index,
                start_sample=start,
                end_sample=end,
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                waveform=chunk,
            )
        )
    return segments


class PhraseSegmenter:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._pre_roll: Deque[float] = deque(maxlen=max(0, self.config.pre_roll_samples()))
        self._phrase_chunks: List[np.ndarray] = []
        self._active_phrase_id = 0
        self._started_at_seconds: Optional[float] = None
        self._pending_voiced_samples = 0
        self._release_samples = 0
        self._active = False
        self._total_samples_seen = 0

    @property
    def is_active(self) -> bool:
        return self._active

    def process_frame(
        self,
        frame: Sequence[float] | np.ndarray,
        level_db: float,
        *,
        now_seconds: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        wav = np.asarray(frame, dtype=np.float32).reshape(-1)
        if wav.size == 0:
            return []

        if now_seconds is None:
            now_seconds = self._total_samples_seen / self.config.sample_rate

        self._pre_roll.extend(float(x) for x in wav)
        self._total_samples_seen += wav.size

        events: List[Dict[str, object]] = []

        if not self._active:
            if level_db >= self.config.gate_open_db:
                self._pending_voiced_samples += wav.size
            else:
                self._pending_voiced_samples = 0

            if self._pending_voiced_samples >= self.config.onset_hold_samples():
                self._start_phrase(now_seconds)
                events.append(
                    {
                        "type": "phrase_started",
                        "phrase_id": self._active_phrase_id,
                        "started_at_seconds": self._started_at_seconds,
                    }
                )
                pre_roll = np.asarray(self._pre_roll, dtype=np.float32)
                if pre_roll.size:
                    self._phrase_chunks.append(pre_roll.copy())
                else:
                    self._phrase_chunks.append(wav.copy())
                self._pending_voiced_samples = 0

        else:
            self._phrase_chunks.append(wav)

            if level_db < self.config.gate_close_db:
                self._release_samples += wav.size
            else:
                self._release_samples = 0

            phrase_samples = sum(chunk.size for chunk in self._phrase_chunks)
            if phrase_samples >= self.config.max_phrase_samples():
                phrase = self._finish_phrase(now_seconds, "max_duration")
                events.extend(self._phrase_to_events(phrase))
            elif self._release_samples >= self.config.release_samples():
                phrase = self._finish_phrase(now_seconds, "vad_release")
                events.extend(self._phrase_to_events(phrase))

        return events

    def flush(self, *, now_seconds: Optional[float] = None) -> List[Dict[str, object]]:
        if not self._active:
            return []
        if now_seconds is None:
            now_seconds = self._total_samples_seen / self.config.sample_rate
        phrase = self._finish_phrase(now_seconds, "flush")
        return self._phrase_to_events(phrase)

    def _start_phrase(self, now_seconds: float) -> None:
        self._active = True
        self._active_phrase_id += 1
        self._started_at_seconds = now_seconds
        self._release_samples = 0
        self._phrase_chunks = []

    def _finish_phrase(self, now_seconds: float, reason: str) -> Optional[SegmentedPhrase]:
        self._active = False
        self._release_samples = 0
        waveform = np.concatenate(self._phrase_chunks, axis=0) if self._phrase_chunks else np.zeros(0, dtype=np.float32)
        self._phrase_chunks = []

        if waveform.size < self.config.min_phrase_samples():
            self._started_at_seconds = None
            return None

        started_at = self._started_at_seconds if self._started_at_seconds is not None else now_seconds
        self._started_at_seconds = None
        segments = split_into_segments(
            waveform,
            sample_rate=self.config.sample_rate,
            segment_seconds=self.config.segment_seconds,
        )
        return SegmentedPhrase(
            phrase_id=self._active_phrase_id,
            started_at_seconds=float(started_at),
            ended_at_seconds=float(now_seconds),
            reason=reason,
            waveform=waveform,
            segments=segments,
        )

    def _phrase_to_events(self, phrase: Optional[SegmentedPhrase]) -> List[Dict[str, object]]:
        if phrase is None:
            return []

        segments_payload = [
            {
                "index": seg.index,
                "start_seconds": seg.start_seconds,
                "end_seconds": seg.end_seconds,
                "start_sample": seg.start_sample,
                "end_sample": seg.end_sample,
                "num_samples": int(seg.waveform.size),
            }
            for seg in phrase.segments
        ]
        return [
            {
                "type": "phrase_ended",
                "phrase_id": phrase.phrase_id,
                "started_at_seconds": phrase.started_at_seconds,
                "ended_at_seconds": phrase.ended_at_seconds,
                "duration_seconds": phrase.waveform.size / self.config.sample_rate,
                "reason": phrase.reason,
            },
            {
                "type": "segments_created",
                "phrase_id": phrase.phrase_id,
                "num_segments": len(phrase.segments),
                "segments": segments_payload,
                "phrase": phrase,
            },
        ]
