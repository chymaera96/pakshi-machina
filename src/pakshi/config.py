from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    sample_rate: int = 32000
    segment_seconds: float = 2.0
    pre_roll_seconds: float = 0.15
    onset_threshold: float = 0.62
    sustain_threshold: float = 0.45
    onset_hold_seconds: float = 0.05
    release_seconds: float = 0.2
    min_phrase_seconds: float = 0.25
    max_phrase_seconds: float = 12.0
    cooldown_seconds: float = 0.15
    output_gain: float = 1.0

    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    def pre_roll_samples(self) -> int:
        return int(round(self.pre_roll_seconds * self.sample_rate))

    def onset_hold_samples(self) -> int:
        return int(round(self.onset_hold_seconds * self.sample_rate))

    def release_samples(self) -> int:
        return int(round(self.release_seconds * self.sample_rate))

    def min_phrase_samples(self) -> int:
        return int(round(self.min_phrase_seconds * self.sample_rate))

    def max_phrase_samples(self) -> int:
        return int(round(self.max_phrase_seconds * self.sample_rate))
