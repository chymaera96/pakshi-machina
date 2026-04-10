from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    sample_rate: int = 16000
    input_frame_seconds: float = 0.1
    segment_seconds: float = 0.25
    pre_roll_seconds: float = 0.15
    gate_open_db: float = -42.0
    gate_close_db: float = -48.0
    onset_hold_seconds: float = 0.05
    release_seconds: float = 0.2
    min_phrase_seconds: float = 0.25
    max_phrase_seconds: float = 12.0
    cooldown_seconds: float = 0.2
    envelope_smoothing: float = 0.72
    stitch_gap_seconds: float = 0.15
    output_gain: float = 1.0
    meter_floor_db: float = -90.0
    calibration_noise_seconds: float = 2.5
    calibration_singing_seconds: float = 8.0
    calibration_min_separation_db: float = 8.0
    onset_sample_rate: int = 16000
    onset_method: str = "hfc"
    onset_window_size: int = 1024
    onset_hop_size: int = 256
    onset_min_interval_seconds: float = 0.07
    onset_threshold: float = 0.12
    onset_silence_db: float = -75.0
    save_onset_debug_plots: bool = True
    onset_debug_dir: str = "debug/onsets"
    embedding_batch_size: int = 32
    embedding_live_max_batch_size: int = 16

    def segment_samples(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    def input_frame_samples(self) -> int:
        return int(round(self.input_frame_seconds * self.sample_rate))

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
