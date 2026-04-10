from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .config import RuntimeConfig

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


CREPE_SAMPLE_RATE = 16000
CREPE_FRAME_SAMPLES = 1024
CREPE_HOP_SAMPLES = 160
CREPE_PAD_SAMPLES = CREPE_FRAME_SAMPLES // 2
CREPE_CENTS_START = 1997.3794084376191
CREPE_BINS = 360


@dataclass
class PitchAnalysis:
    segment_start_offsets_seconds: List[float]
    frame_times_seconds: List[float]
    pitch_hz: List[float]
    pitch_cents: List[float]
    confidence: List[float]
    change_threshold_cents: float
    confidence_floor: float
    ignore_short_gaps: bool


class PitchTracker(Protocol):
    def analyze(self, waveform: Sequence[float] | np.ndarray, sample_rate: int, config: RuntimeConfig) -> PitchAnalysis:
        ...


def _resample_waveform(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    target_len = max(1, int(round(waveform.size * float(target_sr) / float(source_sr))))
    source_x = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
    return np.interp(target_x, source_x, waveform).astype(np.float32)


def _prepare_waveform(waveform: Sequence[float] | np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.reshape(-1)
    if sample_rate != target_sr:
        wav = _resample_waveform(wav, sample_rate, target_sr)
    return wav.astype(np.float32, copy=False)


def _frame_crepe_audio(wav: np.ndarray) -> np.ndarray:
    if wav.size == 0:
        return np.zeros((0, CREPE_FRAME_SAMPLES), dtype=np.float32)
    padded = np.pad(wav, (CREPE_PAD_SAMPLES, CREPE_PAD_SAMPLES), mode="constant")
    n_frames = 1 + max(0, (padded.size - CREPE_FRAME_SAMPLES) // CREPE_HOP_SAMPLES)
    frames = np.stack(
        [padded[i * CREPE_HOP_SAMPLES : i * CREPE_HOP_SAMPLES + CREPE_FRAME_SAMPLES] for i in range(n_frames)],
        axis=0,
    ).astype(np.float32)
    frame_mean = frames.mean(axis=1, keepdims=True)
    centered = frames - frame_mean
    frame_std = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(frame_std, 1e-8)


def _normalize_crepe_activation(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if float(np.min(arr)) >= 0.0 and float(np.max(arr)) <= 1.0:
        return arr
    return 1.0 / (1.0 + np.exp(-arr))


def _local_average_cents(activation: np.ndarray) -> float:
    center = int(np.argmax(activation))
    start = max(0, center - 4)
    stop = min(CREPE_BINS, center + 5)
    weights = activation[start:stop].astype(np.float64)
    if np.sum(weights) <= 0:
        return float(CREPE_CENTS_START + 20.0 * center)
    cents = CREPE_CENTS_START + 20.0 * np.arange(start, stop, dtype=np.float64)
    return float(np.sum(weights * cents) / np.sum(weights))


def _cents_to_hz(cents: float) -> float:
    return float(10.0 * 2.0 ** (cents / 1200.0))


def _smooth_pitch_cents(cents: np.ndarray, confidence: np.ndarray, confidence_floor: float) -> np.ndarray:
    out = cents.astype(np.float32, copy=True)
    for idx in range(cents.size):
        lo = max(0, idx - 2)
        hi = min(cents.size, idx + 3)
        window = cents[lo:hi]
        window_conf = confidence[lo:hi]
        voiced = window_conf >= confidence_floor
        if np.any(voiced):
            out[idx] = float(np.median(window[voiced]))
    return out


def _derive_segment_starts(
    frame_times: np.ndarray,
    cents: np.ndarray,
    confidence: np.ndarray,
    config: RuntimeConfig,
) -> List[float]:
    if frame_times.size == 0:
        return [0.0]

    starts: List[float] = [0.0]
    confidence_floor = float(config.pitch_confidence_floor)
    threshold = float(config.pitch_change_threshold_cents)
    hold_frames = max(1, int(round(config.pitch_stable_hold_seconds * config.pitch_sample_rate / CREPE_HOP_SAMPLES)))
    gap_frames = max(1, int(round(config.pitch_short_gap_seconds * config.pitch_sample_rate / CREPE_HOP_SAMPLES)))
    min_spacing_seconds = float(config.pitch_min_segment_spacing_seconds)
    phrase_end_guard_seconds = float(config.pitch_phrase_end_guard_seconds)
    phrase_end_time = float(frame_times[-1]) if frame_times.size else 0.0

    voiced = confidence >= confidence_floor
    region_values: List[float] = []
    current_ref: Optional[float] = None
    candidate_start: Optional[int] = None
    candidate_count = 0
    gap_count = 0

    for idx in range(frame_times.size):
        if not voiced[idx]:
            gap_count += 1
            if not config.pitch_ignore_short_gaps:
                current_ref = None
                region_values = []
                candidate_start = None
                candidate_count = 0
            continue

        if gap_count > 0:
            if (not config.pitch_ignore_short_gaps) or gap_count > gap_frames:
                start_time = float(frame_times[idx])
                if start_time - starts[-1] > 1e-4:
                    starts.append(start_time)
                region_values = []
                current_ref = None
            gap_count = 0

        value = float(cents[idx])
        if current_ref is None:
            region_values = [value]
            current_ref = value
            continue

        delta = abs(value - current_ref)
        if delta >= threshold:
            if candidate_start is None:
                candidate_start = idx
                candidate_count = 1
            else:
                candidate_count += 1
            if candidate_count >= hold_frames:
                start_time = float(frame_times[candidate_start])
                if (
                    start_time - starts[-1] >= min_spacing_seconds
                    and (phrase_end_time - start_time) >= phrase_end_guard_seconds
                ):
                    starts.append(start_time)
                region_values = [float(v) for v in cents[candidate_start : idx + 1]]
                current_ref = float(np.median(region_values))
                candidate_start = None
                candidate_count = 0
            continue

        candidate_start = None
        candidate_count = 0
        region_values.append(value)
        if len(region_values) > 5:
            region_values.pop(0)
        current_ref = float(np.median(region_values))

    return [max(0.0, value) for value in starts]


class CrepePitchTracker:
    def __init__(
        self,
        model_path: str | Path,
        providers: Optional[Sequence[str]] = None,
        *,
        frame_batch_size: int = 512,
    ):
        if ort is None:  # pragma: no cover
            raise RuntimeError("onnxruntime is not installed. Add onnxruntime to enable CREPE pitch segmentation.")
        self.model_path = str(model_path)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.frame_batch_size = max(1, int(frame_batch_size))
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = list(input_meta.shape)
        self._fixed_batch_size = None
        if input_shape and isinstance(input_shape[0], int):
            self._fixed_batch_size = int(input_shape[0])
        if self.input_name != "frames":
            raise RuntimeError(f"Expected CREPE pitch model input tensor named 'frames', got '{self.input_name}'.")

    def _infer_frames(self, frames: np.ndarray) -> np.ndarray:
        if frames.size == 0:
            return np.zeros((0, CREPE_BINS), dtype=np.float32)
        outputs: List[np.ndarray] = []
        for start in range(0, frames.shape[0], self.frame_batch_size):
            chunk = frames[start : start + self.frame_batch_size]
            if self._fixed_batch_size in (None, chunk.shape[0]):
                out = self.session.run([self.output_name], {self.input_name: chunk})[0]
                outputs.append(np.asarray(out, dtype=np.float32))
            elif self._fixed_batch_size == 1:
                chunk_outputs = []
                for frame in chunk:
                    out = self.session.run([self.output_name], {self.input_name: frame[np.newaxis, ...]})[0]
                    chunk_outputs.append(np.asarray(out[0], dtype=np.float32))
                outputs.append(np.stack(chunk_outputs, axis=0))
            else:
                raise RuntimeError(
                    f"CREPE pitch ONNX expects fixed batch size {self._fixed_batch_size}, "
                    f"but received frame batch size {chunk.shape[0]}."
                )
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, CREPE_BINS), dtype=np.float32)

    def analyze(self, waveform: Sequence[float] | np.ndarray, sample_rate: int, config: RuntimeConfig) -> PitchAnalysis:
        wav = _prepare_waveform(waveform, sample_rate, config.pitch_sample_rate)
        if wav.size == 0:
            return PitchAnalysis([0.0], [], [], [], [], float(config.pitch_change_threshold_cents), float(config.pitch_confidence_floor), bool(config.pitch_ignore_short_gaps))

        frames = _frame_crepe_audio(wav)
        activation = _normalize_crepe_activation(self._infer_frames(frames))
        confidence = activation.max(axis=1).astype(np.float32)
        cents = np.asarray([_local_average_cents(row) for row in activation], dtype=np.float32)
        cents = _smooth_pitch_cents(cents, confidence, float(config.pitch_confidence_floor))
        hz = np.asarray([_cents_to_hz(value) for value in cents], dtype=np.float32)
        frame_times = (np.arange(frames.shape[0], dtype=np.float32) * float(CREPE_HOP_SAMPLES) / float(config.pitch_sample_rate)).astype(np.float32)
        starts = _derive_segment_starts(frame_times, cents, confidence, config)
        return PitchAnalysis(
            segment_start_offsets_seconds=starts,
            frame_times_seconds=frame_times.tolist(),
            pitch_hz=hz.tolist(),
            pitch_cents=cents.tolist(),
            confidence=confidence.tolist(),
            change_threshold_cents=float(config.pitch_change_threshold_cents),
            confidence_floor=float(config.pitch_confidence_floor),
            ignore_short_gaps=bool(config.pitch_ignore_short_gaps),
        )


def analyze_pitch_segments(
    waveform: Sequence[float] | np.ndarray,
    sample_rate: int,
    config: RuntimeConfig,
    tracker: PitchTracker,
) -> PitchAnalysis:
    return tracker.analyze(waveform, sample_rate, config)


def save_pitch_debug_plot(analysis: PitchAnalysis, output_path: str | Path, *, title: str) -> Path:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to save pitch debug plots.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=160)
    times = np.asarray(analysis.frame_times_seconds, dtype=np.float32)
    pitch = np.asarray(analysis.pitch_hz, dtype=np.float32)
    confidence = np.asarray(analysis.confidence, dtype=np.float32)
    if times.size:
        ax1.plot(times, pitch, color="#6fe2b3", linewidth=1.4, label="pitch (Hz)")
        ax1.set_ylabel("Pitch (Hz)")
        ax2 = ax1.twinx()
        ax2.plot(times, confidence, color="#9ab7ff", linewidth=1.1, alpha=0.9, label="confidence")
        ax2.axhline(analysis.confidence_floor, color="#ffd76a", linestyle="--", linewidth=1.1, label=f"confidence {analysis.confidence_floor:.2f}")
        ax2.set_ylabel("Confidence")
    else:
        ax2 = ax1.twinx()
    for idx, start in enumerate(analysis.segment_start_offsets_seconds):
        ax1.axvline(start, color="#ff7f7f", alpha=0.8, linewidth=1.1, linestyle=":" if idx else "-")
    ax1.set_title(title)
    ax1.set_xlabel("Seconds")
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
