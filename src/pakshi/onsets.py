from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .config import RuntimeConfig

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


@dataclass
class OnsetAnalysis:
    onset_offsets_seconds: List[float]
    descriptor_times_seconds: List[float]
    descriptor_values: List[float]
    threshold: float
    silence_db: float
    method: str


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


def _apply_silence_gate(wav: np.ndarray, silence_db: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    amp_floor = float(10 ** (silence_db / 20.0))
    gated = wav.copy()
    gated[np.abs(gated) < amp_floor] = 0.0
    return gated


def _descriptor_from_method(wav: np.ndarray, sr: int, config: RuntimeConfig) -> np.ndarray:
    if librosa is None:  # pragma: no cover
        return np.zeros(0, dtype=np.float32)
    n_fft = int(config.onset_window_size)
    hop = int(config.onset_hop_size)
    if wav.size == 0:
        return np.zeros(0, dtype=np.float32)
    s = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop, center=False))
    if s.size == 0:
        return np.zeros(0, dtype=np.float32)
    if config.onset_method == "hfc":
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)[:, None]
        weighted = s * np.maximum(freqs, 1.0)
        diff = np.maximum(0.0, weighted[:, 1:] - weighted[:, :-1])
        env = diff.mean(axis=0)
    elif config.onset_method == "complex":
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(s, ref=np.max), sr=sr, hop_length=hop, center=False)
    else:
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(s, ref=np.max), sr=sr, hop_length=hop, center=False)
    env = np.asarray(env, dtype=np.float32).reshape(-1)
    if env.size == 0:
        return env
    env = np.maximum(env, 0.0)
    peak = float(np.max(env))
    if peak > 0.0:
        env = env / peak
    return env


def analyze_onsets(waveform: Sequence[float] | np.ndarray, sample_rate: int, config: RuntimeConfig) -> OnsetAnalysis:
    wav = _prepare_waveform(waveform, sample_rate, config.onset_sample_rate)
    if wav.size == 0:
        return OnsetAnalysis([0.0], [], [], float(config.onset_threshold), float(config.onset_silence_db), config.onset_method)
    if librosa is None:  # pragma: no cover
        return OnsetAnalysis([0.0], [], [], float(config.onset_threshold), float(config.onset_silence_db), config.onset_method)

    gated = _apply_silence_gate(wav, float(config.onset_silence_db))
    env = _descriptor_from_method(gated, config.onset_sample_rate, config)
    hop = int(config.onset_hop_size)
    times = (np.arange(env.size, dtype=np.float32) * float(hop) / float(config.onset_sample_rate)).tolist()

    if env.size == 0:
        return OnsetAnalysis([0.0], times, [], float(config.onset_threshold), float(config.onset_silence_db), config.onset_method)

    wait = max(1, int(round(float(config.onset_min_interval_seconds) * float(config.onset_sample_rate) / float(hop))))
    peaks = librosa.util.peak_pick(
        env,
        pre_max=1,
        post_max=1,
        pre_avg=3,
        post_avg=3,
        delta=float(config.onset_threshold),
        wait=wait,
    )
    onset_offsets = (peaks.astype(np.float32) * float(hop) / float(config.onset_sample_rate)).tolist()
    if not onset_offsets:
        onset_offsets = [0.0]

    return OnsetAnalysis(
        onset_offsets_seconds=[max(0.0, float(v)) for v in onset_offsets],
        descriptor_times_seconds=times,
        descriptor_values=env.astype(np.float32).tolist(),
        threshold=float(config.onset_threshold),
        silence_db=float(config.onset_silence_db),
        method=config.onset_method,
    )


def detect_onset_offsets(waveform: Sequence[float] | np.ndarray, sample_rate: int, config: RuntimeConfig) -> List[float]:
    return analyze_onsets(waveform, sample_rate, config).onset_offsets_seconds


def save_onset_debug_plot(analysis: OnsetAnalysis, output_path: str | Path, *, title: str) -> Path:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to save onset debug plots.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    times = np.asarray(analysis.descriptor_times_seconds, dtype=np.float32)
    values = np.asarray(analysis.descriptor_values, dtype=np.float32)
    if values.size:
        ax.plot(times, values, color="#6fe2b3", linewidth=1.6, label=f"{analysis.method} onset curve")
        ax.axhline(analysis.threshold, color="#ffd76a", linestyle="--", linewidth=1.2, label=f"threshold {analysis.threshold:.2f}")
    for idx, onset in enumerate(analysis.onset_offsets_seconds):
        ax.axvline(onset, color="#ff7f7f", alpha=0.8, linewidth=1.1, linestyle=":" if idx else "-")
    ax.set_title(title)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Onset function")
    ax.grid(alpha=0.2)
    if values.size:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
