from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import librosa
import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None

try:
    import torch
    import torchcrepe  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    torchcrepe = None


class CrepeConfidenceEstimator:
    def __init__(self, sample_rate: int, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

    def estimate(self, waveform: Sequence[float] | np.ndarray) -> float:
        if torchcrepe is None or torch is None:  # pragma: no cover
            raise RuntimeError("torchcrepe is not installed. Add torchcrepe to run CREPE-based phrase gating.")
        wav = np.asarray(waveform, dtype=np.float32).reshape(1, -1)
        tensor = torch.from_numpy(wav)
        _, periodicity = torchcrepe.predict(
            tensor,
            self.sample_rate,
            self.hop_length,
            fmin=50.0,
            fmax=1000.0,
            model="full",
            batch_size=1,
            return_periodicity=True,
            device="cpu",
        )
        return float(torch.mean(periodicity).item())


@dataclass
class PlaybackHandle:
    stop_event: threading.Event
    thread: threading.Thread


class NoopSequencePlayer:
    def __init__(self, on_started: Callable[[int, dict], None], on_finished: Callable[[int, dict], None]):
        self.on_started = on_started
        self.on_finished = on_finished
        self._handle: Optional[PlaybackHandle] = None

    def stop(self) -> None:
        if self._handle is None:
            return
        self._handle.stop_event.set()
        self._handle.thread.join(timeout=0.2)
        self._handle = None

    def play_sequence(self, phrase_id: int, matches: Sequence[dict]) -> None:
        self.stop()
        stop_event = threading.Event()

        def runner() -> None:
            for match in matches:
                if stop_event.is_set():
                    return
                self.on_started(phrase_id, match)
                duration = float(match.get("metadata", {}).get("duration_seconds", 0.0) or 0.0)
                if duration > 0:
                    stop_event.wait(timeout=duration)
                self.on_finished(phrase_id, match)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._handle = PlaybackHandle(stop_event=stop_event, thread=thread)


class SoundDeviceSequencePlayer(NoopSequencePlayer):
    def __init__(self, sample_rate: int, output_gain: float, on_started: Callable[[int, dict], None], on_finished: Callable[[int, dict], None]):
        if sd is None:  # pragma: no cover
            raise RuntimeError("sounddevice is not installed. Add sounddevice to enable audio playback.")
        super().__init__(on_started=on_started, on_finished=on_finished)
        self.sample_rate = sample_rate
        self.output_gain = output_gain

    def play_sequence(self, phrase_id: int, matches: Sequence[dict]) -> None:
        self.stop()
        stop_event = threading.Event()

        def runner() -> None:
            for match in matches:
                if stop_event.is_set():
                    return
                path = match.get("metadata", {}).get("path")
                if not path:
                    continue
                self.on_started(phrase_id, match)
                audio, sr = librosa.load(Path(path), sr=self.sample_rate, mono=True)
                if audio.size:
                    sd.play((audio * self.output_gain).astype(np.float32), self.sample_rate, blocking=False)
                    start = time.time()
                    while sd.get_stream().active:
                        if stop_event.wait(timeout=0.05):
                            sd.stop()
                            return
                        if time.time() - start > (len(audio) / self.sample_rate + 1.0):
                            break
                self.on_finished(phrase_id, match)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._handle = PlaybackHandle(stop_event=stop_event, thread=thread)
