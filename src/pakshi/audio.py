from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import librosa
import numpy as np

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None


def rms_dbfs(waveform: Sequence[float] | np.ndarray, *, floor_db: float = -90.0) -> float:
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return floor_db
    rms = float(np.sqrt(np.mean(np.square(wav))))
    if rms <= 1e-9:
        return floor_db
    return float(max(20.0 * np.log10(rms), floor_db))


class LevelEstimator:
    def __init__(self, floor_db: float = -90.0):
        self.floor_db = floor_db

    def estimate_db(self, waveform: Sequence[float] | np.ndarray) -> float:
        return rms_dbfs(waveform, floor_db=self.floor_db)


def build_level_estimator() -> LevelEstimator:
    return LevelEstimator()


class LiveInputStream:
    def __init__(
        self,
        *,
        sample_rate: int,
        frame_samples: int,
        on_frame: Callable[[np.ndarray, float], None],
        on_error: Callable[[str], None],
    ):
        if sd is None:  # pragma: no cover
            raise RuntimeError("sounddevice is not installed. Add sounddevice to enable microphone input.")
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.on_frame = on_frame
        self.on_error = on_error
        self._queue: "queue.Queue[tuple[np.ndarray, float]]" = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stop_event.clear()

        def callback(indata, frames, time_info, status):  # pragma: no cover
            if status:
                self.on_error(str(status))
            frame = np.asarray(indata[:, 0], dtype=np.float32).copy()
            timestamp = time.time()
            try:
                self._queue.put_nowait((frame, timestamp))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait((frame, timestamp))
                except queue.Full:
                    pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._process_frames, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=0.5)
            self._thread = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _process_frames(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.on_frame(frame, timestamp)
            except Exception as exc:  # pragma: no cover
                self.on_error(str(exc))


@dataclass
class PlaybackHandle:
    stop_event: threading.Event
    thread: threading.Thread


class NoopSequencePlayer:
    def __init__(
        self,
        on_started: Callable[[int, dict], None],
        on_finished: Callable[[int, dict], None],
        stitch_gap_seconds: float = 0.15,
    ):
        self.on_started = on_started
        self.on_finished = on_finished
        self.stitch_gap_seconds = stitch_gap_seconds
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
            for index, match in enumerate(matches):
                if stop_event.is_set():
                    return
                self.on_started(phrase_id, match)
                duration = float(match.get("metadata", {}).get("duration_seconds", 0.0) or 0.0)
                if duration > 0:
                    stop_event.wait(timeout=duration)
                self.on_finished(phrase_id, match)
                if index < len(matches) - 1 and self.stitch_gap_seconds > 0:
                    if stop_event.wait(timeout=self.stitch_gap_seconds):
                        return

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._handle = PlaybackHandle(stop_event=stop_event, thread=thread)

    def clear(self) -> None:
        self.stop()


class SoundDeviceSequencePlayer(NoopSequencePlayer):
    def __init__(
        self,
        sample_rate: int,
        output_gain: float,
        on_started: Callable[[int, dict], None],
        on_finished: Callable[[int, dict], None],
        stitch_gap_seconds: float = 0.15,
    ):
        if sd is None:  # pragma: no cover
            raise RuntimeError("sounddevice is not installed. Add sounddevice to enable audio playback.")
        super().__init__(on_started=on_started, on_finished=on_finished, stitch_gap_seconds=stitch_gap_seconds)
        self.sample_rate = sample_rate
        self.output_gain = output_gain

    def play_sequence(self, phrase_id: int, matches: Sequence[dict]) -> None:
        self.stop()
        stop_event = threading.Event()

        def runner() -> None:
            for index, match in enumerate(matches):
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
                if index < len(matches) - 1 and self.stitch_gap_seconds > 0:
                    if stop_event.wait(timeout=self.stitch_gap_seconds):
                        return

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._handle = PlaybackHandle(stop_event=stop_event, thread=thread)
