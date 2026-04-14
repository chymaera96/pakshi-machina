from __future__ import annotations

from collections import OrderedDict
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

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


@dataclass
class ScheduledVoice:
    match: dict
    audio: np.ndarray
    position: int = 0


class NoopSequencePlayer:
    def __init__(
        self,
        on_started: Callable[[int, dict], None],
        on_finished: Callable[[int, dict], None],
        stitch_gap_seconds: float = 0.15,
        on_sequence_finished: Optional[Callable[[int], None]] = None,
    ):
        self.on_started = on_started
        self.on_finished = on_finished
        self.on_sequence_finished = on_sequence_finished
        self.stitch_gap_seconds = stitch_gap_seconds
        self._handle: Optional[PlaybackHandle] = None

    def stop(self) -> None:
        if self._handle is None:
            return
        self._handle.stop_event.set()
        self._handle.thread.join(timeout=0.5)
        self._handle = None

    def play_sequence(self, phrase_id: int, matches: Sequence[dict]) -> None:
        self.stop()
        stop_event = threading.Event()

        def runner() -> None:
            workers: List[threading.Thread] = []
            sequence_started_at = time.monotonic()

            def finish_match(match: dict, duration: float) -> None:
                if duration > 0 and stop_event.wait(timeout=duration):
                    return
                if stop_event.is_set():
                    return
                self.on_finished(phrase_id, match)

            for match in matches:
                if stop_event.is_set():
                    return
                target = max(0.0, float(match.get("scheduled_offset_seconds", 0.0)))
                while not stop_event.is_set():
                    remaining = target - (time.monotonic() - sequence_started_at)
                    if remaining <= 0:
                        break
                    stop_event.wait(timeout=min(remaining, 0.01))
                if stop_event.is_set():
                    return
                self.on_started(phrase_id, match)
                duration = float(match.get("metadata", {}).get("duration_seconds", 0.0) or 0.0)
                thread = threading.Thread(target=finish_match, args=(match, duration), daemon=True)
                thread.start()
                workers.append(thread)

            for thread in workers:
                while thread.is_alive():
                    if stop_event.wait(timeout=0.01):
                        return
            if not stop_event.is_set() and self.on_sequence_finished is not None:
                self.on_sequence_finished(phrase_id)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._handle = PlaybackHandle(stop_event=stop_event, thread=thread)

    def clear(self) -> None:
        self.stop()

    def warmup(self, paths: Sequence[str] | None = None) -> None:
        return None


class SoundDeviceSequencePlayer(NoopSequencePlayer):
    def __init__(
        self,
        sample_rate: int,
        output_gain: float,
        on_started: Callable[[int, dict], None],
        on_finished: Callable[[int, dict], None],
        stitch_gap_seconds: float = 0.15,
        on_sequence_finished: Optional[Callable[[int], None]] = None,
        cache_size: int = 48,
    ):
        if sd is None:  # pragma: no cover
            raise RuntimeError("sounddevice is not installed. Add sounddevice to enable audio playback.")
        super().__init__(
            on_started=on_started,
            on_finished=on_finished,
            stitch_gap_seconds=stitch_gap_seconds,
            on_sequence_finished=on_sequence_finished,
        )
        self.sample_rate = sample_rate
        self.output_gain = output_gain
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def _apply_edge_fade(self, audio: np.ndarray, fade_seconds: float = 0.005) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32).copy()
        if arr.size == 0:
            return arr
        fade_samples = min(int(round(fade_seconds * self.sample_rate)), arr.size // 2)
        if fade_samples <= 1:
            return arr
        ramp = np.linspace(0.0, 1.0, num=fade_samples, endpoint=True, dtype=np.float32)
        arr[:fade_samples] *= ramp
        arr[-fade_samples:] *= ramp[::-1]
        return arr

    def _load_clip(self, path: str) -> np.ndarray:
        if path not in self._cache:
            audio, _ = librosa.load(Path(path), sr=self.sample_rate, mono=True)
            audio = (audio * self.output_gain).astype(np.float32)
            self._cache[path] = self._apply_edge_fade(audio)
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return self._cache[path]
        self._cache.move_to_end(path)
        return self._cache[path]

    def warmup(self, paths: Sequence[str] | None = None) -> None:
        if paths:
            for path in paths:
                try:
                    self._load_clip(path)
                except Exception:
                    continue
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            stream.start()
        except Exception:
            return
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def play_sequence(self, phrase_id: int, matches: Sequence[dict]) -> None:
        self.stop()
        stop_event = threading.Event()

        def runner() -> None:
            active_voices: List[ScheduledVoice] = []
            finished_queue: "queue.Queue[dict]" = queue.Queue()
            lock = threading.Lock()
            stream = None

            def drain_finished() -> None:
                while True:
                    try:
                        match = finished_queue.get_nowait()
                    except queue.Empty:
                        break
                    self.on_finished(phrase_id, match)

            def callback(outdata, frames, time_info, status):  # pragma: no cover
                mixed = np.zeros(frames, dtype=np.float32)
                completed: List[dict] = []
                with lock:
                    survivors: List[ScheduledVoice] = []
                    for voice in active_voices:
                        remaining = voice.audio.size - voice.position
                        if remaining <= 0:
                            completed.append(voice.match)
                            continue
                        take = min(frames, remaining)
                        mixed[:take] += voice.audio[voice.position : voice.position + take]
                        voice.position += take
                        if voice.position >= voice.audio.size:
                            completed.append(voice.match)
                        else:
                            survivors.append(voice)
                    active_voices[:] = survivors
                peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
                if peak > 0.95:
                    mixed *= np.float32(0.95 / peak)
                mixed = np.tanh(mixed * np.float32(1.1)) / np.float32(np.tanh(1.1))
                outdata[:, 0] = np.clip(mixed, -1.0, 1.0)
                for match in completed:
                    finished_queue.put(match)

            try:
                try:
                    stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                        callback=callback,
                    )
                    stream.start()
                except Exception:
                    sequence_started_at = time.monotonic()
                    for match in matches:
                        target = max(0.0, float(match.get("scheduled_offset_seconds", 0.0)))
                        while not stop_event.is_set():
                            remaining = target - (time.monotonic() - sequence_started_at)
                            if remaining <= 0:
                                break
                            stop_event.wait(timeout=min(remaining, 0.01))
                        if stop_event.is_set():
                            return
                        self.on_started(phrase_id, match)
                        if stop_event.is_set():
                            return
                        path = match.get("metadata", {}).get("path")
                        if not path:
                            self.on_finished(phrase_id, match)
                            continue
                        try:
                            audio = self._load_clip(path)
                        except Exception:
                            self.on_finished(phrase_id, match)
                            continue
                        duration = float(audio.size) / float(self.sample_rate) if audio.size else 0.0
                        if duration > 0.0:
                            stop_event.wait(timeout=duration)
                        if stop_event.is_set():
                            return
                        self.on_finished(phrase_id, match)
                    return

                sequence_started_at = time.monotonic()
                for match in matches:
                    target = max(0.0, float(match.get("scheduled_offset_seconds", 0.0)))
                    while not stop_event.is_set():
                        drain_finished()
                        remaining = target - (time.monotonic() - sequence_started_at)
                        if remaining <= 0:
                            break
                        stop_event.wait(timeout=min(remaining, 0.01))
                    if stop_event.is_set():
                        return
                    path = match.get("metadata", {}).get("path")
                    self.on_started(phrase_id, match)
                    if not path:
                        self.on_finished(phrase_id, match)
                        continue
                    audio = self._load_clip(path)
                    if audio.size == 0:
                        self.on_finished(phrase_id, match)
                        continue
                    with lock:
                        active_voices.append(ScheduledVoice(match=match, audio=audio))

                while not stop_event.is_set():
                    drain_finished()
                    with lock:
                        done = not active_voices
                    if done:
                        break
                    stop_event.wait(timeout=0.01)
                drain_finished()
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                if not stop_event.is_set() and self.on_sequence_finished is not None:
                    self.on_sequence_finished(phrase_id)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        self._handle = PlaybackHandle(stop_event=stop_event, thread=thread)
