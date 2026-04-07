from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .audio import LiveInputStream, NoopSequencePlayer, SoundDeviceSequencePlayer, build_confidence_estimator
from .config import RuntimeConfig
from .corpus import CorpusBundle, load_corpus_bundle
from .retrieval import OnnxEmbeddingModel, RetrievalEngine
from .segmentation import PhraseSegmenter, SegmentedPhrase


class SegmentedPhraseWorker:
    def __init__(self, model_path: str | Path, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.model_path = str(model_path)
        self.segmenter = PhraseSegmenter(self.config)
        self.embedder = OnnxEmbeddingModel(self.model_path)
        self.corpus: Optional[CorpusBundle] = None
        self.retrieval: Optional[RetrievalEngine] = None
        self.armed = False
        self._player = self._build_player()
        self._last_state = "idle"
        self._mic: Optional[LiveInputStream] = None
        self._confidence = build_confidence_estimator(self.config.sample_rate)
        self._stream_started_at: Optional[float] = None
        self._smoothed_confidence = 0.0
        self.calibrated = False
        self._calibrating = False
        self._calibration_phrase_target = 3
        self._calibration_phrase_count = 0
        self._calibration_confidences: List[float] = []
        self._calibration_segmenter = self._make_calibration_segmenter()

    def _build_player(self):
        try:
            return SoundDeviceSequencePlayer(
                sample_rate=self.config.sample_rate,
                output_gain=self.config.output_gain,
                stitch_gap_seconds=self.config.stitch_gap_seconds,
                on_started=self._emit_segment_started,
                on_finished=self._emit_segment_finished,
            )
        except Exception:
            return NoopSequencePlayer(
                on_started=self._emit_segment_started,
                on_finished=self._emit_segment_finished,
                stitch_gap_seconds=self.config.stitch_gap_seconds,
            )

    def handle_command(self, command: Dict[str, Any]) -> List[Dict[str, Any]]:
        kind = command.get("command")
        if kind == "load_corpus":
            bundle_dir = command["bundle_dir"]
            self.corpus = load_corpus_bundle(bundle_dir)
            self.retrieval = RetrievalEngine(self.embedder, self.corpus.index, self.corpus.metadata)
            return [self._state_event("idle")]
        if kind == "set_params":
            params = command.get("params", {})
            self._apply_params(params)
            return [self._state_event("idle")]
        if kind == "arm":
            return self._arm_worker()
        if kind == "start_calibration":
            return self._start_calibration()
        if kind == "disarm":
            return self._disarm_worker()
        if kind == "stop_all":
            self._player.stop()
            return [self._state_event("idle")]
        if kind == "clear_queue_restart":
            return self._clear_queue_restart()
        if kind == "get_state":
            return [self._state_event(self._last_state)]
        if kind == "process_frame":
            if not self.armed:
                return []
            frame = np.asarray(command["frame"], dtype=np.float32)
            confidence = float(command["confidence"])
            now_seconds = command.get("now_seconds")
            return self._handle_frame(frame, confidence, now_seconds=now_seconds)
        if kind == "process_phrase":
            if not self.armed:
                return []
            phrase = np.asarray(command["waveform"], dtype=np.float32)
            confidence = float(command.get("confidence", 1.0))
            return self._process_phrase_offline(phrase, confidence)
        raise ValueError(f"Unknown command: {kind}")

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if not hasattr(self.config, key):
                continue
            setattr(self.config, key, value)
        self.segmenter = PhraseSegmenter(self.config)
        self._calibration_segmenter = self._make_calibration_segmenter()
        self._player = self._build_player()
        self._confidence = build_confidence_estimator(self.config.sample_rate)
        self._smoothed_confidence = 0.0
        if self.armed:
            self._stop_mic()
            self._start_mic()

    def _arm_worker(self) -> List[Dict[str, Any]]:
        if not self.calibrated:
            raise RuntimeError("Calibration required before arming. Run the 3-phrase calibration first.")
        self.armed = True
        self._start_mic()
        return [self._state_event("listening")]

    def _start_calibration(self) -> List[Dict[str, Any]]:
        self.armed = False
        self._calibrating = True
        self._calibration_phrase_count = 0
        self._calibration_confidences = []
        self._calibration_segmenter = self._make_calibration_segmenter()
        self._start_mic()
        return [{"type": "calibration_started", "target_phrases": self._calibration_phrase_target}, self._state_event("calibrating")]

    def _disarm_worker(self) -> List[Dict[str, Any]]:
        self.armed = False
        self._calibrating = False
        self._calibration_phrase_count = 0
        self._calibration_confidences = []
        self.segmenter = PhraseSegmenter(self.config)
        self._calibration_segmenter = self._make_calibration_segmenter()
        self._player.clear()
        self._stop_mic()
        return [self._state_event("idle")]

    def _clear_queue_restart(self) -> List[Dict[str, Any]]:
        self._player.clear()
        self.segmenter = PhraseSegmenter(self.config)
        events = [{"type": "queue_cleared"}]
        if self.armed:
            self._stop_mic()
            self._start_mic()
            events.append(self._state_event("listening"))
        elif self._calibrating:
            self._stop_mic()
            self._start_mic()
            events.append(self._state_event("calibrating"))
        else:
            events.append(self._state_event("idle"))
        return events

    def _start_mic(self) -> None:
        if self._mic is not None and self._mic.active:
            return
        self._stream_started_at = time.time()
        self._smoothed_confidence = 0.0
        self._mic = LiveInputStream(
            sample_rate=self.config.sample_rate,
            frame_samples=self.config.input_frame_samples(),
            on_frame=self._handle_live_frame,
            on_error=self._handle_mic_error,
        )
        self._mic.start()
        self.emit({"type": "mic_status", "active": True})

    def _stop_mic(self) -> None:
        if self._mic is not None:
            self._mic.stop()
            self._mic = None
            self.emit({"type": "mic_status", "active": False})

    def _handle_live_frame(self, frame: np.ndarray, timestamp: float) -> None:
        confidence = float(self._confidence.estimate(frame))
        alpha = float(np.clip(self.config.confidence_smoothing, 0.0, 0.999))
        self._smoothed_confidence = alpha * self._smoothed_confidence + (1.0 - alpha) * confidence
        if self._calibrating:
            self._handle_calibration_frame(frame, confidence, timestamp)
            return
        if not self.armed:
            return
        level = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
        now_seconds = (
            timestamp - self._stream_started_at if self._stream_started_at is not None else None
        )
        self.emit(
            {
                "type": "mic_level",
                "level": level,
                "confidence": self._smoothed_confidence,
                "raw_confidence": confidence,
            }
        )
        for event in self._handle_frame(frame, self._smoothed_confidence, now_seconds=now_seconds):
            self.emit(event)

    def _handle_mic_error(self, message: str) -> None:
        self.emit({"type": "error", "message": message})

    def _make_calibration_segmenter(self) -> PhraseSegmenter:
        cfg = RuntimeConfig(**asdict(self.config))
        cfg.onset_threshold = min(cfg.onset_threshold, 0.2)
        cfg.sustain_threshold = min(cfg.sustain_threshold, 0.12)
        cfg.release_seconds = max(cfg.release_seconds, 0.35)
        return PhraseSegmenter(cfg)

    def _handle_calibration_frame(self, frame: np.ndarray, confidence: float, timestamp: float) -> None:
        self._calibration_confidences.append(confidence)
        now_seconds = (
            timestamp - self._stream_started_at if self._stream_started_at is not None else None
        )
        events = self._calibration_segmenter.process_frame(frame, confidence, now_seconds=now_seconds)
        for event in events:
            if event["type"] == "phrase_ended":
                self._calibration_phrase_count += 1
                self.emit(
                    {
                        "type": "calibration_progress",
                        "phrase_count": self._calibration_phrase_count,
                        "target_phrases": self._calibration_phrase_target,
                    }
                )
                if self._calibration_phrase_count >= self._calibration_phrase_target:
                    self._finish_calibration()
                    return

    def _finish_calibration(self) -> None:
        voiced = [x for x in self._calibration_confidences if x > 0.1]
        if voiced:
            onset = float(np.clip(np.percentile(voiced, 65), 0.18, 0.85))
            sustain = float(np.clip(onset * 0.7, 0.12, onset - 0.05))
            self.config.onset_threshold = onset
            self.config.sustain_threshold = sustain
        self.calibrated = True
        self._calibrating = False
        self._calibration_segmenter = self._make_calibration_segmenter()
        self._stop_mic()
        self.emit(
            {
                "type": "calibration_complete",
                "phrase_count": self._calibration_phrase_count,
                "onset_threshold": self.config.onset_threshold,
                "sustain_threshold": self.config.sustain_threshold,
            }
        )
        self.emit(self._state_event("idle"))

    def _handle_frame(self, frame: np.ndarray, confidence: float, *, now_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        events = self.segmenter.process_frame(frame, confidence, now_seconds=now_seconds)
        return self._expand_events(events)

    def _process_phrase_offline(self, waveform: np.ndarray, confidence: float) -> List[Dict[str, Any]]:
        # Feed one voiced frame then a silent tail to reuse the same VAD/segment logic.
        events: List[Dict[str, Any]] = []
        events.extend(self.segmenter.process_frame(waveform, confidence, now_seconds=0.0))
        silent_tail = np.zeros(self.config.release_samples(), dtype=np.float32)
        events.extend(self.segmenter.process_frame(silent_tail, 0.0, now_seconds=float(waveform.size / self.config.sample_rate)))
        return self._expand_events(events)

    def _expand_events(self, events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for event in events:
            phrase = event.pop("phrase", None)
            out.append(event)
            if event["type"] == "phrase_started":
                out.append(self._state_event("in_phrase"))
            elif event["type"] == "phrase_ended":
                out.append(self._state_event("processing"))
            if event["type"] == "segments_created" and isinstance(phrase, SegmentedPhrase):
                retrieval = self._retrieve_phrase(phrase)
                out.append(retrieval.to_event())
                out.append(self._state_event("playing_sequence" if retrieval.matches else "listening"))
                self._schedule_sequence(retrieval.to_event())
        return out

    def _retrieve_phrase(self, phrase: SegmentedPhrase):
        if self.retrieval is None:
            raise RuntimeError("Corpus is not loaded. Call load_corpus before processing phrases.")
        return self.retrieval.query_segments(phrase.phrase_id, phrase.segments)

    def _schedule_sequence(self, event: Dict[str, Any]) -> None:
        self._player.play_sequence(event["phrase_id"], event["matches"])

    def _emit_segment_started(self, phrase_id: int, match: Dict[str, Any]) -> None:
        self.emit(
            {
                "type": "segment_playback_started",
                "phrase_id": phrase_id,
                "segment_index": match["segment_index"],
                "metadata": match["metadata"],
            }
        )

    def _emit_segment_finished(self, phrase_id: int, match: Dict[str, Any]) -> None:
        self.emit(
            {
                "type": "segment_playback_finished",
                "phrase_id": phrase_id,
                "segment_index": match["segment_index"],
                "metadata": match["metadata"],
            }
        )

    def _state_event(self, state: str) -> Dict[str, Any]:
        self._last_state = state
        event: Dict[str, Any] = {
            "type": "state",
            "state": state,
            "armed": self.armed,
            "calibrated": self.calibrated,
            "calibrating": self._calibrating,
            "config": asdict(self.config),
            "model_path": self.model_path,
        }
        if self.corpus is not None:
            event["corpus_backend"] = self.corpus.backend
            event["corpus_dir"] = str(self.corpus.bundle_dir)
            event["num_items"] = len(self.corpus.metadata)
        event["mic_active"] = self._mic.active if self._mic is not None else False
        return event

    def emit(self, event: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()

    def run_stdio(self) -> None:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                command = json.loads(raw_line)
                events = self.handle_command(command)
                for event in events:
                    self.emit(event)
            except Exception as exc:
                self.emit({"type": "error", "message": str(exc)})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pakshi segmented phrase retrieval worker")
    parser.add_argument("--model", type=Path, required=True, help="Path to waveform->embedding ONNX model")
    parser.add_argument("--bundle", type=Path, default=None, help="Optional corpus bundle to load at startup")
    parser.add_argument("--arm", action="store_true", help="Start in armed/listening mode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    worker = SegmentedPhraseWorker(model_path=args.model)
    if args.bundle is not None:
        for event in worker.handle_command({"command": "load_corpus", "bundle_dir": str(args.bundle)}):
            worker.emit(event)
    if args.arm:
        for event in worker.handle_command({"command": "arm"}):
            worker.emit(event)
    worker.run_stdio()


if __name__ == "__main__":
    main()
