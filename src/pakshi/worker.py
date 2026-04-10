from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .audio import LiveInputStream, NoopSequencePlayer, SoundDeviceSequencePlayer, build_level_estimator
from .config import RuntimeConfig
from .corpus import CorpusBundle, load_corpus_bundle
from .onsets import save_onset_debug_plot
from .retrieval import EffNetBioEmbeddingModel, RetrievalEngine
from .segmentation import PhraseSegmenter, SegmentedPhrase


class SegmentedPhraseWorker:
    def __init__(self, model_path: str | Path, config: Optional[RuntimeConfig] = None):
        self.config = config or RuntimeConfig()
        self.model_path = str(model_path)
        self.segmenter = PhraseSegmenter(self.config)
        self.embedder = EffNetBioEmbeddingModel(
            self.model_path,
            input_sample_rate=self.config.sample_rate,
            batch_size=self.config.embedding_live_max_batch_size,
        )
        self.corpus: Optional[CorpusBundle] = None
        self.retrieval: Optional[RetrievalEngine] = None
        self.armed = False
        self.calibrated = False
        self._player = self._build_player()
        self._last_state = "idle"
        self._mic: Optional[LiveInputStream] = None
        self._level_estimator = build_level_estimator()
        self._stream_started_at: Optional[float] = None
        self._smoothed_level_db = self.config.meter_floor_db

        self.setup_mode = False
        self._setup_stage = "idle"
        self._capture_started_at: Optional[float] = None
        self._room_noise_levels: List[float] = []
        self._singing_levels: List[float] = []
        self.noise_floor_db: Optional[float] = None
        self.singing_soft_db: Optional[float] = None
        self.singing_median_db: Optional[float] = None
        self._candidate_gate_open_db: Optional[float] = None
        self._candidate_gate_close_db: Optional[float] = None
        self._resume_after_playback = False
        self._active_playback_phrase_id: Optional[int] = None

    def _build_player(self):
        try:
            return SoundDeviceSequencePlayer(
                sample_rate=self.config.sample_rate,
                output_gain=self.config.output_gain,
                stitch_gap_seconds=self.config.stitch_gap_seconds,
                on_started=self._emit_segment_started,
                on_finished=self._emit_segment_finished,
                on_sequence_finished=self._emit_sequence_finished,
            )
        except Exception:
            return NoopSequencePlayer(
                on_started=self._emit_segment_started,
                on_finished=self._emit_segment_finished,
                stitch_gap_seconds=self.config.stitch_gap_seconds,
                on_sequence_finished=self._emit_sequence_finished,
            )

    def handle_command(self, command: Dict[str, Any]) -> List[Dict[str, Any]]:
        kind = command.get("command")
        if kind == "load_corpus":
            bundle_dir = command["bundle_dir"]
            self.corpus = load_corpus_bundle(bundle_dir)
            self.retrieval = RetrievalEngine(self.embedder, self.corpus.index, self.corpus.metadata)
            return [self._state_event("idle")]
        if kind == "set_params":
            self._apply_params(command.get("params", {}))
            return [self._state_event(self._last_state)]
        if kind == "start_setup":
            return self._start_setup()
        if kind == "capture_noise_floor":
            return self._capture_noise_floor()
        if kind == "capture_singing_level":
            return self._capture_singing_level()
        if kind == "reset_setup":
            return self._reset_setup()
        if kind == "arm":
            return self._arm_worker()
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
            level_db = float(command.get("level_db", self._level_estimator.estimate_db(frame)))
            now_seconds = command.get("now_seconds")
            return self._handle_frame(frame, level_db, now_seconds=now_seconds)
        if kind == "process_phrase":
            if not self.armed:
                return []
            phrase = np.asarray(command["waveform"], dtype=np.float32)
            level_db = float(command.get("level_db", self._level_estimator.estimate_db(phrase)))
            return self._process_phrase_offline(phrase, level_db)
        raise ValueError(f"Unknown command: {kind}")

    def _apply_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.segmenter = PhraseSegmenter(self.config)
        self._player = self._build_player()
        self._smoothed_level_db = self.config.meter_floor_db
        self._level_estimator = build_level_estimator()
        if self.armed or self._setup_stage in {"capture_noise", "capture_singing"}:
            self._stop_mic()
            self._start_mic()

    def _start_setup(self) -> List[Dict[str, Any]]:
        self.armed = False
        self._player.clear()
        self.setup_mode = True
        self.calibrated = False
        self._setup_stage = "setup"
        self._capture_started_at = None
        self._room_noise_levels = []
        self._singing_levels = []
        self.noise_floor_db = None
        self.singing_soft_db = None
        self.singing_median_db = None
        self._candidate_gate_open_db = None
        self._candidate_gate_close_db = None
        self._stop_mic()
        return [{"type": "setup_started"}, self._state_event("setup")]

    def _capture_noise_floor(self) -> List[Dict[str, Any]]:
        if not self.setup_mode:
            self._start_setup()
        self._setup_stage = "capture_noise"
        self._capture_started_at = None
        self._room_noise_levels = []
        self._start_mic()
        return [
            {"type": "noise_floor_capture_started", "duration_seconds": self.config.calibration_noise_seconds},
            self._state_event("setup"),
        ]

    def _capture_singing_level(self) -> List[Dict[str, Any]]:
        if self.noise_floor_db is None:
            raise RuntimeError("Capture Noise Floor first.")
        self._setup_stage = "capture_singing"
        self._capture_started_at = None
        self._singing_levels = []
        self._start_mic()
        return [
            {"type": "singing_level_capture_started", "duration_seconds": self.config.calibration_singing_seconds},
            self._state_event("setup"),
        ]

    def _reset_setup(self) -> List[Dict[str, Any]]:
        self.armed = False
        self.setup_mode = True
        self.calibrated = False
        self._setup_stage = "setup"
        self._capture_started_at = None
        self._room_noise_levels = []
        self._singing_levels = []
        self.noise_floor_db = None
        self.singing_soft_db = None
        self.singing_median_db = None
        self._candidate_gate_open_db = None
        self._candidate_gate_close_db = None
        self._player.clear()
        self.segmenter = PhraseSegmenter(self.config)
        self._stop_mic()
        return [{"type": "setup_reset"}, self._state_event("setup")]

    def _arm_worker(self) -> List[Dict[str, Any]]:
        if not self.calibrated:
            raise RuntimeError("Setup must be completed before arming.")
        self.setup_mode = False
        self._setup_stage = "idle"
        self.armed = True
        self._start_mic()
        return [self._state_event("listening")]

    def _disarm_worker(self) -> List[Dict[str, Any]]:
        self.armed = False
        self._player.clear()
        self.segmenter = PhraseSegmenter(self.config)
        self._stop_mic()
        return [self._state_event("idle"), {"type": "queue_cleared"}]

    def _clear_queue_restart(self) -> List[Dict[str, Any]]:
        self._player.clear()
        self.segmenter = PhraseSegmenter(self.config)
        events = [{"type": "queue_cleared"}]
        if self.armed:
            self._stop_mic()
            self._start_mic()
            events.append(self._state_event("listening"))
        else:
            events.append(self._state_event(self._last_state))
        return events

    def _start_mic(self) -> None:
        if self._mic is not None and self._mic.active:
            return
        self._stream_started_at = time.time()
        self._smoothed_level_db = self.config.meter_floor_db
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
        level_db = float(self._level_estimator.estimate_db(frame))
        alpha = float(np.clip(self.config.envelope_smoothing, 0.0, 0.999))
        self._smoothed_level_db = alpha * self._smoothed_level_db + (1.0 - alpha) * level_db
        self.emit(self._meter_event(level_db, self._smoothed_level_db))

        if self._setup_stage == "capture_noise":
            self._handle_noise_capture(level_db, timestamp)
            return
        if self._setup_stage == "capture_singing":
            self._handle_singing_capture(level_db, timestamp)
            return
        if not self.armed:
            return

        now_seconds = timestamp - self._stream_started_at if self._stream_started_at is not None else None
        for event in self._handle_frame(frame, self._smoothed_level_db, now_seconds=now_seconds):
            self.emit(event)

    def _handle_noise_capture(self, level_db: float, timestamp: float) -> None:
        if self._capture_started_at is None:
            self._capture_started_at = timestamp
        self._room_noise_levels.append(level_db)
        elapsed = (timestamp - self._capture_started_at) + 1e-9
        if elapsed < self.config.calibration_noise_seconds:
            return
        self.noise_floor_db = float(np.percentile(self._room_noise_levels, 95))
        self._capture_started_at = None
        self._setup_stage = "setup"
        self._stop_mic()
        self.emit({"type": "noise_floor_captured", "noise_floor_db": self.noise_floor_db})
        self.emit(self._state_event("setup"))

    def _handle_singing_capture(self, level_db: float, timestamp: float) -> None:
        if self._capture_started_at is None:
            self._capture_started_at = timestamp
        self._singing_levels.append(level_db)
        elapsed = (timestamp - self._capture_started_at) + 1e-9
        if elapsed < self.config.calibration_singing_seconds:
            return
        self._capture_started_at = None
        self._setup_stage = "setup"
        self._stop_mic()
        try:
            payload = self._derive_gate_from_singing()
            self.emit(payload)
            self._commit_setup()
        except RuntimeError as exc:
            self.emit({"type": "setup_error", "message": str(exc)})
        self.emit(self._state_event("setup"))

    def _derive_gate_from_singing(self) -> Dict[str, Any]:
        if self.noise_floor_db is None:
            raise RuntimeError("Noise floor is not available.")
        voiced = [level for level in self._singing_levels if level >= self.noise_floor_db + 3.0]
        if len(voiced) < 5:
            raise RuntimeError("Not enough performance-level singing was captured.")

        self.singing_soft_db = float(np.percentile(voiced, 20))
        self.singing_median_db = float(np.percentile(voiced, 50))
        separation_db = self.singing_soft_db - self.noise_floor_db
        if separation_db < self.config.calibration_min_separation_db:
            raise RuntimeError(f"Singing/noise separation too small ({separation_db:.1f} dB).")

        gate_open_db = max(self.noise_floor_db + 3.0, min(self.singing_soft_db - 1.5, self.noise_floor_db + separation_db * 0.4))
        gate_open_db = min(gate_open_db, self.singing_soft_db)
        gate_close_db = max(self.noise_floor_db + 1.5, gate_open_db - 6.0)
        self._candidate_gate_open_db = float(gate_open_db)
        self._candidate_gate_close_db = float(gate_close_db)
        return {
            "type": "singing_level_captured",
            "singing_soft_db": self.singing_soft_db,
            "singing_median_db": self.singing_median_db,
            "gate_open_db": self._candidate_gate_open_db,
            "gate_close_db": self._candidate_gate_close_db,
        }

    def _commit_setup(self) -> None:
        if self._candidate_gate_open_db is None or self._candidate_gate_close_db is None:
            return
        self.config.gate_open_db = float(self._candidate_gate_open_db)
        self.config.gate_close_db = float(self._candidate_gate_close_db)
        self.segmenter = PhraseSegmenter(self.config)
        self.calibrated = True
        self.setup_mode = False
        self._setup_stage = "idle"
        self.emit(
            {
                "type": "setup_ready",
                "noise_floor_db": self.noise_floor_db,
                "singing_soft_db": self.singing_soft_db,
                "singing_median_db": self.singing_median_db,
                "gate_open_db": self.config.gate_open_db,
                "gate_close_db": self.config.gate_close_db,
            }
        )

    def _meter_event(self, level_db: float, envelope_db: float) -> Dict[str, Any]:
        gate_open_db = self._candidate_gate_open_db if self._candidate_gate_open_db is not None else self.config.gate_open_db
        gate_close_db = self._candidate_gate_close_db if self._candidate_gate_close_db is not None else self.config.gate_close_db
        if envelope_db >= gate_open_db:
            gate_state = "open"
        elif envelope_db >= gate_close_db:
            gate_state = "arming"
        else:
            gate_state = "below"
        return {
            "type": "mic_level",
            "level_db": level_db,
            "envelope_db": envelope_db,
            "gate_state": gate_state,
            "gate_open_db": gate_open_db,
            "gate_close_db": gate_close_db,
        }

    def _handle_mic_error(self, message: str) -> None:
        self.emit({"type": "error", "message": message})

    def _handle_frame(self, frame: np.ndarray, level_db: float, *, now_seconds: Optional[float] = None) -> List[Dict[str, Any]]:
        events = self.segmenter.process_frame(frame, level_db, now_seconds=now_seconds)
        return self._expand_events(events)

    def _process_phrase_offline(self, waveform: np.ndarray, level_db: float) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        events.extend(self.segmenter.process_frame(waveform, level_db, now_seconds=0.0))
        silent_tail = np.zeros(self.config.release_samples(), dtype=np.float32)
        events.extend(
            self.segmenter.process_frame(
                silent_tail,
                self.config.meter_floor_db,
                now_seconds=float(waveform.size / self.config.sample_rate),
            )
        )
        return self._expand_events(events)

    def _save_onset_debug(self, phrase: SegmentedPhrase) -> Optional[Dict[str, Any]]:
        if not self.config.save_onset_debug_plots:
            return None
        try:
            repo_root = Path(self.model_path).resolve().parent
            output_path = repo_root / self.config.onset_debug_dir / f"phrase_{phrase.phrase_id:04d}.png"
            saved = save_onset_debug_plot(
                phrase.onset_analysis,
                output_path,
                title=f"Phrase {phrase.phrase_id} onset debug",
            )
            return {
                "type": "onset_debug_saved",
                "phrase_id": phrase.phrase_id,
                "path": str(saved),
                "num_onsets": len(phrase.onset_offsets_seconds),
            }
        except Exception as exc:
            return {
                "type": "error",
                "message": f"failed to save onset debug plot for phrase {phrase.phrase_id}: {exc}",
            }

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
                out.append(
                    {
                        "type": "phrase_summary",
                        "phrase_id": phrase.phrase_id,
                        "duration_seconds": phrase.waveform.size / self.config.sample_rate,
                        "num_segments": len(phrase.segments),
                        "num_onsets": len(phrase.onset_offsets_seconds),
                        "onset_offsets_seconds": phrase.onset_offsets_seconds,
                    }
                )
                debug_event = self._save_onset_debug(phrase)
                if debug_event is not None:
                    out.append(debug_event)
                retrieval = self._retrieve_phrase(phrase)
                retrieval_event = retrieval.to_event()
                out.append(retrieval_event)
                out.append(self._state_event("playing_sequence" if retrieval.matches else "listening"))
                self._schedule_sequence(retrieval_event)
        return out

    def _retrieve_phrase(self, phrase: SegmentedPhrase):
        if self.retrieval is None:
            raise RuntimeError("Corpus is not loaded. Call load_corpus before processing phrases.")
        return self.retrieval.query_segments(phrase.phrase_id, phrase.segments)

    def _schedule_sequence(self, event: Dict[str, Any]) -> None:
        if self.armed:
            self._resume_after_playback = True
            self._active_playback_phrase_id = int(event["phrase_id"])
            self._stop_mic()
        self._player.play_sequence(event["phrase_id"], event["matches"])

    def _emit_segment_started(self, phrase_id: int, match: Dict[str, Any]) -> None:
        self.emit(
            {
                "type": "segment_playback_started",
                "phrase_id": phrase_id,
                "segment_index": match["segment_index"],
                "scheduled_offset_seconds": match.get("scheduled_offset_seconds", 0.0),
                "metadata": match["metadata"],
            }
        )

    def _emit_segment_finished(self, phrase_id: int, match: Dict[str, Any]) -> None:
        self.emit(
            {
                "type": "segment_playback_finished",
                "phrase_id": phrase_id,
                "segment_index": match["segment_index"],
                "scheduled_offset_seconds": match.get("scheduled_offset_seconds", 0.0),
                "metadata": match["metadata"],
            }
        )

    def _emit_sequence_finished(self, phrase_id: int) -> None:
        self.emit({"type": "sequence_playback_finished", "phrase_id": phrase_id})
        if self._resume_after_playback and self.armed and self._active_playback_phrase_id == phrase_id:
            self._resume_after_playback = False
            self._active_playback_phrase_id = None
            self._start_mic()
            self.emit(self._state_event("listening"))

    def _state_event(self, state: str) -> Dict[str, Any]:
        self._last_state = state
        event: Dict[str, Any] = {
            "type": "state",
            "state": state,
            "armed": self.armed,
            "calibrated": self.calibrated,
            "model_style": self.embedder.model_style,
            "live_sample_rate": self.config.sample_rate,
            "onset_sample_rate": self.config.onset_sample_rate,
            "embedding_sample_rate": 16000,
            "setup_mode": self.setup_mode,
            "setup_stage": self._setup_stage,
            "config": asdict(self.config),
            "model_path": self.model_path,
            "noise_floor_db": self.noise_floor_db,
            "singing_soft_db": self.singing_soft_db,
            "singing_median_db": self.singing_median_db,
            "gate_open_db": self._candidate_gate_open_db if self._candidate_gate_open_db is not None else self.config.gate_open_db,
            "gate_close_db": self._candidate_gate_close_db if self._candidate_gate_close_db is not None else self.config.gate_close_db,
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
