from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .audio import NoopSequencePlayer, SoundDeviceSequencePlayer
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

    def _build_player(self):
        try:
            return SoundDeviceSequencePlayer(
                sample_rate=self.config.sample_rate,
                output_gain=self.config.output_gain,
                on_started=self._emit_segment_started,
                on_finished=self._emit_segment_finished,
            )
        except Exception:
            return NoopSequencePlayer(on_started=self._emit_segment_started, on_finished=self._emit_segment_finished)

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
            self.armed = True
            return [self._state_event("listening")]
        if kind == "disarm":
            self.armed = False
            return [self._state_event("idle")]
        if kind == "stop_all":
            self._player.stop()
            return [self._state_event("idle")]
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
        self._player = self._build_player()

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
            "config": asdict(self.config),
            "model_path": self.model_path,
        }
        if self.corpus is not None:
            event["corpus_backend"] = self.corpus.backend
            event["corpus_dir"] = str(self.corpus.bundle_dir)
            event["num_items"] = len(self.corpus.metadata)
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
