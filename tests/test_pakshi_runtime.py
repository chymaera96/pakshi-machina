import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from setup_ml4bl import write_wav_manifest
from src.pakshi.audio import render_crossfaded_sequence
from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import load_corpus_bundle, write_metadata_jsonl
from src.pakshi.retrieval import NumpyFlatL2Index, OnnxEmbeddingModel, RetrievalEngine, normalize_rows
from src.pakshi.segmentation import PhraseSegmenter, split_into_segments
from src.pakshi.worker import SegmentedPhraseWorker


class DummyEmbedder:
    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        means = waveforms.mean(axis=1, keepdims=True)
        maxes = waveforms.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)


class FakeOnnxEmbeddingModel:
    def __init__(self, model_path):
        self.model_path = str(model_path)

    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        waveforms = np.asarray(waveforms, dtype=np.float32)
        means = waveforms.mean(axis=1, keepdims=True)
        maxes = waveforms.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)


class FakeOrtSession3D:
    def __init__(self):
        self._inputs = [type("Input", (), {"name": "waveform"})()]
        self._outputs = [type("Output", (), {"name": "embedding"})()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        batch = feeds["waveform"].shape[0]
        base = np.arange(batch * 2 * 3, dtype=np.float32).reshape(batch, 2, 3)
        return [base]


class FakeOrtSessionSingletonLeading:
    def __init__(self):
        self._inputs = [type("Input", (), {"name": "waveform"})()]
        self._outputs = [type("Output", (), {"name": "embedding"})()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        batch = feeds["waveform"].shape[0]
        base = np.arange(batch * 3, dtype=np.float32).reshape(1, batch, 3)
        return [base]


class FakeConfidenceEstimator:
    def estimate(self, waveform: np.ndarray) -> float:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size == 0:
            return 0.0
        return 0.8 if float(np.max(np.abs(waveform))) > 0 else 0.0


class FakeLiveInputStream:
    def __init__(self, *, sample_rate, frame_samples, on_frame, on_error):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.on_frame = on_frame
        self.on_error = on_error
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False


class PakshiRuntimeTests(unittest.TestCase):
    def test_runtime_config_exposes_input_frame_samples(self):
        cfg = RuntimeConfig(sample_rate=32000, input_frame_seconds=0.1)
        self.assertEqual(cfg.input_frame_samples(), 3200)

    def test_render_crossfaded_sequence_overlaps_adjacent_clips(self):
        clip_a = np.ones(4, dtype=np.float32)
        clip_b = np.full(4, 2.0, dtype=np.float32)
        rendered, spans = render_crossfaded_sequence(
            [clip_a, clip_b],
            sample_rate=4,
            crossfade_seconds=0.5,
        )
        self.assertEqual(rendered.shape, (6,))
        self.assertEqual(spans, [(0, 4), (2, 6)])

    def test_split_into_segments_exact_and_padded(self):
        sr = 10
        waveform = np.arange(25, dtype=np.float32)
        segments = split_into_segments(waveform, sample_rate=sr, segment_seconds=2.0)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].waveform.size, 20)
        self.assertEqual(segments[1].waveform.size, 20)
        np.testing.assert_array_equal(segments[1].waveform[:5], waveform[20:25])
        np.testing.assert_array_equal(segments[1].waveform[5:], np.zeros(15, dtype=np.float32))

    def test_phrase_segmenter_detects_single_phrase(self):
        cfg = RuntimeConfig(
            sample_rate=10,
            segment_seconds=2.0,
            pre_roll_seconds=0.0,
            onset_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=10.0,
        )
        segmenter = PhraseSegmenter(cfg)
        events = []
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), 0.8, now_seconds=0.0))
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), 0.8, now_seconds=1.0))
        events.extend(segmenter.process_frame(np.zeros(10, dtype=np.float32), 0.0, now_seconds=2.0))
        kinds = [event["type"] for event in events]
        self.assertIn("phrase_started", kinds)
        self.assertIn("phrase_ended", kinds)
        self.assertIn("segments_created", kinds)
        segments_event = next(event for event in events if event["type"] == "segments_created")
        self.assertEqual(segments_event["num_segments"], 1)

    def test_short_phrase_is_dropped(self):
        cfg = RuntimeConfig(
            sample_rate=10,
            segment_seconds=2.0,
            pre_roll_seconds=0.0,
            onset_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=3.0,
        )
        segmenter = PhraseSegmenter(cfg)
        events = []
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), 0.8, now_seconds=0.0))
        events.extend(segmenter.process_frame(np.zeros(10, dtype=np.float32), 0.0, now_seconds=1.0))
        self.assertFalse(any(event["type"] == "segments_created" for event in events))

    def test_retrieval_normalizes_before_search(self):
        metadata = [
            {"path": "a.wav", "name": "A"},
            {"path": "b.wav", "name": "B"},
        ]
        corpus_embeddings = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        index = NumpyFlatL2Index.from_embeddings(corpus_embeddings)
        engine = RetrievalEngine(DummyEmbedder(), index, metadata)
        segments = split_into_segments(np.ones(20, dtype=np.float32), sample_rate=10, segment_seconds=2.0)
        sequence = engine.query_segments(phrase_id=1, segments=segments)
        self.assertEqual(len(sequence.matches), 1)
        self.assertEqual(sequence.matches[0].metadata["name"], "A")

    def test_onnx_embedding_model_pools_time_axis_outputs(self):
        model = OnnxEmbeddingModel.__new__(OnnxEmbeddingModel)
        model.session = FakeOrtSession3D()
        model.input_name = "waveform"
        model.output_name = "embedding"
        pooled = model.embed_batch(np.ones((1, 20), dtype=np.float32))
        self.assertEqual(pooled.shape, (1, 3))
        np.testing.assert_allclose(pooled[0], np.array([1.5, 2.5, 3.5], dtype=np.float32))

    def test_onnx_embedding_model_squeezes_singleton_leading_axis(self):
        model = OnnxEmbeddingModel.__new__(OnnxEmbeddingModel)
        model.session = FakeOrtSessionSingletonLeading()
        model.input_name = "waveform"
        model.output_name = "embedding"
        pooled = model.embed_batch(np.ones((2, 20), dtype=np.float32))
        self.assertEqual(pooled.shape, (2, 3))
        np.testing.assert_allclose(
            pooled,
            np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        )

    def test_corpus_bundle_falls_back_to_numpy_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            np.save(bundle / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            write_metadata_jsonl(
                [
                    {"path": "a.wav", "name": "A"},
                    {"path": "b.wav", "name": "B"},
                ],
                bundle / "metadata.jsonl",
            )
            corpus = load_corpus_bundle(bundle)
            self.assertEqual(corpus.backend, "numpy")
            self.assertEqual(len(corpus.metadata), 2)

    def test_ml4bl_manifest_writer_lists_wavs(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_dir = Path(tmp) / "wavs"
            wav_dir.mkdir(parents=True)
            (wav_dir / "a.wav").write_bytes(b"")
            (wav_dir / "b.wav").write_bytes(b"")
            manifest_path = Path(tmp) / "manifest.jsonl"
            write_wav_manifest(wav_dir, manifest_path)
            rows = manifest_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn('"id": "a"', rows[0])
            self.assertIn('"id": "b"', rows[1])

    def test_repo_default_assets_exist(self):
        repo_root = Path(__file__).resolve().parents[1]
        onnx_files = sorted(repo_root.glob("*.onnx"))
        self.assertEqual(len(onnx_files), 1)
        self.assertTrue((repo_root / "pakshi_bundle").exists())
        self.assertTrue((repo_root / "pakshi_bundle" / "metadata.jsonl").exists())
        self.assertTrue((repo_root / "pakshi_bundle" / "index.faiss").exists())

    def test_worker_can_load_default_bundle_with_fakes(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), \
             mock.patch("src.pakshi.worker.build_confidence_estimator", return_value=FakeConfidenceEstimator()), \
             mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(model_path=model_path)
            events = worker.handle_command(
                {"command": "load_corpus", "bundle_dir": str(repo_root / "pakshi_bundle")}
            )
            self.assertEqual(events[0]["type"], "state")
            self.assertEqual(events[0]["corpus_dir"], str(repo_root / "pakshi_bundle"))
            self.assertGreater(events[0]["num_items"], 0)

    def test_worker_arm_starts_fake_mic_and_restart_keeps_listening(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), \
             mock.patch("src.pakshi.worker.build_confidence_estimator", return_value=FakeConfidenceEstimator()), \
             mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(model_path=model_path)
            worker.handle_command(
                {"command": "load_corpus", "bundle_dir": str(repo_root / "pakshi_bundle")}
            )
            arm_events = worker.handle_command({"command": "arm"})
            self.assertTrue(worker.armed)
            self.assertTrue(worker._mic is not None and worker._mic.active)
            self.assertEqual(arm_events[-1]["state"], "listening")

            restart_events = worker.handle_command({"command": "clear_queue_restart"})
            self.assertEqual(restart_events[0]["type"], "queue_cleared")
            self.assertEqual(restart_events[-1]["state"], "listening")
            self.assertTrue(worker._mic is not None and worker._mic.active)

    def test_worker_process_phrase_returns_retrieval_events(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        cfg = RuntimeConfig(
            sample_rate=10,
            segment_seconds=2.0,
            pre_roll_seconds=0.0,
            onset_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=10.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            embeddings = normalize_rows(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32))
            np.save(bundle / "embeddings.npy", embeddings)
            write_metadata_jsonl(
                [
                    {"path": "a.wav", "name": "A"},
                    {"path": "b.wav", "name": "B"},
                ],
                bundle / "metadata.jsonl",
            )
            with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), \
                 mock.patch("src.pakshi.worker.build_confidence_estimator", return_value=FakeConfidenceEstimator()), \
                 mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
                worker = SegmentedPhraseWorker(model_path=model_path, config=cfg)
                worker.handle_command({"command": "load_corpus", "bundle_dir": str(bundle)})
                worker.handle_command({"command": "arm"})
                events = worker.handle_command(
                    {
                        "command": "process_phrase",
                        "waveform": np.ones(20, dtype=np.float32).tolist(),
                        "confidence": 0.9,
                    }
                )
                event_types = [event["type"] for event in events]
                self.assertIn("phrase_started", event_types)
                self.assertIn("phrase_ended", event_types)
                self.assertIn("segments_created", event_types)
                self.assertIn("retrieval_sequence_ready", event_types)


if __name__ == "__main__":
    unittest.main()
