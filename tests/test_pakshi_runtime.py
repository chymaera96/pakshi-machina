import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from setup_ml4bl import write_wav_manifest
from src.pakshi.audio import rms_dbfs
from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import load_corpus_bundle, write_metadata_jsonl
from src.pakshi.retrieval import (
    EFFICIENTAT_TARGET_SAMPLES,
    EFFICIENTAT_TARGET_SR,
    NumpyFlatL2Index,
    OnnxEmbeddingModel,
    RetrievalEngine,
    normalize_rows,
    preprocess_efficientat_waveform,
)
from src.pakshi.segmentation import PhraseSegmenter, split_into_segments
from src.pakshi.worker import SegmentedPhraseWorker


def db_frame(db: float, size: int) -> np.ndarray:
    amplitude = 10 ** (db / 20.0)
    return np.full(size, amplitude, dtype=np.float32)


class DummyEmbedder:
    def __init__(self):
        self.last_sample_rate = None

    def embed_batch(self, waveforms: np.ndarray, sample_rate=None) -> np.ndarray:
        self.last_sample_rate = sample_rate
        means = waveforms.mean(axis=1, keepdims=True)
        maxes = waveforms.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)


class FakeOnnxEmbeddingModel:
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.model_style = "efficientat_fixed_1s"

    def embed_batch(self, waveforms: np.ndarray, sample_rate=None) -> np.ndarray:
        waveforms = np.asarray(waveforms, dtype=np.float32)
        sample_rate = int(sample_rate or EFFICIENTAT_TARGET_SR)
        processed = np.stack(
            [preprocess_efficientat_waveform(waveform, sample_rate) for waveform in waveforms],
            axis=0,
        )
        means = processed.mean(axis=1, keepdims=True)
        maxes = processed.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)


class FakeOrtSession2D:
    def __init__(self):
        self._inputs = [type("Input", (), {"name": "audio"})()]
        self._outputs = [type("Output", (), {"name": "embedding"})()]
        self.last_feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        self.last_feed = feeds["audio"]
        batch = self.last_feed.shape[0]
        return [np.arange(batch * 32, dtype=np.float32).reshape(batch, 32)]


class FakeOrtSessionSingletonLeading:
    def __init__(self):
        self._inputs = [type("Input", (), {"name": "audio"})()]
        self._outputs = [type("Output", (), {"name": "embedding"})()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        batch = feeds["audio"].shape[0]
        return [np.arange(batch * 3, dtype=np.float32).reshape(1, batch, 3)]


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

    def test_rms_dbfs_tracks_signal_level(self):
        self.assertLess(rms_dbfs(np.zeros(10, dtype=np.float32)), -80.0)
        self.assertAlmostEqual(rms_dbfs(np.ones(10, dtype=np.float32)), 0.0, places=4)

    def test_preprocess_efficientat_repeat_pads_short_waveform(self):
        waveform = np.arange(4000, dtype=np.float32)
        processed = preprocess_efficientat_waveform(waveform, EFFICIENTAT_TARGET_SR)
        self.assertEqual(processed.shape, (EFFICIENTAT_TARGET_SAMPLES,))
        np.testing.assert_array_equal(processed[:4000], waveform)
        np.testing.assert_array_equal(processed[4000:8000], waveform[:4000])

    def test_preprocess_efficientat_crops_long_waveform(self):
        waveform = np.arange(40000, dtype=np.float32)
        processed = preprocess_efficientat_waveform(waveform, EFFICIENTAT_TARGET_SR)
        self.assertEqual(processed.shape, (EFFICIENTAT_TARGET_SAMPLES,))
        np.testing.assert_array_equal(processed, waveform[:EFFICIENTAT_TARGET_SAMPLES])

    def test_preprocess_efficientat_resamples_to_32k(self):
        waveform = np.ones(16000, dtype=np.float32)
        processed = preprocess_efficientat_waveform(waveform, 16000)
        self.assertEqual(processed.shape, (EFFICIENTAT_TARGET_SAMPLES,))

    def test_split_into_segments_exact_and_padded(self):
        sr = 10
        waveform = np.arange(25, dtype=np.float32)
        segments = split_into_segments(waveform, sample_rate=sr, segment_seconds=1.0)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].waveform.size, 10)
        self.assertEqual(segments[1].waveform.size, 10)
        self.assertEqual(segments[2].waveform.size, 10)
        np.testing.assert_array_equal(segments[2].waveform[:5], waveform[20:25])
        np.testing.assert_array_equal(segments[2].waveform[5:], np.zeros(5, dtype=np.float32))

    def test_phrase_segmenter_detects_single_phrase(self):
        cfg = RuntimeConfig(
            sample_rate=10,
            segment_seconds=1.0,
            pre_roll_seconds=0.0,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            onset_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=10.0,
        )
        segmenter = PhraseSegmenter(cfg)
        events = []
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=0.0))
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=1.0))
        events.extend(segmenter.process_frame(np.zeros(10, dtype=np.float32), -90.0, now_seconds=2.0))
        kinds = [event["type"] for event in events]
        self.assertIn("phrase_started", kinds)
        self.assertIn("phrase_ended", kinds)
        self.assertIn("segments_created", kinds)

    def test_phrase_segmenter_ignores_short_subthreshold_dip(self):
        cfg = RuntimeConfig(
            sample_rate=10,
            segment_seconds=1.0,
            pre_roll_seconds=0.0,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            onset_hold_seconds=0.1,
            release_seconds=0.3,
            min_phrase_seconds=0.1,
            max_phrase_seconds=10.0,
        )
        segmenter = PhraseSegmenter(cfg)
        events = []
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=0.0))
        events.extend(segmenter.process_frame(np.ones(1, dtype=np.float32), -60.0, now_seconds=1.0))
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=1.1))
        self.assertFalse(any(event["type"] == "phrase_ended" for event in events))

    def test_retrieval_normalizes_before_search(self):
        metadata = [{"path": "a.wav", "name": "A"}, {"path": "b.wav", "name": "B"}]
        corpus_embeddings = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        index = NumpyFlatL2Index.from_embeddings(corpus_embeddings)
        engine = RetrievalEngine(DummyEmbedder(), index, metadata, sample_rate=EFFICIENTAT_TARGET_SR)
        segments = split_into_segments(np.ones(20, dtype=np.float32), sample_rate=10, segment_seconds=1.0)
        sequence = engine.query_segments(phrase_id=1, segments=segments)
        self.assertEqual(len(sequence.matches), 2)
        self.assertEqual(sequence.matches[0].metadata["name"], "A")

    def test_onnx_embedding_model_preprocesses_to_fixed_window(self):
        model = OnnxEmbeddingModel.__new__(OnnxEmbeddingModel)
        model.session = FakeOrtSession2D()
        model.input_name = "audio"
        model.output_name = "embedding"
        model.model_style = "efficientat_fixed_1s"
        _ = model.embed_batch(np.ones((1, 16000), dtype=np.float32), sample_rate=16000)
        self.assertEqual(model.session.last_feed.shape, (1, EFFICIENTAT_TARGET_SAMPLES))

    def test_onnx_embedding_model_squeezes_singleton_leading_axis(self):
        model = OnnxEmbeddingModel.__new__(OnnxEmbeddingModel)
        model.session = FakeOrtSessionSingletonLeading()
        model.input_name = "audio"
        model.output_name = "embedding"
        model.model_style = "efficientat_fixed_1s"
        pooled = model.embed_batch(np.ones((2, EFFICIENTAT_TARGET_SAMPLES), dtype=np.float32), sample_rate=EFFICIENTAT_TARGET_SR)
        self.assertEqual(pooled.shape, (2, 3))

    def test_corpus_bundle_falls_back_to_numpy_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            np.save(bundle / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}, {"path": "b.wav", "name": "B"}], bundle / "metadata.jsonl")
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

    def test_repo_default_assets_exist(self):
        repo_root = Path(__file__).resolve().parents[1]
        self.assertEqual(len(sorted(repo_root.glob("*.onnx"))), 1)
        self.assertTrue((repo_root / "pakshi_bundle").exists())

    def test_worker_can_load_default_bundle_with_fakes(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), mock.patch(
            "src.pakshi.worker.LiveInputStream", FakeLiveInputStream
        ):
            worker = SegmentedPhraseWorker(model_path=model_path)
            events = worker.handle_command({"command": "load_corpus", "bundle_dir": str(repo_root / "pakshi_bundle")})
            self.assertEqual(events[0]["type"], "state")
            self.assertEqual(events[0]["model_style"], "efficientat_fixed_1s")

    def test_setup_flow_derives_thresholds_and_finishes(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        cfg = RuntimeConfig(sample_rate=100, input_frame_seconds=0.1, calibration_noise_seconds=0.3, calibration_singing_seconds=0.5)
        with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), mock.patch(
            "src.pakshi.worker.LiveInputStream", FakeLiveInputStream
        ):
            worker = SegmentedPhraseWorker(model_path=model_path, config=cfg)
            emitted = []
            worker.emit = lambda event: emitted.append(event)
            worker.handle_command({"command": "start_setup"})
            worker.handle_command({"command": "capture_noise_floor"})
            timestamp = 0.0
            for _ in range(4):
                worker._handle_live_frame(db_frame(-56.0, cfg.input_frame_samples()), timestamp)
                timestamp += cfg.input_frame_seconds
            worker.handle_command({"command": "capture_singing_level"})
            for level in [-28.0, -24.0, -22.0, -26.0, -30.0, -25.0]:
                worker._handle_live_frame(db_frame(level, cfg.input_frame_samples()), timestamp)
                timestamp += cfg.input_frame_seconds
            self.assertTrue(any(event["type"] == "setup_ready" for event in emitted))
            self.assertTrue(worker.calibrated)

    def test_setup_rejects_small_separation(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        cfg = RuntimeConfig(sample_rate=100, input_frame_seconds=0.1, calibration_noise_seconds=0.3, calibration_singing_seconds=0.5)
        with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), mock.patch(
            "src.pakshi.worker.LiveInputStream", FakeLiveInputStream
        ):
            worker = SegmentedPhraseWorker(model_path=model_path, config=cfg)
            emitted = []
            worker.emit = lambda event: emitted.append(event)
            worker.handle_command({"command": "start_setup"})
            worker.handle_command({"command": "capture_noise_floor"})
            timestamp = 0.0
            for _ in range(4):
                worker._handle_live_frame(db_frame(-40.0, cfg.input_frame_samples()), timestamp)
                timestamp += cfg.input_frame_seconds
            worker.handle_command({"command": "capture_singing_level"})
            for level in [-35.0, -34.0, -36.0, -35.0, -34.5, -35.5]:
                worker._handle_live_frame(db_frame(level, cfg.input_frame_samples()), timestamp)
                timestamp += cfg.input_frame_seconds
            self.assertTrue(any(event["type"] == "setup_error" for event in emitted))
            self.assertFalse(worker.calibrated)

    def test_worker_process_phrase_returns_retrieval_events(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = sorted(repo_root.glob("*.onnx"))[0]
        cfg = RuntimeConfig(
            sample_rate=10,
            segment_seconds=1.0,
            pre_roll_seconds=0.0,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            onset_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=12.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            embeddings = normalize_rows(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32))
            np.save(bundle / "embeddings.npy", embeddings)
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}, {"path": "b.wav", "name": "B"}], bundle / "metadata.jsonl")
            with mock.patch("src.pakshi.worker.OnnxEmbeddingModel", FakeOnnxEmbeddingModel), mock.patch(
                "src.pakshi.worker.LiveInputStream", FakeLiveInputStream
            ):
                worker = SegmentedPhraseWorker(model_path=model_path, config=cfg)
                worker.handle_command({"command": "load_corpus", "bundle_dir": str(bundle)})
                worker.calibrated = True
                worker.handle_command({"command": "arm"})
                events = worker.handle_command({"command": "process_phrase", "waveform": np.ones(25, dtype=np.float32).tolist(), "level_db": -12.0})
                retrieval = next(event for event in events if event["type"] == "retrieval_sequence_ready")
                self.assertEqual(retrieval["num_segments"], 3)


if __name__ == "__main__":
    unittest.main()
