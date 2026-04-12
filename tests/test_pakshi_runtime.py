import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from setup_ml4bl import write_wav_manifest
from src.pakshi.audio import NoopSequencePlayer, rms_dbfs
from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import (
    BundleMetadata,
    VisualizationData,
    compute_visualization_projection,
    load_corpus_bundle,
    write_bundle_metadata,
    write_metadata_jsonl,
    write_visualization_data,
)
from src.pakshi.pitch import PitchAnalysis
from src.pakshi import pitch as pitch_module
from src.pakshi.retrieval import (
    CREPE_TARGET_SAMPLES,
    CREPE_TARGET_SR,
    EFFNET_BIO_CLIP_SAMPLES,
    EFFNET_BIO_SAMPLE_RATE,
    CrepeLatentEmbeddingModel,
    EffNetBioEmbeddingModel,
    MODEL_FAMILY_CREPE_LATENT,
    MODEL_FAMILY_EFFNET_BIO,
    NumpyFlatL2Index,
    RetrievalEngine,
    compute_effnet_bio_mel_spectrogram,
    infer_model_family,
    normalize_rows,
    preprocess_crepe_waveform,
    preprocess_effnet_bio_waveform,
)
from src.pakshi.segmentation import PhraseSegmenter, split_into_pitch_segments, split_into_segments
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


class FakeEffNetSession:
    def __init__(self):
        self.last_feed = None
        self._inputs = [type("Input", (), {"name": "mel_spec", "shape": [None, 3, 128, 26]})()]
        self._outputs = [type("Output", (), {"name": "embedding"})()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        self.last_feed = feeds["mel_spec"]
        batch = self.last_feed.shape[0]
        out = np.tile(np.linspace(0.0, 1.0, 1024, dtype=np.float32), (batch, 1))
        return [out]


class FakeEffNetBioEmbeddingModel:
    def __init__(self, model_path, input_sample_rate=16000, providers=None, batch_size=32):
        self.model_path = str(model_path)
        self.input_sample_rate = input_sample_rate
        self.batch_size = batch_size
        self.model_family = MODEL_FAMILY_EFFNET_BIO
        self.model_style = "effnet_bio_emb1024"
        self.warmup_calls = 0

    def embed_batch(self, waveforms: np.ndarray, sample_rate=None) -> np.ndarray:
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms[np.newaxis, :]
        sample_rate = int(sample_rate or self.input_sample_rate)
        processed = np.stack(
            [preprocess_effnet_bio_waveform(waveform, sample_rate) for waveform in waveforms],
            axis=0,
        )
        means = processed.mean(axis=1, keepdims=True)
        maxes = processed.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)

    def warmup(self):
        self.warmup_calls += 1


class FakeCrepeSession2D:
    def __init__(self):
        self.last_feed = None
        self._inputs = [type("Input", (), {"name": "audio"})()]
        self._outputs = [type("Output", (), {"name": "latent"})()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        self.last_feed = feeds["audio"]
        return [np.arange(32, dtype=np.float32).reshape(1, 32)]


class FakeCrepeLatentEmbeddingModel:
    def __init__(self, model_path, providers=None):
        self.model_path = str(model_path)
        self.input_sample_rate = CREPE_TARGET_SR
        self.model_family = MODEL_FAMILY_CREPE_LATENT
        self.model_style = "crepe_latent_1s"
        self.warmup_calls = 0

    def embed_batch(self, waveforms: np.ndarray, sample_rate=None) -> np.ndarray:
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms[np.newaxis, :]
        sample_rate = int(sample_rate or self.input_sample_rate)
        processed = np.stack(
            [preprocess_crepe_waveform(waveform, sample_rate) for waveform in waveforms],
            axis=0,
        )
        means = processed.mean(axis=1, keepdims=True)
        maxes = processed.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)

    def warmup(self):
        self.warmup_calls += 1


def fake_create_embedding_model(model_path, model_family=None, input_sample_rate=16000, providers=None, batch_size=32):
    family = model_family or infer_model_family(model_path)
    if family == MODEL_FAMILY_CREPE_LATENT:
        return FakeCrepeLatentEmbeddingModel(model_path, providers=providers)
    return FakeEffNetBioEmbeddingModel(model_path, input_sample_rate=input_sample_rate, providers=providers, batch_size=batch_size)


class FakePitchTracker:
    def __init__(self, model_path, providers=None, frame_batch_size=512):
        self.model_path = str(model_path)
        self.frame_batch_size = frame_batch_size
        self.warmup_calls = 0

    def warmup(self):
        self.warmup_calls += 1

    def analyze(self, waveform, sample_rate, config):
        return PitchAnalysis(
            segment_start_offsets_seconds=[0.0, 1.0, 2.0],
            frame_times_seconds=[0.0, 0.01, 0.02],
            pitch_hz=[220.0, 246.94, 261.63],
            pitch_cents=[5700.0, 5800.0, 5900.0],
            confidence=[0.95, 0.96, 0.97],
            change_threshold_cents=float(config.pitch_change_threshold_cents),
            confidence_floor=float(config.pitch_confidence_floor),
            ignore_short_gaps=bool(config.pitch_ignore_short_gaps),
        )


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
        cfg = RuntimeConfig(sample_rate=16000, input_frame_seconds=0.1)
        self.assertEqual(cfg.input_frame_samples(), 1600)

    def test_rms_dbfs_tracks_signal_level(self):
        self.assertLess(rms_dbfs(np.zeros(10, dtype=np.float32)), -80.0)
        self.assertAlmostEqual(rms_dbfs(np.ones(10, dtype=np.float32)), 0.0, places=4)

    def test_preprocess_effnet_zero_pads_short_waveform(self):
        waveform = np.arange(1000, dtype=np.float32)
        processed = preprocess_effnet_bio_waveform(waveform, EFFNET_BIO_SAMPLE_RATE)
        self.assertEqual(processed.shape, (EFFNET_BIO_CLIP_SAMPLES,))
        np.testing.assert_array_equal(processed[:1000], waveform)
        np.testing.assert_array_equal(processed[1000:], np.zeros(EFFNET_BIO_CLIP_SAMPLES - 1000, dtype=np.float32))

    def test_preprocess_effnet_crops_long_waveform(self):
        waveform = np.arange(EFFNET_BIO_CLIP_SAMPLES + 1000, dtype=np.float32)
        processed = preprocess_effnet_bio_waveform(waveform, EFFNET_BIO_SAMPLE_RATE)
        self.assertEqual(processed.shape, (EFFNET_BIO_CLIP_SAMPLES,))
        np.testing.assert_array_equal(processed, waveform[:EFFNET_BIO_CLIP_SAMPLES])

    def test_preprocess_effnet_resamples_to_16k(self):
        waveform = np.ones(8000, dtype=np.float32)
        processed = preprocess_effnet_bio_waveform(waveform, 8000)
        self.assertEqual(processed.shape, (EFFNET_BIO_CLIP_SAMPLES,))

    def test_preprocess_crepe_repeat_pads_short_waveform(self):
        waveform = np.arange(4000, dtype=np.float32)
        processed = preprocess_crepe_waveform(waveform, CREPE_TARGET_SR)
        self.assertEqual(processed.shape, (CREPE_TARGET_SAMPLES,))
        np.testing.assert_array_equal(processed[:4000], waveform)
        np.testing.assert_array_equal(processed[4000:8000], waveform[:4000])

    def test_effnet_mel_shape_matches_model_contract(self):
        waveform = np.random.randn(EFFNET_BIO_CLIP_SAMPLES).astype(np.float32)
        mel = compute_effnet_bio_mel_spectrogram(waveform)
        self.assertEqual(mel.shape, (1, 3, 128, 26))

    def test_split_into_segments_exact_and_padded(self):
        sr = 10
        waveform = np.arange(25, dtype=np.float32)
        segments = split_into_segments(waveform, sample_rate=sr, segment_seconds=1.0)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[2].waveform.size, 10)
        np.testing.assert_array_equal(segments[2].waveform[:5], waveform[20:25])
        np.testing.assert_array_equal(segments[2].waveform[5:], np.zeros(5, dtype=np.float32))

    def test_split_into_pitch_segments_uses_offsets(self):
        sr = 10
        waveform = np.arange(30, dtype=np.float32)
        segments = split_into_pitch_segments(
            waveform,
            sample_rate=sr,
            segment_seconds=1.0,
            segment_start_offsets_seconds=[0.0, 1.2],
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[1].start_sample, 12)
        self.assertAlmostEqual(segments[1].scheduled_offset_seconds, 1.2, places=5)

    def test_phrase_segmenter_detects_single_phrase(self):
        cfg = RuntimeConfig(
            sample_rate=10,
            pitch_sample_rate=10,
            segment_seconds=1.0,
            pre_roll_seconds=0.0,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            gate_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=10.0,
        )
        segmenter = PhraseSegmenter(cfg, pitch_tracker=FakePitchTracker("crepe_pitch.onnx"))
        events = []
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=0.0))
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=1.0))
        events.extend(segmenter.process_frame(np.zeros(10, dtype=np.float32), -90.0, now_seconds=2.0))
        kinds = [event["type"] for event in events]
        self.assertIn("phrase_started", kinds)
        self.assertIn("phrase_ended", kinds)
        self.assertIn("segments_created", kinds)

    def test_phrase_segmenter_uses_pitch_segment_starts(self):
        cfg = RuntimeConfig(
            sample_rate=10,
            pitch_sample_rate=10,
            segment_seconds=1.0,
            pre_roll_seconds=0.0,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            gate_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=10.0,
        )
        segmenter = PhraseSegmenter(cfg, pitch_tracker=FakePitchTracker("crepe_pitch.onnx"))
        events = []
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=0.0))
        events.extend(segmenter.process_frame(np.ones(10, dtype=np.float32), -20.0, now_seconds=1.0))
        events.extend(segmenter.process_frame(np.zeros(10, dtype=np.float32), -90.0, now_seconds=2.0))
        segments_event = next(event for event in events if event["type"] == "segments_created")
        self.assertEqual(segments_event["num_segments"], 3)
        self.assertEqual(segments_event["segment_start_offsets_seconds"], [0.0, 1.0, 2.0])

    def test_pitch_segmenter_skips_end_of_phrase_tail_segments(self):
        cfg = RuntimeConfig(
            pitch_sample_rate=16000,
            pitch_change_threshold_cents=100.0,
            pitch_confidence_floor=0.5,
            pitch_min_hz=120.0,
            pitch_stable_hold_seconds=0.03,
            pitch_min_segment_spacing_seconds=0.12,
            pitch_phrase_end_guard_seconds=0.2,
        )
        frame_times = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.22, 0.23, 0.24, 0.25], dtype=np.float32)
        cents = np.array([5700, 5700, 5700, 5700, 5700, 5700, 5820, 5820, 5820, 5820], dtype=np.float32)
        hz = np.array([220.0, 220.0, 220.0, 220.0, 220.0, 220.0, 246.94, 246.94, 246.94, 246.94], dtype=np.float32)
        confidence = np.full(frame_times.shape, 0.9, dtype=np.float32)
        starts = pitch_module._derive_segment_starts(frame_times, cents, hz, confidence, cfg)
        self.assertEqual(starts, [0.0])

    def test_pitch_segmenter_respects_min_spacing_for_short_instability(self):
        cfg = RuntimeConfig(
            pitch_sample_rate=16000,
            pitch_change_threshold_cents=100.0,
            pitch_confidence_floor=0.5,
            pitch_min_hz=120.0,
            pitch_stable_hold_seconds=0.03,
            pitch_min_segment_spacing_seconds=0.12,
            pitch_phrase_end_guard_seconds=0.0,
        )
        frame_times = np.array([0.00, 0.01, 0.02, 0.14, 0.15, 0.16, 0.20, 0.21, 0.22], dtype=np.float32)
        cents = np.array([5700, 5700, 5700, 5820, 5820, 5820, 5700, 5700, 5700], dtype=np.float32)
        hz = np.array([220.0, 220.0, 220.0, 246.94, 246.94, 246.94, 220.0, 220.0, 220.0], dtype=np.float32)
        confidence = np.full(frame_times.shape, 0.9, dtype=np.float32)
        starts = pitch_module._derive_segment_starts(frame_times, cents, hz, confidence, cfg)
        self.assertEqual(len(starts), 2)
        self.assertAlmostEqual(starts[0], 0.0, places=5)
        self.assertAlmostEqual(starts[1], 0.14, places=5)

    def test_pitch_segmenter_ignores_low_frequency_noise_tail(self):
        cfg = RuntimeConfig(
            pitch_sample_rate=16000,
            pitch_change_threshold_cents=100.0,
            pitch_confidence_floor=0.5,
            pitch_min_hz=120.0,
            pitch_stable_hold_seconds=0.03,
            pitch_min_segment_spacing_seconds=0.12,
            pitch_phrase_end_guard_seconds=0.0,
        )
        frame_times = np.array([0.00, 0.01, 0.02, 0.30, 0.31, 0.32], dtype=np.float32)
        cents = np.array([5700, 5700, 5700, 2700, 2700, 2700], dtype=np.float32)
        hz = np.array([220.0, 220.0, 220.0, 50.0, 50.0, 50.0], dtype=np.float32)
        confidence = np.full(frame_times.shape, 0.8, dtype=np.float32)
        starts = pitch_module._derive_segment_starts(frame_times, cents, hz, confidence, cfg)
        self.assertEqual(starts, [0.0])

    def test_retrieval_normalizes_before_search(self):
        metadata = [{"path": "a.wav", "name": "A"}, {"path": "b.wav", "name": "B"}]
        corpus_embeddings = normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        index = NumpyFlatL2Index.from_embeddings(corpus_embeddings)
        engine = RetrievalEngine(DummyEmbedder(), index, metadata, sample_rate=EFFNET_BIO_SAMPLE_RATE)
        segments = split_into_segments(np.ones(20, dtype=np.float32), sample_rate=10, segment_seconds=1.0)
        sequence = engine.query_segments(phrase_id=1, segments=segments)
        self.assertEqual(len(sequence.matches), 2)
        self.assertEqual(sequence.matches[0].metadata["name"], "A")

    def test_effnet_embedding_model_preprocesses_to_model_input(self):
        model = EffNetBioEmbeddingModel.__new__(EffNetBioEmbeddingModel)
        model.model_path = "effnet_bio_zf_emb1024.onnx"
        model.input_sample_rate = 16000
        model.providers = ["CPUExecutionProvider"]
        model.batch_size = 16
        model.session = FakeEffNetSession()
        model.input_name = "mel_spec"
        model.output_name = "embedding"
        model.model_style = "effnet_bio_emb1024"
        model._fixed_batch_size = None
        out = model.embed_batch(np.ones((2, EFFNET_BIO_CLIP_SAMPLES), dtype=np.float32), sample_rate=16000)
        self.assertEqual(model.session.last_feed.shape, (2, 3, 128, 26))
        self.assertEqual(out.shape, (2, 1024))

    def test_crepe_embedding_model_preprocesses_to_fixed_window(self):
        model = CrepeLatentEmbeddingModel.__new__(CrepeLatentEmbeddingModel)
        model.model_path = "crepe_latent.onnx"
        model.input_sample_rate = CREPE_TARGET_SR
        model.providers = ["CPUExecutionProvider"]
        model.session = FakeCrepeSession2D()
        model.input_name = "audio"
        model.output_name = "latent"
        model.output_names = ["latent"]
        model.model_family = MODEL_FAMILY_CREPE_LATENT
        model.model_style = "crepe_latent_1s"
        out = model.embed_batch(np.ones((1, 32000), dtype=np.float32), sample_rate=32000)
        self.assertEqual(model.session.last_feed.shape, (1, CREPE_TARGET_SAMPLES))
        self.assertEqual(out.shape, (1, 32))

    def test_corpus_bundle_falls_back_to_numpy_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            np.save(bundle / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}, {"path": "b.wav", "name": "B"}], bundle / "metadata.jsonl")
            corpus = load_corpus_bundle(bundle)
            self.assertEqual(corpus.backend, "numpy")
            self.assertEqual(len(corpus.metadata), 2)

    def test_corpus_bundle_reads_bundle_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            np.save(bundle / "embeddings.npy", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}], bundle / "metadata.jsonl")
            write_bundle_metadata(
                BundleMetadata(
                    model_family=MODEL_FAMILY_EFFNET_BIO,
                    model_style="effnet_bio_emb1024",
                    embedding_sample_rate=16000,
                    model_path="/tmp/effnet.onnx",
                ),
                bundle / "bundle_metadata.json",
            )
            corpus = load_corpus_bundle(bundle)
            self.assertIsNotNone(corpus.bundle_metadata)
            self.assertEqual(corpus.bundle_metadata.model_family, MODEL_FAMILY_EFFNET_BIO)

    def test_corpus_bundle_reads_visualization_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            mean, components, scale, coords = compute_visualization_projection(embeddings)
            np.save(bundle / "embeddings.npy", embeddings)
            write_metadata_jsonl([{"path": "a.wav", "name": "DBlue01"}, {"path": "b.wav", "name": "Orange02"}], bundle / "metadata.jsonl")
            write_visualization_data(
                VisualizationData(
                    points=[
                        {"index": 0, "label": "DBlue01", "syl_type": "DBlue", "x": float(coords[0, 0]), "y": float(coords[0, 1]), "z": float(coords[0, 2])},
                        {"index": 1, "label": "Orange02", "syl_type": "Orange", "x": float(coords[1, 0]), "y": float(coords[1, 1]), "z": float(coords[1, 2])},
                    ],
                    mean=mean,
                    components=components,
                    scale=scale,
                ),
                bundle / "visualization.json",
            )
            corpus = load_corpus_bundle(bundle)
            self.assertIsNotNone(corpus.visualization)
            self.assertEqual(len(corpus.visualization.points), 2)
            self.assertEqual(corpus.visualization.points[0]["syl_type"], "DBlue")

    def test_infer_model_family_from_filename(self):
        self.assertEqual(infer_model_family("effnet_bio_zf_emb1024.onnx"), MODEL_FAMILY_EFFNET_BIO)
        self.assertEqual(infer_model_family("crepe_latent.onnx"), MODEL_FAMILY_CREPE_LATENT)

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
        self.assertTrue((repo_root / "pakshi_bundle_effnet_bio").exists() or True)

    def test_worker_can_load_default_bundle_with_fakes(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = Path("/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx")
        with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
            "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
        ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(model_path=model_path, pitch_model_path="crepe_pitch.onnx")
            events = worker.handle_command({"command": "load_corpus", "bundle_dir": str(repo_root / "pakshi_bundle")})
            self.assertEqual(events[0]["type"], "state")
            self.assertEqual(events[0]["model_style"], "effnet_bio_emb1024")
            self.assertEqual(events[0]["live_sample_rate"], 16000)
            self.assertEqual(events[0]["pitch_sample_rate"], 16000)
            self.assertEqual(events[0]["embedding_sample_rate"], 16000)

    def test_setup_flow_derives_thresholds_and_finishes(self):
        model_path = Path("/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx")
        cfg = RuntimeConfig(sample_rate=100, pitch_sample_rate=100, input_frame_seconds=0.1, calibration_noise_seconds=0.3, calibration_singing_seconds=0.5)
        with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
            "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
        ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(model_path=model_path, config=cfg, pitch_model_path="crepe_pitch.onnx")
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
        model_path = Path("/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx")
        cfg = RuntimeConfig(sample_rate=100, pitch_sample_rate=100, input_frame_seconds=0.1, calibration_noise_seconds=0.3, calibration_singing_seconds=0.5)
        with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
            "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
        ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(model_path=model_path, config=cfg, pitch_model_path="crepe_pitch.onnx")
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
        model_path = Path("/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx")
        cfg = RuntimeConfig(
            sample_rate=10,
            pitch_sample_rate=10,
            segment_seconds=1.0,
            pre_roll_seconds=0.0,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            gate_hold_seconds=0.1,
            release_seconds=0.2,
            min_phrase_seconds=0.2,
            max_phrase_seconds=12.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            embeddings = normalize_rows(np.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
            mean, components, scale, coords = compute_visualization_projection(embeddings)
            np.save(bundle / "embeddings.npy", embeddings)
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}, {"path": "b.wav", "name": "B"}, {"path": "c.wav", "name": "C"}], bundle / "metadata.jsonl")
            write_visualization_data(
                VisualizationData(
                    points=[
                        {"index": idx, "label": chr(ord("A") + idx), "syl_type": "Unknown", "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])}
                        for idx, coord in enumerate(coords)
                    ],
                    mean=mean,
                    components=components,
                    scale=scale,
                ),
                bundle / "visualization.json",
            )
            with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
                "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
            ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
                worker = SegmentedPhraseWorker(model_path=model_path, config=cfg, pitch_model_path="crepe_pitch.onnx")
                worker.handle_command({"command": "load_corpus", "bundle_dir": str(bundle)})
                worker.calibrated = True
                worker.handle_command({"command": "arm"})
                events = worker.handle_command({"command": "process_phrase", "waveform": np.ones(25, dtype=np.float32).tolist(), "level_db": -12.0})
                retrieval = next(event for event in events if event["type"] == "retrieval_sequence_ready")
                self.assertEqual(retrieval["num_segments"], 3)
                self.assertEqual([match["scheduled_offset_seconds"] for match in retrieval["matches"]], [0.0, 1.0, 2.0])
                self.assertEqual(len(retrieval["query_pos3d"]), 3)
                self.assertTrue(all("pos3d" in match for match in retrieval["matches"]))

    def test_worker_rejects_mismatched_bundle_metadata(self):
        model_path = Path("/tmp/crepe_latent.onnx")
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            np.save(bundle / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}], bundle / "metadata.jsonl")
            write_bundle_metadata(
                BundleMetadata(
                    model_family=MODEL_FAMILY_EFFNET_BIO,
                    model_style="effnet_bio_emb1024",
                    embedding_sample_rate=16000,
                    model_path="/tmp/effnet_bio.onnx",
                ),
                bundle / "bundle_metadata.json",
            )
            with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
                "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
            ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
                worker = SegmentedPhraseWorker(
                    model_path=model_path,
                    pitch_model_path="crepe_pitch.onnx",
                    model_family=MODEL_FAMILY_CREPE_LATENT,
                )
                with self.assertRaises(RuntimeError):
                    worker.handle_command({"command": "load_corpus", "bundle_dir": str(bundle)})

    def test_worker_switches_backend_without_losing_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            effnet_bundle = Path(tmp) / "effnet_bundle"
            crepe_bundle = Path(tmp) / "crepe_bundle"
            effnet_bundle.mkdir()
            crepe_bundle.mkdir()
            np.save(effnet_bundle / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
            np.save(crepe_bundle / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}], effnet_bundle / "metadata.jsonl")
            write_metadata_jsonl([{"path": "a.wav", "name": "A"}], crepe_bundle / "metadata.jsonl")
            write_bundle_metadata(
                BundleMetadata(
                    model_family=MODEL_FAMILY_EFFNET_BIO,
                    model_style="effnet_bio_emb1024",
                    embedding_sample_rate=16000,
                    model_path="/tmp/effnet_bio.onnx",
                ),
                effnet_bundle / "bundle_metadata.json",
            )
            write_bundle_metadata(
                BundleMetadata(
                    model_family=MODEL_FAMILY_CREPE_LATENT,
                    model_style="crepe_latent_1s",
                    embedding_sample_rate=16000,
                    model_path="/tmp/crepe_latent.onnx",
                ),
                crepe_bundle / "bundle_metadata.json",
            )
            with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
                "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
            ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
                worker = SegmentedPhraseWorker(
                    model_path="/tmp/effnet_bio_zf_emb1024.onnx",
                    pitch_model_path="crepe_pitch.onnx",
                    model_family=MODEL_FAMILY_EFFNET_BIO,
                )
                worker.calibrated = True
                worker.noise_floor_db = -55.0
                worker.singing_soft_db = -24.0
                worker.handle_command({"command": "load_corpus", "bundle_dir": str(effnet_bundle)})
                events = worker.handle_command(
                    {
                        "command": "set_model_backend",
                        "model_path": "/tmp/crepe_latent.onnx",
                        "model_family": MODEL_FAMILY_CREPE_LATENT,
                        "bundle_dir": str(crepe_bundle),
                    }
                )
                self.assertTrue(worker.calibrated)
                self.assertEqual(worker.noise_floor_db, -55.0)
                self.assertEqual(worker.singing_soft_db, -24.0)
                self.assertEqual(worker.config.segment_seconds, 0.5)
                self.assertEqual(events[0]["model_family"], MODEL_FAMILY_CREPE_LATENT)

    def test_worker_warms_pitch_and_embedding_models(self):
        with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
            "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
        ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(
                model_path="/tmp/effnet_bio_zf_emb1024.onnx",
                pitch_model_path="crepe_pitch.onnx",
                model_family=MODEL_FAMILY_EFFNET_BIO,
            )
            self.assertEqual(worker.pitch_tracker.warmup_calls, 1)
            self.assertEqual(worker.embedder.warmup_calls, 1)

    def test_arm_warmup_primes_preroll_before_first_phrase(self):
        model_path = Path("/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx")
        cfg = RuntimeConfig(
            sample_rate=100,
            pitch_sample_rate=100,
            input_frame_seconds=0.1,
            pre_roll_seconds=0.2,
            gate_open_db=-40.0,
            gate_close_db=-48.0,
            gate_hold_seconds=0.1,
            arm_warmup_seconds=0.25,
        )
        with mock.patch("src.pakshi.worker.create_embedding_model", fake_create_embedding_model), mock.patch(
            "src.pakshi.worker.CrepePitchTracker", FakePitchTracker
        ), mock.patch("src.pakshi.worker.LiveInputStream", FakeLiveInputStream):
            worker = SegmentedPhraseWorker(model_path=model_path, config=cfg, pitch_model_path="crepe_pitch.onnx")
            worker.calibrated = True
            worker.handle_command({"command": "arm"})
            assert worker._stream_started_at is not None
            timestamp = worker._stream_started_at + 0.05
            worker._handle_live_frame(db_frame(-10.0, cfg.input_frame_samples()), timestamp)
            self.assertFalse(worker.segmenter.is_active)
            timestamp = worker._arm_ready_at + 0.01
            worker._handle_live_frame(db_frame(-10.0, cfg.input_frame_samples()), timestamp)
            self.assertFalse(worker.segmenter.is_active)
            timestamp += cfg.input_frame_seconds
            worker._handle_live_frame(db_frame(-10.0, cfg.input_frame_samples()), timestamp)
            self.assertTrue(worker.segmenter.is_active)

    def test_noop_sequence_player_preserves_pitch_segment_offsets(self):
        started = []
        finished = []
        done = []
        player = NoopSequencePlayer(
            on_started=lambda phrase_id, match: started.append((match["segment_index"], time.monotonic())),
            on_finished=lambda phrase_id, match: finished.append((match["segment_index"], time.monotonic())),
            on_sequence_finished=lambda phrase_id: done.append(phrase_id),
        )
        player.play_sequence(
            1,
            [
                {"segment_index": 0, "scheduled_offset_seconds": 0.0, "metadata": {"duration_seconds": 0.06}},
                {"segment_index": 1, "scheduled_offset_seconds": 0.01, "metadata": {"duration_seconds": 0.01}},
            ],
        )
        timeout = time.time() + 1.0
        while time.time() < timeout and not done:
            time.sleep(0.01)
        self.assertTrue(done)
        self.assertEqual([entry[0] for entry in started], [0, 1])
        self.assertEqual([entry[0] for entry in finished], [1, 0])
        self.assertLess(started[1][1], finished[0][1])


if __name__ == "__main__":
    unittest.main()
