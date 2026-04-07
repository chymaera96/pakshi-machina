import tempfile
import unittest
from pathlib import Path

import numpy as np

from setup_ml4bl import write_wav_manifest
from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import load_corpus_bundle, write_metadata_jsonl
from src.pakshi.retrieval import NumpyFlatL2Index, RetrievalEngine, normalize_rows
from src.pakshi.segmentation import PhraseSegmenter, split_into_segments


class DummyEmbedder:
    def embed_batch(self, waveforms: np.ndarray) -> np.ndarray:
        means = waveforms.mean(axis=1, keepdims=True)
        maxes = waveforms.max(axis=1, keepdims=True)
        return np.concatenate([means, maxes], axis=1).astype(np.float32)


class PakshiRuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
