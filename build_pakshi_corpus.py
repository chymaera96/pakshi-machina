#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np

from src.pakshi.config import RuntimeConfig
from src.pakshi.corpus import write_metadata_jsonl
from src.pakshi.retrieval import FaissFlatL2Index, NumpyFlatL2Index, OnnxEmbeddingModel, normalize_rows


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pakshi corpus bundle from a manifest and ONNX model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to waveform->embedding ONNX model")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON or JSONL manifest with sound file metadata")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output corpus bundle directory")
    parser.add_argument("--sample_rate", type=int, default=RuntimeConfig.sample_rate)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    embedder = OnnxEmbeddingModel(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    for row in manifest:
        path = Path(row["path"])
        audio, _ = librosa.load(path, sr=args.sample_rate, mono=True)
        emb = embedder.embed_batch(np.expand_dims(audio.astype(np.float32), axis=0))
        emb = normalize_rows(emb)[0]
        embeddings.append(emb)
        out_row = dict(row)
        out_row["path"] = str(path)
        out_row.setdefault("duration_seconds", float(audio.size / args.sample_rate))
        rows.append(out_row)

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)
    np.save(args.out_dir / "embeddings.npy", emb_arr)
    write_metadata_jsonl(rows, args.out_dir / "metadata.jsonl")

    try:
        FaissFlatL2Index.from_embeddings(emb_arr).save(args.out_dir / "index.faiss")
    except Exception:
        NumpyFlatL2Index.from_embeddings(emb_arr)

    print(f"Wrote corpus bundle to {args.out_dir}")


if __name__ == "__main__":
    main()
