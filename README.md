# Pakshi Machina

## Overview
Pakshi Machina listens for a sung phrase, detects onset timings inside it, embeds each onset-driven window, and plays back matching bird vocalisations at the same rhythmic offsets.

This branch is `effnet_bio`-only.

- Query window: `0.25 s`
- Live/runtime sample rate: `16 kHz`
- Onset analysis sample rate: `16 kHz`
- Embedding model: `effnet_bio_zf_emb1024.onnx`
- Embedding output: `1024`-dim L2-normalized vector

The current backend uses the preprocessing contract from [`trained_models/effnet_bio/inference.py`](/Users/abhattacharjee/Downloads/trained_models/effnet_bio/inference.py). There is no `README.md` inside `trained_models/effnet_bio`; that inference file is the source of truth.

## Model Contract
The shipped ONNX model expects a mel spectrogram, not raw waveform:

- input name: `mel_spec`
- input shape: `[B, 3, 128, 26]`
- output: `[B, 1024]`

Pakshi computes the frontend locally in NumPy:

- mono audio
- resample to `16 kHz`
- zero-pad or crop to `250 ms` (`4000` samples)
- STFT with `n_fft=800`, `hop=160`, `win_length=800`
- `128` mel bins
- `log(mel + 1e-6)` then min-max normalize to `[0, 1]`
- repeat to `3` channels for EfficientNet input

## Assets
Expected model path:

- `/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx`

Default bundle path for this branch:

- `pakshi_bundle_effnet_bio`

The ML4BL corpus audio is typically `48 kHz` stereo. Pakshi resamples it internally to `16 kHz` so corpus build and live retrieval use the exact same preprocessing.

## Build The Corpus
From the repo root:

```bash
conda activate pakshi-machina
python build_pakshi_corpus.py \
  --model /Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx \
  --manifest data/ml4bl_wavs.jsonl \
  --batch-size 32 \
  --out_dir pakshi_bundle_effnet_bio
```

If you switch to a different ONNX, rebuild the bundle with that same model.

## Run The App
```bash
conda activate pakshi-machina
cd pakshi_ui
npm start
```

Electron defaults to:

- model: `~/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx`
- bundle: `pakshi_bundle_effnet_bio`

You can override them with:

- `PAKSHI_MODEL_PATH`
- `PAKSHI_BUNDLE_PATH`
- `PAKSHI_PYTHON_PATH`

## Runtime Notes
- Setup flow is still:
  - `Capture Noise Floor`
  - `Capture Singing Level`
- Phrase segmentation is still noise-gate driven.
- Inside each phrase, detected onsets create overlapping or non-overlapping `0.25 s` query windows.
- Playback keeps the onset timing from the singer’s phrase.
- The mic is muted during playback to prevent recursive self-triggering.

## Tests
```bash
/Users/abhattacharjee/miniforge3/envs/pakshi-machina/bin/python -m unittest discover -s tests -p 'test_*.py'
```
