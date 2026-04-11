# Pakshi Machina

## Overview
Pakshi Machina listens for a sung phrase, detects pitch-change boundaries inside it, embeds each pitch-driven window, and plays back matching bird vocalisations at the same rhythmic offsets.

This branch supports two retrieval backends:

- `effnet_bio`
- `crepe_latent`

- Query window: `0.25 s`
- Live/runtime sample rate: `16 kHz`
- Pitch analysis sample rate: `16 kHz`
- Retrieval model family is switchable in the UI or via env/CLI overrides

Backend contracts:

- `effnet_bio`: uses the preprocessing contract from [`trained_models/effnet_bio/inference.py`](/Users/abhattacharjee/Downloads/trained_models/effnet_bio/inference.py)
- `crepe_latent`: uses the waveform-to-latent ONNX contract from `v2.0`

## Model Contract
`effnet_bio` expects a mel spectrogram, not raw waveform:

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

`crepe_latent` expects:

- input name: `audio`
- waveform resampled to `16 kHz`
- repeat-pad or crop to `1.0 s`
- output pooled to a single latent vector per query window

## Assets
Expected model paths:

- `/Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx`
- `/Users/abhattacharjee/pakshi-machina/crepe_latent.onnx`

Default bundle paths:

- `pakshi_bundle_effnet_bio`
- `pakshi_bundle_crepe_latent`

The ML4BL corpus audio is typically `48 kHz` stereo. Pakshi resamples it internally to `16 kHz` so corpus build and live retrieval use the exact same preprocessing.

## Build The Corpus
From the repo root:

```bash
conda activate pakshi-machina
python build_pakshi_corpus.py \
  --model /Users/abhattacharjee/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx \
  --model-family effnet_bio \
  --manifest data/ml4bl_wavs.jsonl \
  --batch-size 32 \
  --out_dir pakshi_bundle_effnet_bio
```

For CREPE latent:

```bash
python build_pakshi_corpus.py \
  --model /Users/abhattacharjee/pakshi-machina/crepe_latent.onnx \
  --model-family crepe_latent \
  --manifest data/ml4bl_wavs.jsonl \
  --out_dir pakshi_bundle_crepe_latent
```

If you switch retrieval families, rebuild or load the matching family-specific bundle.

## Run The App
```bash
conda activate pakshi-machina
cd pakshi_ui
npm start
```

Electron defaults to:

- model family: `effnet_bio`
- model: `~/Downloads/trained_models/effnet_bio/effnet_bio_zf_emb1024.onnx`
- bundle: `pakshi_bundle_effnet_bio`

You can override them with:

- `PAKSHI_MODEL_FAMILY`
- `PAKSHI_MODEL_PATH`
- `PAKSHI_PITCH_MODEL_PATH`
- `PAKSHI_BUNDLE_PATH`
- `PAKSHI_PYTHON_PATH`

## Runtime Notes
- Setup flow is still:
  - `Capture Noise Floor`
  - `Capture Singing Level`
- Phrase segmentation is still noise-gate driven.
- Inside each phrase, CREPE pitch changes create overlapping or non-overlapping `0.25 s` query windows.
- Playback keeps the pitch-segment timing from the singer’s phrase.
- The mic is muted during playback to prevent recursive self-triggering.

## Tests
```bash
/Users/abhattacharjee/miniforge3/envs/pakshi-machina/bin/python -m unittest discover -s tests -p 'test_*.py'
```
