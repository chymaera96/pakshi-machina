# pakshi-machina

Realtime segmented phrase retrieval for a voice-and-sound performance system.

## Overview (CREPE branch)
- Capture a sung phrase using a noise-gate envelope VAD tuned through a lightweight setup stage.
- Detect rhythmic onsets inside each completed phrase with a lightweight `librosa` onset curve and peak picker.
- Start one fixed `0.5 s` CREPE-latent retrieval window at each detected onset.
- L2-normalize each segment embedding and retrieve the top-1 match from a static FAISS corpus.
- Schedule the resulting bird sounds at the same onset offsets as the singer.

## Repo Layout
- `src/pakshi/`: Python runtime, phrase segmentation, onset analysis, retrieval, corpus loading, playback helpers
- `pakshi_worker.py`: JSON-lines worker entrypoint
- `build_pakshi_corpus.py`: build a static corpus bundle from audio metadata
- `export_crepe_onnx.py`: helper to export a CREPE-latent ONNX with a Pakshi-friendly contract
- `setup_ml4bl.py`: download and unpack the ML4BL zebra finch dataset into `data/`
- `tests/`: runtime tests
- `pakshi_ui/`: Electron control surface scaffold
- `*.onnx`: the app auto-discovers the single ONNX model in the repo root
- `pakshi_bundle/`: default retrieval bundle used by the app

## Setup
Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate pakshi-machina
```

If you update `environment.yml` later, refresh the env with:

```bash
conda env update -f environment.yml --prune
```

Notes:
- Run the Python commands below inside the active `pakshi-machina` environment.
- The app auto-discovers the single `*.onnx` file in the repo root and uses `pakshi_bundle/` by default.
- You only need `PAKSHI_MODEL_PATH` or `PAKSHI_BUNDLE_PATH` if you want to override those defaults.
- If `faiss-cpu` is unavailable on a machine, the runtime can fall back to `embeddings.npy` with the in-repo NumPy backend.
- `librosa` onset analysis is used for lightweight onset detection inside each VAD-defined phrase.

## ML4BL Dataset Setup
The project uses the ML4BL zebra finch dataset from Zenodo:
- https://zenodo.org/records/5545872

To download and unpack the dataset into `data/ML4BL_ZF/wavs`:

```bash
python setup_ml4bl.py --write-manifest
```

## CREPE ONNX Contract
This branch assumes the model is a CREPE-derived `audio -> latent` ONNX with this contract:
- input tensor name: `audio`
- input shape per query: `1 x 16000`
- waveform: mono, `16 kHz`, `0.5 s` onset window (repeat-padded/cropped for model input)
- preferred output: a single latent vector per query window
- supported fallback output: frame-level latent activations that Pakshi mean-pools across time

The retrieval path intentionally does **not** use the final 360-bin pitch classifier output as the search embedding. It expects the intermediate latent representation before the classifier head.

## Export CREPE To ONNX
Use the helper when you are ready to export a CREPE-latent model:

```bash
python export_crepe_onnx.py --capacity tiny --out crepe_latent.onnx
```

The export helper assumes you have the extra export-time dependencies installed locally. They are separate from the runtime app dependencies.

## Build A Corpus Bundle
```bash
python build_pakshi_corpus.py   --model crepe_latent.onnx   --manifest data/ml4bl_wavs.jsonl   --out_dir pakshi_bundle
```

The bundle contains `metadata.jsonl` plus `index.faiss` and/or `embeddings.npy`.
Whenever you change the ONNX model, rebuild `pakshi_bundle` with that same model.

## Run The Worker (for debugging)

```bash
python pakshi_worker.py   --model crepe_latent.onnx   --bundle pakshi_bundle
```

The worker is the long-running backend. During setup it captures room noise and realistic singing level, then derives gate thresholds automatically. When armed, it opens the microphone, monitors live amplitude in dBFS, closes a phrase via the noise gate, detects onset peaks inside that phrase with `librosa`, runs one CREPE-latent retrieval per onset-driven window, and schedules playback at the same rhythmic offsets.

You do not need to start `pakshi_worker.py` manually when using the Electron app. The app launches it automatically on startup.

## Run The UI
Install the Electron app:

```bash
cd pakshi_ui
npm install
npm start
```

UI notes:
- Use Setup Mode to capture room noise and then capture realistic singing. Setup completes automatically after the singing capture.
- After setup, use the record control to open the mic for live listening.
- During playback, the mic is paused so the system does not trigger on its own bird vocalisations.
- The Playback and Segments panels now show onset counts and onset-triggered bird schedules.
- Tap the lit record control at any time to stop playback and reset the live state.

## Validation
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
