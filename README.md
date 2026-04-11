# Pakshi Machina

## Overview
Pakshi Machina listens for a sung phrase, detects pitch-change boundaries inside it, embeds each pitch-driven window, and plays back matching bird vocalisations at the same rhythmic offsets.

This branch supports two retrieval backends:

- `effnet_bio`
- `crepe_latent`


## Assets

Default bundle paths:

- `pakshi_bundle_effnet_bio`
- `pakshi_bundle_crepe_latent`

The ML4BL corpus audio is typically `48 kHz` stereo. Pakshi resamples it internally to `16 kHz` so corpus build and live retrieval use the exact same preprocessing.

## Build The Corpus
From the repo root:

```bash
conda activate pakshi-machina
bash setup.sh
```

## Run The App
```bash
cd pakshi_ui
npm start
```

## Tests
```bash
/Users/abhattacharjee/miniforge3/envs/pakshi-machina/bin/python -m unittest discover -s tests -p 'test_*.py'
```

