# Pakshi Machina

## BLURB
Pakshi Machina is an improvised set for voice, human and non-human. Using the Machine Learning for Bird Song Learning (ML4BL) dataset, based on zebra finches’ perception of audio similarity, and audio pitch in human perception, the performance explores the anthropocentrism that is often superimposed on other species’ communication. The systems’ different perception models playfully challenge engineering and design norms around working with more-than-human voices. In dialogue, the work explores the plurality of what it means to have a voice, communicate, and be understood.


##  Technical Overview
Pakshi Machina listens for a sung phrase, detects pitch-change boundaries inside it, embeds each pitch-driven window, and plays back matching bird vocalisations at the same rhythmic offsets.

This branch supports two retrieval backends:

- `effnet_bio`
- `crepe_latent`

## Pre-requisites

- Conda https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#regular-installation
- Node https://nodejs.org/en/download

## Assets

Default bundle paths:

- `pakshi_bundle_effnet_bio`
- `pakshi_bundle_crepe_latent`

The ML4BL corpus audio is typically `48 kHz` stereo. Pakshi resamples it internally to `16 kHz` so corpus build and live retrieval use the exact same preprocessing.

## Build The Corpus
From the repo root:

```bash
conda deactivate
conda env remove -n pakshi-machina

conda create -n pakshi-machina python=3.10
conda activate pakshi-machina
conda install numba
conda env update -f environment.yml
```

```bash
conda activate pakshi-machina
bash setup.sh
```

```bash
cd pakshi_ui
npm install
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


## References
- ML4BL dataset: https://zenodo.org/records/5545872 and associated publication: 
Morfi, Veronica, Robert F. Lachlan, and Dan Stowell. **"Deep perceptual embeddings for unlabelled animal sound events."** The Journal of the Acoustical Society of America 150.1 (2021): 2-11.
- EffNet model is trained from the checkpoint available in https://github.com/earthspecies/avex
- 
