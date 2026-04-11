# Debug Notes

## Fixing `numba` / `llvmlite` conflicts in `pakshi-machina`

If the `pakshi-machina` Conda environment is broken because of `numba` / `llvmlite` conflicts, the most reliable fix is to remove the environment and recreate it with Python `3.10`, then install `numba` before applying `environment.yml`.

Run:

```bash
conda deactivate
conda env remove -n pakshi-machina

conda create -n pakshi-machina python=3.10
conda activate pakshi-machina
conda install numba
conda env update -f environment.yml
```

This is the recovery sequence that has been confirmed to work.
