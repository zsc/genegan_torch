# Lessons Learned (GeneGAN PyTorch + Apple Silicon MPS)

This file captures “gotchas” and working command lines discovered while implementing/running this repo.

## Data layout & file formats

- **`img_align_celeba` often has an extra nesting level**:
  - Common: `.../img_align_celeba/img_align_celeba/000001.jpg`
  - This repo auto-detects and uses the nested folder when present.
- **Attributes/landmarks may be CSV, not TXT** (Kaggle-style exports):
  - `list_attr_celeba.csv` header includes `image_id` + 40 attrs.
  - `list_landmarks_align_celeba.csv` header includes `image_id` + 10 numbers.
  - Code supports both `list_attr_celeba.txt` and `.csv`.

## MPS training (Apple Silicon)

- **Force device**: pass `--device mps`.
- **Enable fallback** when you hit unsupported MPS ops:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **Stay float32** (MPS is sensitive to float64); all model paths should keep `float32`.

## DataLoader on macOS (multiprocessing)

- **If you run with `num_workers>0`, `torch_shm_manager` must be executable**:
  - Symptom: `torch_shm_manager ... Permission denied`
  - Fix (conda env example):
    - `chmod +x $(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent/'bin'/'torch_shm_manager')")`
- **Don’t test `num_workers>0` from an interactive `<stdin>` script**:
  - macOS uses spawn; workers require an importable main module.
  - Use the CLI entrypoints instead (`python -m genegan.cli.train ...`).
- **Safe default**: `--num_workers 0` (slow but reliable). After fixing `torch_shm_manager`, `--num_workers 4` works.

## Training schedule & “10min run” trick

- Official schedule does **100 critic steps when `iter % 500 == 0`**.
  - This makes `iter=0` extremely slow (100× D steps) and can dominate short runs.
- For a quick sanity run, override:
  - `--critic_n_500 1`
- For a more “serious” run, keep defaults (100 at every 500 iters), but expect longer wall time.

## Checkpoints, logs, samples

- **Checkpoint size is large** (hundreds of MB) because it stores:
  - `G_splitter`, `G_joiner`, `D_Ax`, `D_Be`, plus both optimizers.
- Outputs:
  - Checkpoints: `outputs/<exp>/checkpoints/iter_XXXXXX.pt` + `latest.pt`
  - Samples: `outputs/<exp>/samples/iter_XXXXXX_j.jpg` (row order: `[Au, B0, A0, Bu, Au_hat, B0_hat]`)
  - TensorBoard: `outputs/<exp>/logs/`

## Resume

- Use `--resume_ckpt outputs/<exp>/checkpoints/latest.pt` to continue training.
  - Loads model + optim state; continues from `iter + 1`.

## Working command templates

### Short smoke run (minutes)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir outputs/celebA_Smiling_smoke \
  --device mps \
  --max_iter 400 \
  --batch_size 64 \
  --num_workers 0 \
  --critic_n_500 1
```

### Longer run (hours)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir outputs/celebA_Smiling_long \
  --device mps \
  --max_iter 12001 \
  --batch_size 64 \
  --num_workers 4
```

### TensorBoard

```bash
tensorboard --logdir outputs/<exp>/logs
```

