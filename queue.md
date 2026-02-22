# Experiment Queue (GeneGAN Multi-Res)

## Ground Rules

- Each experiment runs **~3 hours** (then stop and move to the next).
  - Recommended stop method (graceful, triggers final save): `timeout -s INT 3h <train-cmd>`
  - Recommended runner (auto-prunes after finish): `python scripts/run_3h_experiment.py --exp_dir <exp> --duration 3h --keep 2 -- <train-cmd>`
    - Detach (keeps running if you disconnect): prefix with `setsid -f`
- Disk: checkpoints are large.
  - `latest.pt` is a symlink to the most recent `iter_*.pt` (no duplicate bytes).
  - After an experiment, prune old checkpoints (keep last 2 for safety):
    - `python scripts/prune_checkpoints.py --ckpt_dir <exp_dir>/checkpoints --keep 2`
  - If the last checkpoint’s loss looks exploded / NaN, keep one more older checkpoint manually.
- Evaluation (add FID for every experiment; compare apples-to-apples):
  - We compute two FIDs at **128x128** with `N=5000` samples:
    - `FID(Bu -> with_attr)` and `FID(A0 -> without_attr)`
  - Real split stats are cached under `outputs/_fid_cache/` (first run is slower).
  - Command:
```bash
python scripts/eval_fid.py \
  --ckpt <exp_dir>/checkpoints/latest.pt \
  --attribute Smiling \
  --data_root img_align_celeba \
  --img_size 128 \
  --n 5000 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda \
  --out <exp_dir>/fid_128_n5000.json
```

## Confirmed Training Objective

- Critic is **WGAN-style** (no sigmoid) with **weight clipping** after each D step.

## Queue

### EXP00 (DONE): Baseline Multi-Res Input/Output (64/96/128)

- Purpose: shared trunk + minimal branching; per-resolution BN stats (SwitchableBN).
- Model:
  - Splitter bottleneck spatial size:
    - 64 -> 16x16
    - 96 -> 12x12
    - 128 -> 16x16
  - Joiner shared upsampling trunk (+ shared `to_rgb`), resolution decides whether to run the 3rd up-block.
  - Discriminator fully shared conv trunk + `adaptive_avg_pool2d -> 4x4 -> fc`.
- Training: each iter averages losses across **(64, 96, 128)** batches (same D/G step).
- Command (CUDA, 3h):
```bash
exp=outputs/EXP00_smiling_multires_64_96_128_sharedtrunk_switchbn_wgan_cuda
mkdir -p "$exp"
setsid -f python scripts/run_3h_experiment.py --exp_dir "$exp" --duration 3h --keep 2 -- \
  python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir $exp \
  --device cuda \
  --img_sizes 64 96 128 \
  --img_size 128 \
  --max_iter 300000 \
  --batch_size 64 \
  --num_workers 4 \
  --save_every 20000 \
  --sample_every 20000
```

- Result (3h run):
  - exp: `outputs/EXP00_smiling_multires_64_96_128_sharedtrunk_switchbn_wgan_cuda`
  - last iter: `31318` (ran ~3h twice)
  - last TensorBoard step: `31300` (`loss_D=-0.0115`, `loss_G=0.0849`)
  - checkpoints kept: `iter_020000.pt`, `iter_031318.pt` (+ `latest.pt` symlink)
  - samples: `samples/iter_020000_*.jpg`, `samples/iter_031318_*.jpg`
  - FID @128 N=5000 (batch=8):
    - iter_020000: `FID(Bu->with_attr)=34.68`, `FID(A0->without_attr)=33.74`
    - iter_031318: `FID(Bu->with_attr)=16.22`, `FID(A0->without_attr)=13.87`

### EXP01 (RUN NEXT): Block-Conv At Bottleneck (Partial Sharing Between LocalConv and Conv)

- Purpose: on the bottleneck feature map (16x16 or 12x12), apply **spatial block-wise conv**
  so different blocks have different kernels, but share within each block.
- Implementation idea:
  - Split bottleneck into `GxG` non-overlapping blocks (e.g. `4x4` blocks for 16x16; `3x3` blocks for 12x12),
    run a small conv per block, then stitch back.
  - Use the same block grid for both bg/obj or only for obj (ablation).
- Run 3h with same schedule as EXP00 (new `--bottleneck_blockconv` flag TBD).
  - Current implementation: residual block-wise conv on the **object** subspace at bottleneck:
    - 16x16 uses a 4x4 block grid (16 convs)
    - 12x12 uses a 3x3 block grid (9 convs)
- Command (CUDA, 3h):
```bash
exp=outputs/EXP01_smiling_multires_objblockconv_cuda
mkdir -p "$exp"
setsid -f python scripts/run_3h_experiment.py --exp_dir "$exp" --duration 3h --keep 2 -- \
  python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir $exp \
  --device cuda \
  --img_sizes 64 96 128 \
  --img_size 128 \
  --max_iter 300000 \
  --batch_size 64 \
  --num_workers 4 \
  --obj_blockconv \
  --obj_block_size 4 \
  --save_every 20000 \
  --sample_every 20000
```

### EXP02: 128-In, Multi-Scale Outputs (64/96/128) With Multi-Scale Loss

- Purpose: feed **only 128x128 inputs**, but compute reconstruction/parallelogram (and optionally adv)
  at **64/96/128** scales.
- Implementation idea:
  - Generator produces main output at 128.
  - Derive `out64 = head64(resize(out128, 64))`, `out96 = head96(resize(out128, 96))`.
    - `head64/head96` can be lightweight `1x1 conv` (per your suggestion) before `tanh`/range mapping.
  - Compute multi-scale cycle + parallelogram losses against `resize(Au, s)` / `resize(B0, s)`.
  - Keep adversarial loss only at 128 first (then ablate: add scale-specific critics).
- Run 3h with a new flag `--multiscale_loss` (TBD).

### EXP03: Multi-Res Training Mode Ablation (All-Scales-Per-Iter vs Random-Scale-Per-Iter)

- Purpose: compare stability/speed.
- Variants:
  - `mode=all`: current behavior (each iter averages over 64/96/128)
  - `mode=random`: each iter samples one resolution uniformly (3x cheaper per iter)
- Add CLI flag `--multi_res_mode all|random` (TBD).

### EXP04: Normalization Ablation (SwitchableBN vs GroupNorm)

- Purpose: avoid BN running-stat interference; GN removes need for per-res stats.
- Variants:
  - SwitchableBN (current)
  - GroupNorm (e.g. `32` groups) in G and D

### EXP05: Loss Weighting Ablation Across Scales

- Purpose: balance optimization difficulty (128 tends to dominate compute).
- Variants:
  - equal average (current)
  - weighted by `s^2` (more weight to high-res)
  - weighted by `1/s^2` (more weight to low-res)

### EXP06: Init From Pretrained VQVAE (Ablation: w/ and w/o VQ Init)

- Purpose: warm-start the generator (splitter/joiner) from an existing VQVAE checkpoint, to see if training is more stable / faster.
- Implementation: `--init_vqvae_ckpt <path>` (optional: `--init_vqvae_map <mapping.json>`).
- Note: below commands run in foreground so we can compute FID right after. If you want to detach, prefix the train command with `setsid -f` and run FID later.
- EXP06a (control, no VQ init):
```bash
exp=outputs/EXP06a_smiling_multires_no_vqinit_cuda
mkdir -p "$exp"
python scripts/run_3h_experiment.py --exp_dir "$exp" --duration 3h --keep 2 -- \
  python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir $exp \
  --device cuda \
  --img_sizes 64 96 128 \
  --img_size 128 \
  --max_iter 300000 \
  --batch_size 64 \
  --num_workers 4 \
  --save_every 20000 \
  --sample_every 20000
python scripts/eval_fid.py --ckpt $exp/checkpoints/latest.pt --attribute Smiling --data_root img_align_celeba --img_size 128 --n 5000 --batch_size 8 --num_workers 4 --device cuda --out $exp/fid_128_n5000.json
```
- EXP06b (treatment, VQ init):
```bash
exp=outputs/EXP06b_smiling_multires_vqinit_cuda
vq_ckpt=/path/to/vqvae.pt
mkdir -p "$exp"
python scripts/run_3h_experiment.py --exp_dir "$exp" --duration 3h --keep 2 -- \
  python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir $exp \
  --device cuda \
  --img_sizes 64 96 128 \
  --img_size 128 \
  --max_iter 300000 \
  --batch_size 64 \
  --num_workers 4 \
  --init_vqvae_ckpt $vq_ckpt \
  --save_every 20000 \
  --sample_every 20000
python scripts/eval_fid.py --ckpt $exp/checkpoints/latest.pt --attribute Smiling --data_root img_align_celeba --img_size 128 --n 5000 --batch_size 8 --num_workers 4 --device cuda --out $exp/fid_128_n5000.json
```
