# GeneGAN PyTorch (MPS)

PyTorch 复现 **GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data**（arXiv:1705.04932），以作者官方 TensorFlow 复现为精确对齐基准（网络结构/超参/训练 schedule/WGAN-style loss + weight clipping）。

## 安装

Python ≥ 3.10。

```bash
pip install -e .
```

> Apple Silicon 建议：如遇到个别算子 MPS 不支持，可设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 让其回退到 CPU（会变慢）。

## 数据

默认按官方仓库的 CelebA 结构：

```
datasets/celebA/
  data/                       # 原始 jpg（可选：用于 5 点对齐）
  list_attr_celeba.txt
  list_landmarks_celeba.txt
  align_5p/                   # 预处理输出（256x256）
```

本仓库也支持你已有的 `img_align_celeba/`（通常是 178×218 的官方对齐图）。训练时通过 `--image_dir` 指向它即可。

例如（本仓库根目录下有 `img_align_celeba/` 数据集软链接时）：

```bash
python -m genegan.cli.train \
  --attribute Bangs \
  --data_root img_align_celeba \
  --exp_dir outputs/celeba_bangs_smoke \
  --max_iter 200 \
  --max_images 2000 \
  --critic_n_500 1
```

## 预处理（5 点对齐，输出 256×256）

```bash
python -m genegan.cli.preprocess_celebA \
  --data_dir datasets/celebA \
  --out_dir  datasets/celebA/align_5p \
  --num_workers 0
```

## 训练

```bash
python -m genegan.cli.train \
  --attribute Bangs \
  --data_root datasets/celebA \
  --exp_dir outputs/celebA_Bangs \
  --device mps
```

默认训练/推理分辨率为 `128×128`（`--img_size 128`），并相应扩展了网络结构；如需使用原始 `64×64` 版本，传 `--img_size 64` 即可。

CUDA（Linux/Windows）：

```bash
python -m genegan.cli.train \
  --attribute Bangs \
  --data_root img_align_celeba \
  --exp_dir outputs/celebA_Bangs_cuda \
  --device cuda
```

> 备注：为了避免 GPU 训练过快导致 checkpoint 占满磁盘，CUDA 下默认 `--save_every=20000`、`--sample_every=5000`。如需严格对齐官方 TF 的保存频率，可显式设回 `--save_every 500 --sample_every 500`。

## 推理 demo

Swap / interpolation / matrix：

```bash
python -m genegan.cli.test --help
```

## 备注：论文 vs 官方 TF 复现

论文展示的是标准 GAN（log D）形式；作者官方 TF 复现实际使用 **WGAN-style loss + weight clipping**，并有特定的 critic 更新 schedule。本实现按官方 TF 复现对齐，优先复现作者代码产出。
