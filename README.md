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

## 当前实验结果（定性）

我们在 CelebA（`img_align_celeba/`）上做了一个 multi-res（64/96/128）共享主干 + SwitchableBN 的版本（WGAN + weight clipping）。

- **EXP00 / Smiling / 64+96+128 混训**：训到 ~31k iter 时，重建（`Au_hat`/`B0_hat`）已经比较稳定；`Bu`（给 B0 加 smile）通常自然；`A0`（从 Au 去 smile）偶尔会出现肤色/色调漂移（还在继续排查/对比实验）。
- 其它实验队列与运行方式见 `queue.md`。

样例（iter=31318，列顺序：`Au | B0 | A0 | Bu | Au_hat | B0_hat`；用 `kimi -p` 按“笑/不笑/重建像/无漂移”约束挑选）：

![EXP00 Smiling multi-res sample](assets/exp00_smiling_iter31318_row3.jpg)

FID（128x128，N=5000，batch=8）：

| checkpoint iter | FID(Bu -> with_attr) | FID(A0 -> without_attr) |
|---:|---:|---:|
| 20000 | 34.68 | 33.74 |
| 31318 | 16.22 | 13.87 |

## 推理 demo

Swap / interpolation / matrix：

```bash
python -m genegan.cli.test --help
```

## 评估：FID（可选）

本仓库提供一个简单的 FID 评估脚本（对 CelebA 的 attribute 两个 split）：

```bash
python scripts/eval_fid.py \
  --ckpt outputs/<exp>/checkpoints/latest.pt \
  --attribute Smiling \
  --data_root img_align_celeba \
  --img_size 128 \
  --n 5000 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda \
  --out outputs/<exp>/fid_128_n5000.json
```

输出包含两项：
- `fid_Bu_vs_with_attr`：把属性从 `Au` 移植到 `B0` 生成的 `Bu`，与真实 `with_attr` 分布的 FID
- `fid_A0_vs_without_attr`：从 `Au` 去属性生成的 `A0`，与真实 `without_attr` 分布的 FID

## VQVAE 初始化（可选）

如果你已有一个训练好的 VQVAE checkpoint，可用它初始化 GeneGAN 的生成器（splitter/joiner）：

```bash
python -m genegan.cli.train \
  --attribute Smiling \
  --data_root img_align_celeba \
  --exp_dir outputs/exp_vqinit \
  --device cuda \
  --init_vqvae_ckpt /path/to/vqvae.pt
```

如 VQVAE 的 key 命名不匹配，可额外提供显式 mapping：
`--init_vqvae_map mapping.json`（key 必须以 `splitter.` / `joiner.` 开头）。

## 备注：论文 vs 官方 TF 复现

论文展示的是标准 GAN（log D）形式；作者官方 TF 复现实际使用 **WGAN-style loss + weight clipping**，并有特定的 critic 更新 schedule。本实现按官方 TF 复现对齐，优先复现作者代码产出。
