# GeneGAN（arXiv:1705.04932）PyTorch MPS 复现 SPEC

> 目标：把论文 **GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data** 复现为 **PyTorch 实现**，并且在 **Apple Silicon 的 PyTorch MPS** 上可训练/可推理；同时给出清晰的工程结构与 CLI，使 **gemini-cli/codex** 这类代码生成器可以直接按规范落地。

参考依据：论文方法与损失定义（含 Parallelogram constraint、训练数据流），以及作者提供的官方 TensorFlow 复现代码（架构/超参/训练 schedule/实现细节以该代码为准）([GitHub][1])。

---

## 1. 复现范围

### 1.1 必做（MVP）

* 数据集：CelebA（按属性二分类 0/1 拆分为两套“有属性/无属性”的非配对集合）。
* 属性：默认支持任意 CelebA attribute（如 `Bangs` / `Smiling` / `Eyeglasses`），通过 CLI 参数选择。([GitHub][1])
* 训练：在 MPS 上可跑（无 CUDA 依赖），可保存 checkpoint、TensorBoard 日志、样例拼图。([GitHub][2])
* 推理：实现三种 demo 模式

  * swap（属性交换）
  * interpolation（线性插值）
  * matrix（子空间矩阵插值）([GitHub][1])

### 1.2 选做（非 MVP）

* Multi-PIE 复现实验（论文包含，但官方 TF 复现仓库主要给 CelebA 工具链）。
* 更严格的定量指标（论文主要展示定性图）。

---

## 2. 关键事实对齐（“以谁为准”）

论文给出方法框架与损失项，但明确表示“loss 权重细节留给 online implementation”。
因此，本复现 SPEC 采用 **官方 TF 复现代码** 作为“精确实现基准”，包括：

* 网络结构（通道数/层数/卷积核/BN 放置）([GitHub][3])
* 超参（batch、lr、iter、second_ratio、weight_decay）([GitHub][4])
* 训练策略（WGAN-style critic loss + weight clipping、D 更新次数 schedule）([GitHub][3])

同时，论文实验段落提到 RMSProp、lr=5e-5、momentum=0、使用 BatchNorm，有助于交叉验证。

---

## 3. 交付物清单（仓库应包含）

```
genegan-pytorch/
  README.md
  pyproject.toml 或 requirements.txt
  configs/
    celeba_default.yaml
  genegan/
    __init__.py
    cli/
      preprocess_celebA.py
      train.py
      test.py
    data/
      celeba.py
      transforms.py
    models/
      genegan.py          # Splitter/Joiner/Discriminators
      init.py             # weight init
    losses.py
    trainer.py
    utils/
      io.py               # save_image_grid, ckpt IO
      seed.py
      device.py
  outputs/                # 默认输出目录（可配置）
  tests/
    test_shapes.py
    test_forward.py
```

---

## 4. 环境与设备要求

### 4.1 Python 依赖

* Python ≥ 3.10
* torch、torchvision
* numpy、Pillow
* opencv-python（人脸对齐需要）
* tqdm
* tensorboard（或 torch.utils.tensorboard）

### 4.2 设备策略（必须）

* 默认 device 选择顺序：`mps`（可用则用）→ `cuda`（可用则用）→ `cpu`
* 强制使用 `float32`（MPS 上避免 float64）
* 在 README 里说明：如遇到个别算子不支持，可让用户设置环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`（不强制，但建议写入 troubleshooting）

---

## 5. 数据集与目录规范（CelebA）

### 5.1 原始数据目录（与官方 TF 复现一致）

用户下载 CelebA 后，要求目录结构：

```
datasets/
  celebA/
    data/                   # 原始 jpg，000001.jpg ...
    list_attr_celeba.txt
    list_landmarks_celeba.txt
```

并建议检查原始图片尺寸应为 409×687（用于确认下载版本一致）。([GitHub][1])

### 5.2 对齐后目录（preprocess 产物）

```
datasets/
  celebA/
    align_5p/               # preprocess 生成：000001.jpg ...
```

---

## 6. 预处理：5 点人脸对齐（必须复刻）

### 6.1 行为定义

* 输入：`data/*.jpg` + `list_landmarks_celeba.txt`（每张图 5 点，共 10 个数）
* 输出：对齐到 **256×256** 的人脸图，写入 `align_5p/*.jpg`
* 对齐方法：使用官方 TF 复现 `preprocess.py` 的 mean face 与仿射矩阵计算逻辑（必须逐行 port，避免“看起来差不多”的替代实现）。([GitHub][5])

### 6.2 CLI（必须实现）

```bash
python -m genegan.cli.preprocess_celebA \
  --data_dir datasets/celebA \
  --out_dir  datasets/celebA/align_5p \
  --num_workers 30
```

对齐实现依据：官方 preprocess 脚本。([GitHub][5])

---

## 7. 数据加载：无配对二集合采样（必须）

### 7.1 属性拆分规则（与官方一致）

* 读取 `list_attr_celeba.txt`
* 按指定 attribute 列：

  * 值为 `1` → 集合 **A**（有属性，论文记为 `Au`）
  * 值为 `-1` → 集合 **B**（无属性，论文记为 `B0`）([GitHub][4])

### 7.2 训练输入

每个 iteration 需要 **两个 batch**：

* `Ax`：从集合 A 随机采样 batch（Au）
* `Be`：从集合 B 随机采样 batch（B0）

### 7.3 图像处理（与官方 TF 复现对齐）

* 从 `align_5p/*.jpg` 读取
* resize 到 **64×64**
* 输出 tensor 值域：`float32`、范围 **[0, 255]**
* PyTorch 采用 NCHW：`[B, 3, 64, 64]`
  官方 TF 复现使用 `nhwc=[64,64,64,3]` 且网络内部会 `image/255.0`。([GitHub][4])

---

## 8. 模型：网络结构精确规范（以官方 TF 复现为准）

> 说明：论文仅概述“encoder 3 conv、decoder 3 deconv”，实现细节以官方 TF 复现的 `model.py` 为准。

### 8.1 组件与共享关系

* `G_splitter`：Encoder（同一个模块复用在 Au 与 B0 上）
* `G_joiner`：Decoder（同一个模块复用在 Au/A0/Bu/B0 上）
* `D_Ax`：判别器/critic（区分 Au vs Bu）
* `D_Be`：判别器/critic（区分 B0 vs A0）([GitHub][3])

### 8.2 Splitter（Encoder）结构

输入：`[B, 3, 64, 64]`（内部先 /255）

* Conv1: k=4,s=2,p=1, out=128，**无 BN**，LeakyReLU(0.2)
* Conv2: k=4,s=2,p=1, out=256，**BN**，LeakyReLU(0.2)
* Conv3: k=4,s=2,p=1, out=512，**BN**，LeakyReLU(0.2)

输出 feature map：`[B, 512, 8, 8]`
按 channel 切分为：

* 背景 A：`[B, 512 - num_ch, 8, 8]`
* 属性/对象 x：`[B, num_ch, 8, 8]`

其中 `num_ch = int(512 * second_ratio)`，默认 `second_ratio=0.25` → `num_ch=128`。([GitHub][3])

### 8.3 Joiner（Decoder）结构

输入：concat(A, x) → `[B, 512, 8, 8]`

* Deconv1: k=4,s=2,p=1, out=512，BN，ReLU → `[B,512,16,16]`
* Deconv2: k=4,s=2,p=1, out=256，BN，ReLU → `[B,256,32,32]`
* Deconv3: k=4,s=2,p=1, out=3，**无 BN** → `[B,3,64,64]`
* 输出激活：`tanh`，并映射到像素域：`(tanh(x)+1)*255/2`（输出范围 [0,255]）([GitHub][3])

> 注意：官方 TF 中卷积/反卷积权重没有 bias，最后单独加一个 `b`。PyTorch 可选择等价实现：
>
> * conv/deconv bias=False + 单独 Parameter b
> * 或直接让最后一层 deconv bias=True（效果接近但严格不完全等价）；本 SPEC 要求走“等价实现”路线（单独 b）。([GitHub][3])

### 8.4 Discriminator（Critic）结构（每个 D 各一套）

输入：`[B, 3, 64, 64]`（内部先 /255），输出标量 `[B,1]`（无 sigmoid）

* Conv1: k=4,s=2,p=1, out=128，LeakyReLU(0.2)
* Conv2: k=4,s=2,p=1, out=256，BN，LeakyReLU(0.2)
* Conv3: k=4,s=2,p=1, out=512，BN，LeakyReLU(0.2)
* Conv4: k=4,s=2,p=1, out=512，BN，LeakyReLU(0.2)
* Flatten + Linear → 1 ([GitHub][3])

### 8.5 权重初始化（必须）

* 所有 conv/deconv/linear 权重：`Normal(mean=0, std=0.02)`
* BN：gamma=1, beta=0（或 PyTorch 默认后显式设置）([GitHub][3])

---

## 9. 前向图（训练时生成 6 张图）

给定：

* `Au`（代码中 `Ax`）：有属性集合 batch
* `B0`（代码中 `Be`）：无属性集合 batch

计算：

1. `(A, x) = Splitter(Au)`
2. `(B, e) = Splitter(B0)`
3. 重建：

   * `Au_hat = Joiner(A, x)`
   * `B0_hat = Joiner(B, 0)`
4. 交叉重组：

   * `Bu = Joiner(B, x)`（把属性从 Au 移植到 B）
   * `A0 = Joiner(A, 0)`（从 Au 移除属性）

并且对 `e` 施加 “nulling” 约束（让无属性图编码出的对象分量趋近 0）。

---

## 10. 损失函数（以官方 TF 复现为准：WGAN-style）

> 论文写的是 log-based GAN 损失，但官方 TF 复现使用 **WGAN + weight clipping** 的形式；本 SPEC 复刻官方实现。

### 10.1 Generator 总损失 `L_G`

组成（全部为 batch mean）：

* `L_e = mean(|e|)`  （nulling loss）
* `L_cycle_A = mean(|Au - Au_hat|) / 255`
* `L_cycle_B = mean(|B0 - B0_hat|) / 255`
* `L_adv_Bu = - mean(D_Ax(Bu))`
* `L_adv_A0 = - mean(D_Be(A0))`
* `L_para = 0.01 * mean(|Au + B0 - Bu - A0|)`（parallelogram，官方系数 0.01）
* `L_wd = Σ 0.5 * weight_decay * mean(w^2)`（仅对 G 的卷积/反卷积权重；weight_decay 默认 5e-5）([GitHub][4])

最终：

* `L_G = L_e + L_cycle_A + L_cycle_B + L_adv_Bu + L_adv_A0 + L_para + L_wd` ([GitHub][3])

### 10.2 Discriminator/Critic 总损失 `L_D`

两个 critic 各一项：

* `L_D_Ax = mean(D_Ax(Bu) - D_Ax(Au))`
* `L_D_Be = mean(D_Be(A0) - D_Be(B0))`
* `L_D = L_D_Ax + L_D_Be` ([GitHub][3])

### 10.3 Weight clipping（必须）

每次更新 D 之后，对 D 的所有参数执行：

* `param.clamp_(-0.01, 0.01)` ([GitHub][3])

---

## 11. 训练超参与 schedule（必须对齐官方 TF）

默认 config（来自官方 dataset.py）：

* `batch_size = 64`
* `img_size = 64`
* `max_iter = 100000`
* `g_lr = d_lr = 5e-5`（无衰减）
* `weight_decay = 5e-5`
* `second_ratio = 0.25` ([GitHub][4])

优化器：

* `torch.optim.RMSprop(lr=5e-5, alpha=0.8, momentum=0)`（G 与 D 都用）

每个 iteration 的更新策略（来自官方 train.py）：

* 若 `iter % 500 == 0`：D 更新 `100` 次，否则 `1` 次
* G 更新 `1` 次
* 每次 D/G 更新都应获取新的 `(Au_batch, B0_batch)`（可用无限 dataloader iterator 实现）([GitHub][2])

---

## 12. 日志、采样图、checkpoint 规范（必须）

### 12.1 TensorBoard

记录以下标量（名称保持一致，方便对照）：

* G_loss：`e`, `cycle_Ax`, `cycle_Be`, `Bx`, `Ae`, `parallelogram`, `loss_G_nodecay`, `loss_G_decay`, `loss_G`
* D_loss：`Ax_Bx`, `Be_Ae`, `loss_D`
* learning rate：`g_learning_rate`, `d_learning_rate` ([GitHub][2])

### 12.2 采样图输出（必须）

每 500 iter 保存样例图（至少保存 5 张），每张横向拼接 6 幅图，顺序严格一致：

`[Au, B0, A0, Bu, Au_hat, B0_hat]`

文件名建议：

* `iter_{iter:06d}_{j}.jpg`，j=0..4 ([GitHub][2])

### 12.3 Checkpoint

每 500 iter 保存一次（另可保存 latest）：

* `checkpoints/iter_{iter:06d}.pt`
* 内容必须包含：

  * `G_splitter.state_dict`
  * `G_joiner.state_dict`
  * `D_Ax.state_dict`
  * `D_Be.state_dict`
  * `opt_G.state_dict`
  * `opt_D.state_dict`
  * `iter`
  * `config`（或 config hash）

---

## 13. CLI 规范（gemini-cli/codex 友好）

### 13.1 训练

```bash
python -m genegan.cli.train \
  --attribute Bangs \
  --data_root datasets/celebA \
  --exp_dir outputs/celebA_Bangs \
  --device mps \
  --max_iter 100000 \
  --batch_size 64
```

属性选择与 CelebA 文件结构按官方 readme/dataset.py。([GitHub][1])

### 13.2 测试：swap

```bash
python -m genegan.cli.test \
  --mode swap \
  --ckpt outputs/celebA_Bangs/checkpoints/iter_100000.pt \
  --input  datasets/celebA/align_5p/182929.jpg \
  --target datasets/celebA/align_5p/022344.jpg \
  --device mps
```

该模式语义对齐官方 test.py：输出 `out1`（input 加属性）与 `out2`（target 去属性）。([GitHub][6])

### 13.3 测试：interpolation

```bash
python -m genegan.cli.test \
  --mode interpolation \
  --ckpt outputs/celebA_Bangs/checkpoints/iter_100000.pt \
  --input  datasets/celebA/align_5p/182929.jpg \
  --target datasets/celebA/align_5p/035460.jpg \
  --num 5 \
  --device mps
```

### 13.4 测试：matrix

```bash
python -m genegan.cli.test \
  --mode matrix \
  --ckpt outputs/celebA_Bangs/checkpoints/iter_100000.pt \
  --input datasets/celebA/align_5p/182929.jpg \
  --targets \
      datasets/celebA/align_5p/035460.jpg \
      datasets/celebA/align_5p/035451.jpg \
      datasets/celebA/align_5p/035463.jpg \
      datasets/celebA/align_5p/035474.jpg \
  --size 5 5 \
  --device mps
```

矩阵插值逻辑可参考官方实现。([GitHub][6])

---

## 14. 可复现性要求（最低标准）

### 14.1 随机种子

* CLI 支持 `--seed`
* 同时设置：

  * `random.seed`
  * `numpy.random.seed`
  * `torch.manual_seed`
* DataLoader shuffle 与采样应受 seed 控制（尽可能一致；注意多线程 worker 会引入差异）

### 14.2 数值稳定性

* 训练中如出现 NaN：

  * 首先检查是否把输入正确保持在 [0,255] 且模型内部再 /255
  * 检查 D 的 weight clipping 是否在每次 D step 后执行 ([GitHub][3])

---

## 15. 验收标准（让实现“可判定完成”）

### 15.1 单元测试（必须）

* `test_shapes.py`：

  * Splitter 输出 shape：A `[B,384,8,8]`，x `[B,128,8,8]`（默认 second_ratio=0.25）
  * Joiner 输出 `[B,3,64,64]`
  * Discriminator 输出 `[B,1]` ([GitHub][3])

### 15.2 冒烟训练（必须）

* 用小数据子集（如各 2k 张）跑 200 iter：

  * loss 不为 NaN
  * sample 图能正常保存（拼接顺序正确）
  * 输出像素范围在 [0,255]（允许轻微越界但需 clamp 保存）([GitHub][2])

### 15.3 完整训练（建议）

* 全量 CelebA + 指定 attribute（如 Bangs）跑到 100k iter：

  * swap/interpolation/matrix 三个 demo 能生成合理的人脸属性迁移效果（定性对照论文图示与 repo 示例）

---

## 16. 已知坑位与实现提示（MPS 特别说明）

* MPS 下 DataLoader 多进程可能不稳定：默认 `num_workers=0`，并在 README 写清楚如何调大。
* BatchNorm：训练用 `model.train()`，推理用 `model.eval()`；推理阶段不应更新 BN 统计。
* 不要用 AMP/half 作为默认（先保证对齐与稳定）。

---

## 17. 备注：论文 vs 官方 TF 复现差异说明（必须写入 README）

* 论文公式展示的是标准 GAN（log D）形式，并给出 parallelogram 与 nulling/重建等项。
* 官方 TF 复现实际用 WGAN-style 损失 + weight clipping + 特定 D 更新次数 schedule；本 PyTorch 复现按官方实现对齐（优先“复现作者代码产出”）。([GitHub][3])

---

如果你接下来希望我把这个 SPEC 再“压缩成更适合 codex/gemini-cli 的任务分解格式”（例如：逐文件接口、每个函数签名、伪代码、实现优先级与断言），我也可以在同一套约束下给你一个 **Implementation Checklist 版本**。

[1]: https://github.com/Prinsphield/GeneGAN "GitHub - Prinsphield/GeneGAN: GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data"
[2]: https://raw.githubusercontent.com/Prinsphield/GeneGAN/master/train.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/Prinsphield/GeneGAN/master/model.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Prinsphield/GeneGAN/master/dataset.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Prinsphield/GeneGAN/master/preprocess.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Prinsphield/GeneGAN/master/test.py "raw.githubusercontent.com"

