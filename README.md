# Diffusion & Inversion Models — Study Notes & Code

个人学习 Diffusion Models 与 Image Inversion 的笔记与代码实现。

---

## 📂 项目结构

```
vslatex/
├── README.md
│
├── diffusion.tex / diffusion.pdf    # DDPM & DDIM 学习笔记
├── ddpm.py                          # DDPM 完整实现
├── ddim.py                          # DDIM 采样器
│
└── inversion/
    ├── inversion.tex / inversion.pdf  # Image Inversion 学习笔记
    └── images/                        # 笔记配图
```

---

## 1. Diffusion Models（扩散模型）

| 文件 | 内容 |
|------|------|
| `diffusion.tex` / `diffusion.pdf` | 学习笔记：DDPM & DDIM 的公式推导与直观理解 |
| `ddpm.py` | DDPM 完整实现（U-Net + 训练 + 采样） |
| `ddim.py` | DDIM 采样器（复用 DDPM checkpoint，支持少步采样） |

### 快速开始

```bash
pip install torch torchvision tqdm
```

```bash
# 在 MNIST 上训练 DDPM
python ddpm.py --mode train --epochs 20

# DDPM 采样（1000 步）
python ddpm.py --mode sample --ckpt ddpm_mnist.pt

# DDIM 采样（默认 50 步，η=0 确定性）
python ddim.py --ckpt ddpm_mnist.pt --steps 50

# DDIM 对比实验：不同步数下的生成质量
python ddim.py --ckpt ddpm_mnist.pt --compare
```

### 关键公式速查

**DDPM:**
- 前向：$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$
- 损失：$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$

**DDIM:**
- 采样：$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta + \sigma_t z$
- $\eta=0$：确定性采样，可大幅减少步数
- 训练：与 DDPM 完全相同，无需额外训练

---

## 2. Image Inversion（图像反演）

| 文件 | 内容 |
|------|------|
| `inversion/inversion.tex` / `inversion/inversion.pdf` | 学习笔记：Prompt-to-Prompt、Pivotal Inversion、DDIM Inversion 的原理与推导 |

### 涵盖内容

1. **Introduction** — 什么是 Inversion？为什么需要它？重建精度 vs 可编辑性 trade-off
2. **Prompt-to-Prompt** — 基于 Cross-Attention 的图像编辑（Word Swap / Adding Phrase / Re-weighting）
3. **Pivotal Inversion** — 直接优化 latent code，配合 pixel loss + perceptual loss + regularization
4. **DDIM Inversion** — 基于 ODE 逆向求解，Classifier-Free Guidance 下的挑战与 Null-text Inversion
5. **比较与联系** — 三种方法的关系、组合使用方式、实际使用建议
6. **相关进展** — Null-text Inversion、Textual Inversion、Encoder-based Inversion、Consistency Models

### 编译

```bash
cd inversion
latexmk -xelatex inversion.tex
```

---

## 参考

**Diffusion:**
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al. (2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al. (2021)
- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — Song et al. (2021)

**Inversion:**
- [Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — Hertz et al. (2023)
- [Pivotal Tuning for Latent-based Editing of Real Images](https://arxiv.org/abs/2106.05744) — Roich et al. (2022)
- [Null-text Inversion for Editing Real Images](https://arxiv.org/abs/2211.09794) — Mokady et al. (2023)
- [An Image is Worth One Word: Textual Inversion](https://arxiv.org/abs/2208.01618) — Gal et al. (2023)

## License

MIT
