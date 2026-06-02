# Diffusion Models — Study Notes & Code

个人学习 Diffusion Models 的笔记与代码实现，涵盖 DDPM 和 DDIM。

## 内容

| 文件 | 内容 |
|------|------|
| `diffusion.tex` / `diffusion.pdf` | 学习笔记：DDPM & DDIM 的公式推导与直观理解 |
| `ddpm.py` | DDPM 完整实现（U-Net + 训练 + 采样） |
| `ddim.py` | DDIM 采样器（复用 DDPM checkpoint，支持少步采样） |

## 快速开始

### 环境

```bash
pip install torch torchvision tqdm
```

### 训练 DDPM

```bash
# 在 MNIST 上训练（约 20 个 epoch，CPU/MPS 均可）
python ddpm.py --mode train --epochs 20
```

训练完成后生成 `ddpm_mnist.pt` 和采样图片 `samples_ddpm_epoch*.png`。

### 采样

```bash
# DDPM 采样（1000 步）
python ddpm.py --mode sample --ckpt ddpm_mnist.pt

# DDIM 采样（默认 50 步，η=0 确定性）
python ddim.py --ckpt ddpm_mnist.pt --steps 50

# DDIM 对比实验：不同步数下的生成质量
python ddim.py --ckpt ddpm_mnist.pt --compare

# 加点随机性（η>0）
python ddim.py --ckpt ddpm_mnist.pt --steps 20 --eta 0.5
```

## 关键公式速查

### DDPM

- **前向**：$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$
- **反向**：$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t), \sigma_t^2 I)$
- **损失**：$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$

### DDIM

- **采样**：$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta + \sigma_t z$
- **$\eta=0$**：确定性采样，可大幅减少步数
- **训练**：与 DDPM 完全相同，无需额外训练

## 参考

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al. (2020)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al. (2021)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Song et al. (2021)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Nichol & Dhariwal (2021)

## License

MIT
