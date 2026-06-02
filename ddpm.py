"""
DDPM: Denoising Diffusion Probabilistic Models
==============================================
Reference: Ho et al. (2020) — "Denoising Diffusion Probabilistic Models"
           https://arxiv.org/abs/2006.11239

This implementation is written for educational clarity.
Train on MNIST; the same code works for CIFAR-10 with a larger U-Net.

Key equations (cf. diffusion.tex for full derivations):
  Forward:  q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)·x_{t-1}, β_t·I)
            x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε          … easily sample any step
  Reverse:  p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²·I)
            μ_θ(x_t, t) = 1/sqrt(α_t) · (x_t - β_t/sqrt(1-ᾱ_t) · ε_θ(x_t,t))
  Loss:     L_simple = E[|| ε - ε_θ(x_t, t) ||²]          … predict the noise

Usage:
  python ddpm.py          # train on MNIST, save samples & checkpoint
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


# ═════════════════════════════════════════════════════════════════════════════
# U-Net  —  a lightweight denoising backbone for 28×28 (MNIST)
# The network takes a noisy image x_t  AND the timestep t  as inputs.
# ═════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    """Transformer-style sinusoidal position embedding for continuous diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,)  integer timesteps in [0, T-1]  →  (B, dim)"""
        device = t.device
        half_dim = self.dim // 2
        # frequencies: 1 / 10000^(2i/dim)
        exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
        emb = t.float().unsqueeze(1) * exponent.exp().unsqueeze(0)   # (B, half_dim)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)            # (B, dim)


class ResBlock(nn.Module):
    """Simple residual block with group-norm + time embedding conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # project time embedding into channel-wise scale + shift
        self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)
        # skip-connection 1×1 conv when channel dimensions differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb shape: (B, time_emb_dim)  →  (B, 2*out_ch, 1, 1)
        scale, shift = self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * (1 + scale) + shift       # time conditioning
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class UNet(nn.Module):
    """Tiny U-Net for 28×28 grayscale images (enough to demo DDPM on MNIST).

    Architecture: 2 down-sampling blocks → bottleneck → 2 up-sampling blocks
    with skip connections and time embeddings injected at every block.
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 64, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # -------- encoder --------
        self.enc1 = ResBlock(in_ch,      base_ch,     time_emb_dim)   # 28×28
        self.enc2 = ResBlock(base_ch,    base_ch * 2, time_emb_dim)   # 14×14

        # -------- bottleneck --------
        self.bottleneck = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)  # 7×7

        # -------- decoder --------
        self.dec2 = ResBlock(base_ch * 4, base_ch,     time_emb_dim)   # 7→14×14
        self.dec1 = ResBlock(base_ch * 2, base_ch,     time_emb_dim)   # 14→28×28

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

        self.down = nn.MaxPool2d(2)
        self.up  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # encoder
        e1 = self.enc1(x, t_emb)                     # (B, 64,  28, 28)
        e2 = self.enc2(self.down(e1), t_emb)          # (B, 128, 14, 14)

        # bottleneck
        b = self.bottleneck(self.down(e2), t_emb)     # (B, 128, 7, 7)

        # decoder with skip connections
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1), t_emb)   # (B, 64, 14, 14)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)  # (B, 64, 28, 28)

        return self.out_conv(d1)     # predicted noise ε_θ(x_t, t)


# ═════════════════════════════════════════════════════════════════════════════
# DDPM  —  noise schedule, forward / reverse kernels, training & sampling
# ═════════════════════════════════════════════════════════════════════════════

class DDPM:
    """DDPM diffusion process.

    Args:
        model:        noise-prediction network  ε_θ(x_t, t) → ε
        timesteps:    T  (default 1000)
        beta_start:   β_1
        beta_end:     β_T   (linear schedule)
    """

    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.model = model
        self.T = timesteps

        # noise schedule  —  linear β schedule as in the paper
        self.beta = torch.linspace(beta_start, beta_end, timesteps)   # (T,)
        self.alpha = 1.0 - self.beta                                  # α_t
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)             # ᾱ_t = ∏ α_s

        # pre-compute DDPM posterior terms
        self.sqrt_alpha_bar     = self.alpha_bar.sqrt()               # √ᾱ_t
        self.sqrt_one_minus_alpha_bar = (1 - self.alpha_bar).sqrt()   # √(1-ᾱ_t)

        # posterior variance  σ_t² = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])
        self.posterior_var = (1 - alpha_bar_prev) / (1 - self.alpha_bar) * self.beta

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion:  x_t = √ᾱ_t · x_0  +  √(1-ᾱ_t) · ε"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._gather(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_one_minus_t = self._gather(self.sqrt_one_minus_alpha_bar, t, x0.shape)

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_t * noise

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion  x_t → x_{t-1}  (one step).

        μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
        x_{t-1} = μ_θ + σ_t · z   (z ~ N(0,I) when t>0 else 0)
        """
        eps_pred = self.model(xt, t)

        # gather coefficients for each sample in the batch
        alpha_t      = self._gather(self.alpha, t, xt.shape)           # √α_t
        beta_t       = self._gather(self.beta, t, xt.shape)
        sqrt_one_minus_alpha_bar_t = self._gather(self.sqrt_one_minus_alpha_bar, t, xt.shape)

        # predicted mean
        mu_theta = (xt - beta_t / sqrt_one_minus_alpha_bar_t * eps_pred) / alpha_t.sqrt()

        if (t == 0).all():
            return mu_theta   # no noise at t=0

        # add posterior variance noise
        sigma_t = self._gather(self.posterior_var, t, xt.shape).sqrt()
        return mu_theta + sigma_t * torch.randn_like(xt)

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, device: str = 'cpu') -> torch.Tensor:
        """Full reverse chain:  x_T ~ N(0,I) → x_{T-1} → … → x_0"""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        return x

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        """Return loss for one batch.

        Algorithm 1 from Ho et al.:
          1. Sample t ~ Uniform({1,…,T})
          2. Sample ε ~ N(0,I)
          3. x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
          4. loss = MSE(ε_θ(x_t, t), ε)
        """
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, noise=noise)
        eps_pred = self.model(xt, t)

        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, n: int, device: str = 'cpu', img_ch: int = 1,
               img_size: int = 28) -> torch.Tensor:
        """Convenience: generate n images."""
        self.model.eval()
        samples = self.p_sample_loop((n, img_ch, img_size, img_size), device)
        self.model.train()
        return samples

    @staticmethod
    def _gather(tensor: torch.Tensor, t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Index tensor by t and reshape for broadcasting."""
        out = tensor.to(t.device)[t]           # (B,)
        # reshape to (B, 1, 1, 1)
        return out.reshape(out.shape[0], *((1,) * (len(target_shape) - 1)))


# ═════════════════════════════════════════════════════════════════════════════
# Training script
# ═════════════════════════════════════════════════════════════════════════════

def train_ddpm(
    epochs: int        = 20,
    batch_size: int    = 128,
    lr: float          = 2e-4,
    T: int             = 1000,
    img_size: int      = 28,
    img_ch: int        = 1,
    sample_interval: int = 5,
):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- data ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- model & diffusion ---
    unet = UNet(in_ch=img_ch).to(device)
    ddpm = DDPM(unet, timesteps=T).to(device)
    opt = torch.optim.Adam(unet.parameters(), lr=lr)

    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"Training DDPM (T={T}) for {epochs} epochs on MNIST…\n")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = ddpm.train_step(x)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:3d}/{epochs}  |  loss = {avg_loss:.6f}")

        if epoch % sample_interval == 0:
            ddpm.model.eval()
            samples = ddpm.sample(16, device, img_ch, img_size)
            vutils.save_image(samples, f'samples_ddpm_epoch{epoch}.png',
                              nrow=4, normalize=True, value_range=(-1, 1))
            ddpm.model.train()

    # --- save checkpoint ---
    torch.save({'model': unet.state_dict(), 'T': T, 'config': {'img_ch': img_ch, 'img_size': img_size}},
               'ddpm_mnist.pt')
    print("\nCheckpoint saved → ddpm_mnist.pt")
    return ddpm


# ═════════════════════════════════════════════════════════════════════════════
# Sampling from a saved checkpoint  (run standalone)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_from_checkpoint(ckpt_path: str = 'ddpm_mnist.pt', n: int = 64):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt['config']

    unet = UNet(in_ch=config['img_ch']).to(device)
    unet.load_state_dict(ckpt['model'])
    unet.eval()

    ddpm = DDPM(unet, timesteps=ckpt['T']).to(device)
    samples = ddpm.sample(n, device, config['img_ch'], config['img_size'])

    vutils.save_image(samples, 'samples_ddpm_final.png',
                      nrow=int(math.sqrt(n)), normalize=True, value_range=(-1, 1))
    print(f"Saved {n} samples → samples_ddpm_final.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'sample'], default='train')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--ckpt', type=str, default='ddpm_mnist.pt')
    args = parser.parse_args()

    if args.mode == 'train':
        train_ddpm(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        sample_from_checkpoint(args.ckpt)
