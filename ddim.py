"""
DDIM: Denoising Diffusion Implicit Models
=========================================
Reference: Song et al. (2021) — "Denoising Diffusion Implicit Models"
           https://arxiv.org/abs/2010.02502

DDIM  shares the exact same training objective as DDPM.
The only difference is the sampling procedure:
  - DDPM:  Markovian reverse chain, requires full T=1000 steps
  - DDIM:  non-Markovian "implicit" sampler, can skip steps

Key insight  (cf. diffusion.tex §4 for derivations):
  The DDPM training objective L_simple only depends on the marginal
  q(x_t | x_0), not on the joint q(x_{1:T} | x_0).  Therefore any
  reverse process that matches the same marginals is valid.

DDIM sampling (equation 4.1 in diffusion.tex):
  x_{t-1} = √ᾱ_{t-1} · x̂_0           … predicted clean image
          + √(1-ᾱ_{t-1} - σ_t²) · ε_θ  … "direction to x_t"
          + σ_t · z                      … random noise  (z=0 when η=0)

  where  x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
         σ_t = η · √(1-ᾱ_{t-1})/(1-ᾱ_t) · √(1 - ᾱ_t/ᾱ_{t-1})

  η = 0  →  fully deterministic (DDIM), few-step sampling possible
  η = 1  →  recovers DDPM stochasticity

Usage:
  # Need a trained DDPM checkpoint first:
  python ddpm.py --mode train           # train & save ddpm_mnist.pt
  python ddim.py                        # sample with DDIM (50 steps, η=0)

  # Or specify options:
  python ddim.py --steps 20 --eta 0.0   # faster, deterministic
  python ddim.py --steps 100 --eta 0.5  # more steps, slightly stochastic
"""

import math
import torch
import torch.nn as nn
from torchvision import utils as vutils

# Re-use the U-Net from DDPM
from ddpm import UNet, SinusoidalEmbedding, ResBlock


# ═════════════════════════════════════════════════════════════════════════════
# DDIM Sampler
# ═════════════════════════════════════════════════════════════════════════════

class DDIMSampler:
    """DDIM reverse sampler that can use a trained DDPM model as-is.

    No re-training needed — just load a DDPM checkpoint and sample with
    this class using far fewer steps.

    Args:
        model:     noise-prediction network  ε_θ(x_t, t)
        timesteps: original DDPM T (usually 1000)
        beta_start, beta_end:  same as used during DDPM training
    """

    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.model = model
        self.T = timesteps

        # -------- same schedule as DDPM --------
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(alpha, dim=0)          # ᾱ_t,  t=0..T-1

    @staticmethod
    def _gather(tensor: torch.Tensor, t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        out = tensor.to(t.device)[t]
        return out.reshape(out.shape[0], *((1,) * (len(target_shape) - 1)))

    @torch.no_grad()
    def sample(self, shape: tuple, steps: int = 50, eta: float = 0.0,
               device: str = 'cpu') -> torch.Tensor:
        """DDIM sampling with a sub-sequence of `steps` timesteps.

        Args:
            shape:  (B, C, H, W)
            steps:  number of sampling steps  (≤ T, e.g., 50)
            eta:    0.0 = deterministic DDIM
                    1.0 = recovers DDPM variance
            device: 'cpu' / 'cuda' / 'mps'

        Returns:
            Generated images in [-1, 1], same shape as input.
        """
        self.model.eval()
        B = shape[0]

        # -------- choose a sub-sequence of timesteps --------
        # e.g. T=1000, steps=50  →  [980, 960, …, 20, 0]
        indices = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        # prepend the final step (x_T = N(0,I) at index T)
        times = torch.cat([indices, torch.tensor([-1], device=device)])

        # -------- sub-sequence ᾱ values --------
        alpha_bar = self.alpha_bar.to(device)

        x = torch.randn(shape, device=device)     # x_T ~ N(0, I)

        for i in range(steps):
            t_curr = times[i].item()                         # current step index
            t_prev = times[i + 1].item()                     # next step index  (smaller)

            t_batch = torch.full((B,), t_curr, device=device, dtype=torch.long)

            # predict noise
            eps = self.model(x, t_batch)

            # predicted x_0  (Tweedie's formula / one-step denoising)
            alpha_bar_curr = self._gather(alpha_bar, t_batch, shape)
            x0_pred = (x - (1 - alpha_bar_curr).sqrt() * eps) / alpha_bar_curr.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)    # optional clipping for stability

            # -------- compute σ_t (equation 4.1 in diffusion.tex) --------
            if t_prev < 0:
                # final step: return predicted x_0 directly
                x = x0_pred
                continue

            alpha_bar_prev = alpha_bar[t_prev]

            # σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
            sigma_t = eta * math.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_curr.item()) *
                (1 - alpha_bar_curr.item() / alpha_bar_prev)
            )

            # -------- direction pointing to x_t --------
            # √(1-ᾱ_{t-1} - σ_t²) · ε_θ
            direction = (1 - alpha_bar_prev - sigma_t ** 2).sqrt() * eps

            # -------- random noise --------
            noise = sigma_t * torch.randn_like(x) if eta > 0 else 0.0

            # -------- DDIM step --------
            # x_{t-1} = √ᾱ_{t-1} · x̂_0  +  direction  +  noise
            x = alpha_bar_prev.sqrt() * x0_pred + direction + noise

        self.model.train()
        return x


# ═════════════════════════════════════════════════════════════════════════════
# Demo: sample from a DDPM checkpoint using DDIM
# ═════════════════════════════════════════════════════════════════════════════

def demo_ddim(ckpt_path: str = 'ddpm_mnist.pt',
              steps_list: tuple = (5, 10, 20, 50, 100),
              eta: float = 0.0,
              n: int = 64):
    """Load a DDPM checkpoint and run DDIM sampling at various step budgets.

    Saves a grid comparing quality vs. step count.
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # load trained DDPM model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt['config']

    unet = UNet(in_ch=config['img_ch']).to(device)
    unet.load_state_dict(ckpt['model'])
    unet.eval()

    sampler = DDIMSampler(unet, timesteps=ckpt['T'])
    shape = (n, config['img_ch'], config['img_size'], config['img_size'])

    for steps in steps_list:
        print(f"Sampling with DDIM:  steps={steps},  η={eta}  … ", end='', flush=True)
        samples = sampler.sample(shape, steps=steps, eta=eta, device=device)
        fname = f'samples_ddim_T{steps}_eta{eta}.png'
        vutils.save_image(samples, fname, nrow=int(math.sqrt(n)),
                          normalize=True, value_range=(-1, 1))
        print(f"saved → {fname}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   type=str,   default='ddpm_mnist.pt')
    parser.add_argument('--steps',  type=int,   default=50)
    parser.add_argument('--eta',    type=float, default=0.0)
    parser.add_argument('--n',      type=int,   default=64)
    parser.add_argument('--compare', action='store_true',
                        help='Run multiple step-counts and compare')
    args = parser.parse_args()

    if args.compare:
        demo_ddim(ckpt_path=args.ckpt, n=args.n)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
        config = ckpt['config']
        unet = UNet(in_ch=config['img_ch']).to(device)
        unet.load_state_dict(ckpt['model'])
        unet.eval()

        sampler = DDIMSampler(unet, timesteps=ckpt['T'])
        samples = sampler.sample(
            (args.n, config['img_ch'], config['img_size'], config['img_size']),
            steps=args.steps, eta=args.eta, device=device,
        )
        fname = f'samples_ddim_T{args.steps}_eta{args.eta}.png'
        vutils.save_image(samples, fname, nrow=int(math.sqrt(args.n)),
                          normalize=True, value_range=(-1, 1))
        print(f"Saved {args.n} samples → {fname}")
