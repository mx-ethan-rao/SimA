import math
from typing import Callable, Optional

import torch
from typing import Callable




class EpsGetter:
    def __init__(self, model):
        self.model = model

    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        raise NotImplementedError


class Attacker:
    def __init__(
        self,
        diffusion,               # <-- GaussianDiffusion instance
        interval: int,
        attack_num: int,
        eps_getter: EpsGetter,              # your EpsGetter wrapper
        normalize:  Callable = None,
        denormalize: Callable = None,
    ):
        self.diffusion   = diffusion
        self.eps_getter  = eps_getter
        self.interval    = interval
        self.attack_num  = attack_num
        self.normalize   = normalize
        self.denormalize = denormalize

        self.sqrt_alphas_cumprod      = diffusion.sqrt_alphas_cumprod      # √ᾱₜ
        self.sqrt_one_minus_alphas_cp = diffusion.sqrt_one_minus_alphas_cumprod
        self.T = diffusion.num_timesteps

    def __call__(self, x0, xt, condition):
        raise NotImplementedError

    def get_xt_coefficient(self, step):
        """Return (√ᾱₜ, √(1−ᾱₜ)) as a tuple of scalar tensors."""
        return (
            self.sqrt_alphas_cumprod[step],
            self.sqrt_one_minus_alphas_cp[step],
        )

    def get_xt(self, x0, step, eps):
        """
        Forward noising using diffusion.q_sample – this guarantees
        bit-exact consistency with the checkpoint’s training code.
        """
        # step = torch.tensor(step, device=x0.device)
        if not torch.is_tensor(step):
            step = torch.tensor(step, device=x0.device)
        if step.ndim == 0:                      # scalar → broadcast
            step = step.expand(x0.shape[0])
        return self.diffusion.q_sample(x0, step, noise=eps)

    def _normalize(self, x):
        if self.normalize is not None:
            return self.normalize(x)
        return x

    def _denormalize(self, x):
        if self.denormalize is not None:
            return self.denormalize(x)
        return x


class DDIMAttacker(Attacker):
    # part of the DDIM parameterisation:  y_t = x_t / √ᾱₜ
    def get_y(self, x_t: torch.Tensor, step: int):
        return x_t / self.sqrt_alphas_cumprod[step]


    # inverse mapping  x_t = √ᾱₜ · y_t
    def get_x(self, y_t: torch.Tensor, step: int):
        return y_t * self.sqrt_alphas_cumprod[step]

    # DDIM “p” term  √(1/ᾱₜ − 1)
    def get_p(self, step, device=None):
        # scalar value: ᾱ_t^0.5
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[step]          # numpy.float64
        
        # p_t = sqrt((1 − ᾱ_t) / ᾱ_t)  — same as 1/ᾱ_t − 1 under the square-root
        val = 1.0 / (sqrt_alpha_bar ** 2) - 1.0                  # still numpy scalar
        
        # safest: turn it directly into a PyTorch tensor
        return torch.sqrt(torch.tensor(val, dtype=torch.float32, device=device))

    def get_reverse_and_denoise(self, x0, condition, step=None):
        x0 = self._normalize(x0)
        intermediates = self.ddim_reverse(x0, condition)
        intermediates_denoise = self.ddim_denoise(x0, intermediates, condition)
        return torch.stack(intermediates), torch.stack(intermediates_denoise)

    def __call__(self, x0, condition=None):
        intermediates, intermediates_denoise = self.get_reverse_and_denoise(x0, condition)
        return self.distance(intermediates, intermediates_denoise)

    def distance(self, x0, x1):
        return ((x0 - x1).abs()**2).flatten(2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        raise NotImplementedError

    def ddim_denoise(self, x0, intermediates, condition):
        raise NotImplementedError
    
class SimA(DDIMAttacker):
    def ddim_reverse(self, x0, condition=None):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        for step in range(0, terminal_step + self.interval, self.interval):
            eps = self.eps_getter(x0, condition, step)
            intermediates.append(eps)
        return intermediates
    
    def ddim_denoise(self, x0, intermediates, condition):
        # return dummy data
        return [torch.zeros_like(x0)] * 2
    
    def distance(self, x0, x1):
        return (x0.abs()**4).flatten(2).sum(dim=-1)


class SecMI(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        x = x0
        intermediates.append(x0)

        for step in range(0, terminal_step, self.interval):
            y_next = self.eps_getter(x, condition, step) * (self.get_p(step + self.interval) - self.get_p(step)) + self.get_y(x, step)
            x = self.get_x(y_next, step + self.interval)
            intermediates.append(x)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        ternimal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, ternimal_step + self.interval, self.interval), 1):
            x = intermediates[idx]
            y_next = self.eps_getter(x, condition, step) * (self.get_p(step + self.interval) - self.get_p(step)) + self.get_y(x, step)
            x_intermediate = self.get_x(y_next, step + self.interval)
            y_prev = self.eps_getter(x_intermediate, condition, step + self.interval) * (self.get_p(step) - self.get_p(step + self.interval)) + self.get_y(x_intermediate, step + self.interval)
            x_recon = self.get_x(y_prev, step)
            # x = x_recon
            intermediates_denoise.append(x_recon)

            if idx == len(intermediates) - 1:
                del intermediates[-1]
        return intermediates_denoise

    def get_prev_from_eps(self, x0, eps_x0, eps, t):
        t = t + self.interval
        xta1 = self.get_xt(x0, t, eps_x0)

        y_prev = eps * (self.get_p(t - self.interval) - self.get_p(t)) + self.get_y(xta1, t)
        x_prev = self.get_x(y_prev, t - self.interval)
        return x_prev


class PIA(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None, lp=4):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.lp = lp

    def distance(self, x0, x1):
        return ((x0 - x1).abs()**self.lp).flatten(2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, 0)
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class PIAN(DDIMAttacker):

    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None, lp=4):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.lp = lp

    def distance(self, x0, x1):
        return ((x0 - x1).abs()**self.lp).flatten(2).sum(dim=-1)

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, 0)
        eps = eps / eps.abs().mean(list(range(1, eps.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise

class PFAMI(DDIMAttacker):  # noqa: F821 – DDIMAttacker is defined earlier in components.py
    """PFAMI — Probabilistic‑Fluctuation Assessing MIA (metric variant).

    * Inherits from **DDIMAttacker** so we can reuse the helper functions that
      sample noisy latents / run the model’s ε‑predictor, just like PIA & SecMI.
    * Keeps the public constructor identical to every other attacker in
      *components.py*: `(betas, interval, attack_num, eps_getter, normalize,
      denormalize)`.
    * All PFAMI‑specific knobs requested by the user are *hard‑coded* – the
      benchmark driver still chooses only `interval` and `attack_num`.

    The implementation corresponds to the *statistical* flavour (PFAMIMet)
    described in Section 4.3 of the paper – essentially the **mean
    probabilistic‑fluctuation** over 10 crop perturbations and 10 diffusion
    timesteps.  This is sufficient for apples‑to‑apples comparison with the
    other distance attacks in the benchmark.
    """

    # --------------------------- PFAMI‑specific constants --------------------
    _TIMESTEPS = torch.tensor([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])
    _N_PERT     = 10                       # number of perturbations (λ)
    _L_START    = 0.95                     # crop strength range (1.0 == no crop)
    _L_END      = 0.70

    # ------------------------------ utils -----------------------------------
    @staticmethod
    def _center_crop(img: torch.Tensor, strength: float) -> torch.Tensor:
        """Crop the *center* region of size `strength * HxW`, then resize back."""
        c, h, w = img.shape
        tgt = int(h * strength)
        top = (h - tgt) // 2
        left = (w - tgt) // 2
        cropped = img[:, top : top + tgt, left : left + tgt]
        return torch.nn.functional.interpolate(
            cropped.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

    # ----------------------------- init -------------------------------------
    def __init__(
        self,
        betas: torch.Tensor,
        interval: int,
        attack_num: int,
        eps_getter: "EpsGetter",  # noqa: F821 – defined elsewhere in file
        normalize: Callable | None = None,
        denormalize: Callable | None = None,
    ) -> None:
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        # Pre‑compute λ list once (tensor on same device as inputs later).
        self._lambdas = torch.linspace(self._L_START, self._L_END, steps=self._N_PERT)

    # ------------------------- core helper methods --------------------------
    def _single_step_loss(self, x0: torch.Tensor, cond, t: int) -> torch.Tensor:
        """Per‑image DDPM loss *L_t* (Eq. 10).

        This reproduces the ``ddpm_loss`` routine in the PFAMI reference code,
        **not** the ``ddim_singlestep`` shown in your screenshot (that helper
        is used for deterministic sampling, not likelihood estimation).
        Steps:
            1. Sample ε ~ N(0, I).
            2. Build x_t with the closed‑form forward equation.
            3. Query the victim to get ε̂(x_t, t).
            4. Return MSE(ε̂, ε) for each item in batch.
        """
        eps_true = torch.randn_like(x0)                # 1) ground‑truth noise
        x_t = self.get_xt(x0, t, eps_true)            # 2) x_0 → x_t
        eps_pred = self.eps_getter(x_t, cond, t)  # 3) ε̂
        return torch.mean((eps_pred - eps_true) ** 2, dim=(1, 2, 3))  # 4) [B]

    def _prob_fluctuation(self, x: torch.Tensor, cond) -> torch.Tensor:
        """Compute PFAMI *overall probabilistic fluctuation* for one batch."""
        # Baseline losses (no perturbation) – shape [steps, B]
        base_losses = torch.stack([self._single_step_loss(x, cond, int(t)) for t in self._TIMESTEPS])
        base_mean   = base_losses.mean(dim=0)  # [B]

        # Neighbour losses for each λ – shape [λ, steps, B]
        neigh_losses = []
        for lam in self._lambdas.tolist():
            x_pert = torch.stack([self._center_crop(img, lam) for img in x])
            step_losses = [self._single_step_loss(x_pert, cond, int(t)) for t in self._TIMESTEPS]
            neigh_losses.append(torch.stack(step_losses))
        neigh_losses = torch.stack(neigh_losses)            # [λ, steps, B]
        neigh_mean   = neigh_losses.mean(dim=1)             # [λ, B]

        # Δp̂ per λ, then mean over λ (Eq. 19 in paper) → [B]
        fluct = (base_mean.unsqueeze(0) - neigh_mean).mean(dim=0)
        return fluct

    # ----------------------------- public API -------------------------------
    def __call__(self, x0: torch.Tensor, condition: torch.Tensor | None = None):
        """Compute PFAMI *distance* for a batch (lower ⇒ more member‑like)."""
        x0_n = self._normalize(x0)  # reuse helper from parent class
        fluct = self._prob_fluctuation(x0_n, condition)
        # Sign convention: existing attackers treat *smaller* distance as
        # *more likely member*.  A *larger* fluctuation means the sample is
        # probably a **member** (local maximum).  Thus we negate.

        # Since PFAMI is time invariant method, unsqueeze(0) make sure it compatible to other method
        return fluct.unsqueeze(0)


# Estimator: \hat\epsilon(x_t,t)
class Loss(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        # x = x0
        terminal_step = self.interval * self.attack_num
        for _ in reversed(range(0, terminal_step, self.interval)):
            eps = torch.randn_like(x0)
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, terminal_step + self.interval, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class Epsilon(DDIMAttacker):
    def ddim_reverse(self, x0, condition=None):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        for step in range(0, terminal_step + self.interval, self.interval):
            eps_add = torch.randn_like(x0)

            eps = self.eps_getter(self.get_xt(x0, step, eps_add), condition, step)
            intermediates.append(eps)
        return intermediates
    
    def ddim_denoise(self, x0, intermediates, condition):
        # return dummy data
        return [torch.zeros_like(x0)] * 2
    
    def distance(self, x0, x1):
        return (x0.abs()**4).flatten(2).sum(dim=-1)