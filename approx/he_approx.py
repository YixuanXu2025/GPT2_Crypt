# he_approx.py
# HE-friendly approximations for GELU and LayerNorm (simulation in plaintext; later can be bound to HE backend)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import os
import torch
import torch.nn as nn

# ----------------------------
# Polynomial utilities
# ----------------------------
def poly_eval_horner(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Horner evaluation: coeffs=[c0..ck] => sum_i c_i x^i (NOT HE-friendly in general)."""
    cs = coeffs.view(-1)
    y = torch.zeros_like(x)
    for c in reversed(cs):
        y = y * x + c
    return y

def poly_eval_md3(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate polynomial with multiplicative depth <= 3 (supports degree <= 8).
    coeffs=[c0..cN], y=sum_i c_i x^i.

    Depth plan:
      x2 = x*x (1)
      x4 = x2*x2 (2)
      x8 = x4*x4 (3)
      x3 = x2*x (2)
      x5 = x4*x (3)
      x6 = x4*x2 (3)
      x7 = x4*x3 (3)
    """
    cs = coeffs.view(-1)
    n = cs.numel()

    x1 = x
    x2 = x1 * x1          # depth 1
    x4 = x2 * x2          # depth 2
    x8 = x4 * x4          # depth 3

    x3 = x2 * x1          # depth 2
    x5 = x4 * x1          # depth 3
    x6 = x4 * x2          # depth 3
    x7 = x4 * x3          # depth 3

    # coefficient fetch
    def c(i: int):
        return cs[i] if i < n else 0.0

    y = c(0)
    if n > 1: y = y + c(1) * x1
    if n > 2: y = y + c(2) * x2
    if n > 3: y = y + c(3) * x3
    if n > 4: y = y + c(4) * x4
    if n > 5: y = y + c(5) * x5
    if n > 6: y = y + c(6) * x6
    if n > 7: y = y + c(7) * x7
    if n > 8: y = y + c(8) * x8
    return y

# ----------------------------
# Piecewise-linear (SIM ONLY)
# ----------------------------
def pwl_eval_sim(
    x: torch.Tensor,
    knots: torch.Tensor,
    slopes: torch.Tensor,
    intercepts: torch.Tensor
) -> torch.Tensor:
    """
    Piecewise linear (SIM ONLY):
      knots: [K+1] increasing
      slopes/intercepts: [K]
      y = slopes[i]*x + intercepts[i] for x in [knots[i], knots[i+1])
    """
    x_clamped = torch.clamp(x, float(knots[0].item()), float(knots[-1].item()))
    y = torch.empty_like(x_clamped)
    K = int(slopes.numel())
    for i in range(K):
        left = knots[i]
        right = knots[i + 1]
        if i < K - 1:
            mask = (x_clamped >= left) & (x_clamped < right)
        else:
            mask = (x_clamped >= left) & (x_clamped <= right)
        y[mask] = slopes[i] * x_clamped[mask] + intercepts[i]
    return y

# ----------------------------
# GELU approximation
# ----------------------------
class HEGELU(nn.Module):
    """
    HE-friendly GELU replacement.

    approx:
      - "poly": polynomial approximation (HE-friendly, md<=3 for degree<=8)
      - "pwl":  piecewise linear (SIM ONLY)
      - "identity": y=x (debug)
    """
    def __init__(
        self,
        approx: str = "poly",
        poly_coeffs: Optional[torch.Tensor] = None,
        pwl_knots: Optional[torch.Tensor] = None,
        pwl_slopes: Optional[torch.Tensor] = None,
        pwl_intercepts: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.approx = approx

        # Default polynomial coefficients:
        # A weighted least-squares fit of GELU on [-3, 3], degree=6.
        # This is a reasonable *starting point* for training stability.
        if poly_coeffs is None:
            poly_coeffs = torch.tensor([
                0.0044944344,   # c0
                0.5000000000,   # c1
                0.3718453755,   # c2
                0.0,            # c3
                -0.0420038767,  # c4
                0.0,            # c5
                0.0021857134,   # c6
            ], dtype=torch.float32)
        self.register_buffer("poly_coeffs", poly_coeffs)

        # Default PWL (SIM ONLY) example over [-6,6]
        if pwl_knots is None:
            pwl_knots = torch.tensor([-6.0, -3.0, 0.0, 3.0, 6.0], dtype=torch.float32)
        if pwl_slopes is None:
            pwl_slopes = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float32)
        if pwl_intercepts is None:
            pwl_intercepts = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self.register_buffer("pwl_knots", pwl_knots)
        self.register_buffer("pwl_slopes", pwl_slopes)
        self.register_buffer("pwl_intercepts", pwl_intercepts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approx == "identity":
            return x
        if self.approx == "pwl":
            return pwl_eval_sim(x, self.pwl_knots, self.pwl_slopes, self.pwl_intercepts)
        if self.approx == "poly":
            # IMPORTANT:
            # In real HE you cannot clamp with comparisons.
            # Here we clamp in plaintext *simulation* to keep x inside the fitted interval
            # and avoid polynomial explosion. Later, in HE backend, you will need
            # scale management / distribution control to ensure |x| stays bounded.
            clip = float(os.environ.get("GELU_POLY_CLIP", "3.0"))
            if clip > 0:
                x = torch.clamp(x, -clip, clip)
            return poly_eval_md3(x, self.poly_coeffs)
        # fallback (should not happen)
        return torch.nn.functional.gelu(x)

# ----------------------------
# LayerNorm approximation
# ----------------------------
class HELayerNorm(nn.Module):
    """
    HE-friendly LayerNorm approximation.

    approx:
      - "affine_only": skip normalization, keep y = x*gamma + beta
      - "static_calib": y = ((x - mu_hat) * invstd_hat) * gamma + beta
        mu_hat, invstd_hat are offline-calibrated and fixed (buffers).
    """
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        approx: str = "affine_only",
        mu: Optional[torch.Tensor] = None,
        inv_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        assert len(self.normalized_shape) == 1, "This HELayerNorm assumes last-dim normalization."
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.approx = approx

        H = self.normalized_shape[0]

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(H, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(H, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if mu is not None:
            self.register_buffer("mu", mu.view(-1).to(dtype=torch.float32))
        else:
            self.register_buffer("mu", torch.zeros(H, dtype=torch.float32))

        if inv_std is not None:
            self.register_buffer("inv_std", inv_std.view(-1).to(dtype=torch.float32))
        else:
            self.register_buffer("inv_std", torch.ones(H, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H] for GPT-2
        if self.approx == "affine_only":
            y = x
        elif self.approx == "static_calib":
            mu = self.mu.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            inv = self.inv_std.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            y = (x - mu) * inv
        else:
            # fallback to real LN (debug)
            y = torch.nn.functional.layer_norm(x, self.normalized_shape, None, None, self.eps)

        if self.elementwise_affine:
            w = self.weight.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            b = self.bias.view(1, 1, -1).to(dtype=x.dtype, device=x.device)
            y = y * w + b
        return y
