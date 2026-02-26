# he_approx.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn as nn

# ----------------------------
# Utils
# ----------------------------
#多项式计算函数，y=(((ckx+ck−1)x+ck−2)x+⋯+c0)
def poly_eval(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Horner: coeffs = [c0, c1, ..., ck] for sum_{i=0}^k c_i x^i"""
    #创建一个与x同形状的全0张量
    y = torch.zeros_like(x)
    for c in reversed(coeffs):
        y = y * x + c
    return y


def poly_eval_md3(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    HE-friendly polynomial evaluation with multiplicative depth <= 3.

    This avoids Horner (sequential multiplications) and instead precomputes powers
    using a squaring/multiply schedule. Supports degree <= 8.

    coeffs: [c0, c1, ..., ck] for y = sum_{i=0}^k c_i x^i
    """
    k = int(coeffs.numel() - 1)
    if k < 0:
        raise ValueError("coeffs must have at least one element (c0).")
    if k > 8:
        raise ValueError(f"poly_eval_md3 supports degree <= 8, got degree={k}.")

    y = coeffs[0] * torch.ones_like(x)

    if k >= 1:
        y = y + coeffs[1] * x

    x2 = None
    if k >= 2:
        x2 = x * x  # depth 1
        y = y + coeffs[2] * x2

    x3 = None
    if k >= 3:
        if x2 is None:
            x2 = x * x
        x3 = x2 * x  # depth 2
        y = y + coeffs[3] * x3

    x4 = None
    if k >= 4:
        if x2 is None:
            x2 = x * x
        x4 = x2 * x2  # depth 2
        y = y + coeffs[4] * x4

    if k >= 5:
        if x4 is None:
            if x2 is None:
                x2 = x * x
            x4 = x2 * x2
        x5 = x4 * x  # depth 3
        y = y + coeffs[5] * x5

    if k >= 6:
        if x3 is not None:
            x6 = x3 * x3  # depth 3
        else:
            # fallback: x6 = x4 * x2 (also depth 3)
            if x4 is None:
                if x2 is None:
                    x2 = x * x
                x4 = x2 * x2
            if x2 is None:
                x2 = x * x
            x6 = x4 * x2
        y = y + coeffs[6] * x6

    if k >= 7:
        if x4 is None:
            if x2 is None:
                x2 = x * x
            x4 = x2 * x2
        if x3 is None:
            if x2 is None:
                x2 = x * x
            x3 = x2 * x
        x7 = x4 * x3  # depth 3
        y = y + coeffs[7] * x7

    if k >= 8:
        if x4 is None:
            if x2 is None:
                x2 = x * x
            x4 = x2 * x2
        x8 = x4 * x4  # depth 3
        y = y + coeffs[8] * x8

    return y

#分段线性knots表示所分段落。如：[-6,0,6]表示分段为[-6,0][0,6]两段曲线。
#      slopes表示斜率，在上面例子中，slopes为[a0,a1]
#      intercepts表示截距，在上面例子中，intercepts为[b0,b1]
def pwl_eval_sim(x: torch.Tensor, knots: torch.Tensor, slopes: torch.Tensor, intercepts: torch.Tensor) -> torch.Tensor:
    """
    Piecewise linear (SIM ONLY):
    knots: [K+1] increasing, define K segments [k0,k1),...,[kK-1,kK]
    slopes/intercepts: [K]
    """
    #如果 x 超过拟合区间，我们把它强行压缩回区间内。例如 x=10，knots[-1]=6 → clamp 后变 6
    x_clamped = torch.clamp(x, knots[0].item(), knots[-1].item())
    #初始化输出张量，创建一个与x_clamped形状相同的未初始化张量。我们将在循环中填充它。
    y = torch.empty_like(x_clamped)
    # naive loop; later you can vectorize if needed
    #循环每个分段对于每个分段，我们定义区间的左边界和右边界：left=knots[i]  right=knots[i+1]
    for i in range(len(slopes)):
        left, right = knots[i], knots[i + 1]
        #创建掩码，对于前 K-1 个区间，区间为 [left, right)，即包括左边界但不包括右边界。对于最后一个区间（i == len(slopes)-1），区间为 [left, right]，即包括右边界。
        mask = (x_clamped >= left) & (x_clamped < right) if i < len(slopes) - 1 else (x_clamped >= left) & (x_clamped <= right)
        #算当前区间内点的线性函数值：slopes[i] * x_clamped + intercepts[i]。并使用 torch.where 将结果赋值给 y 中对应位置。
        y = torch.where(mask, slopes[i] * x_clamped + intercepts[i], y)
    return y

# ----------------------------
# GELU
# ----------------------------
class HEGELU(nn.Module):
    """
    HE-friendly GELU replacement.
    approx:
      - "poly": polynomial approximation evaluated with md<=3 (HE-friendly) 多项式近似（乘法深度≤3）
      - "poly_horner": Horner polynomial eval (SIM/DEBUG) Horner多项式（乘法深度随次数增长）
      - "pwl": piecewise linear (SIM ONLY) 分段线性
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

        #多项式系数，有默认值[0.0, 0.5, 0.0, 0.0] y=0+0.5x+0x2+0x3=0.5x
        if poly_coeffs is None:
            # placeholder; replace with your fitted coeffs
            poly_coeffs = torch.tensor([0.0, 0.5, 0.0, 0.0], dtype=torch.float32)
        #把poly_coeffs变成self.poly_coeffs属性。后面可以直接self.poly_coeffs访问。登记为 buffer。 它会：跟着 model.to("cuda") 自动去 GPU跟着 state_dict() 一起保存/加载默认不参与训练更新（optimizer 不会更新它）这很适合“拟合出来的常量系数”。
        self.register_buffer("poly_coeffs", poly_coeffs)

        # SIM ONLY 区间段、斜率、截距。
        if pwl_knots is None:
            pwl_knots = torch.tensor([-6.0, 0.0, 6.0], dtype=torch.float32)
        if pwl_slopes is None:
            pwl_slopes = torch.tensor([0.0, 1.0], dtype=torch.float32)
        if pwl_intercepts is None:
            pwl_intercepts = torch.tensor([0.0, 0.0], dtype=torch.float32)

        self.register_buffer("pwl_knots", pwl_knots)
        self.register_buffer("pwl_slopes", pwl_slopes)
        self.register_buffer("pwl_intercepts", pwl_intercepts)

    #计算部分
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approx == "identity":
            return x
        if self.approx == "poly":
            # HE-friendly evaluation: multiplicative depth <= 3
            return poly_eval_md3(x, self.poly_coeffs)
        if self.approx == "poly_horner":
            # SIM/DEBUG ONLY: Horner chain (depth grows with degree)
            return poly_eval(x, self.poly_coeffs)
        if self.approx == "pwl":
            # SIM ONLY
            return pwl_eval_sim(x, self.pwl_knots, self.pwl_slopes, self.pwl_intercepts)
        raise ValueError(f"Unknown approx: {self.approx}")

# ----------------------------
# LayerNorm
# ----------------------------
class HELayerNorm(nn.Module):
    """
    HE-friendly LayerNorm replacement.

    approx:
      - "affine_only": y = x * gamma + beta
      - "static_calib": y = (x - mu) * inv_std * gamma + beta
    """
    def __init__(
        self,
        normalized_shape,   #对应hidden size，可以是int或者tuple
        eps: float = 1e-5,  #保留，暂未使用 在标准LayerNorm用于使数值稳定 常数
        elementwise_affine: bool = True,    #用于控制LayerNorm中的weight与bias
        approx: str = "affine_only",    #指定近似方法。有两种选择："affine_only"（仅仿射变换）和"static_calib"（静态校准）。
        mu: Optional[torch.Tensor] = None,  #用于静态校准的均值 初始化0
        inv_std: Optional[torch.Tensor] = None, #用于静态校准的逆标准差 初始化1
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.approx = approx

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # static calibration buffers (optional)
        if mu is None:
            mu = torch.zeros(self.normalized_shape, dtype=torch.float32)
        if inv_std is None:
            inv_std = torch.ones(self.normalized_shape, dtype=torch.float32)
        self.register_buffer("mu", mu)
        self.register_buffer("inv_std", inv_std)

    #无需梯度计算，设置mu inv_std
    @torch.no_grad()
    def set_static_calibration(self, mu: torch.Tensor, inv_std: torch.Tensor):
        self.mu.copy_(mu)
        self.inv_std.copy_(inv_std)

    #前向传播
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #affine_only进行仿射变换（如果elementwise_affine为True，则使用weight和bias，否则直接返回x）。注意，这里没有进行任何归一化操作，只是简单的缩放和偏移
        if self.approx == "affine_only":
            if self.elementwise_affine:
                return x * self.weight + self.bias
            return x

        #"static_calib"：则使用预先校准的mu和inv_std对输入x进行归一化，即 y = (x - mu) * inv_std。然后再根据elementwise_affine决定是否进行仿射变换。
        if self.approx == "static_calib":
            # NOTE: no runtime mean/var; only constant shift/scale
            y = (x - self.mu) * self.inv_std
            if self.elementwise_affine:
                y = y * self.weight + self.bias
            return y
        #如果approx不是上述两种，则抛出错误。
        raise ValueError(f"Unknown approx: {self.approx}")


# ----------------------------
# Offline fitting helpers (SIM / calibration)
# ----------------------------
@torch.no_grad()
def gelu_interval_from_samples(samples: torch.Tensor, central_mass: float = 0.90) -> Tuple[float, float]:
    """
    Choose an interval [lo, hi] that contains `central_mass` of the samples.

    This is NOT (min, max)*central_mass. It is the central quantile interval:
      central_mass=0.90  ->  [5%, 95%] quantiles.
    """
    if not (0.0 < central_mass < 1.0):
        raise ValueError("central_mass must be in (0, 1).")
    x = samples.detach().reshape(-1).to(dtype=torch.float32)
    tail = (1.0 - central_mass) / 2.0
    lo = torch.quantile(x, tail).item()
    hi = torch.quantile(x, 1.0 - tail).item()
    return float(lo), float(hi)


@torch.no_grad()
def fit_gelu_poly_from_samples(
    samples: torch.Tensor,
    degree: int = 7,
    central_mass: float = 0.90,
    clip_for_fit: bool = True,
) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """
    Fit GELU(x) with a polynomial y = sum_{i=0}^degree c_i x^i on a data-driven interval.

    Interval: central quantile interval containing `central_mass` of samples.
      e.g., central_mass=0.90 -> [5%, 95%].

    Notes:
      - Fitting is offline / in cleartext. In HE inference you typically do NOT clip,
        because comparisons are expensive. `clip_for_fit` only affects the regression target.
      - For multiplicative depth <= 3 with poly_eval_md3, require degree <= 8. Typical: 6/7/8.

    Returns:
      coeffs: tensor [c0..c_degree]
      interval: (lo, hi)
    """
    if degree > 8:
        raise ValueError("degree must be <= 8 for poly_eval_md3 (md<=3).")
    lo, hi = gelu_interval_from_samples(samples, central_mass=central_mass)

    x = samples.detach().reshape(-1).to(dtype=torch.float32)
    if clip_for_fit:
        x_fit = torch.clamp(x, lo, hi)
    else:
        x_fit = x

    y = torch.nn.functional.gelu(x_fit)

    # Design matrix [1, x, x^2, ..., x^degree]
    Phi = torch.stack([x_fit ** i for i in range(degree + 1)], dim=1)  # [N, degree+1]
    coeffs = torch.linalg.lstsq(Phi, y.unsqueeze(1)).solution.squeeze(1).to(dtype=torch.float32)
    return coeffs, (lo, hi)
