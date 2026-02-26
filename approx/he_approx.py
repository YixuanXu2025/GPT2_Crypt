"""HE 近似模块（精简版）。

本文件只保留当前项目真正需要的两类近似：
1) GELU 的多项式近似（HEGELU）
2) LayerNorm 的静态近似（HELayerNorm）

删除了未使用的 PWL、identity、冗余工具函数，便于维护与阅读。
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


def _poly_eval_md3(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """以乘法深度 <= 3 计算多项式（支持最高 8 次项）。

    参数：
        x: 输入张量。
        coeffs: 系数向量 [c0, c1, ..., cN]，表示 sum(ci * x^i)。

    说明：
        该实现为 HE 场景准备，尽量减少乘法层数，降低同态计算噪声增长。
    """
    cs = coeffs.view(-1)
    n = cs.numel()

    x1 = x
    x2 = x1 * x1
    x4 = x2 * x2
    x8 = x4 * x4

    x3 = x2 * x1
    x5 = x4 * x1
    x6 = x4 * x2
    x7 = x4 * x3

    def c(i: int):
        return cs[i] if i < n else 0.0

    y = c(0)
    if n > 1:
        y = y + c(1) * x1
    if n > 2:
        y = y + c(2) * x2
    if n > 3:
        y = y + c(3) * x3
    if n > 4:
        y = y + c(4) * x4
    if n > 5:
        y = y + c(5) * x5
    if n > 6:
        y = y + c(6) * x6
    if n > 7:
        y = y + c(7) * x7
    if n > 8:
        y = y + c(8) * x8
    return y


class HEGELU(nn.Module):
    """GELU 的 HE 友好近似层（仅保留 poly 方式）。"""

    def __init__(self, approx: str = "poly", poly_coeffs: Optional[torch.Tensor] = None):
        super().__init__()
        if approx != "poly":
            raise ValueError("精简版 HEGELU 仅支持 approx='poly'")
        self.approx = approx

        # 最小二乘拟合得到的 6 阶 GELU 近似系数（拟合区间约为 [-3, 3]）。
        if poly_coeffs is None:
            poly_coeffs = torch.tensor(
                [
                    0.0044944344,
                    0.5000000000,
                    0.3718453755,
                    0.0,
                    -0.0420038767,
                    0.0,
                    0.0021857134,
                ],
                dtype=torch.float32,
            )
        self.register_buffer("poly_coeffs", poly_coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 说明：裁剪仅用于明文仿真阶段，避免输入超出拟合区间导致数值爆炸。
        # 在真实 HE 部署中，通常需要通过尺度管理/数据分布控制替代该操作。
        clip = float(os.environ.get("GELU_POLY_CLIP", "3.0"))
        if clip > 0:
            x = torch.clamp(x, -clip, clip)
        return _poly_eval_md3(x, self.poly_coeffs)


class HELayerNorm(nn.Module):
    """LayerNorm 的 HE 友好近似层。"""

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
        if approx not in {"affine_only", "static_calib"}:
            raise ValueError("HELayerNorm 仅支持 'affine_only' 或 'static_calib'")

        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        if len(self.normalized_shape) != 1:
            raise ValueError("当前 HELayerNorm 仅支持最后一维归一化")

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.approx = approx

        hidden_size = self.normalized_shape[0]
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if mu is None:
            mu = torch.zeros(hidden_size, dtype=torch.float32)
        if inv_std is None:
            inv_std = torch.ones(hidden_size, dtype=torch.float32)

        self.register_buffer("mu", mu.view(-1).to(dtype=torch.float32))
        self.register_buffer("inv_std", inv_std.view(-1).to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT2 常见输入为 [B, T, H]，这里按最后一维进行近似归一化。
        if self.approx == "affine_only":
            y = x
        else:  # static_calib
            mu = self.mu.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
            inv = self.inv_std.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
            y = (x - mu) * inv

        if self.elementwise_affine:
            w = self.weight.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
            b = self.bias.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
            y = y * w + b
        return y
