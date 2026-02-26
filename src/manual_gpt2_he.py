"""手写 GPT2（支持 HE 近似）的核心模型定义。

说明：
1) 本文件尽量按照 HuggingFace GPT2 的模块命名来组织，便于加载预训练权重。
2) 非线性部分（GELU、LayerNorm）支持替换为近似实现，方便后续同态加密推理研究。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

from approx.he_approx import HEGELU, HELayerNorm


@dataclass
class ManualGPT2Config:
    """简化版配置，字段名尽量与 GPT2 配置一致。"""

    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class ManualGPT2Attention(nn.Module):
    """GPT2 的自注意力层（仅实现 causal self-attention）。"""

    def __init__(self, cfg: ManualGPT2Config):
        super().__init__()
        self.embed_dim = cfg.n_embd
        self.num_heads = cfg.n_head
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("n_embd 必须能被 n_head 整除")

        # 与 HF GPT2 一致：一次性产生 q/k/v
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,C] -> [B,nh,T,hd]
        b, t, c = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,nh,T,hd] -> [B,T,C]
        b, nh, t, hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, t, nh * hd)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # 缩放点积注意力
        att = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # 因果 mask：保证只看见过去 token
        causal_mask = torch.tril(torch.ones(t, t, device=hidden_states.device, dtype=torch.bool))
        att = att.masked_fill(~causal_mask.view(1, 1, t, t), torch.finfo(att.dtype).min)

        # 外部 mask（padding）可选
        if attention_mask is not None:
            # attention_mask: [B, T], 1=有效, 0=padding
            ext = (1.0 - attention_mask[:, None, None, :].to(dtype=att.dtype)) * -1e4
            att = att + ext

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        y = self._merge_heads(y)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class ManualGPT2MLP(nn.Module):
    """GPT2 的 FFN 层，可将激活函数替换为 HEGELU。"""

    def __init__(self, cfg: ManualGPT2Config, use_he_gelu: bool = True):
        super().__init__()
        inner = 4 * cfg.n_embd
        self.c_fc = Conv1D(inner, cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, inner)
        self.act = HEGELU(approx="poly") if use_he_gelu else nn.GELU()
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class ManualGPT2Block(nn.Module):
    """一个标准 GPT2 Block：LN + Attn + LN + MLP。"""

    def __init__(self, cfg: ManualGPT2Config, use_he_gelu: bool = True, use_he_ln: bool = True):
        super().__init__()
        ln_cls = HELayerNorm if use_he_ln else nn.LayerNorm

        if use_he_ln:
            self.ln_1 = ln_cls(cfg.n_embd, eps=cfg.layer_norm_epsilon, approx="static_calib")
            self.ln_2 = ln_cls(cfg.n_embd, eps=cfg.layer_norm_epsilon, approx="static_calib")
        else:
            self.ln_1 = ln_cls(cfg.n_embd, eps=cfg.layer_norm_epsilon)
            self.ln_2 = ln_cls(cfg.n_embd, eps=cfg.layer_norm_epsilon)

        self.attn = ManualGPT2Attention(cfg)
        self.mlp = ManualGPT2MLP(cfg, use_he_gelu=use_he_gelu)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ManualGPT2Transformer(nn.Module):
    """GPT2 主体（不含 lm_head）。"""

    def __init__(self, cfg: ManualGPT2Config, use_he_gelu: bool = True, use_he_ln: bool = True):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList(
            [ManualGPT2Block(cfg, use_he_gelu=use_he_gelu, use_he_ln=use_he_ln) for _ in range(cfg.n_layer)]
        )
        if use_he_ln:
            self.ln_f = HELayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon, approx="static_calib")
        else:
            self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = input_ids.shape
        pos_ids = torch.arange(0, t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.drop(x)
        for block in self.h:
            x = block(x, attention_mask=attention_mask)
        return self.ln_f(x)


class ManualGPT2LMHeadModel(nn.Module):
    """完整语言模型：Transformer + tied lm_head。"""

    def __init__(self, cfg: ManualGPT2Config, use_he_gelu: bool = True, use_he_ln: bool = True):
        super().__init__()
        self.transformer = ManualGPT2Transformer(cfg, use_he_gelu=use_he_gelu, use_he_ln=use_he_ln)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # 与 GPT2 一致：词嵌入和输出层共享权重
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        out: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # 自回归语言模型损失：预测下一个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out


def _copy_layernorm_to_he(src: nn.LayerNorm, dst: HELayerNorm) -> None:
    """把标准 LayerNorm 参数拷贝到 HELayerNorm。"""

    if src.weight is not None:
        dst.weight.data.copy_(src.weight.data)
    if src.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def build_manual_model_from_hf(
    model_name: str,
    use_he_gelu: bool = True,
    use_he_ln: bool = True,
) -> Tuple[ManualGPT2LMHeadModel, ManualGPT2Config]:
    """从 HuggingFace GPT2 权重初始化手写模型。

    这里采用“结构同名 + state_dict 加载”的方式，尽量减少手写映射错误。
    """

    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_cfg = hf_model.config
    cfg = ManualGPT2Config(
        vocab_size=hf_cfg.vocab_size,
        n_positions=hf_cfg.n_positions,
        n_embd=hf_cfg.n_embd,
        n_layer=hf_cfg.n_layer,
        n_head=hf_cfg.n_head,
        layer_norm_epsilon=hf_cfg.layer_norm_epsilon,
        embd_pdrop=hf_cfg.embd_pdrop,
        resid_pdrop=hf_cfg.resid_pdrop,
        attn_pdrop=hf_cfg.attn_pdrop,
    )
    model = ManualGPT2LMHeadModel(cfg, use_he_gelu=use_he_gelu, use_he_ln=use_he_ln)

    # 先直接加载同名参数（大部分层会对齐）
    missing, unexpected = model.load_state_dict(hf_model.state_dict(), strict=False)

    # 当 use_he_ln=True 时，LayerNorm -> HELayerNorm 的参数名一般对齐；
    # 这里做一次显式保障，避免某些版本差异导致未拷贝。
    if use_he_ln:
        for i in range(cfg.n_layer):
            _copy_layernorm_to_he(hf_model.transformer.h[i].ln_1, model.transformer.h[i].ln_1)
            _copy_layernorm_to_he(hf_model.transformer.h[i].ln_2, model.transformer.h[i].ln_2)
        _copy_layernorm_to_he(hf_model.transformer.ln_f, model.transformer.ln_f)

    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")

    return model, cfg
