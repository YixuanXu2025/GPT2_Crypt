# -*- coding: utf-8 -*-
"""
One-off distribution probe for GELU-input (GPT-2 MLP c_fc output).

Goals:
  1) Confirm GELU is replaced by HEGELU (for transformers NewGELUActivation).
  2) Compute FULL-DATASET ratio of GELU-input scalars within [-CLIP, CLIP].

Usage example (probe only, full pass over cached dataset):
  USE_HE_APPROX=1 REPLACE_GELU=1 REPLACE_LN=0 PROBE_FULL_DATASET=1 PROBE_ONLY=1 \
  PROBE_CLIP_CHECK=3.0 PROBE_BS=2 \
  python m1_lora_clm_chinese_he_distribution_full_v2.py

Optional: load a LoRA adapter before probing:
  ADAPTER_PATH=./out_gelu_only_1000/checkpoint-1000

Notes:
  - This script assumes you already have the cached chunked dataset at ./data/lm_ds_he
    (created by the training script). If not found, it will abort with a clear message.
  - masked=True (default) means only attention_mask==1 positions are counted.
"""

import os
import math
import torch
import torch.nn as nn
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from peft import PeftModel


# -------------------- Config via env --------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "uer/gpt2-distil-chinese-cluecorpussmall")
LM_DS_PATH = os.environ.get("LM_DS_PATH", "./data/lm_ds_he")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip()

USE_HE_APPROX = os.environ.get("USE_HE_APPROX", "0") == "1"
REPLACE_GELU = os.environ.get("REPLACE_GELU", "0") == "1"
REPLACE_LN = os.environ.get("REPLACE_LN", "0") == "1"   # not used here, kept for compatibility

# Probe controls
PROBE_FULL_DATASET = os.environ.get("PROBE_FULL_DATASET", "1") == "1"
PROBE_ONLY = os.environ.get("PROBE_ONLY", "1") == "1"
PROBE_BS = int(os.environ.get("PROBE_BS", "2"))
PROBE_CLIP_CHECK = float(os.environ.get("PROBE_CLIP_CHECK", "3.0"))
PROBE_MASKED = os.environ.get("PROBE_MASKED", "1") == "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- HEGELU replacement --------------------
def apply_he_gelu_only(model: nn.Module, gelu_approx: str = "poly"):
    """Replace all GELU-like activations (including transformers NewGELUActivation) with HEGELU."""
    from approx.he_approx import HEGELU

    replaced = set()

    def _is_gelu_like(mod) -> bool:
        if isinstance(mod, nn.GELU):
            return True
        cls = mod.__class__.__name__.lower()
        if "gelu" in cls:
            return True
        if callable(mod) and (not isinstance(mod, nn.Module)):
            fn = getattr(mod, "__name__", "").lower()
            if "gelu" in fn:
                return True
        return False

    def _replace(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if _is_gelu_like(child):
                setattr(parent, name, HEGELU(approx=gelu_approx))
                replaced.add(full)
            else:
                _replace(child, full)

        # Some GPT2 MLPs store act as attribute
        if parent.__class__.__name__.lower().endswith("mlp") and hasattr(parent, "act"):
            act = getattr(parent, "act")
            if _is_gelu_like(act) and (not isinstance(act, HEGELU)):
                setattr(parent, "act", HEGELU(approx=gelu_approx))
                replaced.add(f"{prefix}.act" if prefix else "act")

    _replace(model, prefix="")

    print(f"[GELU REPLACED] unique_count={len(replaced)}")
    for n in sorted(list(replaced))[:50]:
        print("  -", n)
    if len(replaced) > 50:
        print("  ...")

    # Count HEGELU modules
    num_hegelu = sum(1 for m in model.modules() if m.__class__.__name__ == "HEGELU")
    print(f"[COUNT] HEGELU modules={num_hegelu}")

    return model


# -------------------- Full dataset probe --------------------
@torch.no_grad()
def probe_ratio_inrange_full(model, dataset, data_collator, clip_check: float = 3.0, batch_size: int = 2, masked: bool = True):
    """Compute the ratio of c_fc outputs within [-clip_check, clip_check] across the full dataset."""
    from torch.utils.data import DataLoader

    model.eval()

    total = 0
    in_range = 0

    # To verify we really cover all layers: collect hooked module names
    hooked_names = []

    cur_mask_cpu = None  # [B,T] on CPU

    handles = []

    def hook_factory(mod_name: str):
        def _hook(mod, inp, out):
            nonlocal total, in_range, cur_mask_cpu
            x = out.detach()  # [B,T,H]
            if masked and (cur_mask_cpu is not None):
                m = cur_mask_cpu.to(device=x.device, dtype=torch.bool).unsqueeze(-1)  # [B,T,1]
                tot = int(m.sum().item()) * x.size(-1)
                cond = (x >= -clip_check) & (x <= clip_check)
                inn = int((cond & m).sum().item())
            else:
                tot = x.numel()
                inn = int(((x >= -clip_check) & (x <= clip_check)).sum().item())

            total += tot
            in_range += inn
        return _hook

    # Hook all GPT-2 MLP c_fc modules (should be 6 for GPT2-distil)
    for name, mod in model.named_modules():
        if name.endswith("mlp.c_fc"):
            handles.append(mod.register_forward_hook(hook_factory(name)))
            hooked_names.append(name)

    print(f"[HOOK] c_fc hooked count={len(hooked_names)}")
    for n in hooked_names:
        print("  -", n)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    for batch in dl:
        if masked and ("attention_mask" in batch):
            cur_mask_cpu = batch["attention_mask"].cpu()
        else:
            cur_mask_cpu = None

        batch = {k: v.to(DEVICE) for k, v in batch.items() if torch.is_tensor(v)}
        _ = model(**batch)

    for h in handles:
        h.remove()

    ratio = in_range / max(1, total)
    print(f"[PROBE FULL] masked={masked} clip={clip_check} total_scalars={total} in_range={in_range} ratio_in[-{clip_check},{clip_check}]={ratio*100:.4f}%")


def main():
    if not os.path.isdir(LM_DS_PATH):
        raise FileNotFoundError(
            f"Cached dataset not found: {LM_DS_PATH}. " 
            "Please run your training script once to create ./data/lm_ds_he via save_to_disk()."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Optional LoRA adapter (your finetuned checkpoint)
    if ADAPTER_PATH:
        print(f"[LOAD] Loading adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.to(DEVICE)

    # Load cached dataset
    lm_ds = load_from_disk(LM_DS_PATH)
    dataset = lm_ds["train"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Apply HE-friendly GELU replacement (only if requested)
    if USE_HE_APPROX and REPLACE_GELU:
        model = apply_he_gelu_only(model, gelu_approx="poly")
    else:
        print(f"[INFO] USE_HE_APPROX={USE_HE_APPROX} REPLACE_GELU={REPLACE_GELU} -> no GELU replacement applied.")

    # Full-dataset ratio
    if PROBE_FULL_DATASET:
        probe_ratio_inrange_full(
            model=model,
            dataset=dataset,
            data_collator=data_collator,
            clip_check=PROBE_CLIP_CHECK,
            batch_size=PROBE_BS,
            masked=PROBE_MASKED,
        )

    if PROBE_ONLY:
        print("[DONE] PROBE_ONLY=1, exiting.")
        return


if __name__ == "__main__":
    main()
