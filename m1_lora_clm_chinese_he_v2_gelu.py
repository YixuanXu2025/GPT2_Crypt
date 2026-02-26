#%%
import os
import math
import random
import torch
import time
import torch.nn as nn
from transformers.activations import NewGELUActivation, GELUActivation
from typing import Optional
from torch.utils.data import DataLoader
from approx.he_approx import HEGELU, HELayerNorm, fit_gelu_poly_from_samples
from torch.utils.data import IterableDataset
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    BertTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    BloomForCausalLM,
    AutoConfig,
)
from transformers.activations import NewGELUActivation, GELUActivation
from peft import LoraConfig, get_peft_model, PeftModel

# ----------------------------
# Device helpers (GPU-only replacements)
# ----------------------------
def _module_device(mod: nn.Module) -> torch.device:
    """Infer a module's device for placing replacement modules on the same GPU."""
    for p in mod.parameters(recurse=False):
        return p.device
    for b in mod.buffers(recurse=False):
        return b.device
    for p in mod.parameters():
        return p.device
    for b in mod.buffers():
        return b.device
    return torch.device("cpu")


def _unwrap_core_gpt2(m: nn.Module) -> nn.Module:
    """Unwrap PeftModel / wrappers to access underlying GPT2 {transformer.h}."""
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m
    if hasattr(m, "base_model"):
        bm = m.base_model
        if hasattr(bm, "model") and hasattr(bm.model, "transformer"):
            return bm.model
        if hasattr(bm, "transformer"):
            return bm
    if hasattr(m, "model") and hasattr(m.model, "transformer"):
        return m.model
    return m




DATA_PATH = "./data/CLUECorpusSmall.txt"
MODEL_NAME = "uer/gpt2-distil-chinese-cluecorpussmall"
USE_HE_APPROX = os.environ.get("USE_HE_APPROX", "0") == "1"
REPLACE_LN = os.environ.get("REPLACE_LN", "1") == "1"
REPLACE_GELU = os.environ.get("REPLACE_GELU", "1") == "1"

BENCH_STEPS = int(os.environ.get("BENCH_STEPS", "200"))   # 200 步通常 10~20 分钟级（视GPU）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./ft_out_bench")
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")           # 为空则不resume
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")         # 为空则从头训练 LoRA；否则加载 adapter 继续训练
LR = float(os.environ.get("LR", "5e-5"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "50"))
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"
TRAIN_LN_AFFINE = os.environ.get("TRAIN_LN_AFFINE", "0") == "1"


LN_APPROX = os.environ.get("LN_APPROX", "static_calib")  # affine_only / static_calib
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "20"))  # 校准用多少个 batch
LN_SCOPE = os.environ.get("LN_SCOPE", "all")  # all / ln_2_only
LN1_TOPK = int(os.environ.get("LN1_TOPK", "0"))  # when LN_SCOPE=all, only replace ln_1 for last K blocks; 0 means all
CALIB_CLAMP_MIN = float(os.environ.get("CALIB_CLAMP_MIN", "0.1"))
CALIB_CLAMP_MAX = float(os.environ.get("CALIB_CLAMP_MAX", "5.0"))

GELU_APPROX = os.environ.get("GELU_APPROX", "poly")  # poly / poly_horner / pwl / identity
GELU_DEGREE = int(os.environ.get("GELU_DEGREE", "7"))  # degree<=8 for md<=3
GELU_CENTRAL_MASS = float(os.environ.get("GELU_CENTRAL_MASS", "0.90"))  # central quantile mass
GELU_CALIB_BATCHES = int(os.environ.get("GELU_CALIB_BATCHES", str(CALIB_BATCHES)))
GELU_MAX_SAMPLES = int(os.environ.get("GELU_MAX_SAMPLES", "200000"))
GELU_CLIP_FOR_FIT = os.environ.get("GELU_CLIP_FOR_FIT", "1") == "1"



raw = load_dataset("text", data_files={"train": "./data/CLUECorpusSmall.txt"})["train"]
raw = raw.train_test_split(test_size=0.01, seed=42)
holdout = raw["test"].shuffle(seed=42)
raw = DatasetDict({
    "train": raw["train"].select(range(5000)),
    "validation": holdout.select(range(50)),
    "test": holdout.select(range(50, 50 + 50)),  # 也可扩大
})
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

cls_id = tokenizer.cls_token_id
sep_id = tokenizer.sep_token_id

block_size = 1024
# 如果你希望每个块都形如 [CLS] ... [SEP]，那中间内容最多 block_size-2
max_content_len = block_size - 2

stride = 128
step = max_content_len - stride

@torch.no_grad()
def calibrate_ln_stats(model, dataset, data_collator, device, batch_size: int, num_batches: int):
    """
    静态校准：为每个 nn.LayerNorm 统计一个固定的 (mu, inv_std)，用于 HELayerNorm(static_calib)。

    注意：这里统计的是“按通道（hidden dim）在 (batch, seq) 维度上的均值/方差”，得到 shape=(H,) 的向量。
    这不是严格等价于 LayerNorm 的“每 token 归一化”，但在 HE-friendly 约束下是更稳、更常用的近似起点。
    """
    from torch.utils.data import DataLoader

    model.eval()
    handles = []
    sums_mu = {}
    sums_invstd = {}
    counts = {}

    def make_hook(name: str, eps: float):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (B,1,H)
            # 统计用 fp32
            x = x.float()

            # 按 (batch, seq) 聚合，得到每通道统计量 (H,)
            mu = x.mean(dim=(0, 1))
            var = x.var(dim=(0, 1), unbiased=False)
            inv_std = torch.rsqrt(var + eps)

            # 过滤非有限值
            mu = torch.where(torch.isfinite(mu), mu, torch.zeros_like(mu))
            inv_std = torch.where(torch.isfinite(inv_std), inv_std, torch.ones_like(inv_std))

            # 累计到 CPU，节省显存
            sums_mu[name] += mu.detach()
            sums_invstd[name] += inv_std.detach()
            counts[name] += 1
        return hook

    # 注册 hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
            shape = mod.normalized_shape
            if isinstance(shape, int):
                shape = (shape,)
            H = int(shape[-1])
            sums_mu[name] = torch.zeros(H, dtype=torch.float32, device=device)
            sums_invstd[name] = torch.zeros(H, dtype=torch.float32, device=device)
            counts[name] = 0
            handles.append(mod.register_forward_hook(make_hook(name, mod.eps)))

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    it = iter(dl)
    for _ in range(num_batches):
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        # 校准 forward 一律用 fp32，避免统计被 fp16 误差污染
        with torch.cuda.amp.autocast(enabled=False):
            _ = model(**batch)

    for h in handles:
        h.remove()

    stats = {}
    for name in sums_mu.keys():
        c = max(1, counts[name])
        mu = sums_mu[name] / c
        inv_std = sums_invstd[name] / c

        # 安全夹紧：避免极端放大导致后续爆炸（尤其是 ln_1）
        inv_std = torch.clamp(inv_std, CALIB_CLAMP_MIN, CALIB_CLAMP_MAX)

        stats[name] = (mu, inv_std)

    return stats


@torch.no_grad()
def calibrate_gelu_poly_coeffs(
    model,
    dataset,
    data_collator,
    device,
    batch_size: int,
    num_batches: int,
    degree: int = 7,
    central_mass: float = 0.90,
    max_samples: int = 200000,
    clip_for_fit: bool = True,
):
    """
    Collect representative samples of GELU input (pre-activation) and fit a polynomial.

    For GPT2, MLP activation is typically `transformers.activations.NewGELUActivation` as `mlp.act`.
    We register forward_pre_hook on those activation modules to capture their input tensors.

    The fitting uses a central quantile interval containing `central_mass` of samples (e.g. 0.90 -> [5%,95%])
    and performs least-squares regression via `fit_gelu_poly_from_samples`.

    Returns:
      coeffs: [c0..c_degree] float32 tensor (CPU)
      interval: (lo, hi)
    """
    model.eval()

    # Collect on CPU to avoid GPU memory growth
    collected = []
    seen = 0
    handles = []

    def pre_hook(mod, inputs):
        nonlocal seen
        if seen >= max_samples:
            return
        if not inputs:
            return
        x = inputs[0].detach()
        if x.dim() >= 2:
            x = x.reshape(-1)
        x = x.to(dtype=torch.float32)
        if x.numel() == 0:
            return
        take = min(int(max_samples - seen), int(x.numel()))
        if take <= 0:
            return
        collected.append(x[:take].clone())
        seen += take

    # Register hooks on all GELU activation modules
    for m in model.modules():
        if isinstance(m, (NewGELUActivation, GELUActivation, nn.GELU)):
            handles.append(m.register_forward_pre_hook(pre_hook))

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    it = iter(dl)

    for _ in range(num_batches):
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.cuda.amp.autocast(enabled=False):
            _ = model(**batch)
        if seen >= max_samples:
            break

    for h in handles:
        h.remove()

    if not collected:
        raise RuntimeError("No GELU samples collected. Check that activation modules exist and hooks are registered.")

    samples = torch.cat(collected, dim=0)
    coeffs, interval = fit_gelu_poly_from_samples(
        samples=samples,
        degree=degree,
        central_mass=central_mass,
        clip_for_fit=clip_for_fit,
    )
    return coeffs.detach().cpu(), interval

def tokenize_lines(examples):
    return tokenizer(examples["text"], add_special_tokens=False)
tok = raw.map(tokenize_lines, batched=True, remove_columns=["text"])
def split_to_chunks(examples):
    input_ids_out, attn_out = [], []
    for ids in examples["input_ids"]:
        if not ids:
            continue
        start = 0
        while start < len(ids):
            chunk = ids[start : start + max_content_len]
            # 组装成 [CLS] chunk [SEP]
            chunk_ids = [cls_id] + chunk + [sep_id]
            input_ids_out.append(chunk_ids)
            attn_out.append([1] * len(chunk_ids))

            if start + max_content_len >= len(ids):
                break
            start += step

    return {"input_ids": input_ids_out, "attention_mask": attn_out}

lm_ds = tok.map(split_to_chunks, batched=True, remove_columns=tok["train"].column_names)
save_path = "./data/lm_ds_he"
lm_ds.save_to_disk(save_path)


def apply_he_approx(
    model,
    gelu_approx: str = "poly",
    ln_approx: str = "static_calib",
    ln_stats=None,
    gelu_poly_coeffs: Optional[torch.Tensor] = None,
):
    """
    Replace LayerNorm and GELU activations with HE-friendly approximations.

    GPU-only guarantee:
      - Replacement modules are created and placed on the SAME device as the module they replace.
      - This avoids CPU<->GPU copy errors and keeps the entire experiment on GPU.
    """
    ln_total = 0
    ln_loaded = 0
    replaced_names = []

    def _replace_ln(parent: nn.Module, prefix: str = ""):
        nonlocal ln_total, ln_loaded
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.LayerNorm):
                ln_total += 1
                if ln_approx == "static_calib" and ln_stats is not None and full_name in ln_stats:
                    ln_loaded += 1

                # Scope controls (keep your existing behavior)
                if LN_SCOPE == "ln_2_only" and not full_name.endswith("ln_2"):
                    continue
                if LN_SCOPE == "all" and LN1_TOPK > 0 and full_name.endswith("ln_1"):
                    # only replace last K blocks' ln_1
                    import re
                    m = re.search(r"\.h\.(\d+)\.ln_1$", full_name)
                    if m and hasattr(model, "config") and hasattr(model.config, "n_layer"):
                        layer_idx = int(m.group(1))
                        if layer_idx < int(model.config.n_layer) - LN1_TOPK:
                            continue

                # Put new module on same device
                dev = _module_device(child)
                if dev.type == "cpu":
                    dev = _module_device(parent)

                mu = None
                inv_std = None
                if ln_approx == "static_calib" and ln_stats is not None and full_name in ln_stats:
                    mu_vec, inv_std_vec = ln_stats[full_name]
                    mu = mu_vec.to(device=dev, dtype=torch.float32)
                    inv_std = inv_std_vec.to(device=dev, dtype=torch.float32)

                new_ln = HELayerNorm(
                    child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                    approx=ln_approx,
                    mu=mu,
                    inv_std=inv_std,
                ).to(dev)

                # 关键：把新 LN 放到被替换 LN 同一设备（你要全程 GPU，必须做）
                if child.elementwise_affine:
                    new_ln = new_ln.to(child.weight.device)
                    new_ln.weight.data.copy_(child.weight.data)
                    new_ln.bias.data.copy_(child.bias.data)
                else:
                    # 没有 affine 时也尽量跟随 child 的 device
                    new_ln = new_ln.to(next(child.parameters(), torch.tensor(0,
                                                                             device="cuda" if torch.cuda.is_available() else "cpu")).device)

                setattr(parent, name, new_ln)
                replaced_names.append(full_name)
            else:
                _replace_ln(child, full_name)

    def _make_he_gelu(dev: torch.device) -> nn.Module:
        if gelu_poly_coeffs is not None and gelu_approx in ("poly", "poly_horner"):
            return HEGELU(approx=gelu_approx, poly_coeffs=gelu_poly_coeffs.to(device=dev))
        return HEGELU(approx=gelu_approx)

    def _replace_gelu(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, (nn.GELU, NewGELUActivation, GELUActivation)):
                dev = _module_device(child)
                if dev.type == "cpu":
                    dev = _module_device(parent)
                setattr(parent, name, _make_he_gelu(dev).to(dev))
            else:
                _replace_gelu(child, full_name)

        # Extra safety: GPT2MLP has attribute `act` holding NewGELUActivation
        if hasattr(parent, "act"):
            act = getattr(parent, "act")
            dev = _module_device(parent)
            if isinstance(act, (nn.GELU, NewGELUActivation, GELUActivation)):
                setattr(parent, "act", _make_he_gelu(dev).to(dev))
            elif callable(act) and not isinstance(act, nn.Module):
                setattr(parent, "act", _make_he_gelu(dev).to(dev))

    if REPLACE_LN:
        _replace_ln(model)
    if REPLACE_GELU:
        _replace_gelu(model)

    print(f"[LN STATS] scope={LN_SCOPE} replaced={ln_total} stats_loaded={ln_loaded}")
    print(f"[LN REPLACED] count={len(replaced_names)}")
    for n in replaced_names:
        print("  -", n)

    return model


print("saved to:", save_path)

# 1) tokenizer / model
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],
)
if ADAPTER_PATH:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, is_trainable=True)
else:
    model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# 确保 pad_token_id 存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 2) （可选）LoRA



# 打印一下替换是否生效（非常重要，避免“其实没替换到”）
num_he_ln = sum(isinstance(m, HELayerNorm) for m in model.modules())
num_he_gelu = sum(isinstance(m, HEGELU) for m in model.modules())
print(f"[HE_APPROX] enabled={USE_HE_APPROX}  HELayerNorm={num_he_ln}  HEGELU={num_he_gelu}")


# 4) collator：CausalLM 训练必须 mlm=False
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

ln_stats = None
if USE_HE_APPROX and REPLACE_LN and LN_APPROX == "static_calib":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 用训练集做少量校准；不需要很大
    ln_stats = calibrate_ln_stats(
        model=model,
        dataset=lm_ds["train"],
        data_collator=data_collator,
        device=device,
        batch_size=2,           # 对齐你的 per_device_train_batch_size
        num_batches=CALIB_BATCHES,
    )

    if ln_stats:
        # inv_std 是向量（Tensor），不能直接用 Python 的 min/max 比较 Tensor；这里取所有层所有通道的全局最小/最大
        inv_mins = [float(v[1].min().item()) for v in ln_stats.values()]
        inv_maxs = [float(v[1].max().item()) for v in ln_stats.values()]
        print(f"[CALIB] ln_stats={len(ln_stats)} inv_std range: {min(inv_mins):.4f} ~ {max(inv_maxs):.4f}")

gelu_poly_coeffs = None
gelu_interval = None
if USE_HE_APPROX and REPLACE_GELU and GELU_APPROX in ("poly", "poly_horner"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Calibrate with fp32 forward to avoid fp16 noise in statistics
    gelu_poly_coeffs, gelu_interval = calibrate_gelu_poly_coeffs(
        model=model,
        dataset=lm_ds["train"],
        data_collator=data_collator,
        device=device,
        batch_size=2,  # align with per_device_train_batch_size
        num_batches=GELU_CALIB_BATCHES,
        degree=GELU_DEGREE,
        central_mass=GELU_CENTRAL_MASS,
        max_samples=GELU_MAX_SAMPLES,
        clip_for_fit=GELU_CLIP_FOR_FIT,
    )
    print(f"[CALIB GELU] degree={GELU_DEGREE} central_mass={GELU_CENTRAL_MASS} interval={gelu_interval} coeffs_len={len(gelu_poly_coeffs)}")

if USE_HE_APPROX:
    model = apply_he_approx(model, gelu_approx=GELU_APPROX, ln_approx=LN_APPROX, ln_stats=ln_stats, gelu_poly_coeffs=gelu_poly_coeffs)
    # 替换后再数一次，确认确实生效
    num_he_ln_after = sum(isinstance(m, HELayerNorm) for m in model.modules())
    num_he_gelu_after = sum(isinstance(m, HEGELU) for m in model.modules())
    print(f"[HE_APPROX AFTER] HELayerNorm={num_he_ln_after}  HEGELU={num_he_gelu_after}")
    core = _unwrap_core_gpt2(model)
    print(type(core.transformer.h[0].mlp.act))
    print(type(core.transformer.h[1].mlp.act))

if TRAIN_LN_AFFINE and USE_HE_APPROX and REPLACE_LN:
    # 只解冻 LN 的仿射参数（非常少），让模型更容易适配 LN 近似
    for n, p in model.named_parameters():
        if ("ln_" in n) and (n.endswith(".weight") or n.endswith(".bias")):
            p.requires_grad = True


# 5) TrainingArguments（先跑通，后面再加大 max_steps）

args = TrainingArguments(

    output_dir=OUTPUT_DIR,
    max_steps=BENCH_STEPS,
    overwrite_output_dir=False,
    # Batch size设置
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,  # 防止梯度爆炸

    # 学习率与优化
    learning_rate=LR,

    num_train_epochs=1,
    # 热身步骤需要大幅增加（对于1600万数据）
    warmup_steps=min(WARMUP_STEPS, max(0, BENCH_STEPS // 2)),
    # 大幅增加日志间隔（从20到10000步）
    logging_steps=1000,  # 每1万步打印一次（不是10万，这样可以看到进展）
    # 评估和保存间隔调整为每10万步
    eval_strategy="steps",
    eval_steps=100,  # 每10万步评估一次
    save_strategy="steps",
    save_steps=100,  # 每10万步保存一次
    save_total_limit=10,  # 保存5个检查点（50万步时会有5个）
    # 其他设置
    fp16=(USE_FP16 and torch.cuda.is_available() and (not USE_HE_APPROX)),
    report_to="tensorboard",  # 建议使用tensorboard记录
    seed=42,
    # 新增优化参数
    optim="adamw_torch",
    lr_scheduler_type="cosine",  # 余弦退火学习率
    weight_decay=0.01,  # 权重衰减防止过拟合
    gradient_checkpointing=False,  # 节省显存
    dataloader_num_workers=4,  # 加速数据加载
    group_by_length=True,  # 按长度分组提高效率
)

# 6) Trainer
trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=lm_ds["train"],
    eval_dataset=lm_ds["validation"],
    data_collator=data_collator,
)

#trainer.train()
start_step = trainer.state.global_step

torch.cuda.synchronize()
t0 = time.perf_counter()

if RESUME_CKPT:
    trainer.train(resume_from_checkpoint=RESUME_CKPT)
else:
    trainer.train()

torch.cuda.synchronize()
t1 = time.perf_counter()

end_step = trainer.state.global_step
steps = max(1, end_step - start_step)
total_sec = t1 - t0

# 估算 tokens/s（近似即可，比较用）
# 单卡：effective_batch = per_device_train_batch_size * gradient_accumulation_steps
effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
approx_tokens = steps * effective_batch * block_size  # block_size=1024（你上面定义的）
print(f"[TIME] use_he={USE_HE_APPROX} ln={REPLACE_LN} gelu={REPLACE_GELU} "
      f"steps={steps} total_sec={total_sec:.1f} sec/step={total_sec/steps:.4f} "
      f"approx_tokens/s={approx_tokens/total_sec:.0f}")



# 最终测试集：只做一次
test_metrics = trainer.evaluate(eval_dataset=lm_ds["test"])
test_loss = test_metrics["eval_loss"]
test_ppl = math.exp(test_loss) if test_loss < 20 else float("inf")
print(f"[TEST] loss={test_loss:.4f}, ppl={test_ppl:.2f}")
# 保存 LoRA adapter
model.save_pretrained("./ft_out_he/lora_adapter")
tokenizer.save_pretrained("./ft_out_he/lora_adapter")