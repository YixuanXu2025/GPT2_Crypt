#%%
import os
import math
import random
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from approx.he_approx import HEGELU, HELayerNorm
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
from peft import LoraConfig, get_peft_model

DATA_PATH = "./data/CLUECorpusSmall.txt"
MODEL_NAME = "uer/gpt2-distil-chinese-cluecorpussmall"
USE_HE_APPROX = os.environ.get("USE_HE_APPROX", "0") == "1"
REPLACE_LN = os.environ.get("REPLACE_LN", "1") == "1"
REPLACE_GELU = os.environ.get("REPLACE_GELU", "1") == "1"

BENCH_STEPS = int(os.environ.get("BENCH_STEPS", "200"))   # 200 步通常 10~20 分钟级（视GPU）
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./ft_out_bench")
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")           # 为空则不resume

LN_APPROX = os.environ.get("LN_APPROX", "static_calib")  # affine_only / static_calib
CALIB_BATCHES = int(os.environ.get("CALIB_BATCHES", "20"))  # 校准用多少个 batch
LN_SCOPE = os.environ.get("LN_SCOPE", "all")  # all / ln_2_only



raw = load_dataset("text", data_files={"train": "./data/CLUECorpusSmall.txt"})["train"]
raw = raw.train_test_split(test_size=0.01, seed=42)
holdout = raw["test"].shuffle(seed=42)
raw = DatasetDict({
    "train": raw["train"].select(range(5000)),
    "validation": holdout.select(range(50)),
    "test": holdout.select(range(50, 50 + 50)),  # 也可扩大
})
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")

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
    为每个 nn.LayerNorm 统计一个“token-wise mean 的平均值”和“rsqrt(var+eps) 的平均值”。
    这样 static_calib 的 (x - mu) * inv_std 在平均意义上更接近真实 LN 的归一化尺度。
    返回: dict[name] = (mu_scalar, inv_std_scalar)
    """
    model.eval()
    stats = {}
    handles = []
    sums_mean = {}
    sums_rsqrt = {}
    counts = {}

    def make_hook(name: str, eps: float):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            if x.dim() == 2:
                x_ = x.unsqueeze(1)
            else:
                x_ = x
            x_ = x_.float()  # 关键：统计用 fp32

            mean = x_.mean(dim=-1)
            var = x_.var(dim=-1, unbiased=False)
            rsqrt = torch.rsqrt(var + eps)

            # 可选但很推荐：过滤掉非有限值
            rsqrt = torch.where(torch.isfinite(rsqrt), rsqrt, torch.zeros_like(rsqrt))

            # 只取标量累计，避免存大张量
            sums_mean[name] += mean.sum().item()
            sums_rsqrt[name] += rsqrt.sum().item()
            counts[name] += mean.numel()
        return hook

    # 注册 hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
            sums_mean[name] = 0.0
            sums_rsqrt[name] = 0.0
            counts[name] = 0
            handles.append(mod.register_forward_hook(make_hook(name, mod.eps)))

    # 用少量 batch 跑一遍 forward 做统计
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    it = iter(dl)
    for _ in range(num_batches):
        batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.cuda.amp.autocast(enabled=False):  # 关键：校准用 fp32
            _ = model(**batch)

    # 移除 hook
    for h in handles:
        h.remove()

    # 汇总
    for name in sums_mean.keys():
        c = max(1, counts[name])
        mu_scalar = sums_mean[name] / c
        inv_std_scalar = sums_rsqrt[name] / c

        # 安全保护：避免 inf / 0 / 极端放大
        if not math.isfinite(mu_scalar):
            mu_scalar = 0.0
        if (not math.isfinite(inv_std_scalar)) or (inv_std_scalar <= 0):
            inv_std_scalar = 1.0
        inv_std_scalar = max(min(inv_std_scalar, 10.0), 0.05)  # 经验范围，可按需调整

        stats[name] = (mu_scalar, inv_std_scalar)

    return stats

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

def apply_he_approx(model, gelu_approx="poly", ln_approx="static_calib", ln_stats=None):
    ln_total = 0
    ln_loaded = 0
    def _replace_ln(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.LayerNorm):
                nonlocal ln_total, ln_loaded
                ln_total += 1
                ...
                if ln_approx == "static_calib" and ln_stats is not None and full_name in ln_stats:
                    ln_loaded += 1

                # scope 控制：只替换 ln_2
                if LN_SCOPE == "ln_2_only" and not full_name.endswith("ln_2"):
                    continue

                mu = None
                inv_std = None
                if ln_approx == "static_calib" and ln_stats is not None and full_name in ln_stats:
                    mu_scalar, inv_std_scalar = ln_stats[full_name]
                    # 用标量填满到 normalized_shape，保证广播
                    mu = torch.full(child.normalized_shape, float(mu_scalar), dtype=torch.float32)
                    inv_std = torch.full(child.normalized_shape, float(inv_std_scalar), dtype=torch.float32)
                    inv_list = [v[1] for v in ln_stats.values()]
                    print(f"[CALIB] ln_stats={len(ln_stats)} inv_std range: {min(inv_list):.4f} ~ {max(inv_list):.4f}")

                new_ln = HELayerNorm(
                    child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                    approx=ln_approx,
                    mu=mu,
                    inv_std=inv_std,
                )
                if child.elementwise_affine:
                    new_ln.weight.data.copy_(child.weight.data)
                    new_ln.bias.data.copy_(child.bias.data)

                setattr(parent, name, new_ln)
            else:
                _replace_ln(child, full_name)

    def _replace_gelu(parent: nn.Module):
        # 你下一轮先关掉 GELU 替换（REPLACE_GELU=0），这里保留不动
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.GELU):
                setattr(parent, name, HEGELU(approx=gelu_approx))
            else:
                _replace_gelu(child)

        if parent.__class__.__name__.lower().endswith("mlp"):
            if hasattr(parent, "act") and callable(getattr(parent, "act")) and not isinstance(getattr(parent, "act"), nn.Module):
                setattr(parent, "act", HEGELU(approx=gelu_approx))

    if REPLACE_LN:
        _replace_ln(model)
    if REPLACE_GELU:
        _replace_gelu(model)
    print(f"[LN STATS] scope={LN_SCOPE} replaced={ln_total} stats_loaded={ln_loaded}")
    return model


print("saved to:", save_path)

# 1) tokenizer / model

# 确保 pad_token_id 存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 2) （可选）LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "c_fc"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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

if USE_HE_APPROX:
    model = apply_he_approx(model, gelu_approx="poly", ln_approx=LN_APPROX, ln_stats=ln_stats)


# 5) TrainingArguments（先跑通，后面再加大 max_steps）

args = TrainingArguments(

    output_dir=OUTPUT_DIR,
    max_steps=BENCH_STEPS,
    overwrite_output_dir=False,
    # Batch size设置
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    # 学习率与优化
    learning_rate=5e-5,

    num_train_epochs=1,
    # 热身步骤需要大幅增加（对于1600万数据）
    warmup_steps=min(50, BENCH_STEPS // 10),
    # 大幅增加日志间隔（从20到10000步）
    logging_steps=1000,  # 每1万步打印一次（不是10万，这样可以看到进展）
    # 评估和保存间隔调整为每10万步
    eval_strategy="steps",
    eval_steps=100,  # 每10万步评估一次
    save_strategy="steps",
    save_steps=100,  # 每10万步保存一次
    save_total_limit=10,  # 保存5个检查点（50万步时会有5个）
    # 其他设置
    fp16=torch.cuda.is_available(),
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
