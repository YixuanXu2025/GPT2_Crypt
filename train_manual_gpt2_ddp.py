"""分布式训练入口（支持 8 卡 4090 的 torchrun 方式）。

使用示例：
torchrun --nproc_per_node=8 train_manual_gpt2_ddp.py \
  --model_name uer/gpt2-distil-chinese-cluecorpussmall \
  --output_dir ./outputs/manual_he_ddp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.manual_gpt2_he import HELayerNorm, build_manual_model_from_hf


def set_seed(seed: int) -> None:
    """固定随机种子，提升实验可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> Dict[str, int]:
    """初始化分布式环境。

    兼容单卡和多卡：
    - 单卡直接返回 rank=0。
    - 多卡由 torchrun 注入环境变量进行初始化。
    """

    if "RANK" not in os.environ:
        return {"rank": 0, "local_rank": 0, "world_size": 1}

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def cleanup_distributed() -> None:
    """结束分布式通信。"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def rank0_print(rank: int, msg: str) -> None:
    """只在 rank0 打印，避免多卡重复刷屏。"""
    if rank == 0:
        print(msg, flush=True)


def load_and_clean_text_dataset(max_samples: int = 20000) -> Dataset:
    """加载并清洗文本数据。

    优先使用 CLUE（与你的原始场景一致），若不可用则回退到 wikitext。
    清洗策略：
    1) 去掉前后空白
    2) 去掉空行
    3) 把连续空格压缩为单空格
    """

    datasets: List[Dataset] = []
    try:
        clue = load_dataset("clue", "tnews")
        for split in ["train", "validation"]:
            if split in clue:
                ds = clue[split].map(lambda x: {"text": f"{x.get('sentence', '')}"})
                datasets.append(ds.remove_columns([c for c in ds.column_names if c != "text"]))
    except Exception:
        pass

    if not datasets:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
        datasets.append(wiki["train"].remove_columns([c for c in wiki["train"].column_names if c != "text"]))

    raw = concatenate_datasets(datasets)

    def _clean(batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        cleaned = []
        for x in batch["text"]:
            x = " ".join((x or "").strip().split())
            if len(x) >= 5:
                cleaned.append(x)
        return {"text": cleaned}

    raw = raw.map(_clean, batched=True, remove_columns=raw.column_names)
    if len(raw) > max_samples:
        raw = raw.select(range(max_samples))
    return raw.train_test_split(test_size=0.01, seed=42)["train"]


def build_lm_dataset(dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    """把清洗文本转成 GPT 训练块。"""

    def _tok(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tok = dataset.map(_tok, batched=True, remove_columns=["text"])

    def _group(batch):
        all_ids = []
        for ids in batch["input_ids"]:
            all_ids.extend(ids)
        total_len = (len(all_ids) // block_size) * block_size
        all_ids = all_ids[:total_len]

        chunks = [all_ids[i : i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks, "attention_mask": [[1] * block_size for _ in chunks]}

    return tok.map(_group, batched=True)


@torch.no_grad()
def calibrate_he_ln(model, dataloader, device, max_batches: int = 20) -> None:
    """对 HELayerNorm 做静态统计（mu 与 inv_std）。"""

    ln_modules = [(n, m) for n, m in model.named_modules() if isinstance(m, HELayerNorm)]
    if not ln_modules:
        return

    sums_mu = {name: 0.0 for name, _ in ln_modules}
    sums_inv = {name: 0.0 for name, _ in ln_modules}
    counts = {name: 0 for name, _ in ln_modules}

    hooks = []
    for name, mod in ln_modules:
        def _hook(module, inputs, output, n=name):
            x = inputs[0].float()
            mu = x.mean(dim=-1)
            var = x.var(dim=-1, unbiased=False)
            inv = torch.rsqrt(var + module.eps)
            sums_mu[n] += mu.sum().item()
            sums_inv[n] += inv.sum().item()
            counts[n] += mu.numel()

        hooks.append(mod.register_forward_hook(_hook))

    model.eval()
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch["labels"])

    for h in hooks:
        h.remove()

    for name, mod in ln_modules:
        c = max(1, counts[name])
        mu = sums_mu[name] / c
        inv = sums_inv[name] / c
        inv = max(0.05, min(10.0, float(inv)))
        mod.mu.data.fill_(mu)
        mod.inv_std.data.fill_(inv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="uer/gpt2-distil-chinese-cluecorpussmall")
    parser.add_argument("--output_dir", type=str, default="./outputs/manual_he_ddp")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_he_gelu", action="store_true")
    parser.add_argument("--use_he_ln", action="store_true")
    args = parser.parse_args()

    dist = setup_distributed()
    rank, local_rank, world_size = dist["rank"], dist["local_rank"], dist["world_size"]
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_and_clean_text_dataset(max_samples=args.max_samples)
    lm_ds = build_lm_dataset(raw, tokenizer=tokenizer, block_size=args.block_size)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sampler = DistributedSampler(lm_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        lm_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    model, cfg = build_manual_model_from_hf(
        model_name=args.model_name,
        use_he_gelu=args.use_he_gelu,
        use_he_ln=args.use_he_ln,
    )
    model.to(device)

    if args.use_he_ln:
        calibrate_he_ln(model, loader, device=device, max_batches=20)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss = out["loss"] / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 20 == 0:
                    # 多卡下做 loss 平均，观察更稳定
                    loss_item = out["loss"].detach()
                    if world_size > 1:
                        torch.distributed.all_reduce(loss_item, op=torch.distributed.ReduceOp.SUM)
                        loss_item = loss_item / world_size
                    rank0_print(rank, f"[epoch={epoch}] step={global_step} loss={loss_item.item():.4f}")

    # 只在 rank0 保存
    if rank == 0:
        unwrapped = model.module if isinstance(model, DDP) else model
        ckpt_dir = os.path.join(args.output_dir, "manual_model")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ckpt_dir)

        with open(os.path.join(ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(args), "model_cfg": asdict(cfg)}, f, ensure_ascii=False, indent=2)

        rank0_print(rank, f"训练完成，模型已保存到: {ckpt_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
