#%%
import os
import math
import random
import torch
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


raw = load_dataset("text", data_files={"train": "./data/CLUECorpusSmall.txt"})["train"]
raw = raw.train_test_split(test_size=0.01, seed=42)
holdout = raw["test"].shuffle(seed=42)
raw = DatasetDict({
    "train": raw["train"].select(range(500000)),
    "validation": holdout.select(range(5000)),
    "test": holdout.select(range(5000, 5000 + 5000)),  # 也可扩大
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
save_path = "./data/lm_ds"
lm_ds.save_to_disk(save_path)
print("saved to:", save_path)

# 1) tokenizer / model

# 确保 pad_token_id 存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 2) （可选）LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4) collator：CausalLM 训练必须 mlm=False
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5) TrainingArguments（先跑通，后面再加大 max_steps）

args = TrainingArguments(
    output_dir="./ft_out",
    overwrite_output_dir=False,
    # Batch size设置
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    # 学习率与优化
    learning_rate=2e-4,
    max_steps=50000,
    num_train_epochs=1,
    # 热身步骤需要大幅增加（对于1600万数据）
    warmup_steps=500,
    # 大幅增加日志间隔（从20到10000步）
    logging_steps=1000,  # 每1万步打印一次（不是10万，这样可以看到进展）
    # 评估和保存间隔调整为每10万步
    eval_strategy="steps",
    eval_steps=10000,  # 每10万步评估一次
    save_strategy="steps",
    save_steps=10000,  # 每10万步保存一次
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
trainer.train(resume_from_checkpoint="./ft_out/checkpoint-30000")


# 最终测试集：只做一次
test_metrics = trainer.evaluate(eval_dataset=lm_ds["test"])
test_loss = test_metrics["eval_loss"]
test_ppl = math.exp(test_loss) if test_loss < 20 else float("inf")
print(f"[TEST] loss={test_loss:.4f}, ppl={test_ppl:.2f}")
# 保存 LoRA adapter
model.save_pretrained("./ft_out/lora_adapter")
tokenizer.save_pretrained("./ft_out/lora_adapter")
