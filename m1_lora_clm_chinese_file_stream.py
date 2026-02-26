import os
import math
import random
import torch
from torch.utils.data import IterableDataset

from transformers import (
    BertTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CLMBlockDataset(IterableDataset):
    """
    从一个超大的 txt 文件中逐行读取中文文本，分词后拼接成 token 流，
    再按 block_size 切成固定长度样本，用于 CausalLM 训练。

    好处：不需要一次性读入内存，也不需要预处理成箭头/缓存文件。
    """
    def __init__(self, file_path, tokenizer, block_size=128, max_lines=None):
        super().__init__()
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_lines = max_lines  # 先跑通可设一个小值，比如 20000；正式实验设 None

    def __iter__(self):
        buffer_ids = []
        line_count = 0

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if self.max_lines is not None and line_count >= self.max_lines:
                    break

                line = line.strip()
                line_count += 1

                if not line:
                    continue

                # 分词：不加特殊 token，直接拼接成长序列
                #ids = self.tokenizer(line, add_special_tokens=False)["input_ids"]
                ids = self.tokenizer(
                    line,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=1024,  # GPT-2 的最大上下文通常是 1024
                )["input_ids"]

                if len(ids) == 0:
                    continue

                buffer_ids.extend(ids)

                # 不断切块
                while len(buffer_ids) >= self.block_size:
                    chunk = buffer_ids[: self.block_size]
                    buffer_ids = buffer_ids[self.block_size :]

                    yield {
                        "input_ids": chunk,
                        "labels": chunk.copy(),
                        "attention_mask": [1] * self.block_size,
                    }


def main():
    set_seed(42)

    # ======= 你只需要改这里：数据路径 =======
    DATA_PATH = "./data/CLUECorpusSmall.txt"  # 改成你的真实路径
    # =======================================

    model_name = "uer/gpt2-chinese-cluecorpussmall"
    output_dir = "./m1_output_chinese"

    # 新手先跑通：先用少量行数，确认训练流程正确
    # 跑通后把它设为 None，就会读完整文件（但训练 epoch/steps 要控制）
    max_train_lines = 20000
    max_eval_lines = 2000

    block_size = 128

    # 1) 加载 tokenizer / model（模型卡推荐 BertTokenizer + GPT2LMHeadModel）
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT2 常见：补齐 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 2) LoRA 配置（先和你 M1 baseline 保持一致）
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

    # 3) 构造训练/验证 IterableDataset
    train_ds = CLMBlockDataset(
        file_path=DATA_PATH,
        tokenizer=tokenizer,
        block_size=block_size,
        max_lines=max_train_lines,
    )

    eval_ds = CLMBlockDataset(
        file_path=DATA_PATH,
        tokenizer=tokenizer,
        block_size=block_size,
        max_lines=max_eval_lines,
    )

    # 4) collator：CausalLM -> mlm=False
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) 训练参数：IterableDataset 必须用 max_steps 控制训练长度（不要用 epochs 依赖长度）
    #    这里先跑少量 steps，确保 eval loss / ppl 能出。
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,

        max_steps=200,          # 先跑 200 steps 验证流程；正式实验再调大
        warmup_steps=50,

        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,

        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval loss: {eval_loss:.4f}, Perplexity: {ppl:.2f}")

    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    main()
