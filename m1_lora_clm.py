import math
import os
import torch
import random
import numpy as np


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BertTokenizer,
    GPT2LMHeadModel,
)

from peft import LoraConfig, get_peft_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)



def main():
    # ========= 0. 基础设置 =========
    model_name = "uer/gpt2-chinese-cluecorpussmall"
    output_dir = "./m1_output"

    # 建议初次先用很小的数据和很少的 step，跑通流程即可
    max_train_samples = 2000
    max_eval_samples = 500

    # 训练超参（初次先保守）
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 8  # 等效 batch = 2*8=16
    learning_rate = 2e-4
    num_train_epochs = 1

    # GPT-2 通常没有 pad_token，需要手动处理
    block_size = 128  # 每个样本的最大 token 长度（越大越吃显存）

    # ========= 1. 载入 tokenizer 和模型 =========
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    #wtokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # GPT-2 类模型很多没有 pad_token，这会导致 batch padding 报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #model = AutoModelForCausalLM.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")

    # 同样，把模型的 pad_token_id 对齐
    model.config.pad_token_id = tokenizer.pad_token_id

    # ========= 2. 配置 LoRA，并把 LoRA “插”到模型上 =========
    # GPT-2 的线性层命名在不同实现中略有差异。初学者最稳的策略：
    # - 先尝试 target_modules=["c_attn", "c_proj"]（常见于 GPT-2）
    # - 如果报错/找不到模块，我们再用“打印模块名”的方式定位
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],
    )

    model = get_peft_model(model, lora_config)

    # 打印一下：确认 LoRA 参数量（应远小于全量参数）
    model.print_trainable_parameters()

    # ========= 3. 准备数据（这里用一个公开小数据：wikitext） =========
    # 注意：这是英文数据，只是为了跑通流程。
    # 你后续换成中文语料也完全一样的流程。
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_texts = raw["train"]
    eval_texts = raw["validation"]

    # 初次只取小部分，避免训练太久
    if max_train_samples is not None:
        train_texts = train_texts.select(range(min(max_train_samples, len(train_texts))))
    if max_eval_samples is not None:
        eval_texts = eval_texts.select(range(min(max_eval_samples, len(eval_texts))))

    # ========= 4. 分词 + 拼接成固定长度 block（适合语言模型训练） =========
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_train = train_texts.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_texts.map(tokenize_function, batched=True, remove_columns=["text"])

    # 把很多短句拼接后再切块，提升训练效率
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        # 截断到 block_size 的整数倍
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        # CausalLM 的 labels 就是 input_ids 的拷贝（Trainer 会算 shift）
        result["labels"] = result["input_ids"].copy()
        return result

    lm_train = tokenized_train.map(group_texts, batched=True)
    lm_eval = tokenized_eval.map(group_texts, batched=True)

    # ========= 5. DataCollator（这里 MLM=False，因为是 CausalLM） =========
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ========= 6. 训练参数 =========
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # 训练
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_steps=50,
        weight_decay=0.0,
        seed=42,

        # 评估与日志
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        logging_dir="./m1_output/logs",

        # 性能相关
        fp16=torch.cuda.is_available(),  # 有 GPU 就开混合精度
        report_to="none",  # 不用 wandb
    )

    # ========= 7. Trainer =========
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_train,
        eval_dataset=lm_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ========= 8. 开始训练 =========
    trainer.train()

    # ========= 9. 评估并计算 perplexity =========
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics["eval_loss"]
    ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval loss: {eval_loss:.4f}, Perplexity: {ppl:.2f}")

    # ========= 10. 保存 LoRA adapter（不是保存整个大模型） =========
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))


if __name__ == "__main__":
    main()
