import math
import random
import numpy as np
import torch
from itertools import chain

from datasets import load_dataset
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    # =========================
    # 1) 数据路径：改成你的真实文件路径
    #    你说文件名是：web_text_zh_train / web_text_zh_testa / web_text_zh_valid
    #    它们通常是 JSONL（每行一个 JSON 对象）
    # =========================
    data_files = {
        "train": "./data/web_text_zh_train",
        "validation": "./data/web_text_zh_valid",
        "test": "./data/web_text_zh_testa",
    }

    model_name = "uer/gpt2-chinese-cluecorpussmall"
    output_dir = "./m1_output_webtext_zh"

    # 你可以先用小子集跑通（新手推荐）
    max_train_samples = 200_000
    max_eval_samples = 20_000

    block_size = 128  # 先用 128，显存允许再试 256

    # =========================
    # 2) 加载数据集（不使用 streaming，方便后续处理/可复现）
    # =========================
    ds = load_dataset("json", data_files=data_files, streaming=False)

    # ds 是 DatasetDict，包含 ds["train"], ds["validation"], ds["test"]
    print("Columns:", ds["train"].column_names)
    print("Example:", ds["train"][0])

    # =========================
    # 3) 把一条样本转换成“训练文本”
    #    你的样本字段有：title/desc/topic/content 等
    #    对语言模型训练，常见做法是把这些字段拼成一段自然语言
    # =========================
    def build_text(batch):
        """
        batch 是一个“批”，例如 batch["title"] 是 list[str]
        我们返回一个新的字段 "text": list[str]
        """
        titles = batch.get("title", [""] * len(batch["content"]))
        descs = batch.get("desc", [""] * len(batch["content"]))
        topics = batch.get("topic", [""] * len(batch["content"]))
        contents = batch.get("content", [""] * len(batch["content"]))

        texts = []
        for t, d, tp, c in zip(titles, descs, topics, contents):
            # 你可以按需要调整模板：只用 content，或 title+content，或加入 topic/desc
            # 这里给一个比较通用的拼接方式（更像“网页文本”）
            t = "" if t is None else str(t).strip()
            d = "" if d is None else str(d).strip()
            tp = "" if tp is None else str(tp).strip()
            c = "" if c is None else str(c).strip()

            # 跳过空内容
            if c == "" and t == "":
                texts.append("")
                continue

            # 模板（可改）
            text = f"标题：{t}\n话题：{tp}\n描述：{d}\n内容：{c}"
            texts.append(text)

        return {"text": texts}

    # 先生成 "text" 字段
    ds = ds.map(build_text, batched=True)

    # 过滤空文本（避免后面 tokenize 出空 input_ids）
    ds = ds.filter(lambda x: x["text"] is not None and x["text"].strip() != "")

    # =========================
    # 4) tokenizer / model
    #    这个模型常用 BertTokenizer + GPT2LMHeadModel
    # =========================
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # pad_token 兜底（通常 BertTokenizer 有 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # =========================
    # 5) LoRA 配置（与你前面的 baseline 一致）
    # =========================
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

    # =========================
    # 6) tokenize（非常关键：截断到 1024，避免极长样本触发 GPT-2 上下文上限问题）
    # =========================
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)

    # =========================
    # 7) group_texts：把很多短文本拼接后切成固定 block
    #    语言模型训练的标准做法（提升效率、减少 padding）
    # =========================
    def group_texts(examples):
        # examples["input_ids"] : list[list[int]]
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    train_dataset = lm_ds["train"]
    eval_dataset = lm_ds["validation"]

    # collator：mlm=False 表示 CausalLM（labels 会从 input_ids 拷贝并对 pad 置 -100）
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =========================
    # 8) Trainer 训练参数
    #    为了后续 LN/GELU ablation 公平对比，建议用 max_steps 控制训练量
    # =========================
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=200,          # 先跑通；论文 baseline 可提高到 2000/10000
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Eval loss: {eval_loss:.4f}, Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
