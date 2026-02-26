# GPT2_Crypt（手写 GPT2 + HE 近似 + 分布式训练）

这个版本做了重构，目标：

1. **手动实现 GPT2 结构**（便于你继续替换任意非线性模块）。
2. 保留并接入你已有的 **HE 近似模块**（`HEGELU`、`HELayerNorm`）。
3. 提供可直接用于 **8 卡 4090** 的 DDP 训练脚本（`torchrun`）。

---

## 目录说明

- `src/manual_gpt2_he.py`：手写 GPT2 的核心结构（Attention / MLP / Block / LMHead）。
- `train_manual_gpt2_ddp.py`：分布式训练入口脚本。
- `approx/he_approx.py`：你原有的 GELU 与 LayerNorm 近似实现。

---

## 环境依赖

建议 Python 3.10+，并安装：

```bash
pip install torch transformers datasets
```

---

## 训练方式

### 单卡快速验证

```bash
python train_manual_gpt2_ddp.py \
  --model_name uer/gpt2-distil-chinese-cluecorpussmall \
  --output_dir ./outputs/manual_he_ddp \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --use_he_gelu \
  --use_he_ln
```

### 8 卡 4090 分布式训练

```bash
torchrun --nproc_per_node=8 train_manual_gpt2_ddp.py \
  --model_name uer/gpt2-distil-chinese-cluecorpussmall \
  --output_dir ./outputs/manual_he_ddp \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 8 \
  --block_size 256 \
  --max_samples 20000 \
  --use_he_gelu \
  --use_he_ln
```

---

## 数据集说明

脚本会优先尝试加载 CLUE（`clue/tnews`），失败时自动回退到 `wikitext-2-raw-v1`。

并做了基础清洗：

1. 去除首尾空白。
2. 压缩连续空格。
3. 删除过短文本（长度 < 5）。

你可以把数据加载函数替换成你自己的清洗版 CLUE 管线。

---

## 说明

为了方便你继续做 HE 改造，注释全部使用中文，并尽量保留 GPT2 原命名（如 `ln_1`、`c_attn`、`c_fc`）。
这样你后续可以更容易地：

- 继续替换更多非线性模块。
- 观察替换前后 loss 与吞吐差异。
- 迁移到同态加密后端时，定位代价最高的算子。
