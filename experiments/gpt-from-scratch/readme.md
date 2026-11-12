# 从零实现 GPT-2 + SMS 垃圾短信分类

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 可训练（指标没记录） |
| 日期 | 2025-06 |
| 环境 | conda `nn`（torch 2.7.0 + cu126） |
| 入口 | `main.py` |

## 目的

不调 transformers，自己把 GPT 的每一层写出来，再拿一个小数据集跑通完整的训练循环，
确认「注意力 + 残差 + LayerNorm + loss 反传」这一套真的能收敛。

## 方法

- **模型**：`model/gpt.py` 的 `GPTModel`，按 GPT-2 small（124M）配置搭：
  vocab 50257、context 1024、emb 768、12 层、12 头、dropout 0.1。
  `model/modelv1.py` 里是 Bigram 基线（默认用它，先把流程跑通）。
- **任务**：SMS Spam Collection 二分类，取序列最后一个 token 的 logits 算 cross entropy。
- **训练**：AdamW(lr=5e-5, weight_decay=0.1)，10 个 epoch，每 50 步记一次 train / val loss。
- **LoRA**：`utils/lora.py` 是低秩旁路注入，用来对比全量微调和参数高效微调。

## 怎么跑

```bash
conda activate nn
python experiments/gpt-from-scratch/main.py
```

首次运行会从 UCI 下载数据到 `dataset/SMS/`；训练完权重存 `weight/model_10.pth`（约 670 MB，不入库）。

## 注意

- `main.py:125` 默认用 `BigramLanguageModel`，想跑 GPT 就把 125 / 126 两行的注释对调。
- 权重路径已改成相对脚本目录，在哪个目录执行都一样。
- 训练结束只存权重，没把准确率落盘 —— 要留实验记录就自己填下面的表。

## 结果

| 模型 | train acc | val acc |
| :-- | :-- | :-- |
| BigramLanguageModel |  |  |
| GPTModel（124M 配置） |  |  |

## 结论

（跑完补一句话。）

## 待办

- [ ] 跑 GPTModel 那版，把 train / val accuracy 记下来
- [ ] 加一组 LoRA 对比实验
