# jieba 分词

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 可运行 |
| 日期 | 2025-05 |
| 环境 | `pip install jieba` |
| 入口 | `main.py` |

## 目的

最小可跑的中文分词示例，顺手确认 jieba 的 `cut()` 返回的是生成器（只能遍历一次）。

## 方法

`jieba.cut(text, cut_all=False)` 精确模式，对同一句话打印两种结果：
一种用 `/` 拼起来，一种直接 `"".join()` —— 第二种能看出来生成器已经被第一次遍历消耗完了。

## 怎么跑

```bash
conda activate nn
python experiments/jieba-tokenize/main.py
```
