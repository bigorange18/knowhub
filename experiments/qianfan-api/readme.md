# 百度千帆（文心一言）API 调用

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 可运行 |
| 日期 | 2025-05 |
| 环境 | `pip install qianfan python-dotenv requests` |
| 入口 | `test.py` |

## 目的

把国内大模型 API 的接法摸清楚：官方 SDK 怎么用、底层 HTTP 到底长什么样、embedding 接口怎么调。

## 方法

`test.py` 里是三段：

1. **SDK 对话**：`qianfan.ChatCompletion().do(...)` 调 ERNIE-Bot，构造 `messages`，带一个 system 人设。
2. **原生 HTTP embedding**：先用 `aip.baidubce.com/oauth/2.0/token` 拿 access_token，
   再打 `embeddings/embedding-v1` 接口 —— 这段是手写 `requests`，用来对照 SDK 背后发的请求。
3. AK/SK 全部从环境变量读，不硬编码。

## 怎么跑

在仓库根目录的 `.env` 里填 `QIANFAN_AK` / `QIANFAN_SK`（模板见 `.env.example`）：

```bash
conda activate nn
python experiments/qianfan-api/test.py
```

## 注意

- `test.py` **在模块层就执行了 embedding 请求**（文件末尾那两行），所以一跑就会消耗一次调用。
- `load_dotenv()` 从当前工作目录向上找 `.env`，在仓库根目录执行最稳。
- 千帆的 `temperature` 不能设成 0，最小 0.01。

## 结果

能拿到 ERNIE-Bot 的回复和 embedding 向量。

## 结论

SDK 和原生 HTTP 两条路都通了；密钥走环境变量，不入库。
