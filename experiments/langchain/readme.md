# LangChain 消息模板 + 千帆 ChatEndpoint

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 可运行 |
| 日期 | 2025-05 |
| 环境 | `pip install langchain langchain-community langchain-openai qianfan python-dotenv` |
| 入口 | `main.py` |

## 目的

搞清 LangChain 的消息抽象：模板变量怎么填、`MessagesPlaceholder` 怎么塞历史对话、
以及怎么把国内模型接进 `ChatModel` 接口。

## 方法

- `create_prompt()`：`ChatPromptTemplate.from_messages([("system", ...), MessagesPlaceholder("msgs")])`，
  演示把 `HumanMessage` 列表塞进占位符。
- `main()`：`QianfanChatEndpoint(streaming=True)` 流式对话，问一句验证链路通。

## 怎么跑

```bash
conda activate nn
python experiments/langchain/main.py
```

## 注意

- 代码里把 `HTTP(S)_PROXY` 写死成 `127.0.0.1:7890`，本机代理是 7897 —— 跑之前改掉或删掉那两行。
- 需要根目录 `.env` 里的千帆 AK/SK。

## 结果

能拿到流式返回的中文回复。

## 结论

LangChain 的模板/消息抽象确实比手拼 prompt 清楚，但换模型时依赖链比较重。
