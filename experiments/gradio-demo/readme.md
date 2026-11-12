# Gradio 最小 Web 界面

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 可运行 |
| 日期 | 2025-01 |
| 环境 | `pip install gradio` |
| 入口 | `main.py` |

## 目的

用最少代码把一个 Python 函数变成能用浏览器访问的界面，看看 `gr.Interface` 的输入输出映射有多直接。

## 方法

`greet(name, intensity)`：输入 text + slider，返回重复名字的字符串，`gr.Interface` 自动生成页面。
`demo.launch(share=True)` 会额外开一条公网临时链接。

## 怎么跑

```bash
conda activate nn
python experiments/gradio-demo/main.py
```

浏览器打开终端打印的 `http://127.0.0.1:7860`。

## 注意

- `share=True` 需要能连外网；只想本地看就改成 `demo.launch()`。
- 文件里的旧代码块（`start()`）是注释掉的早期版本，可以删。
