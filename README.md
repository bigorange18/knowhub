<!-- profile logo 个人资料徽标 -->

<div align="center">
    <a href="https://twitter.com/chenorange2219/"><img src="https://img.shields.io/badge/Twitter-推特-blue" /></a>&emsp;
    <a href="https://www.youtube.com/@chenorange2219"><img src="https://img.shields.io/badge/YouTube-油管-c32136" /></a>&emsp;
    <a href="https://box.sunguoqi.com/weixin_mp"><img src="https://img.shields.io/badge/WeChat-微信-07c160" /></a>&emsp;
    <a href="https://space.bilibili.com/321166054/"><img src="https://img.shields.io/badge/Bilibili-B站-ff69b4" /></a>&emsp;
    <a href="https://blog.csdn.net/royal_killer?spm=1011.2124.3001.5343/"><img src="https://img.shields.io/badge/CSDN-论坛-c32136" /></a>&emsp;
    <a href="https://www.zhihu.com/people//ha-lou-kti-44-92/"><img src="https://img.shields.io/badge/Zhihu-知乎-blue" /></a>&emsp;
    <!-- visitor statistics logo 访问量统计徽标 -->
    <img src="https://komarev.com/ghpvc/?username=bigorange18&label=Views&color=0e75b6&style=flat" alt="访问量统计" />
</div>

# knowhub

实验项目集。每个实验一个目录，自带 `readme.md` 记录**目的 / 方法 / 怎么跑 / 结果 / 结论**，代码都能直接跑。
不放第三方仓库副本，权重和数据集也不进 git。

## 实验

| 实验 | 方向 | 状态 | 一句话结论 |
| :-- | :-- | :-- | :-- |
| [stereo-vision](experiments/stereo-vision/) | 双目视觉：标定 → 校正 → 匹配 → 深度 | 🔬 进行中 | 路线图已列，方向待定 |
| [gpt-from-scratch](experiments/gpt-from-scratch/) | 从零实现 GPT-2（124M 配置）+ SMS 垃圾短信分类 | ✅ 可训练 | 手写注意力与 LoRA；指标还没记录 |
| [qianfan-api](experiments/qianfan-api/) | 百度千帆 / 文心一言 API | ✅ 可运行 | SDK 与原生 HTTP 两种调用方式 |
| [lightning-mnist](experiments/lightning-mnist/) | PyTorch Lightning + MNIST 自编码器 | ⚠️ 待整理 | `test()` 里 ckpt 路径写死 |
| [langchain](experiments/langchain/) | LangChain 消息模板 + 千帆 ChatEndpoint | ✅ 可运行 | 代理地址写死在代码里 |
| [pytest-mlp](experiments/pytest-mlp/) | pytest 单元测试（sklearn 随机森林） | ✅ 4 passed | 覆盖训练 / 预测 / 存取 |
| [pandas-basics](experiments/pandas-basics/) | pandas Series / 多层索引 练习 | 📓 笔记 | 32 个 cell 的练习 |
| [jieba-tokenize](experiments/jieba-tokenize/) | jieba 分词 | ✅ 可运行 | 最小示例 |
| [gradio-demo](experiments/gradio-demo/) | Gradio 最小 Web 界面 | ✅ 可运行 | `launch(share=True)` |

## 笔记

| 笔记 | 内容 |
| :-- | :-- |
| [notes/llm.md](notes/llm.md) | Transformer 结构、LoRA / QLoRA、RLHF、Prompt 工程、大模型面试题 |
| [notes/python-interview.md](notes/python-interview.md) | Python 面试题（第 6 题还没写完） |

## C++

| 路径 | 内容 |
| :-- | :-- |
| [cpp/readme.md](cpp/readme.md) | CMake 速查：常用宏 + 一份完整 CMakeLists 的注释示例 |
| [cpp/webserver/](cpp/webserver/) | C++ 工程骨架（gflags + 多级 CMake），内含文件读写 / 指针 / 模板交换等练习 |
| [cpp/cvtest/](cpp/cvtest/) | C++ 编译环境验证 |

## 目录约定

```
experiments/<名字>/
├── readme.md        # 目的、方法、怎么跑、结果、结论
├── main.py          # 入口
└── requirements.txt
```

- 新实验从 [experiments/_template](experiments/_template/) 复制。
- 权重（`*.pth` / `*.ckpt` / `*.safetensors`）、数据集（`*.zip` / `*.csv`）、CMake 构建产物都不入库，见 [.gitignore](.gitignore)。

## 环境

| 用途 | 环境 |
| :-- | :-- |
| 深度学习（torch 2.7.0 + cu126，CUDA 可用） | conda `nn`（Python 3.11.11） |
| 单元测试 | `python -m pytest experiments/pytest-mlp` |

```bash
conda activate nn
python experiments/gpt-from-scratch/main.py
```

## 使用

```bash
git clone git@github.com:bigorange18/knowhub.git

# 环境变量：复制模板后填自己的 key，.env 已被 gitignore
cp .env.example .env
```
