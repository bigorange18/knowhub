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

技术笔记 + 练手代码。笔记按主题归档，代码都是自己手写的实验项目，不放第三方仓库副本。

## 目录

### llm —— 大模型

| 路径 | 内容 |
| :-- | :-- |
| [readme.md](llm/readme.md) | LLM 学习笔记：Transformer 结构、LoRA / QLoRA / RLHF 等微调方法、Prompt 工程、面试题 |
| [gptv1/](llm/gptv1) | 从零实现 GPT：手写因果多头注意力 + LoRA，在 SMS Spam 数据集上做分类微调 |
| [llm_qianfan/](llm/llm_qianfan) | 百度千帆 API 调用 |
| [langechain_study.py](llm/langechain_study.py) | LangChain 提示词模板与消息占位符 |
| [torchlightning/](llm/torchlightning) | PyTorch Lightning 训练 MNIST |
| [nlp/](llm/nlp) | jieba 分词 |

### cpptest —— C++

| 路径 | 内容 |
| :-- | :-- |
| [readme.md](cpptest/readme.md) | CMake 速查：常用宏 + 一份完整 CMakeLists 的注释示例 |
| [webserver/](cpptest/webserver) | C++ 工程骨架（gflags + 多级 CMake），内含文件读写 / 指针 / 模板交换等练习 |
| [cvtest/](cpptest/cvtest) | C++ 编译环境验证 |

### 其它

| 路径 | 内容 |
| :-- | :-- |
| [pytest/base.ipynb](pytest/base.ipynb) | Python 基础实验 |
| [pytest/example/](pytest/example) | 用 pytest 测一个 sklearn 随机森林小模型 |
| [gradio/base.py](gradio/base.py) | Gradio 最小示例 |
| [other/](other/readme.md) | 杂项 |

## 使用

```bash
# 克隆
git clone https://github.com/bigorange18/knowhub.git
# 国内加速
git clone https://ghp.ci/https://github.com/bigorange18/knowhub.git
```

```bash
# 环境变量：复制模板后填入自己的 key，.env 已被 gitignore
cp .env.example .env
```

> 模型权重（`*.pth` / `*.ckpt` / `*.safetensors`）、数据集（`*.zip` / `*.csv`）、
> CMake 构建产物均不入库，需要时按各目录的 README 自行下载或重新生成。
