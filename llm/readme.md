# 编程基础

## 开发与并行编程

异步编程、多线程、多进程，提升程序性能；

## 经典算法

线性回归、逻辑回归、SVM、朴素贝叶斯、随机森林、K-means、PCA降维

## 深度学习框架

pytorch

### 提示工程

提示词工程(Prompt Engineering) 是有技巧的使用提示词，从而最大限度地提高LLM相应的有效性、准确性。设计有效的提示词以指导模型执行期望任务的方法被称为**提示工程**。

1. 编写清晰的说明
2. 提供参考文本
3. 将复杂的任务才分为更为简单的子任务
4. 给模型思考
5. 使用外部工具
6. 系统地测试变化

**Temperature**：简单来说，`temperature` 的参数值越小，模型就会返回越确定的一个结果。如果调高该参数值，大语言模型可能会返回更随机的结果，也就是说这可能会带来更多样化或更具创造性的产出。（调小 `temperature`）实质上，你是在增加其他可能的 token 的权重。在实际应用方面，对于质量保障（QA）等任务，我们可以设置更低的 `temperature` 值，以促使模型基于事实返回更真实和简洁的结果。 对于诗歌生成或其他创造性任务，适度地调高 `temperature` 参数值可能会更好。

**Top_p**：同样，使用 `top_p`（与 `temperature` 一起称为核采样（nucleus sampling）的技术），可以用来控制模型返回结果的确定性。如果你需要准确和事实的答案，就把参数值调低。如果你在寻找更多样化的响应，可以将其值调高点。

**Max Length**：您可以通过调整 `max length` 来控制大模型生成的 token 数。指定 Max Length 有助于防止大模型生成冗长或不相关的响应并控制成本。

**Frequency Penalty**：`frequency penalty` 是对下一个生成的 token 进行惩罚，这个惩罚和 token 在响应和提示中已出现的次数成比例， `frequency penalty` 越高，某个词再次出现的可能性就越小，这个设置通过给 重复数量多的 Token 设置更高的惩罚来减少响应中单词的重复。

#### 框架

角色：明确模型扮演的角色

指令：希望模型做什么

背景/上下文：

格式限制

实例

大模型本质=分词+运算+预测+生成文本

### AGT

##### embedding

语义相似的实体在空间中映射得更近，而不相似的实体映射得更远。

```

```

padding:

    输入的句子长度不一样，为了保存句子长度一致，需要将长句子截断。

## 预训练

Transformer、 BERT、 GPT及其变种



# Transformer



1. 对输入的文字或信息进行编码：tokenized &embdding;
2. multi-head attention;



# 大语言模型LLM

STAGE1:

1. 准备数据&采样
2. Attention mechanism
3. LLM architecture

pretraining

STAGE2:

1. training loop

2. model evaluation

3. load pretrained weights

   



STAGE3:

- classifier
- personal assistant









## 主流大语言模型

LLAMA ChatGLM Qwen、OpenAI等模型

构建网络的经典流程：

1. 定义一个可以学习参数的神经网络；
2. 遍历训练数据集
3. 计算损失；
4. 将网络参数的梯度进行反向传播；
5. 以一定规则更新网络的权重；

文本处理：

文本张量表示方法：

one-hot

word2vec

word Embedding

Liama 3、 chatGLM

### Bert模型

1. One-hot(独热编码)
2. 词袋模型（bag of world model,BOW)
3. TF-IDF(Term Frequency-Inverse Docement)
4. N-Gram

神经网络语言模型

word2vec

### 模型微调

是指在预训练阶段之后，使用特定任务的有标签数据对模型进行进一步的训练和调整参数。

指令微调（IT）来增强模型的额能力和可控力。IT使用（

#### P-Tuning领域模型微调

#### Instruct-Tuning指令微调

#### Loar-QLora微调

#### RLHF基于人类反馈的强化学习微调

实战代码演

## 提升工程（Prompt Engineering)

## 大模型的微调与预训练

LLam-Factory

## 大模型的量化

## 模型的部署

# 面试

### 工作要求

- [X] python
- [ ] Java
- [X] Linux
- [X] docker
- [ ] Mysql/PostgreSQL
- [ ] openAi
- [ ] Embedding优化
- [ ] RAG开发流程，具备检索（Retrieval）与生成（Generation）模块的开发
- [ ] 具备 Prompt 工程经验，能够编写高效、精准的 Prompt 模板，
- [ ] Fine-tuning
- [ ] Langchain
- [ ] Llama
- [ ] Qwen-Agent
- [ ] Hugging Face
- [ ] SFT
- [ ] RLHF

## 技术面

CNN模型中池化层的作用？

Max Pooling是如何反向传递梯度的？

LSTM Transformer GPT

深度学习的三种并行方式：数据并行，模型并行，流水线并行
