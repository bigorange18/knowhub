# 大语言模型LLM



## 数据集



1. 中文问答数据集（如CMRC 2018、DRCD等），用于训练问答系统。
2. 中文情感分析数据集（如ChnSentiCorp、Fudan News等），用于训练情感分类模型。
3. 中文文本相似度数据集（如LCQMC、BQ Corpus等），用于训练句子对匹配和相似度判断任务。
4. 中文摘要生成数据集（如LCSTS、NLPCC等），用于训练文本摘要生成模型。
5. 中文对话数据集（如LCCC、ECDT等），用于训练聊天机器人或对话系统。

## Transformer



1. 对输入的文字或信息进行编码：tokenized &embdding;
2. multi-head attention;





## 模型微调

是指在预训练阶段之后，使用特定任务的有标签数据对模型进行进一步的训练和调整参数，使得模型能够学习到与任务相关的特定特征好的知识。

**全参数微调发展**

| 微调方法 |                 创新点                  |      |
| :------: | :-------------------------------------: | ---- |
|   Mezo   | 零阶随机梯度下降，梯度估计更新模型参数  |      |
|   LOMO   | 融合梯度计算+参数更新对目标函数更新参数 |      |
|    UT    |         叙述数据重引，数据混合          |      |
|   POUF   |                                         |      |





1. 数据准备：收集和准备特定任务的数据集
2. 模型选择：选择一个预训练模型作为基座模型
3. 迁移学习：在新数据集上继续训练模型，同时保留预训练模型的知识
4. 模型参数调整：根据需要调整模型的参数
5. 模型评估：在验证集上评估模型性能，并根据反馈进行调整

指令微调（IT）来增强模型的额能力和可控力。



### LORA微调(Low-Rank Adaption， 低秩自适应)

解决问题：

1. 节省计算资源
2. 在小数据集上面微调大模型时，会出现过拟合情况



矩阵秩表示其线性无关的行或列的数量。研究发现，微调过程中权重的变化ΔW可以用远低于其原始维度的向量组合表示。

在微软发表的原始LoRA论文中，作者通过奇异值分解(SVD)分析了GPT-2和GPT-3等模型微调时ΔW矩阵的奇异值分布，发现有效秩远低于矩阵全秩。

可以表示如下：
$$
W_{ft}=W_{pt}+\Delta W
$$
其中， $$\Delta W=BA$$, $$B\in R^{(r,k)}$$,  $$A\in R^{( d,r)}$$,且 $$r<<min(k,d)$$。为了控制更新的幅度，引入缩放因子$$\alpha$$。
$$
W_{ft}=W_{pt}+\alpha \Delta W  \\

W_{ft}=W_{pt}+\alpha BA    \\
$$
1、初始化

- 矩阵A：通常使用高斯分布N(0, σ²)初始化，其中σ = 1/√r
- 矩阵B：初始化为0，确保训练初始时ΔW = 0





Prefix Tuning

​	在保持预训练参数固定的基础上，通过每个任务引入额外的embedding来实现任务适配。



#### P-Tuning领域模型微调

#### Instruct-Tuning指令微调

#### Loar-QLora微调

#### RLHF基于人类反馈的强化学习微调

实战代码演







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





## Transformer



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



1. each training epoch
2. batch training
3. 上次迭代的梯度重置
4. 计算当前batch的loss
5. 反向传播计算loss梯度
6. 更新loss梯度
7. 打印验证集loss





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

是指在预训练阶段之后，使用特定任务的有标签数据对模型进行进一步的训练和调整参数，使得模型能够学习到与任务相关的特定特征好的知识。

1. 数据准备：收集和准备特定任务的数据集
2. 模型选择：选择一个预训练模型作为基座模型
3. 迁移学习：在新数据集上继续训练模型，同时保留预训练模型的知识
4. 模型参数调整：根据需要调整模型的参数
5. 模型评估：在验证集上评估模型性能，并根据反馈进行调整

指令微调（IT）来增强模型的额能力和可控力。











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
