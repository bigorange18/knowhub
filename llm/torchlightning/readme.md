## Pytorch Linghtning

Torch Lightning 是一个用于简化 PyTorch 代码的库，旨在简化和规范深度学习模型的训练过程。它提供了一组模块和接口，使用户能够更容易地组织和训练模型，同时减少样板代码的数量。PyTorch Lightning的设计目标是提高代码的可读性和可维护性，同时保持灵活性。它通过将训练循环的组件拆分为独立的模块（如模型、优化器、调度器等），以及提供默认实现来简化用户代码。这使得用户可以专注于模型的定义和高级训练策略，而不必处理底层训练循环的细节。

安装PL

```bash
pip install lightning

conda install lightning -c conda-forge
```





### 数据







### 模型





### 训练



trainer参数



|                           |                                                              |      |
| ------------------------- | ------------------------------------------------------------ | ---- |
| accelerator               | 可设置cpu,gpu,tpu以及auto                                    |      |
| default_root_dir          | 设置保存训练log以及checkpoint的目录，默认为当前目录          |      |
| enable_checkpointing      | 开启后会在Trainer的回调函数列表`callbacks`中加入一个`ModelCheckpoint`回调。默认保存的是最新一个epoch的检查点 |      |
| enable_progress_bar       | 显示进度条                                                   |      |
| progress_bar_refresh_rate | 进度条刷新频率                                               |      |
| auto_lr_find              | 自动学习率查找                                               |      |
|                           |                                                              |      |
|                           |                                                              |      |
|                           |                                                              |      |
|                           |                                                              |      |
|                           |                                                              |      |
|                           |                                                              |      |
|                           |                                                              |      |
|                           |                                                              |      |

