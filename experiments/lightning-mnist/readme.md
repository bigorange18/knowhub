# PyTorch Lightning + MNIST 自编码器

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ⚠️ 能跑，但 `test()` 没整理 |
| 日期 | 2025-04 |
| 环境 | conda `nn`（lightning + torchvision） |
| 入口 | `example.py` |

## 目的

把训练循环里的样板代码（device 搬运、optimizer、日志、checkpoint）全交给 `Trainer`，
看看 Lightning 到底省了什么、又藏了什么。

## 方法

- `model/net.py`：`Encoder` 28×28 → 64 → 3，`Decoder` 3 → 64 → 28×28，组成 `LitAutoEncoder`，
  损失就是重构误差（把 784 维压到 3 维再还原）。
- `example.py`：MNIST + `DataLoader`，`Trainer(max_epochs=10)`，日志走 `TensorBoardLogger(save_dir="./logs/")`。

## 怎么跑

```bash
conda activate nn
cd experiments/lightning-mnist
python example.py
tensorboard --logdir logs
```

## 注意

- MNIST 下载到 `os.getcwd()`，所以要先 `cd` 进这个目录再跑。
- `test()` 里写死了 `lightning_logs/version_0/checkpoints/epoch=59-step=3600000.ckpt`，
  那是以前某次跑的产物，现在肯定找不到 —— 要么改成真实路径，要么先注释掉。
- `net.py` 里还有一个 `LightningTransformer`（WikiText2 的例子），没启用。

## 结果

（没留日志和指标。）

## 待办

- [ ] 把 `test()` 改成从 `Trainer` 最新的 checkpoint 加载
- [ ] 跑完把重构 loss 曲线截个图
