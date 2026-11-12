# pytest 单元测试：MLPredictor

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 4 passed |
| 日期 | 2025-05 |
| 环境 | `pip install pytest scikit-learn` |
| 入口 | `test_model.py` |

## 目的

给一个小模型类写单元测试，顺便把 pytest 的用例发现规则、临时文件夹具这些搞明白。

## 方法

- `model.py`：`MLPredictor` 封装 iris 数据集 + `RandomForestClassifier`，
  方法有 `train()` / `predict()` / `save_model()` / `load_model()`。
- `test_model.py`：4 个用例 —— 训练后能预测、没训练时抛异常、保存到临时文件、加载回来结果一致。

## 怎么跑

```bash
python -m pytest experiments/pytest-mlp -v
```

## 注意

- 文件名必须是 `test_*.py` 才会被收集。
- 用例里用的是 `tmpdir` 而不是 `tmp_path`（本机 pytest 版本较老）。
- 测试文件会把自身目录插到 `sys.path`，所以在仓库根目录直接跑也行。
