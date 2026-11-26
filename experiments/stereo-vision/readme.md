# 双目视觉（Stereo Vision）

| 项目 | 内容 |
| :-- | :-- |
| 状态 | 🔬 规划中 |
| 环境 | 待定（当前 conda `nn` 里没有 opencv，需要先装） |
| 入口 | 待定 |

## 目标

跑通「相机模型 → 极线校正 → 立体匹配 → 深度 / 点云」这条链路，先做出可信的传统方法基线，
再往上叠深度学习匹配网络。每一步都有可量化的产出，而不是"看着差不多"。

文献先读什么、每篇读完能拿到什么，见 **[综述清单](surveys.md)**。

## 路线图

| 阶段 | 内容 | 产出 / 验收 |
| :-- | :-- | :-- |
| 1 相机模型 | 针孔模型、径向/切向畸变、内参 K 与外参 R,t；棋盘格标定 | 标定脚本 + `intrinsics.yaml`，重投影误差 < 0.5 px |
| 2 极线校正 | Bouguet 立体校正，左右图极线水平对齐 | 校正前后对比图；校正后极线行差 ≈ 0 |
| 3 传统匹配 | BM / SGBM 基线，理解代价计算 → 代价聚合 → 视差计算 → 视差优化 | 视差图 + 参数敏感性表 |
| 4 深度与点云 | $Z = f\,B / d$，反投影到 3D，点云滤波与可视化 | 点云文件 + 可视化截图 |
| 5 深度学习匹配 | PSMNet / RAFT-Stereo / IGEV / CREStereo 选一个复现 | 已从 [raft-stereo](../raft-stereo/) 开始：从零实现，Middlebury 5.7 px；SceneFlow 预训练后零样本迁移 Middlebury 5.8 px |
| 6 评测 | 视差误差曲线、坏点率、深度误差 | 汇总表，写进本文件的「结果」 |

## 数据

- **Middlebury**：图像小、有真值，适合验证标定和校正流程。
- **KITTI 2015**：真实车载双目，有 LiDAR 稀疏真值，评测深度最常用。
- **SceneFlow**：合成数据、稠密真值，训深度模型的主力数据集。
- **ETH3D**：高分辨率、多场景，用来测泛化。
- **自采**：两个同型号相机刚性固定 + 棋盘格标定板。

## 环境

```bash
conda activate nn
pip install opencv-python-headless matplotlib
# 需要点云时再装：pip install open3d
```

> 本机 pip 走代理经常失败（ProxyError）。装不上时先用 `--proxy http://127.0.0.1:7897`，
> 或者直接在 `nn` 环境里用 conda 装 `conda install -c conda-forge opencv`。

## 待确认

1. 现在具体在做哪一块：标定 / 校正、传统匹配（SGBM），还是深度学习匹配网络（复现某篇论文）？
2. 数据从哪来：学校实验室的双目相机自采，还是先用 KITTI / SceneFlow 公共数据集？
3. 目标是课程作业 / 论文复现 / 自己攒一个小系统？deadline 是什么时候？
4. 硬件：GPU 型号和显存、相机型号（焦距、基线）。

## 结果

| 阶段 | 结果 |
| :-- | :-- |
| 5 深度匹配 | [raft-stereo](../raft-stereo/) 从零实现：Middlebury 验证 EPE 5.7 px；SceneFlow 预训练零样本迁移到 Middlebury 5.8 px（0.84 M 参数） |

（其余跑完再填。）

## 待办

- [ ] 确定方向与数据集
- [ ] 装 opencv，跑通标定脚本
- [ ] 传统匹配基线（SGBM）跑出第一张视差图
