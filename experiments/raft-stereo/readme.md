# RAFT-Stereo 学习与最小复现

| 项目 | 内容 |
| :-- | :-- |
| 状态 | ✅ 跑通（合成数据过拟合验证） |
| 日期 | 2026-11-14 |
| 环境 | conda `nn`（torch 2.7.0 + cu126，RTX 4070） |
| 入口 | `train_overfit.py` |
| 论文 | Lahav Lipson, Zachary Teed, Jia Deng. *RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching*. 3DV 2021（Best Student Paper）· [arXiv:2109.07547](https://arxiv.org/abs/2109.07547) |
| 官方代码 | [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) |

## 目的

不调用官方权重，从零写一个能跑通的 RAFT-Stereo，确认自己真的理解了它的四个部件——
**特征网络、相关体、GRU 迭代更新、凸上采样**——并用一对已知真值的合成图验证：
网络能不能靠迭代把视差推到亚像素精度。

## 一、论文读什么

RAFT-Stereo 把光流网络 [RAFT](https://arxiv.org/abs/2003.12039) 搬到双目。迁移的关键差别只有一句：
**光流的位移是二维的，而校正后的双目视差只沿 x 方向**，所以相关体只需要在水平方向搜，
代价从 $O(W^2)$ 降到 $O(D)$，同样的算力就能做分辨率更高、范围更宽的匹配。

四个部件（对应论文 Fig. 2）：

**1. 特征网络（feature / context net）**
左右图各过一个共享权重的编码器，得到 1/4 分辨率特征 $f_1, f_2$；左图再单独过 context net 得到 GRU 的初始隐状态 $h_0$。
用 InstanceNorm 而不是 BatchNorm——训练时 batch 很小，BN 的统计量不稳。

**2. 相关体（correlation volume）**
只沿视差方向做全对相关：

$$
C[i, d, y, x] = \frac{1}{\sqrt{C}} \sum_{c} f_1[c, y, x] \cdot f_2[c, y, x - d], \quad d \in [-D, D]
$$

形状 $[2D+1, H, W]$。为了让感受野更大，论文再沿视差轴做平均池化得到几级金字塔（每级范围翻倍、步长翻倍），
查表时用双线性插值取连续视差处的值——这样网络能"看到"比当前视差范围更远的候选。

**3. 迭代更新（核心，也是论文的主要贡献）**
每次迭代：

$$
h_{k+1} = \mathrm{GRU}\big(h_k,\ \text{Encoder}(C[\cdot, d_k + \Delta])\big), \qquad
d_{k+1} = d_k + \sum_{i} p_i \cdot o_i
$$

- 只在当前视差 $d_k$ 附近取 9 个点（$d_k - 4 \ldots d_k + 4$）的相关特征，而不是全网搜；
- GRU 是**卷积 GRU**（`ConvGRU`），隐状态保留空间结构，信息能在邻域间传播；
- 输出 $2\times4+1=9$ 个 softmax 权重 $p_i$，对视差偏移 $o_i \in \{-4,\ldots,4\}$ 做**凸组合**，得到连续（亚像素）增量 $\Delta d$。

论文的贡献是 **multi-level GRU**：把 GRU 拆成几级、不同级在不同分辨率 / 不同视差范围上更新，
让大范围运动和细节能同时被修正，比单级 GRU 收敛更快、精度更高。

**4. 凸上采样（convex upsampling）**
1/4 分辨率的视差用左图特征预测一组 3×3 权重，对粗视差邻域做加权求和放大回原图。
比双线性插值锐利得多，能保住物体边缘。

**损失**：每次迭代都监督，$L = \sum_{k} \gamma^{K-k} \| d_k - d^{gt} \|_1$，$\gamma = 0.9$（越晚的迭代权重越大）。

**论文报告的结果**：Middlebury 榜首，1px 误差比当时第二名好 29%；ETH3D two-view 上优于所有已发表方法；
realtime 版本在 1080p 上可实时。

## 二、我的实现

代码三个文件，没有依赖官方仓库：

| 文件 | 内容 |
| :-- | :-- |
| `model/raft_stereo.py` | `FeatureEncoder`、`SepConvGRU`、`corr_volume` / `sample_corr` / `CorrPyramid`、`UpdateBlock`、`ConvexUpsampler`、`RAFTStereo`、`sequence_loss` |
| `synthetic.py` | 造一对有真值的合成双目图（随机纹理 + 斜面 + 两个方块，按视差反向 warp） |
| `train_overfit.py` | 在单对图上过拟合 300 步，打印 EPE / bad-pixel，画 `figures/overfit.png` |

和官方实现的差距（有意为之，为了单卡小图能跑通、代码一眼看得懂）：

| 组件 | 论文 / 官方 | 这里 |
| :-- | :-- | :-- |
| 特征通道 | 128，6 个残差块 | 64，每级 2 个残差块 |
| GRU | multi-level（多级、多分辨率） | 单级单分辨率（`UpdateBlock` 里一层 `SepConvGRU`） |
| 相关体 | 用 `grid_sample` 的 `alt` 实现，可切 `reg_cuda` | Python 循环逐 $d$ 移位相乘，$O(D)$ 次访存，小图够用 |
| 上采样 | 8× 凸上采样 + 1/4 分辨率输出 | 4× 凸上采样 |
| 附加特征 | motion features、context 分层 | 无 |

`CorrPyramid` 沿视差轴做 `avg_pool3d(kernel=(4,1,1))` 得到三级，每级取 ±4 共 9 个点，
所以送进 GRU 的相关特征是 $3 \times 9 = 27$ 通道。

## 三、怎么跑

```bash
conda activate nn
cd experiments/raft-stereo

python train_overfit.py --steps 300     # 约 1 分钟（4070）
```

只依赖 `torch` / `numpy` / `matplotlib`，不需要 opencv 和数据集。

## 四、结果（300 步，单对 192×384 合成图）

| 指标 | 数值 |
| :-- | :-- |
| 参数量 | 0.84 M |
| 训练耗时 | 300 步 / 66 s（≈0.22 s/step） |
| loss | 9.69 → 0.90 |
| 最终 EPE | **0.07 px** |
| bad-1px / bad-3px | **0.0% / 0.0%** |

逐次迭代的 EPE（验证"迭代确实在收敛"）：

```
10.2  6.4  3.9  2.5  1.8  1.4  0.9  0.5  0.1  0.1  0.1  0.1
```

![过拟合结果](figures/overfit.png)

30 步时的中间结果用来对照：EPE 2.68 px、bad-1px 67.5%——说明**模型先学"大概对"，再靠迭代磨精度**。
30 步时最后几轮迭代的 EPE 还会轻微回升（2.3 → 2.4 → 2.7），300 步后变成单调收敛，
说明这种"迭代越多次越好"的性质是**训出来的**，不是结构天然保证的。

## 五、结论

1. 四个部件都验证过了：相关体只沿视差轴搜、GRU 在隐状态里传播空间信息、softmax 凸组合给出亚像素增量、凸上采样保住边缘。
2. 在合成数据上能过拟合到 0.07 px，说明**前向 + 反向 + 序列损失**这条链路是对的；真实数据上精度取决于特征网络和训练数据。
3. 合成图的右侧边缘误差最大——那是遮挡区（右图该像素在左图里看不到），`valid` 掩掉了，真实评测时这类区域本来也不参与打分。

## 六、下一步

- [ ] 下载官方权重跑真实图片，和本实现对比（`bash download_models.sh` 走 Google Drive，需要能过墙）
- [ ] 在 KITTI 2015 / SceneFlow 上算 EPE 与 bad-pixel，看和论文的差距
- [ ] 把 `UpdateBlock` 扩成 multi-level GRU（论文的主要贡献点）
- [ ] 把相关体换成 `grid_sample` 版，测速对比
- [ ] 视差转深度：$Z = \dfrac{f \cdot B}{d - (c_{x1} - c_{x0})}$，$f$ 用像素单位

## 七、参考

- 论文：<https://arxiv.org/abs/2109.07547>
- 官方代码：<https://github.com/princeton-vl/RAFT-Stereo>
- 光流前作 RAFT：<https://arxiv.org/abs/2003.12039>
- 官方代码结构：`demo.py`（推理）、`train_stereo.py`（训练）、`core/`（`raft_stereo.py` 主模型、`corr.py` 相关体、`update.py` GRU + 凸上采样、`extractor.py` 特征网络）
