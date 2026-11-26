# 双目立体视觉 · 综述清单

> 这张表是给本目录（以及 [raft-stereo](../raft-stereo/)）配的阅读地图：从哪篇开始、
> 每篇读完能拿到什么。标题、年份、出处都核对过（Crossref / arXiv），链接可直接打开。
>
> 一句话结论：**先读 2025 年那篇 IJCV 的《Twenties》，再回头补 2002 年那篇分类学的源头。**

## 一、先读这四篇（按顺序）

| # | 论文 | 出处 | 链接 | 读完能拿到什么 |
| :-: | :-- | :-- | :-- | :-- |
| 1 | Tosi, Bartolomei, Poggi, **A Survey on Deep Stereo Matching in the Twenties** | IJCV 133:4245–4276 (2025) | [arXiv:2407.07816](https://arxiv.org/abs/2407.07816) · [DOI](https://doi.org/10.1007/s11263-024-02331-0) | 2020 年代这一轮的全景：代价体 / 迭代细化 / Transformer / 基础模型这几条线怎么演化，以及领域当前公认的难点分类（域偏移、置信度、实时性、大视差）。配套的 [Awesome-Deep-Stereo-Matching](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching) 是一直在更新的论文索引 |
| 2 | Scharstein, Szeliski, **A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms** | IJCV 47:7–42 (2002) | [Middlebury 上的 PDF](https://vision.middlebury.edu/stereo/taxonomy-IJCV.pdf) | 术语的源头：匹配代价 → 代价聚合 → 视差计算/优化 → 后处理这四步，`local / global / semi-global` 的说法都从这里来；也是 Middlebury 基准的起点。读综述遇到不认识的词，回来查它 |
| 3 | Poggi, Tosi, Batsos, Mordohai, Mattoccia, **On the Synergies between Machine Learning and Binocular Stereo for Depth Estimation from Images: a Survey** | TPAMI (2021) | [arXiv:2004.08566](https://arxiv.org/abs/2004.08566) | 传统几何与深度学习"谁帮谁"的双向关系：立体匹配喂给自监督单目深度，单目先验又回头帮立体匹配处理无纹理/遮挡 |
| 4 | Laga, Jospin, Boussaid, Bennamoun, **A Survey on Deep Learning Techniques for Stereo-based Depth Estimation** | TPAMI (2020/2022) | [arXiv:2006.02535](https://arxiv.org/abs/2006.02535) | 2014–2019 这 150+ 篇的整理：常见 pipeline、损失、数据集，以及"哪些设计其实没用"的讨论。比第 1 篇旧，但把基础概念铺得更平 |

## 二、再补两篇经典（深度学习之前）

| 论文 | 出处 | 链接 | 为什么还要读 |
| :-- | :-- | :-- | :-- |
| Brown, Burschka, Hager, **Advances in Computational Stereo** | TPAMI 25(8):993–1008 (2003) | [DOI](https://doi.org/10.1109/TPAMI.2003.1217603) | 深度学习之前最完整的一篇：匹配代价、聚合、优化、遮挡处理、实时实现。今天论文里的"经典方法基线"基本都能在这找到出处 |
| Zhang, Su, Liu, Sun, **Review of stereo matching based on deep learning** | Displays 87:102940 (2025) | [DOI](https://doi.org/10.1016/j.displa.2024.102940) | 2025 年的英文综述，作为第 1 篇的补充；Elsevier 需要机构权限，先记 DOI |

## 三、中文综述（写开题 / 答辩用得上）

| 论文 | 出处 | 链接 | 覆盖 |
| :-- | :-- | :-- | :-- |
| 《双目视觉的匹配算法综述》 | 图学学报 2020 年第 5 期 | [知网](https://www.cnki.com.cn/Article/CJFDTOTAL-GCTX202005002.htm) | 传统匹配算法（区域 / 特征 / 相位）的分类与比较，术语的中文对照比较全 |
| 《基于深度学习的双目立体匹配研究进展与分析》 | 南昌航空大学学报（自然科学版）2025 年第 1 期 | [知网](https://www.cnki.com.cn/Article/CJFDTOTAL-NCHK202501001.htm) | 深度学习方向的中文进展梳理，适合快速对上国内的做法和数据集 |

> 中文综述更适合"交代背景"，技术细节仍以英文那四篇为准。知网需要校园网或机构账号。

## 四、专题综述（按需再读）

| 方向 | 论文 | 出处 | 什么时候需要 |
| :-- | :-- | :-- | :-- |
| 多视图立体（MVS） | Wang et al., **Learning-based Multi-View Stereo: A Survey** | TPAMI (2026) · [arXiv:2408.15235](https://arxiv.org/abs/2408.15235) | 从两帧双目扩到多视角重建时；里面把 depth-map / voxel / NeRF / 3DGS 几条线分得很清楚 |
| 事件相机立体 | Ghosh, Gallego, **Event-based Stereo Depth Estimation: A Survey** | TPAMI (2025) · [arXiv:2409.17680](https://arxiv.org/abs/2409.17680) | 要做高速运动、高动态范围场景（事件相机）时 |
| 几何约束 | Vats, Crandall, **Geometric Constraints in Deep Learning Frameworks: A Survey** | ACM CSUR (2025) · [arXiv:2403.12431](https://arxiv.org/abs/2403.12431) | 想把极线约束、平面扫描这类几何先验显式塞进网络时 |

## 五、持续跟踪

- [Awesome-Deep-Stereo-Matching](https://github.com/fabiotosi92/Awesome-Deep-Stereo-Matching)：上面第 1 篇作者维护，按任务/年份分类，比翻 arXiv 高效。
- 三个榜单：Middlebury（小场景、亚像素精度）、[KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)（车载）、[ETH3D](https://www.eth3d.net/)（高分辨率、多场景）。
- CVPR 2024 的 tutorial「Deep Stereo Matching in the Twenties」有录像/讲义：<https://sites.google.com/view/stereo-twenties>。

## 六、对应到本仓库已有的工作

读过 [raft-stereo](../raft-stereo/) 的 readme 再看这些综述，可以对上号：

- 我们的 `corr_volume` + `sample_corr` 是**代价体 + 沿视差轴取特征**这条线（第 1 篇里的 cost-volume 家族）。
- `SepConvGRU` + `UpdateBlock` 是**迭代细化**这条线；论文的原始出处是 RAFT-Stereo（3DV 2021，[arXiv:2109.07547](https://arxiv.org/abs/2109.07547)），不在综述清单里但要一起读。
- 现在卡住的"视差 > 192 px 追不上"，正是第 1 篇里 **hierarchical / multi-level / cascade** 那一类方法要解决的问题——也是我们下一步要做的 multi-level GRU。
- 1/4 分辨率的 cost volume 导致大视差被压缩成 80 个格子（max_disp 320 / 4），这是"分辨率 vs 显存"的经典取舍，第 4 篇里讨论得比较多。

## 七、怎么读（省时间的顺序）

1. 花 40 分钟读第 1 篇的**分类图和挑战那一节**，不要逐段读全篇；
2. 顺着分类图，把它引用的 5–8 篇代表作下下来（RAFT-Stereo、PSMNet、IGEV、CREStereo、StereoNet 之类），只看方法和实验表；
3. 遇到术语不清楚，回第 2 篇查定义；
4. 写文档时用第 3、4 篇的中文/英文表述对齐。
