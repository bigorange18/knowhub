# 数据准备

`prepare.py` 把公开双目数据集转成训练用的 `.npz`，`stereo_dataset.py` 负责随机裁剪和增强。
数据本身不放进 git（`.gitignore` 里已忽略 `*.npz`、`*.zip`），只提交代码和本文档。

## 为什么选这两个

| 数据集 | 场景 | GT 来源 | 分辨率 | 特点 |
| :-- | :-- | :-- | :-- | :-- |
| Middlebury 2014 | 室内静物，10 个训练场景 | 结构光 + 亚像素优化 | 2880×1988 | 亚像素精度，和 RAFT-Stereo 论文的评测集一致 |
| KITTI 2015 | 车载街景，200 对 | 激光雷达投影 | 1242×375 | 稀疏 GT、有遮挡、动态物体多 |

先用 Middlebury：它同时具备**稠密 + 亚像素**的真值，最适合验证一个从零写的匹配网络。
KITTI 的 GT 只有约 30% 像素有效，且下载慢（见下），后面再补。

## 下载

下面命令里的 `-x http://127.0.0.1:7897` 是本机代理；如果你的网络不需要代理就去掉。
`-C -` 表示断点续传，中断后重跑同一条命令即可。

### Middlebury 2014

页面：<https://vision.middlebury.edu/stereo/data/scenes2014/>

10 个带真值的训练场景：`Adirondack Jadeplant Motorcycle Piano Pipes Playroom Playtable Recycle Shelves Vintage`。
每份压缩包约 90–100 MB：

```powershell
$scenes = 'Adirondack','Jadeplant','Motorcycle','Piano','Pipes','Playroom','Playtable','Recycle','Shelves','Vintage'
foreach ($s in $scenes) {
    curl.exe -L -C - --retry 10 -x http://127.0.0.1:7897 `
        -o "F:\datasets\middlebury2014\_download\$s-perfect.zip" `
        "https://vision.middlebury.edu/stereo/data/scenes2014/zip/$s-perfect.zip"
}
```

实测约 3 MB/s，十个场景总共约 900 MB。

### KITTI 2015

官方地址不需要登录：

```powershell
curl.exe -L -C - --retry 10 -x http://127.0.0.1:7897 `
    -o F:\datasets\kitti2015\_download\data_scene_flow.zip `
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
```

文件 1.57 GiB。**实测经代理只有 0.1 MB/s**（直连 0.03 MB/s），要几小时以上，所以这次没下完。

## 转换

```powershell
cd experiments\raft-stereo
python -m data.prepare --src F:\datasets\middlebury2014\_download `
    --out F:\datasets\stereo\prepared\middlebury --dataset middlebury --width 768
```

`--width` 把长边统一到 768（高度按比例，再裁到 8 的倍数，保证能被 1/4 分辨率整除）。
脚本直接从 zip 里读，不需要解压。

输出目录：

```
F:\datasets\stereo\prepared\middlebury\
    Adirondack.npz  Jadeplant.npz  ...
    meta.json                 # 每个场景的分辨率、有效像素比例、焦距、基线
```

10 个场景转换后的实际情况（`--width 768`）：

| 场景 | 尺寸 | GT 有效像素 | 视差范围 | 焦距 (px) | 基线 (mm) |
| :-- | :-- | --: | --: | --: | --: |
| Adirondack | 768×528 | 95.2% | 8.0–64.7 | 1109.7 | 176.3 |
| Jadeplant | 768×576 | 86.1% | 13.2–170.0 | 2134.5 | 380.1 |
| Motorcycle | 768×512 | 86.6% | 7.6–62.1 | 1031.2 | 193.0 |
| Piano | 768×520 | 87.6% | 10.9–56.8 | 769.7 | 178.1 |
| Pipes | 768×496 | 85.6% | 8.9–70.0 | 1029.6 | 236.9 |
| Playroom | 768×520 | 84.5% | 10.9–80.6 | 1105.2 | 342.8 |
| Playtable | 768×520 | 90.3% | 9.3–74.7 | 656.8 | 193.0 |
| Recycle | 768×512 | 93.3% | 10.1–59.1 | 789.8 | 178.2 |
| Shelves | 768×520 | 83.0% | 14.0–52.1 | 1405.1 | 176.5 |
| Vintage | 768×504 | 89.7% | 11.6–167.7 | 1052.3 | 195.2 |

合计 29.5 MB（原始 zip 约 925 MB）。注意 Jadeplant / Vintage 的最近处视差到 170 px，
超过默认的 `--max_disp 128`，那部分像素会被当成无效掩掉；要全用上就把 `max_disp` 调到 176。

## npz 里有什么

| 键 | 形状 | 类型 | 说明 |
| :-- | :-- | :-- | :-- |
| `left` | H×W×3 | uint8 | 左图，RGB |
| `right` | H×W×3 | uint8 | 右图，RGB |
| `disp` | H×W | float32 | 视差（像素），无效处为 0 |
| `valid` | H×W | bool | 真值有效的像素 |
| `focal` | 标量 | float32 | 缩放后的焦距（像素） |
| `baseline` | 标量 | float32 | 基线（mm，Middlebury 原始标定单位） |
| `doffs` | 标量 | float32 | 主点差（像素），缩放后 |

深度换算：`Z = focal * baseline / (disp + doffs)`。`disp + doffs` 必须 > 0。

## 原始格式（为什么要自己写解析）

Middlebury 的压缩包里没有常见图片格式的真值：

- `disp0.pfm`：头三行是 `Pf\n<宽> <高>\n<scale>\n`。`scale` 为负表示小端 float32；
  **行序自下而上**，读进来必须上下翻转。值为 `inf` 的地方没有真值。
- `disp0-n.pgm`：P5 格式的 8 位图，每个像素记录该点参与优化的采样次数，`0` 表示无真值。
  所以有效掩码是 `disp0-n > 0`，不是 `disp > 0`。
- `calib.txt`：`cam0=[fx 0 cx; ...]`、`baseline`（mm）、`doffs`、`ndisp`、`isint`（0 = 亚像素）。

KITTI 2015 的视差是 16 位 PNG，**除以 256** 得到像素单位，`0` 表示无真值。

## 训练

```powershell
python train_real.py --data F:\datasets\stereo\prepared\middlebury --steps 1200
```

`split_scenes()` 按场景切分，正常是最后 25% 的场景做验证（这 10 个场景里是
Shelves、Vintage），避免同一场景的 patch 同时出现在训练和验证里。

## 许可

- Middlebury 2014：仅限研究与教学用途，引用 Middlebury Stereo 数据集论文。
- KITTI 2015：CC BY-NC-SA 3.0，仅限非商业用途。

数据只放本地磁盘，不提交到仓库。
