"""最小可跑的 RAFT-Stereo 复现（学习用，不是官方实现）。

论文：Lahav Lipson, Zachary Teed, Jia Deng.
      RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching, 3DV 2021.
      https://arxiv.org/abs/2109.07547

整体流程（对应论文 Fig.2）：
    左右图 --feature net--> 1/4 分辨率特征 f1, f2
    左图   --context net--> 1/4 分辨率上下文特征 --> 初始化 GRU 隐状态 h
    全对相关体 corr[i, d] = <f1[i], f2[i-d]>  --池化--> 多级相关金字塔
    迭代 iters 次：
        (1) 在当前视差 d 附近做 9 点一维采样，得到相关特征
        (2) GRU 更新隐状态
        (3) 输出 softmax 权重，对视差偏移 [-4..4] 做凸组合 --> d = d + Δd
    最后把 1/4 分辨率的视差凸上采样回原图分辨率。

与官方实现的差异（为了单卡/小图能跑通，且代码一眼能看懂）：
    * 特征通道 128 -> 64，残差块数量减少
    * 只保留一层 GRU（论文的 multi-level GRU；README 里写了怎么扩成两层）
    * 迭代表达式相同：d_{k+1} = d_k + Σ_i p_i * offset_i
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """标准残差块：conv3x3 -> InstanceNorm -> ReLU -> conv3x3 -> InstanceNorm -> +x"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        idt = x if self.downsample is None else self.downsample(x)
        return self.relu(out + idt)


class FeatureEncoder(nn.Module):
    """输入原图，输出 1/4 分辨率的特征图（两次 stride=2 下采样）。"""

    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, 2, 3, bias=False)
        self.norm1 = nn.InstanceNorm2d(32, affine=True)
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
        self.layer3 = ResidualBlock(64, 64)
        self.layer4 = ResidualBlock(64, 64)
        self.conv3 = nn.Conv2d(64, out_ch, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))   # 1/2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.relu(self.norm2(self.conv2(x)))   # 1/4
        x = self.layer3(x)
        x = self.layer4(x)
        return self.conv3(x)


class SepConvGRU(nn.Module):
    """卷积 GRU：把空间结构保留在隐状态里（RAFT 的做法，不是 nn.GRU）。"""

    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


# --------------------------------------------------------------------------- #
# 相关体与金字塔
# --------------------------------------------------------------------------- #
def corr_volume(fmap1: torch.Tensor, fmap2: torch.Tensor, max_disp: int) -> torch.Tensor:
    """一维全对相关：corr[b, d, y, x] = <f1[y,x], f2[y,x-d]> / sqrt(C)。

    返回 [B, 2*max_disp+1, H, W]，维度 1 的下标 i 对应视差 d = i - max_disp。
    """
    b, c, h, w = fmap1.shape
    f2 = F.pad(fmap2, (max_disp, max_disp, 0, 0), mode="replicate")
    outs = []
    for d in range(-max_disp, max_disp + 1):
        shifted = f2[:, :, :, max_disp - d: max_disp - d + w]
        outs.append((fmap1 * shifted).sum(dim=1) / (c ** 0.5))
    return torch.stack(outs, dim=1)


def sample_corr(corr: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """沿视差轴做双线性插值。corr: [B,D,H,W]，pos: [B,1,H,W]（下标，不是像素）。"""
    b, d, h, w = corr.shape
    table = corr.permute(0, 2, 3, 1).contiguous().reshape(b * h * w, d)
    p = pos.reshape(-1, 1)
    p0 = torch.floor(p)
    w0 = (p - p0).clamp(0.0, 1.0)
    p0 = p0.clamp(0, d - 1).long()
    p1 = (p0 + 1).clamp(0, d - 1)
    v0 = torch.gather(table, 1, p0)
    v1 = torch.gather(table, 1, p1)
    val = (1.0 - w0) * v0 + w0 * v1
    return val.reshape(b, 1, h, w)


class CorrPyramid:
    """对相关体沿视差轴做平均池化，得到 3 级"金字塔"（论文里用来扩大感受野）。"""

    def __init__(self, corr: torch.Tensor, levels: int = 3, radius: int = 4):
        self.radius = radius
        self.pyramid = [corr]
        for _ in range(levels - 1):
            corr = F.avg_pool3d(
                corr.unsqueeze(1), kernel_size=(4, 1, 1), stride=(4, 1, 1)
            ).squeeze(1)
            self.pyramid.append(corr)
        self.channels = levels * (2 * radius + 1)

    def lookup(self, disp: torch.Tensor) -> torch.Tensor:
        """disp: [B,1,h,w]，单位为 1/4 分辨率像素。返回 3 级 × 9 点的相关特征。"""
        feats = []
        for lvl, corr in enumerate(self.pyramid):
            step = 4 ** lvl
            d = corr.shape[1]
            base = disp / step + (d - 1) / 2.0
            for off in range(-self.radius, self.radius + 1):
                feats.append(sample_corr(corr, base + off))
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# 迭代更新 + 凸上采样
# --------------------------------------------------------------------------- #
class UpdateBlock(nn.Module):
    """一次迭代：相关特征 -> GRU -> softmax 权重 -> 视差增量。"""

    def __init__(self, hidden_dim: int = 64, corr_channels: int = 27, max_offset: int = 4):
        super().__init__()
        self.max_offset = max_offset
        self.encoder = nn.Sequential(
            nn.Conv2d(corr_channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, 3, padding=1),
        )
        self.gru = SepConvGRU(hidden_dim, 96)
        self.conv_delta = nn.Conv2d(hidden_dim, 2 * max_offset + 1, 3, padding=1)
        self.register_buffer(
            "offsets", torch.arange(-max_offset, max_offset + 1).float().view(1, -1, 1, 1)
        )

    def forward(self, h, corr_feat, disp):
        h = self.gru(h, self.encoder(corr_feat))
        logits = self.conv_delta(h)
        prob = torch.softmax(logits, dim=1)
        delta = (prob * self.offsets).sum(dim=1, keepdim=True)
        return h, disp + delta


class ConvexUpsampler(nn.Module):
    """论文的凸上采样：用左图特征预测 3×3 的权重，对粗视差做加权（比双线性更锐利）。"""

    def __init__(self, feat_channels: int = 64, hidden: int = 32):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(feat_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 9, 1),
        )

    def forward(self, disp: torch.Tensor, feat: torch.Tensor, scale: int = 4) -> torch.Tensor:
        """disp 是 1/4 分辨率上的视差（低分辨率像素），返回原分辨率的视差（像素）。"""
        b, _, h, w = disp.shape
        mask = self.mask(feat)                                  # [B,9,h,w]
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False)
        mask = torch.softmax(mask, dim=1)                       # [B,9,H,W]

        # 3×3 邻域的粗视差：unfold + 最近邻放大到原图分辨率
        padded = F.pad(disp, (1, 1, 1, 1), mode="replicate")
        neigh = F.unfold(padded, kernel_size=3)                 # [B,9,h*w]
        neigh = neigh.view(b, 9, h, w)
        neigh = F.interpolate(neigh, scale_factor=scale, mode="nearest")
        # 低分辨率像素 → 原图像素要乘 scale（和 RAFT 里 flow 上采样一致）
        return (mask * neigh).sum(dim=1, keepdim=True) * scale


# --------------------------------------------------------------------------- #
# 完整模型
# --------------------------------------------------------------------------- #
class RAFTStereo(nn.Module):
    def __init__(self, hidden_dim: int = 64, max_disp: int = 64, iters: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_disp = max_disp
        self.iters = iters
        self.fnet = FeatureEncoder(64)
        self.cnet = FeatureEncoder(hidden_dim)
        self.update = UpdateBlock(hidden_dim, corr_channels=3 * 9)
        self.upsampler = ConvexUpsampler(64)

    def forward(self, left: torch.Tensor, right: torch.Tensor, iters: int | None = None,
                return_low_res: bool = False):
        iters = iters or self.iters
        fmap1 = self.fnet(left)
        fmap2 = self.fnet(right)
        corr = corr_volume(fmap1, fmap2, self.max_disp)
        pyramid = CorrPyramid(corr, levels=3, radius=4)

        h = torch.tanh(self.cnet(left))
        disp = torch.zeros_like(h[:, :1])
        outs = []
        for _ in range(iters):
            feat = pyramid.lookup(disp)
            h, disp = self.update(h, feat, disp)
            outs.append(disp)

        if return_low_res:
            return outs
        return [self.upsampler(d, fmap1) for d in outs]


# --------------------------------------------------------------------------- #
# 损失：序列损失（论文式 (1)，γ 后期权重更高）
# --------------------------------------------------------------------------- #
def sequence_loss(preds, gt, valid=None, gamma: float = 0.9):
    n = len(preds)
    loss = 0.0
    for i, pred in enumerate(preds):
        w = gamma ** (n - i - 1)
        diff = (pred - gt).abs()
        if valid is not None:
            diff = diff * valid
            loss = loss + w * diff.sum() / valid.sum().clamp(min=1)
        else:
            loss = loss + w * diff.mean()
    return loss / n
