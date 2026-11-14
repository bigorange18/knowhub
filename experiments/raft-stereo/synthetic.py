"""合成一对"已经校正过"的立体图 + 真值视差，用来验证网络能不能学会。

生成方式：先造一张随机纹理当左图，再按视差图 d(x,y) 做反向 warp 得到右图，
即 right(x) = left(x + d)。这样真值是已知的、不用下数据集。
"""

import torch
import torch.nn.functional as F


def gaussian_kernel(size: int = 9, sigma: float = 3.0) -> torch.Tensor:
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def make_pair(height: int = 192, width: int = 384, seed: int = 0):
    """返回 left, right, disp, valid，形状分别是 [1,3,H,W]、[1,3,H,W]、[1,1,H,W]、[1,1,H,W]。"""
    gen = torch.Generator().manual_seed(seed)

    # 1) 随机纹理 -> 模糊一下，避免纯高频噪声
    tex = torch.rand(1, 1, height, width, generator=gen)
    tex = F.conv2d(F.pad(tex, (4, 4, 4, 4), mode="replicate"), gaussian_kernel())
    tex = (tex - tex.amin()) / (tex.amax() - tex.amin() + 1e-6)
    left = tex.repeat(1, 3, 1, 1)

    # 2) 视差图：地面斜面 + 两个"方块"近景
    yy, xx = torch.meshgrid(
        torch.arange(height).float(), torch.arange(width).float(), indexing="ij"
    )
    disp = 6.0 + 10.0 * (yy / height)
    disp[60:120, 40:170] = 36.0
    disp[128:170, 210:330] = 20.0
    disp = disp.unsqueeze(0).unsqueeze(0)

    # 3) 右图 = 在左图上按 +d 采样
    src_x = xx + disp.squeeze(0).squeeze(0)
    x_norm = 2.0 * src_x / (width - 1) - 1.0
    y_norm = 2.0 * yy / (height - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]
    right = F.grid_sample(left, grid, mode="bilinear", padding_mode="border", align_corners=True)

    valid = ((src_x <= width - 1) & (disp.squeeze(0).squeeze(0) >= 0)).float()
    return left, right, disp, valid.unsqueeze(0).unsqueeze(0)


def colorize_disp(disp: torch.Tensor, vmax: float = 40.0) -> torch.Tensor:
    """把视差图转成 RGB（turbo 风格的简化版），返回 [H,W,3] 的 uint8。"""
    d = disp.squeeze().clamp(0, vmax).float() / vmax
    r = (1.5 - (d - 0.75).abs() * 4).clamp(0, 1)
    g = (1.5 - (d - 0.45).abs() * 4).clamp(0, 1)
    b = (1.5 - (d - 0.15).abs() * 4).clamp(0, 1)
    return (torch.stack([r, g, b], dim=-1) * 255).byte().numpy()
