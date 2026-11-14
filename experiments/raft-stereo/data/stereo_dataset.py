"""双目数据加载：读 prepare.py 生成的 npz，随机裁剪成训练用的 patch。

不用 opencv，全部 numpy + PIL + torch。返回的 disp 与 valid 都是 1/1 分辨率，
网络内部自己下采样。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class StereoPairs(Dataset):
    """一堆 npz，每个含 left / right / disp / valid。

    Args:
        root: prepare.py 的输出目录
        crop: (H, W) 随机裁剪尺寸，None 表示整张
        augment: 随机水平翻转（翻转时左右图交换，视差不变）
        min_valid: 裁剪区域里有效视差像素的最低占比，太低就重裁
        max_disp: 超过这个视差的像素算无效（模型表示不了）
    """

    def __init__(self, root: str | Path | None = None, crop=(192, 384), augment: bool = False,
                 min_valid: float = 0.05, max_disp: float | None = None, files=None,
                 center: bool = False):
        if files is None:
            self.root = Path(root)
            self.files = sorted(self.root.glob("*.npz"))
        else:
            self.root = Path(files[0]).parent
            self.files = [Path(f) for f in files]
        if not self.files:
            raise FileNotFoundError(f"{self.root} 里没有 npz，先跑 data/prepare.py")
        self.crop = crop
        self.augment = augment
        self.min_valid = min_valid
        self.max_disp = max_disp
        self.center = center  # True 时始终取中心裁剪，验证集用它保证每次评测一致
        self._cache: dict[int, dict] = {}

    def __len__(self):
        return len(self.files)

    def _load(self, idx: int) -> dict:
        if idx not in self._cache:
            with np.load(self.files[idx]) as z:
                disp = z["disp"].astype(np.float32)
                disp = np.where(np.isfinite(disp), disp, 0.0)  # 早先版本可能残留 inf
                self._cache[idx] = {
                    "left": z["left"], "right": z["right"],
                    "disp": disp,
                    "valid": z["valid"].astype(bool),
                }
            if len(self._cache) > 8:  # 别把内存吃满
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def __getitem__(self, idx: int):
        s = self._load(idx)
        left, right = s["left"], s["right"]
        disp, valid = s["disp"].copy(), s["valid"].copy()

        if self.max_disp is not None:
            valid &= disp <= self.max_disp

        h, w = disp.shape
        ch, cw = self.crop if self.crop else (h, w)
        ch, cw = min(ch, h), min(cw, w)

        top = left_c = 0
        if (ch, cw) != (h, w):
            if self.center:
                top, left_c = (h - ch) // 2, (w - cw) // 2
            else:
                best = None
                for _ in range(20):
                    t = np.random.randint(0, h - ch + 1)
                    l = np.random.randint(0, w - cw + 1)
                    ratio = valid[t:t + ch, l:l + cw].mean()
                    if ratio >= self.min_valid:
                        top, left_c = t, l
                        break
                    if best is None or ratio > best[0]:
                        best = (ratio, t, l)
                else:
                    top, left_c = best[1], best[2]

        sl = (slice(top, top + ch), slice(left_c, left_c + cw))
        left = left[sl]
        right = right[sl]
        disp = disp[sl]
        valid = valid[sl]

        if self.augment and np.random.rand() < 0.5:
            # 水平翻转 = 左右图互换再翻转，视差方向不变
            left, right = right[:, ::-1], left[:, ::-1]
        if self.augment and np.random.rand() < 0.3:
            gain = np.random.uniform(0.85, 1.15)
            left = np.clip(left.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        left = torch.from_numpy(np.ascontiguousarray(left)).permute(2, 0, 1).float() / 255.0
        right = torch.from_numpy(np.ascontiguousarray(right)).permute(2, 0, 1).float() / 255.0
        disp = torch.from_numpy(np.ascontiguousarray(disp)).unsqueeze(0)
        valid = torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0).float()
        return {"left": left, "right": right, "disparity": disp, "valid": valid}


def split_scenes(root: str | Path, val_ratio: float = 0.25) -> tuple[list, list]:
    """按文件把数据集切成 train / val（同一场景的 patch 不跨集，避免泄漏）。"""
    files = sorted(Path(root).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"{root} 里没有 npz")
    n_val = max(1, int(round(len(files) * val_ratio)))
    if len(files) == 1:  # 只有一张图时两边都用它，仅用于冒烟
        return files, files
    return files[:-n_val], files[-n_val:]
