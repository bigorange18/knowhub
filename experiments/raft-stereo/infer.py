"""用训练好的权重对一整对图像推理，输出视差图（有标定时再输出深度图）。

    python infer.py --npz F:\\datasets\\stereo\\prepared\\middlebury\\Adirondack.npz \\
        --weights weight/real.pth --out figures/adirondack.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from model.raft_stereo import RAFTStereo


def pad_to_multiple(x: torch.Tensor, m: int = 4):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


@torch.no_grad()
def predict(model, left: np.ndarray, right: np.ndarray, dev, iters: int):
    lt = torch.from_numpy(left).permute(2, 0, 1)[None].float().to(dev) / 255.0
    rt = torch.from_numpy(right).permute(2, 0, 1)[None].float().to(dev) / 255.0
    lt, (h, w) = pad_to_multiple(lt)
    rt, _ = pad_to_multiple(rt)
    preds = model(lt, rt, iters=iters)
    return [p[0, 0, :h, :w].cpu().numpy() for p in preds]


def colorize(disp: np.ndarray, valid: np.ndarray | None = None, vmax: float | None = None):
    if vmax is None:
        if valid is not None and valid.any():
            base = disp[valid]
        else:
            base = disp[np.isfinite(disp)]
        vmax = float(np.percentile(base, 99)) if base.size else 1.0
    vmax = max(vmax, 1e-6)
    return plt.cm.magma(np.clip(disp / vmax, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", help="含 left/right/disp/valid/focal/baseline/doffs 的 npz")
    p.add_argument("--left", help="只有图片时用：左图路径")
    p.add_argument("--right")
    p.add_argument("--weights", default="weight/real.pth")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--max_disp", type=int, default=128)
    p.add_argument("--out", default="figures/infer.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location=dev)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = RAFTStereo(hidden_dim=64, max_disp=args.max_disp, iters=args.iters).to(dev)
    model.load_state_dict(state)
    model.eval()

    if args.npz:
        z = np.load(args.npz)
        left, right = z["left"], z["right"]
        gt = z["disp"] if "disp" in z.files else None
        valid = z["valid"] if "valid" in z.files else None
        focal = float(z["focal"]) if "focal" in z.files else None
        baseline = float(z["baseline"]) if "baseline" in z.files else None
        doffs = float(z["doffs"]) if "doffs" in z.files else 0.0
        title = Path(args.npz).stem
    else:
        left = np.array(Image.open(args.left).convert("RGB"))
        right = np.array(Image.open(args.right).convert("RGB"))
        gt = valid = None
        focal = baseline = None
        doffs = 0.0
        title = Path(args.left).stem

    preds = predict(model, left, right, dev, args.iters)
    disp = preds[-1]

    panels = [("左图", left, None), ("预测视差", disp, "disp")]
    if gt is not None:
        err = np.abs(disp - gt)
        m = valid if valid is not None else np.ones_like(gt, bool)
        epe = float((err * m).sum() / max(m.sum(), 1))
        bad1 = float(((err > 1) * m).sum() / max(m.sum(), 1)) * 100
        bad3 = float(((err > 3) * m).sum() / max(m.sum(), 1)) * 100
        print(f"EPE {epe:.3f} px   bad-1px {bad1:.1f}%   bad-3px {bad3:.1f}%")
        panels.append(("真值视差", gt, "disp"))
        panels.append((f"误差 |Δ|（EPE {epe:.2f}px）", err * m, "err"))

    if focal and baseline:
        depth = focal * baseline / np.maximum(disp + doffs, 1e-3)  # mm
        panels.append(("深度 (mm)", depth, "depth"))

    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4.2))
    ax = np.atleast_1d(ax)
    for a, (name, data, kind) in zip(ax, panels):
        if kind == "disp":
            a.imshow(colorize(data, valid))
        elif kind == "err":
            a.imshow(data, cmap="hot", vmax=max(3.0, float(np.percentile(data[data > 0], 95)) if (data > 0).any() else 3.0))
        elif kind == "depth":
            v = data[(data > 0) & np.isfinite(data)]
            a.imshow(np.clip(data, 0, np.percentile(v, 98) if v.size else 1), cmap="viridis")
        else:
            a.imshow(data)
        a.set_title(name)
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
