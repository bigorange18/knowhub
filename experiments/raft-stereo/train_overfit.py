"""在一对合成图上过拟合，验证 RAFT-Stereo 的迭代更新能不能把视差推到真值。

用法：
    python train_overfit.py --steps 300
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让图里的中文标题正常显示（Windows 自带雅黑/黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from model.raft_stereo import RAFTStereo, sequence_loss
from synthetic import make_pair


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def epe(pred, gt, valid):
    return masked_mean((pred - gt).abs(), valid)


def bad_rate(pred, gt, valid, thr=1.0):
    return masked_mean(((pred - gt).abs() > thr).float(), valid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="figures/overfit.png")
    p.add_argument("--save", default="weight/overfit.pth")
    args = p.parse_args()

    torch.manual_seed(0)
    dev = torch.device(args.device)

    left, right, disp, valid = make_pair(height=192, width=384, seed=0)
    left, right, disp, valid = left.to(dev), right.to(dev), disp.to(dev), valid.to(dev)

    model = RAFTStereo(hidden_dim=64, max_disp=64, iters=args.iters).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"参数量 {n_params / 1e6:.2f} M   设备 {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        preds = model(left, right, iters=args.iters)
        loss = sequence_loss(preds, disp, valid)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                e = epe(preds[-1], disp, valid).item()
            hist.append((step, loss.item(), e))
            print(f"step {step:4d}   loss {loss.item():.3f}   EPE {e:6.2f} px   ({time.time() - t0:.1f}s)")

    model.eval()
    with torch.no_grad():
        preds = model(left, right, iters=args.iters)
        final = preds[-1]
        e_final = epe(final, disp, valid).item()
        b1 = bad_rate(final, disp, valid, 1.0).item() * 100
        b3 = bad_rate(final, disp, valid, 3.0).item() * 100
        per_iter = [epe(q, disp, valid).item() for q in preds]

    print(f"\n最终：EPE {e_final:.2f} px   bad-1px {b1:.1f}%   bad-3px {b3:.1f}%")
    print("逐次迭代 EPE: " + "  ".join(f"{v:.1f}" for v in per_iter))

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)

    # ---------------- 画图 ----------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    axes[0, 0].imshow(left[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("左图（输入）")
    axes[0, 1].imshow(right[0].permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title("右图（输入）")
    im = axes[0, 2].imshow(disp[0, 0].cpu().numpy(), cmap="viridis", vmin=0, vmax=40)
    axes[0, 2].set_title("真值视差")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(final[0, 0].cpu().numpy(), cmap="viridis", vmin=0, vmax=40)
    axes[1, 0].set_title(f"预测视差（EPE {e_final:.2f} px）")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    err = (final - disp).abs()[0, 0].cpu().numpy()
    im = axes[1, 1].imshow(err, cmap="hot", vmin=0, vmax=5)
    axes[1, 1].set_title("误差 |预测 - 真值|")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].plot(range(1, len(per_iter) + 1), per_iter, "o-")
    axes[1, 2].set_xlabel("迭代次数")
    axes[1, 2].set_ylabel("EPE (px)")
    axes[1, 2].set_title("每次迭代的视差误差")
    axes[1, 2].grid(alpha=0.3)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 2].set_xticks(range(1, len(per_iter) + 1))
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"图已保存到 {args.out}")


if __name__ == "__main__":
    main()
