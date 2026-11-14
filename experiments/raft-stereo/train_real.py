"""在真实双目数据（Middlebury 2014 / KITTI 2015）上训练 RAFTStereo。

数据先用 data/prepare.py 转成 npz：
    python -m data.prepare --src ... --out F:\\datasets\\stereo\\prepared\\middlebury --dataset middlebury

然后：
    python train_real.py --data F:\\datasets\\stereo\\prepared\\middlebury --steps 1200

训练分几段跑：--steps 是学习率余弦曲线的总长度，--stop_at 是这一段停在哪，
--resume 从 --save 指向的 checkpoint 接着训。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# 图里的中文标题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

from data import StereoPairs, split_scenes
from model.raft_stereo import RAFTStereo, sequence_loss


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, dev, iters: int):
    """返回每个迭代步的 EPE（按样本数加权平均）。"""
    model.eval()
    per_iter = np.zeros(iters)
    counts = 0
    for batch in loader:
        left = batch["left"].to(dev, non_blocking=True)
        right = batch["right"].to(dev, non_blocking=True)
        gt = batch["disparity"].to(dev, non_blocking=True)
        valid = batch["valid"].to(dev, non_blocking=True)
        preds = model(left, right, iters=iters)
        for i, p in enumerate(preds):
            per_iter[i] += masked_mean((p - gt).abs(), valid).item() * left.shape[0]
        counts += left.shape[0]
    return per_iter / max(counts, 1)


def save_figure(model, batch, dev, iters, path: Path):
    model.eval()
    with torch.no_grad():
        left = batch["left"].to(dev)
        preds = model(left, batch["right"].to(dev), iters=iters)
    pred = preds[-1][0, 0].cpu().numpy()
    gt = batch["disparity"][0, 0].numpy()
    valid = batch["valid"][0, 0].numpy() > 0.5
    err = np.abs(pred - gt) * valid

    def norm(x):
        v = x[valid]
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(1, 4, figsize=(20, 3.4))
    ax[0].imshow(batch["left"][0].permute(1, 2, 0).numpy())
    ax[0].set_title("左图")
    ax[1].imshow(norm(gt), cmap="magma")
    ax[1].set_title("真值视差")
    ax[2].imshow(norm(pred), cmap="magma")
    ax[2].set_title(f"预测（EPE {err[valid].mean():.2f} px）")
    im = ax[3].imshow(err, cmap="hot", vmax=max(3.0, float(err[valid].max()) * 0.3))
    ax[3].set_title("误差")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"F:\datasets\stereo\prepared\middlebury")
    p.add_argument("--steps", type=int, default=1200, help="学习率余弦曲线的总步数")
    p.add_argument("--stop_at", type=int, default=None, help="这一段停在哪一步")
    p.add_argument("--resume", action="store_true", help="从 --save 指向的 checkpoint 接着训")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--crop", type=int, nargs=2, default=[192, 384])
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--max_disp", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--val_limit", type=int, default=40,
                   help="验证集最多用多少张图（SceneFlow 有上千张，全评太慢）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default="weight/real.pth")
    p.add_argument("--fig", default="figures/real.png")
    args = p.parse_args()
    stop_at = args.stop_at or args.steps

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device(args.device)

    train_files, val_files = split_scenes(args.data, val_ratio=0.25)
    if args.val_limit and len(val_files) > args.val_limit:
        idx = np.linspace(0, len(val_files) - 1, args.val_limit).round().astype(int)
        val_files = [val_files[i] for i in np.unique(idx)]
    print(f"train files: {len(train_files)} ({len({f.stem.rsplit('_', 1)[0] for f in train_files})} 组)", flush=True)
    print(f"val   files: {len(val_files)}  {[f.stem for f in val_files[:3]]} ...", flush=True)

    crop = tuple(args.crop)
    train_ds = StereoPairs(files=train_files, crop=crop, augment=True, max_disp=args.max_disp)
    val_ds = StereoPairs(files=val_files, crop=crop, augment=False, max_disp=args.max_disp, center=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = RAFTStereo(hidden_dim=args.hidden, max_disp=args.max_disp, iters=args.iters).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M   device: {dev}   max_disp: {args.max_disp}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step: int) -> float:
        """手动余弦退火；checkpoint 里存 step，续训时学习率能接上（OneCycle 做不到）。"""
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(step, args.steps) / args.steps))

    hist_path = Path(args.save).parent / "real_hist.json"
    hist = []
    if args.resume and hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            hist = []

    start_step = 0
    if args.resume and Path(args.save).exists():
        ckpt = torch.load(args.save, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"resume from step {start_step}", flush=True)

    it = iter(train_dl)
    t0 = time.time()
    for step in range(start_step + 1, stop_at + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl)
            batch = next(it)

        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        left = batch["left"].to(dev, non_blocking=True)
        right = batch["right"].to(dev, non_blocking=True)
        gt = batch["disparity"].to(dev, non_blocking=True)
        valid = batch["valid"].to(dev, non_blocking=True)

        preds = model(left, right, iters=args.iters)
        loss = sequence_loss(preds, gt, valid, gamma=0.9)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0 or step == start_step + 1:
            executed = step - start_step
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time() - t0) / executed:.2f}s/step", flush=True)

        if step % args.eval_every == 0 or step == stop_at:
            per_iter = evaluate(model, val_dl, dev, args.iters)
            print("  val EPE per iter: " + " ".join(f"{v:5.2f}" for v in per_iter), flush=True)
            hist = [h for h in hist if h["step"] != step]
            hist.append({"step": step, **{f"it{i + 1}": float(v) for i, v in enumerate(per_iter)}})
            hist.sort(key=lambda h: h["step"])
            try:
                save_figure(model, batch, dev, args.iters, Path(args.fig))
            except Exception as exc:  # 画图失败不该中断训练
                print(f"  [warn] save_figure failed: {exc!r}", flush=True)
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, args.save)
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "step": stop_at}, args.save)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")

    print(f"done in {time.time() - t0:.1f}s -> {args.save}", flush=True)
    if hist:
        print(f"final val EPE: {hist[-1][f'it{args.iters}']:.3f} px", flush=True)


if __name__ == "__main__":
    main()
