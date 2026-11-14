"""在准备好的 npz 数据集上整体评测：EPE / bad-1px / bad-3px。

    # 在自己训练的数据集上看验证集
    python eval.py --data F:\\datasets\\stereo\\prepared\\middlebury \\
        --weights weight/real.pth --json results/eval_middlebury.json

    # 跨数据集：用 SceneFlow 训出来的权重直接测 Middlebury（零样本）
    python eval.py --data F:\\datasets\\stereo\\prepared\\middlebury \\
        --weights weight/sceneflow.pth --max_disp 272 --json results/eval_sf2mb.json

指标按有效像素加权平均，和论文里的 EPE 一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from infer import predict
from model.raft_stereo import RAFTStereo
from data import split_scenes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prepare.py 的输出目录")
    p.add_argument("--weights", default="weight/real.pth")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--max_disp", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="只测前 N 张")
    p.add_argument("--split", choices=["all", "val", "train"], default="all",
                   help="用 split_scenes 的切分只测验证/训练部分")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", help="把逐图结果和汇总写到这个 json")
    args = p.parse_args()

    files = sorted(Path(args.data).glob("*.npz"))
    if not files:
        raise SystemExit(f"{args.data} 里没有 npz")
    if args.split != "all":
        train_files, val_files = split_scenes(args.data, val_ratio=0.25)
        files = sorted(train_files if args.split == "train" else val_files)
    if args.limit and len(files) > args.limit:
        idx = np.linspace(0, len(files) - 1, args.limit).round().astype(int)
        files = [files[i] for i in np.unique(idx)]
    print(f"split={args.split}")

    dev = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location=dev)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = RAFTStereo(hidden_dim=args.hidden, max_disp=args.max_disp, iters=args.iters).to(dev)
    model.load_state_dict(state)
    model.eval()
    print(f"{len(files)} 张  权重 {args.weights}  max_disp {args.max_disp}  iters {args.iters}")

    rows, tot_err, tot_valid, tot_b1, tot_b3 = [], 0.0, 0, 0, 0
    for f in files:
        with np.load(f) as z:
            left, right = z["left"], z["right"]
            gt = z["disp"].astype(np.float32)
            valid = z["valid"].astype(bool) & np.isfinite(gt) & (gt > 0)
        pred = predict(model, left, right, dev, args.iters)[-1]
        err = np.abs(pred - gt)
        n = int(valid.sum())
        if n == 0:
            continue
        epe = float(err[valid].sum() / n)
        b1 = float((err[valid] > 1).sum() / n) * 100
        b3 = float((err[valid] > 3).sum() / n) * 100
        rows.append({"file": f.name, "epe": epe, "bad1": b1, "bad3": b3, "valid_px": n})
        tot_err += err[valid].sum()
        tot_valid += n
        tot_b1 += (err[valid] > 1).sum()
        tot_b3 += (err[valid] > 3).sum()
        print(f"  {f.stem:32s} EPE {epe:7.3f}  bad-1px {b1:5.1f}%  bad-3px {b3:5.1f}%")

    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "iters": args.iters,
        "max_disp": args.max_disp,
        "images": len(rows),
        "epe": tot_err / max(tot_valid, 1),
        "bad1": tot_b1 / max(tot_valid, 1) * 100,
        "bad3": tot_b3 / max(tot_valid, 1) * 100,
    }
    print(f"ALL  EPE {summary['epe']:.3f} px   bad-1px {summary['bad1']:.1f}%   "
          f"bad-3px {summary['bad3']:.1f}%   ({len(rows)} 张)")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "per_image": rows}, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
