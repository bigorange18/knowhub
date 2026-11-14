"""把下载到的双目数据集转成训练用的 .npz（左图 / 右图 / 视差 / 有效掩码）。

支持两种来源：
    Middlebury 2014  zip（im0.png, im1.png, disp0.pfm, disp0-n.pgm, calib.txt）
    KITTI 2015       zip（training/image_2/*_10.png, image_3/, disp_occ_0/）

输出目录里每个场景一个 npz，键：left, right (uint8 HWC)，disp (float32 HW)，valid (bool HW)。

用法：
    python -m data.prepare --src F:\\datasets\\middlebury2014\\_download \\
        --out F:\\datasets\\stereo\\prepared\\middlebury --dataset middlebury --width 768
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# 读 Middlebury 的 pf/pgm
# --------------------------------------------------------------------------- #
def read_pfm(buf: bytes) -> np.ndarray:
    """读 .pfm（Middlebury 的视差真值）。行序是自下而上，这里翻正。"""
    f = io.BytesIO(buf)
    magic = f.readline().strip()
    if magic not in (b"PF", b"Pf"):
        raise ValueError(f"不是 pfm: {magic!r}")
    w, h = (int(x) for x in f.readline().split())
    scale = float(f.readline().strip())
    endian = "<" if scale < 0 else ">"
    n_ch = 3 if magic == b"PF" else 1
    data = np.frombuffer(f.read(w * h * n_ch * 4), dtype=endian + "f4")
    data = data.reshape(h, w, n_ch) if n_ch == 3 else data.reshape(h, w)
    data = data[::-1]  # pfm 底行在前
    if n_ch == 3:
        data = data[:, :, 1]  # 三个通道相同，取一个
    return np.ascontiguousarray(data, dtype=np.float32)


def read_pgm(buf: bytes) -> np.ndarray:
    """读 8 位 P5 pgm（Middlebury 的 disp0-n.pgm：每个像素的采样次数）。"""
    f = io.BytesIO(buf)
    magic = f.readline().strip()
    if magic != b"P5":
        raise ValueError(f"不是 P5 pgm: {magic!r}")
    line = f.readline()
    while line.startswith(b"#"):
        line = f.readline()
    w, h = (int(x) for x in line.split())
    f.readline()  # maxval
    return np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w).copy()


# --------------------------------------------------------------------------- #
# 缩放
# --------------------------------------------------------------------------- #
def _resize_rgb(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    height = int(round(h * width / w))
    return np.array(Image.fromarray(img).resize((width, height), Image.LANCZOS))


def _resize_float(arr: np.ndarray, size, scale: float, mode=Image.BILINEAR) -> np.ndarray:
    out = np.array(Image.fromarray(arr.astype(np.float32), mode="F").resize(size, mode))
    return out * scale


def _resize_mask(mask: np.ndarray, size) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    out = np.array(Image.fromarray(m).resize(size, Image.NEAREST))
    return out > 127


def _trim_to_multiple(arr: np.ndarray, m: int = 8) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[: (h // m) * m, : (w // m) * m]


def _make_sample(left: np.ndarray, right: np.ndarray, disp: np.ndarray,
                 valid: np.ndarray, width: int) -> dict:
    scale = width / left.shape[1]
    left = _resize_rgb(left, width)
    right = _resize_rgb(right, width)
    size = (left.shape[1], left.shape[0])  # PIL 是 (w, h)

    disp = _resize_float(disp, size, scale)
    valid = _resize_mask(valid, size)
    valid &= np.isfinite(disp) & (disp > 0)
    disp = np.where(np.isfinite(disp), disp, 0).astype(np.float32)

    left = _trim_to_multiple(left)
    right = _trim_to_multiple(right)
    disp = _trim_to_multiple(disp)
    valid = _trim_to_multiple(valid)
    return {"left": left, "right": right, "disp": disp, "valid": valid}


# --------------------------------------------------------------------------- #
# Middlebury
# --------------------------------------------------------------------------- #
def parse_calib(text: str) -> dict:
    """解析 Middlebury 的 calib.txt，返回焦距/baseline/doffs 等（单位 mm、像素）。"""
    info = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if key in ("cam0", "cam1"):
            nums = [float(x) for x in val.strip("[]").replace(";", " ").split()]
            info[key] = {"fx": nums[0], "cx": nums[2]}
        elif key in ("width", "height", "ndisp", "isint", "vmin", "vmax"):
            info[key] = int(val)
        else:
            try:
                info[key] = float(val)
            except ValueError:
                pass
    # 统一成训练时要用的两个数：焦距（像素）、基线（与视差同单位，通常是 mm）
    info["focal"] = info.get("cam0", {}).get("fx", 0.0)
    info["doffs"] = info.get("doffs", 0.0)
    return info


def load_middlebury(zip_path: Path, width: int = 768) -> tuple[dict, dict]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        prefix = names[0].split("/")[0]

        def read(suffix: str) -> bytes:
            return z.read(f"{prefix}/{suffix}")

        left = np.array(Image.open(io.BytesIO(read("im0.png"))).convert("RGB"))
        right = np.array(Image.open(io.BytesIO(read("im1.png"))).convert("RGB"))
        disp = read_pfm(read("disp0.pfm"))
        samples = read_pgm(read("disp0-n.pgm"))
        calib = parse_calib(read("calib.txt").decode("utf-8", "ignore"))
    sample = _make_sample(left, right, disp, samples > 0, width)
    sample["focal"] = np.float32(calib["focal"] * width / calib["width"])
    sample["baseline"] = np.float32(calib.get("baseline", 0.0))
    sample["doffs"] = np.float32(calib["doffs"] * width / calib["width"])
    return sample, calib


# --------------------------------------------------------------------------- #
# KITTI 2015
# --------------------------------------------------------------------------- #
def load_kitti_zip(zip_path: Path, width: int = 768) -> list[dict]:
    """KITTI 的 zip 一次装 200 对，返回列表。"""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        lefts = sorted(n for n in names if "/image_2/" in n and n.endswith("_10.png"))
        for i, name in enumerate(lefts):
            stem = Path(name).stem  # 000000_10
            right_name = name.replace("/image_2/", "/image_3/")
            disp_name = name.replace("/image_2/", "/disp_occ_0/")

            left = np.array(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
            right = np.array(Image.open(io.BytesIO(z.read(right_name))).convert("RGB"))
            raw = np.array(Image.open(io.BytesIO(z.read(disp_name))), dtype=np.uint16)
            disp = raw.astype(np.float32) / 256.0
            out.append(_make_sample(left, right, disp, disp > 0, width))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(lefts)} 对已转换")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="zip 所在目录")
    p.add_argument("--out", required=True, help="npz 输出目录")
    p.add_argument("--dataset", choices=["middlebury", "kitti"], required=True)
    p.add_argument("--width", type=int, default=768, help="统一缩放到这个宽度")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zips = sorted(src.glob("*.zip"))
    if not zips:
        raise SystemExit(f"{src} 里没有 .zip")

    meta = {"dataset": args.dataset, "width": args.width, "scenes": []}
    for zp in zips:
        if args.dataset == "middlebury":
            name = zp.stem.replace("-perfect", "")
            sample, calib = load_middlebury(zp, args.width)
            np.savez_compressed(out / f"{name}.npz", **sample)
            h, w = sample["disp"].shape
            meta["scenes"].append({
                "name": name, "h": int(h), "w": int(w),
                "valid_ratio": round(float(sample["valid"].mean()), 4),
                "focal_px": round(float(sample["focal"]), 2),
                "baseline_mm": calib.get("baseline", 0.0),
                "doffs_px": round(float(sample["doffs"]), 2),
                "ndisp_orig": calib.get("ndisp", 0),
            })
            print(f"{name}: {w}x{h}, 有效像素 {sample['valid'].mean() * 100:.1f}%, "
                  f"f={float(sample['focal']):.1f}px B={calib.get('baseline', 0):.1f}mm")
        else:
            samples = load_kitti_zip(zp, args.width)
            for i, s in enumerate(samples):
                np.savez_compressed(out / f"{zp.stem}_{i:06d}.npz", **s)
            h, w = samples[0]["disp"].shape
            meta["scenes"].append({"name": zp.stem, "count": len(samples), "h": int(h), "w": int(w)})
            print(f"{zp.stem}: {len(samples)} 对，{w}x{h}")

    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成，写到 {out}")


if __name__ == "__main__":
    main()
