"""多线程分段下载器：慢速源（每连接被限速）时把带宽压满，支持中断续传。

单连接 0.2 MB/s、6 连接 ~3 MB/s —— 这就是写它的原因。

    python data/fetch_url.py --url https://... --out F:\\datasets\\x.tar --threads 6

进度记在 `--out` 旁边的 `.state.json` 里：已经下完的分段会跳过，
所以随时可以 Ctrl-C / 杀进程，再用同样的命令继续。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024  # 每个分段的字节数


class Downloader:
    def __init__(self, url: str, out: Path, threads: int, proxy: str | None):
        self.url = url
        self.out = out
        self.state_path = out.with_suffix(out.suffix + ".state.json")
        self.threads = threads
        self.lock = threading.Lock()
        self.done: set[int] = set()
        self.size = 0
        self.total = 0
        self.started = time.time()
        self.failed: str | None = None
        self._opener = self._make_opener(proxy)

    @staticmethod
    def _make_opener(proxy: str | None):
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        return urllib.request.build_opener(*handlers)

    def _save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"size": self.size, "chunk": CHUNK,
                                   "done": sorted(self.done)}), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _load_state(self) -> bool:
        if not self.state_path.exists() or not self.out.exists():
            return False
        try:
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if st.get("chunk") != CHUNK or self.out.stat().st_size != st.get("size"):
            return False
        self.size = int(st["size"])
        self.done = set(int(i) for i in st["done"])
        return True

    def probe(self):
        """拿总长度；不支持 Range 就直接报错，别默默下成 200。"""
        req = urllib.request.Request(self.url, method="HEAD")
        with self._opener.open(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            accept = resp.headers.get("Accept-Ranges")
        if not length:
            raise SystemExit("服务器没给 Content-Length，没法分段")
        self.size = int(length)
        self.total = (self.size + CHUNK - 1) // CHUNK
        if accept and accept.lower() != "bytes":
            print(f"警告：Accept-Ranges={accept}，仍然试试分段")

    def _fetch_chunk(self, index: int):
        start = index * CHUNK
        end = min(start + CHUNK, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(1, 5):
            try:
                with self._opener.open(req, timeout=90) as resp:
                    if resp.status != 206:
                        raise RuntimeError(f"分段请求返回 {resp.status}（服务器不支持 Range？）")
                    data = resp.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"分段长度不符：{len(data)} != {end - start + 1}")
                with self.lock, open(self.out, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                    self.done.add(index)
                    self._save_state()
                return
            except Exception as exc:  # 网络抖动，退避重试
                if attempt == 4:
                    raise RuntimeError(f"分段 {index} 失败：{exc!r}") from exc
                time.sleep(2 * attempt)

    def run(self):
        if self._load_state():
            print(f"续传：已完成 {len(self.done)}/{self.total} 段", flush=True)
        else:
            self.probe()
            print(f"总长 {self.size / 1024 ** 3:.2f} GB，{self.total} 段 × 8 MB", flush=True)
            with open(self.out, "wb") as f:
                f.truncate(self.size)
            self.done = set()
            self._save_state()

        pending = [i for i in range(self.total) if i not in self.done]
        q = iter(pending)
        q_lock = threading.Lock()

        def worker():
            while True:
                with q_lock:
                    idx = next(q, None)
                if idx is None:
                    return
                self._fetch_chunk(idx)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.threads)]
        for t in threads:
            t.start()

        last = 0.0
        done_at_start = len(self.done)
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            now = time.time()
            if now - last >= 5:
                last = now
                got = len(self.done) * CHUNK / 1024 ** 2
                pct = len(self.done) / max(self.total, 1) * 100
                rate = (len(self.done) - done_at_start) * CHUNK / 1024 ** 2 / max(now - self.started, 1e-6)
                print(f"  {pct:5.1f}%  {got:8.1f} MB  {rate:5.2f} MB/s  "
                      f"剩余 {self.total - len(self.done)} 段", flush=True)
        for t in threads:
            t.join()
        print(f"完成 {self.out}（{self.size / 1024 ** 3:.2f} GB，"
              f"{time.time() - self.started:.0f}s）", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        Downloader(args.url, out, args.threads, args.proxy).run()
    except KeyboardInterrupt:
        print("中断，进度已保存", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
