"""探测 hf-mirror / modelscope 的下载吞吐（绕过系统代理）。"""
from __future__ import annotations

import time
import urllib.request

# 清掉可能存在的代理环境变量，直连
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TARGETS = [
    ("hf-mirror Anx", "https://hf-mirror.com/AnxForever/chinese-ai-detector-bert/resolve/main/pytorch_model.bin"),
    ("hf-mirror zhv3", "https://hf-mirror.com/yuchuantian/AIGC_detector_zhv3/resolve/main/pytorch_model.bin"),
    ("modelscope Qwen", "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B/repo?Revision=master&FilePath=model.safetensors"),
    ("hf-mirror desklib", "https://hf-mirror.com/desklib/ai-text-detector-v1.01/resolve/main/pytorch_model.bin"),
]
SIZE = 4 * 1024 * 1024


def probe(label: str, url: str) -> None:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{SIZE - 1}", "User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with _opener.open(req, timeout=30) as r:
            total = 0
            while total < SIZE:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                total += len(chunk)
            dt = time.time() - t0
            mb = total / 1048576
            print(f"{label:18s} {mb:5.2f}MB in {dt:5.1f}s -> {mb / dt:5.2f} MB/s  status={r.status}")
    except Exception as exc:  # noqa: BLE001
        dt = time.time() - t0
        print(f"{label:18s} FAIL after {dt:.1f}s: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    for label, url in TARGETS:
        probe(label, url)
