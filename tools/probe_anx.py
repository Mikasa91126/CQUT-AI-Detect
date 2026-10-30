"""用 Anx 中文 AI 检测数据集（含 2025 现代大模型输出）评估 cmj 的窗口级表现。"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
from aidetect.engines.classifier import CmjClassifierEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/anx/test.csv"
WIN = 240
LINES = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]

GROUPS = {
    "human_hc3": ("hc3_human", "0"),
    "human_thucnews": ("thucnews", "0"),
    "ai_gpt35": ("hc3_chatgpt", "1"),
    "ai_gpt41mini": ("parallel_gpt-4.1-mini", "1"),
    "ai_deepseek32": ("parallel_deepseek-v3.2", "1"),
    "ai_gemini25flash": ("parallel_gemini-2.5-flash", "1"),
    "ai_claude_haiku45": ("parallel_claude-haiku-4-5-20251001", "1"),
    "ai_kimi_k2": ("parallel_Kimi-K2", "1"),
    "ai_qwen_max": ("parallel_qwen-max-latest", "1"),
    "ai_deepseek_auto": ("auto_deepseek", "1"),
}
PER = 60


class Win:
    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


def windows(text: str, size: int = WIN) -> list[str]:
    out: list[str] = []
    cur = ""
    for sent_obj in segment.split_sentences(segment.normalize(text)):
        sent = sent_obj.text
        if not sent:
            continue
        if len(sent) > size:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(sent), size):
                out.append(sent[i : i + size])
            continue
        if len(cur) + len(sent) > size and cur:
            out.append(cur)
            cur = sent
        else:
            cur += sent
    if cur:
        out.append(cur)
    return out


def load_groups() -> dict[str, list[str]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("text") or "").strip()
            if len(text) < 200:
                continue
            buckets[(row.get("source") or "", row.get("label") or "")].append(text[:1200])
    rng = random.Random(20241025)
    out: dict[str, list[str]] = {}
    for name, key in GROUPS.items():
        texts = buckets.get(key, [])
        rng.shuffle(texts)
        out[name] = texts[:PER]
    return out


def main() -> int:
    cmj = CmjClassifierEngine()
    if not cmj.available():
        print("cmj 不可用")
        return 3
    cmj._load()  # noqa: SLF001

    groups = load_groups()
    report: dict[str, dict] = {}
    for name, texts in groups.items():
        if not texts:
            continue
        flat: list[str] = []
        for t in texts:
            flat.extend(windows(t))
        scores = [float(s) for s in cmj.score([Win(w) for w in flat]).scores]
        total = sum(len(w) for w in flat) or 1
        shares = {f"{ln:.2f}": round(sum(len(w) for w, s in zip(flat, scores) if s >= ln) / total, 4) for ln in LINES}
        report[name] = {
            "docs": len(texts),
            "n": len(flat),
            "mean": round(sum(scores) / len(scores), 4),
            "shares": shares,
        }
        print(
            f"{name:20s} docs={len(texts):3d} win={len(flat):4d} mean={report[name]['mean']:.3f} "
            + " ".join(f"@{k}={v:.2f}" for k, v in shares.items()),
            flush=True,
        )

    out = ROOT / "out/_probe_anx.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
