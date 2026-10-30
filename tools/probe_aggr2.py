"""探针 2：窗口级（细粒度）疑似度分布。

思路：商业 AIGC 检测站通常按更细的片段判定，只要片段里有 AI 痕迹就会计入。
本探针把文档按句子切成 ≤150 字的窗口，用 v2 判别器逐窗打分，统计
「窗口级疑似占比」，并与人类对照集比较，用于设计 aggressive 档位。

只读。输出 out/_aggr_probe2.json。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aidetect import textio, segment, config  # noqa: E402
from aidetect.engines.classifier import CmjClassifierEngine  # noqa: E402

LINES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
WIN = 150


class Win:
    """最小 unit 兼容层：判别器只用到 .text 与 .chars。"""

    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


def score_windows(engine, wins: list[str]) -> list[float]:
    return list(engine.score([Win(w) for w in wins]).scores)


def windows_of(text: str) -> list[str]:
    """按句子边界把正文切成 ≤WIN 字的窗口（超长句硬切）。"""
    out: list[str] = []
    buf = ""
    for sent_obj in segment.split_sentences(segment.normalize(text)):
        sent = sent_obj.text
        if len(sent) > WIN:
            if buf:
                out.append(buf)
                buf = ""
            for i in range(0, len(sent), WIN):
                out.append(sent[i:i + WIN])
            continue
        if len(buf) + len(sent) > WIN and buf:
            out.append(buf)
            buf = sent
        else:
            buf += sent
    if buf:
        out.append(buf)
    return [w for w in out if len(w.strip()) >= 10]


def doc_windows(path: Path) -> list[str]:
    text, _ = textio.load(path)
    doc = segment.build_doc(text, "x")
    body = "".join(p.text for p in doc.paras if p.kind == "body")
    return windows_of(body)


def share(wins, scores, line):
    tot = sum(len(w) for w in wins) or 1
    hit = sum(len(w) for w, s in zip(wins, scores) if s >= line)
    return hit / tot


def summarise(name, wins, scores):
    order = sorted(scores)
    n = len(scores)
    return {
        "name": name,
        "n_windows": n,
        "chars": sum(len(w) for w in wins),
        "mean": round(sum(scores) / n, 4),
        "median": round(order[n // 2], 4),
        "p75": round(order[int(n * 0.75)], 4),
        "p90": round(order[min(n - 1, int(n * 0.9))], 4),
        "max": round(max(scores), 4),
        "share": {str(l): round(share(wins, scores, l), 4) for l in LINES},
    }


def hc3_human(sample=40, seed=7):
    path = config.DATA_DIR / "hc3zh" / "all.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            for a in d.get("human_answers") or []:
                t = (a or "").strip()
                if 200 <= len(t) <= 1200:
                    rows.append(t)
    random.Random(seed).shuffle(rows)
    return rows[:sample]


def main() -> int:
    cmj = CmjClassifierEngine()
    docs = {
        "刘颐刚": Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx"),
        "夏崇僖": Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx"),
        "张昊坤": Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx"),
        "班会总结第七周": Path(r"C:\Users\mikasa\Documents\DeepSeek Harness\班会记录\药学与生物工程学院生物医学工程402班第七周班会总结.docx"),
        "mixed_hc3": ROOT / "tests" / "fixtures" / "mixed_hc3.txt",
        "essay_3000": ROOT / "tests" / "fixtures" / "essay_3000.txt",
    }
    results = []
    for name, path in docs.items():
        if not path.exists():
            print("missing", path, flush=True)
            continue
        wins = doc_windows(path)
        if not wins:
            print("no windows", name, flush=True)
            continue
        res = None
        scores = score_windows(cmj, wins)
        # 引擎按 unit 对象取 chars；这里用文本长度手动加权，故直接复用 share()
        s = summarise(name, wins, scores)
        results.append(s)
        print(f"{name}: n={s['n_windows']} mean={s['mean']} p90={s['p90']} max={s['max']} "
              f"share@0.5={s['share']['0.5']} @0.35={s['share']['0.35']} @0.2={s['share']['0.2']}",
              flush=True)

    humans = hc3_human()
    wins = [w for t in humans for w in windows_of(t)]
    res = None
    scores = score_windows(cmj, wins)
    s = summarise("HC3人类回答40条", wins, scores)
    results.append(s)
    print(f"HC3人类: n={s['n_windows']} mean={s['mean']} p90={s['p90']} "
          f"share@0.5={s['share']['0.5']} @0.35={s['share']['0.35']} @0.2={s['share']['0.2']}", flush=True)

    out = ROOT / "out" / "_aggr_probe2.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
