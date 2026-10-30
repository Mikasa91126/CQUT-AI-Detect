"""对比 PaperPass 报告的句级 AI 分与我们 cmj 的分，看看能否做单调映射。"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidetect.engines.classifier import CmjClassifierEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "out/_site_sentences.json"


class Win:
    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


def rank(data: list[float]) -> list[float]:
    order = sorted(range(len(data)), key=lambda i: data[i])
    out = [0.0] * len(data)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and data[order[j + 1]] == data[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main() -> int:
    items = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    cmj = CmjClassifierEngine()
    cmj._load()  # noqa: SLF001
    for it in items:
        text = (it.get("content") or "").strip()
        score = float(it.get("score") or 0.0)
        if score < 0 or len(text) < 15:
            continue
        rows.append((text, score))
    print("usable sentences:", len(rows), flush=True)

    ours = [float(cmj.score([Win(t)]).scores[0]) for t, _ in rows]
    theirs = [s for _, s in rows]

    bands = {
        "0": [i for i, s in enumerate(theirs) if s == 0],
        "(0,60)": [i for i, s in enumerate(theirs) if 0 < s < 60],
        "[60,75)": [i for i, s in enumerate(theirs) if 60 <= s < 75],
        "[75,90)": [i for i, s in enumerate(theirs) if 75 <= s < 90],
        ">=90": [i for i, s in enumerate(theirs) if s >= 90],
    }
    for name, idx in bands.items():
        if not idx:
            print(f"{name:10s} n=0")
            continue
        vals = [ours[i] for i in idx]
        print(
            f"{name:10s} n={len(idx):3d} 我们cmj: mean={statistics.fmean(vals):.3f} "
            f"median={statistics.median(vals):.3f} max={max(vals):.3f}",
            flush=True,
        )
    print("spearman rho =", round(spearman(ours, theirs), 3))

    out = ROOT / "out/_site_alignment.json"
    out.write_text(
        json.dumps(
            {
                "n": len(rows),
                "spearman": round(spearman(ours, theirs), 4),
                "bands": {
                    name: (len(idx), round(statistics.fmean([ours[i] for i in idx]), 4) if idx else None)
                    for name, idx in bands.items()
                },
                "rows": [
                    {"text": t, "site": s, "cmj": round(o, 5)} for (t, s), o in zip(rows, ours)
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
