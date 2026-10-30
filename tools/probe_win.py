"""窗口粒度扫描：不同窗口长度下，疑似占比随判定线的变化。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402
from aidetect.engines.classifier import CmjClassifierEngine  # noqa: E402

LINES = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
WIN_SIZES = [60, 100, 150, 240, 400]


class Win:
    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


DOCS = [
    ("刘颐刚", Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx")),
    ("夏崇僖", Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx")),
    ("张昊坤", Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx")),
    ("mixed_hc3", Path(__file__).resolve().parent.parent / "tests/fixtures/mixed_hc3.txt"),
    ("essay_3000", Path(__file__).resolve().parent.parent / "tests/fixtures/essay_3000.txt"),
]


def body_text(path: Path) -> str:
    text, _ = textio.load(str(path))
    doc = segment.build_doc(text, path.stem)
    return "\n".join(p.text for p in doc.paras if p.kind == "body")


def windows(text: str, size: int) -> list[str]:
    """按句子打包成 <=size 字符的窗口。"""
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


def shares(scores: list[float], wins: list[str]) -> dict[str, float]:
    total = sum(len(w) for w in wins) or 1
    res = {}
    for line in LINES:
        hit = sum(len(w) for w, s in zip(wins, scores) if s >= line)
        res[f"{line:.2f}"] = round(hit / total, 4)
    return res


def hc3_human(limit: int = 60) -> list[str]:
    import json as _json

    path = Path(__file__).resolve().parent.parent / "data/hc3zh/all.jsonl"
    out: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if len(out) >= limit:
                break
            row = _json.loads(line)
            for ans in row.get("human_answers", [])[:1]:
                if not ans:
                    continue
                ans = str(ans).strip()
                if len(ans) >= 200:
                    out.append(ans[:1200])
    return out


def main() -> int:
    cmj = CmjClassifierEngine()
    if not cmj.available():
        print("cmj 不可用")
        return 3
    cmj._load()  # noqa: SLF001

    report = {}
    for size in WIN_SIZES:
        entry = {}
        for name, path in DOCS:
            if not path.exists():
                continue
            wins = windows(body_text(path), size)
            if not wins:
                continue
            scores = [float(s) for s in cmj.score([Win(w) for w in wins]).scores]
            entry[name] = {
                "n": len(wins),
                "mean": round(sum(scores) / len(scores), 4),
                "max": round(max(scores), 4),
                "shares": shares(scores, wins),
            }
        humans = hc3_human()
        wins = []
        for h in humans:
            wins.extend(windows(h, size))
        scores = [float(s) for s in cmj.score([Win(w) for w in wins]).scores]
        entry["HC3人类"] = {
            "n": len(wins),
            "mean": round(sum(scores) / len(scores), 4),
            "max": round(max(scores), 4),
            "shares": shares(scores, wins),
        }
        report[str(size)] = entry
        print(f"== window {size} ==", flush=True)
        for key, val in entry.items():
            sh = val["shares"]
            print(
                f"  {key:10s} n={val['n']:4d} mean={val['mean']:.3f} max={val['max']:.3f} "
                f"@.10={sh['0.10']:.3f} @.20={sh['0.20']:.3f} @.35={sh['0.35']:.3f} @.50={sh['0.50']:.3f}",
                flush=True,
            )

    out = Path(__file__).resolve().parent.parent / "out/_probe_win.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
