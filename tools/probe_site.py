"""站点对照档探针：窗口级（240 字）判别分 → 用 HC3 校验人类误报并给出判定线。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402
from aidetect.engines.classifier import CmjClassifierEngine  # noqa: E402

WIN = 240
TRUNC = 1200
DEV = ("finance", "law", "medicine")
TEST = ("open_qa", "baike", "psychology", "nlpcc_dbqa")
PER = 60
LINES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
ROOT = Path(__file__).resolve().parent.parent


class Win:
    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


def windows(text: str, size: int) -> list[str]:
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


def load_samples() -> dict[str, list[tuple[str, str, str]]]:
    """返回 {split: [(text, label, domain)]}。"""
    path = ROOT / "data/hc3zh/all.jsonl"
    rng = random.Random(20241025)
    buckets: dict[tuple[str, str, str], list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            dom = row.get("source") or "?"
            if dom not in DEV and dom not in TEST:
                continue
            split = "dev" if dom in DEV else "test"
            for side, key in (("human", "human_answers"), ("ai", "chatgpt_answers")):
                for ans in row.get(key, [])[:1]:
                    if not ans:
                        continue
                    ans = str(ans).strip()
                    if len(ans) < 200:
                        continue
                    buckets.setdefault((split, side, dom), []).append(ans[:TRUNC])
    out: dict[str, list[tuple[str, str, str]]] = {"dev": [], "test": []}
    for (split, side, dom), texts in buckets.items():
        rng.shuffle(texts)
        for t in texts[:PER]:
            out[split].append((t, side, dom))
    return out


def score_texts(cmj, texts: list[str]) -> list[list[float]]:
    """每条文本返回窗口级分数列表。"""
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in texts:
        wins = windows(t, WIN)
        spans.append((len(flat), len(flat) + len(wins)))
        flat.extend(wins)
    scores = [float(s) for s in cmj.score([Win(w) for w in flat]).scores]
    return [scores[a:b] for a, b in spans]


def char_share(pairs: list[tuple[float, int]], line: float) -> float:
    total = sum(c for _, c in pairs) or 1
    return sum(c for p, c in pairs if p >= line) / total


def main() -> int:
    cmj = CmjClassifierEngine()
    if not cmj.available():
        print("cmj 不可用")
        return 3
    cmj._load()  # noqa: SLF001

    samples = load_samples()
    print(f"dev {len(samples['dev'])} 篇 / test {len(samples['test'])} 篇", flush=True)

    data: dict[str, list[tuple[str, list[float]]]] = {}
    for split in ("dev", "test"):
        texts = [t for t, _, _ in samples[split]]
        per = score_texts(cmj, texts)
        rows: list[tuple[str, list[float]]] = []
        for (text, side, _dom), sc in zip(samples[split], per):
            rows.append((side, sc))
        data[split] = rows
        print(f"{split} 打分完成", flush=True)

    def win_pairs(split: str, side: str) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        for s, sc in data[split]:
            if s != side:
                continue
            out.extend((p, WIN) for p in sc)
        return out

    def auc(neg: list[float], pos: list[float]) -> float:
        pairs = [(x, 0) for x in neg] + [(y, 1) for y in pos]
        pairs.sort()
        rank_sum = 0.0
        n = len(pairs)
        i = 0
        while i < n:
            j = i
            while j < n and pairs[j][0] == pairs[i][0]:
                j += 1
            avg = (i + j + 1) / 2
            for k in range(i, j):
                if pairs[k][1] == 1:
                    rank_sum += avg
            i = j
        npos = sum(1 for _, l in pairs if l == 1)
        nneg = n - npos
        if npos == 0 or nneg == 0:
            return 0.5
        return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)

    dev_h = [p for p, _ in win_pairs("dev", "human")]
    dev_a = [p for p, _ in win_pairs("dev", "ai")]
    test_h_pairs = win_pairs("test", "human")
    test_a_pairs = win_pairs("test", "ai")
    print(f"窗口数 dev human {len(dev_h)} / dev ai {len(dev_a)} / test human {len(test_h_pairs)} / test ai {len(test_a_pairs)}")
    print(f"窗口级 AUC dev {auc(dev_h, dev_a):.4f} test {auc([p for p,_ in test_h_pairs], [p for p,_ in test_a_pairs]):.4f}")

    table = []
    for target in (0.05, 0.10, 0.15, 0.20):
        line = sorted(dev_h)[max(0, int(len(dev_h) * (1 - target)) - 1)]
        table.append({
            "target_fpr": target,
            "line": round(line, 4),
            "dev_fpr_win": round(sum(1 for p in dev_h if p >= line) / len(dev_h), 4),
            "test_fpr_win": round(sum(1 for p in test_h_pairs if p[0] >= line) / len(test_h_pairs), 4),
            "test_fpr_char": round(char_share(test_h_pairs, line), 4),
            "test_tpr_win": round(sum(1 for p, _ in test_a_pairs if p >= line) / len(test_a_pairs), 4),
            "test_tpr_char": round(char_share(test_a_pairs, line), 4),
        })
        print(f"  目标FPR {target:.0%} -> 线 {line:.4f} | test 人类窗口 {table[-1]['test_fpr_win']:.3%} 字数 {table[-1]['test_fpr_char']:.3%} | test AI 字数 {table[-1]['test_tpr_char']:.3%}")

    docs = [
        ("刘颐刚", Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx")),
        ("夏崇僖", Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx")),
        ("张昊坤", Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx")),
        ("mixed_hc3", ROOT / "tests/fixtures/mixed_hc3.txt"),
        ("essay_3000", ROOT / "tests/fixtures/essay_3000.txt"),
    ]
    anchor = {}
    for name, path in docs:
        if not path.exists():
            continue
        text, _ = textio.load(str(path))
        doc = segment.build_doc(text, path.stem)
        body = "\n".join(p.text for p in doc.paras if p.kind == "body")
        wins = windows(body, WIN)
        sc = [float(x) for x in cmj.score([Win(w) for w in wins]).scores]
        pairs = [(p, len(w)) for p, w in zip(sc, wins)]
        anchor[name] = {
            "n": len(wins),
            "mean": round(sum(sc) / len(sc), 4),
            **{f"@fpr{t}": round(char_share(pairs, row["line"]), 4) for t, row in zip((0.05, 0.10, 0.15, 0.20), table)},
        }
        print(f"  {name:10s} n={len(wins):3d} mean={anchor[name]['mean']:.3f} " +
              " ".join(f"{k}={v:.3f}" for k, v in anchor[name].items() if k.startswith("@") and isinstance(v, float)))

    out = ROOT / "out/_probe_site.json"
    out.write_text(json.dumps({"window": WIN, "per": PER, "table": table, "anchor": anchor}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
