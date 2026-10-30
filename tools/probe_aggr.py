"""探针：在目标论文 + 人类对照集上，测量不同「疑似线」下的疑似占比。

只读，不改任何被测代码。输出 JSON 到 out/_aggr_probe.json。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aidetect import textio, segment, config  # noqa: E402
from aidetect.engines.classifier import CmjClassifierEngine, ClassifierEngine  # noqa: E402

LINES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]


def load_doc(path: Path):
    text, name = textio.load(path)
    doc = segment.build_doc(text, name)
    units = segment.make_units(doc)
    return doc, units


def unit_scores(engine, units):
    res = engine.score(units)
    return res, list(res.scores)


def shares(units, scores, line):
    tot = sum(u.chars for u in units) or 1
    hit_chars = sum(u.chars for u, s in zip(units, scores) if s >= line)
    hit_paras = sum(1 for s in scores if s >= line)
    return hit_chars / tot, hit_paras / max(1, len(scores))


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


def build_one(engine, texts):
    units = []
    for i, t in enumerate(texts):
        doc = segment.build_doc(t, f"h{i}")
        units.extend(segment.make_units(doc))
    res = engine.score(units)
    return units, list(res.scores)


def summarize(name, units, scores):
    out = {"name": name, "n_units": len(scores), "chars": sum(u.chars for u in units),
           "max": round(max(scores), 4), "mean": round(sum(scores) / len(scores), 4),
           "median": round(sorted(scores)[len(scores) // 2], 4)}
    out["share"] = {str(l): round(shares(units, scores, l)[0], 4) for l in LINES}
    out["para_share"] = {str(l): round(shares(units, scores, l)[1], 4) for l in LINES}
    return out


def main() -> int:
    results = []
    cmj = CmjClassifierEngine()
    clf = ClassifierEngine()
    print("loading engines ...", flush=True)

    targets = {
        "刘颐刚_导论论文": Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx"),
        "夏崇僖_导论论文": Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx"),
        "张昊坤_测控小论文": Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx"),
        "班会总结第七周(人写)": Path(r"C:\Users\mikasa\Documents\DeepSeek Harness\班会记录\药学与生物工程学院生物医学工程402班第七周班会总结.docx"),
    }
    for name, path in targets.items():
        if not path.exists():
            print("missing", path, flush=True)
            continue
        doc, units = load_doc(path)
        for eng, tag in ((cmj, "cmj"), (clf, "clf")):
            res, scores = unit_scores(eng, units)
            if not scores:
                continue
            s = summarize(f"{name}|{tag}", units, scores)
            s["doc"] = name
            s["engine"] = tag
            results.append(s)
            print(tag, name, "max", s["max"], "share@0.5", s["share"]["0.5"],
                  "share@0.3", s["share"]["0.3"], flush=True)

    for fname in ("mixed_hc3.txt", "essay_3000.txt"):
        p = ROOT / "tests" / "fixtures" / fname
        if not p.exists():
            continue
        doc, units = load_doc(p)
        for eng, tag in ((cmj, "cmj"), (clf, "clf")):
            res, scores = unit_scores(eng, units)
            if not scores:
                continue
            s = summarize(f"{fname}|{tag}", units, scores)
            s["doc"] = fname
            s["engine"] = tag
            results.append(s)
            print(tag, fname, "max", s["max"], "share@0.5", s["share"]["0.5"], flush=True)

    human_texts = hc3_human()
    units, scores = build_one(cmj, human_texts)
    s = summarize("HC3人类回答40条|cmj", units, scores)
    s["doc"] = "HC3人类（域外人类对照）"
    s["engine"] = "cmj"
    results.append(s)
    print("cmj HC3-human share@0.35", s["share"]["0.35"], "@0.5", s["share"]["0.5"], flush=True)

    out = ROOT / "out" / "_aggr_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
