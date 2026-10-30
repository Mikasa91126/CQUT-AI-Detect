"""文体特征探针：比较目标论文与人类/AI 文本的文体统计量。"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402
from aidetect.engines.stylometry import StylometryEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/anx/test.csv"
SAMPLE = 25

DOCS = [
    ("刘颐刚", Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx")),
    ("夏崇僖", Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx")),
    ("张昊坤", Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx")),
]


def load_group(source: str, label: str, per: int) -> list[str]:
    texts: list[str] = []
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("source") or "") != source or (row.get("label") or "") != label:
                continue
            t = (row.get("text") or "").strip()
            if len(t) >= 400:
                texts.append(t)
    random.Random(20241025).shuffle(texts)
    return texts[:per]


def units_of_text(text: str) -> list:
    doc = segment.build_doc(text, "x")
    return segment.make_units(doc)


def measure(stylo: StylometryEngine, units: list) -> dict:
    res = stylo.score(units)
    scores = getattr(res, "scores", None) or []
    return {
        "score": round(float(sum(scores) / len(scores)), 4) if scores else 0.0,
        **{k: round(v, 4) for k, v in res.features.items() if isinstance(v, (int, float))},
    }


def main() -> int:
    stylo = StylometryEngine()
    rows: dict[str, dict] = {}
    for name, path in DOCS:
        text, _ = textio.load(str(path))
        doc = segment.build_doc(text, name)
        units = segment.make_units(doc)
        rows[name] = measure(stylo, units)
        print(name, json.dumps(rows[name], ensure_ascii=False), flush=True)

    for name, source, label in (
        ("human_thucnews", "thucnews", "0"),
        ("human_hc3", "hc3_human", "0"),
        ("ai_deepseek", "parallel_deepseek-v3.2", "1"),
        ("ai_gpt41mini", "parallel_gpt-4.1-mini", "1"),
    ):
        vals = [measure(stylo, units_of_text(t)) for t in load_group(source, label, SAMPLE)]
        keys = ["score", "cv_slen", "short_ratio", "conn100", "tmpl100", "repeat3", "comma_per_sent", "mixed_en"]
        agg = {}
        for k in keys:
            series = [v[k] for v in vals if k in v]
            if series:
                agg[k] = round(sum(series) / len(series), 4)
        rows[name] = agg
        print(name, json.dumps(agg, ensure_ascii=False), flush=True)

    out = ROOT / "out/_probe_stylo.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
