"""混合特征探针：窗口级 cmj 分 + GPT2-Chinese 惊奇度，比较目标论文与人类/现代 AI 文本。"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402
from aidetect.engines.classifier import CmjClassifierEngine  # noqa: E402
from aidetect.engines.perplexity import PerplexityEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WIN = 240
CSV_PATH = ROOT / "data/anx/test.csv"


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


def load_group(source: str, label: str, per: int) -> list[str]:
    texts: list[str] = []
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("source") or "") != source or (row.get("label") or "") != label:
                continue
            t = (row.get("text") or "").strip()
            if len(t) >= 200:
                texts.append(t[:1200])
    random.Random(20241025).shuffle(texts)
    return texts[:per]


def body_of(path: Path) -> str:
    text, _ = textio.load(str(path))
    doc = segment.build_doc(text, path.stem)
    return "\n".join(p.text for p in doc.paras if p.kind == "body")


def main() -> int:
    cmj = CmjClassifierEngine()
    cmj._load()  # noqa: SLF001
    ppl = PerplexityEngine()
    if not ppl.available():
        print("ppl 不可用")
        return 3
    ppl._load()  # noqa: SLF001

    docs = [("刘颐刚", body_of(Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx")))]
    for name, source, label, per in (
        ("human_thucnews", "thucnews", "0", 12),
        ("human_hc3", "hc3_human", "0", 12),
        ("ai_deepseek", "parallel_deepseek-v3.2", "1", 12),
        ("ai_gpt41mini", "parallel_gpt-4.1-mini", "1", 12),
    ):
        for i, t in enumerate(load_group(source, label, per)):
            docs.append((f"{name}#{i}", t))

    report: dict[str, dict] = {}
    for name, text in docs:
        wins = windows(text)
        if len(wins) < 2:
            continue
        p_cmj = [float(s) for s in cmj.score([Win(w) for w in wins]).scores]
        lps: list[float] = []
        for w in wins:
            feats = ppl._window_features([w])  # noqa: SLF001
            lp = feats[0].get("meanlprob") if feats else None
            lps.append(float(lp) if lp is not None else -9.0)
        group = name.split("#")[0]
        report.setdefault(group, {"docs": 0, "cmj": [], "lp": [], "ppl": []})
        report[group]["docs"] += 1
        report[group]["cmj"].extend(p_cmj)
        report[group]["lp"].extend(lps)

    summary = {}
    for group, val in report.items():
        cmj_vals = val["cmj"]
        lp_vals = val["lp"]
        entry = {
            "docs": val["docs"],
            "n": len(cmj_vals),
            "cmj_mean": round(sum(cmj_vals) / len(cmj_vals), 4),
            "cmj_share50": round(sum(1 for x in cmj_vals if x >= 0.5) / len(cmj_vals), 4),
            "cmj_share20": round(sum(1 for x in cmj_vals if x >= 0.2) / len(cmj_vals), 4),
            "cmj_share10": round(sum(1 for x in cmj_vals if x >= 0.1) / len(cmj_vals), 4),
            "lp_mean": round(sum(lp_vals) / len(lp_vals), 4),
            "ppl_median": round(sorted(__import__("math").exp(-x) for x in lp_vals)[len(lp_vals) // 2], 2),
        }
        summary[group] = entry
        print(f"{group:18s} docs={entry['docs']:3d} n={entry['n']:4d} cmj_mean={entry['cmj_mean']:.3f} "
              f"share50={entry['cmj_share50']:.2f} share20={entry['cmj_share20']:.2f} "
              f"lp={entry['lp_mean']:.3f} ppl_med={entry['ppl_median']:.1f}", flush=True)

    out = ROOT / "out/_probe_mix.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
