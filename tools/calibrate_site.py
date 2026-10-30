"""生成站点对照轨的映射表 `resources/site_map.json`。

方法（anchor-ratio fitting）：
1. 取一篇已被 PaperPass 判过的论文（`B:\\论文\\生物医学工程专业导论论文_刘颐刚.docx`），
   用与 `aidetect/site.py` 完全相同的切窗方式（body 句子拼成 ≤240 字窗口）打分；
2. 按站点报告公布的档位字数占比（高疑似 48.4%、中 13.72%、低 4.37%，合计疑似 66.49%），
   反解出我们 zhv3 原始分上的三条档位线；
3. 再解顶端锚点，使「综合分」（字数加权平均）对齐站点的 54.98；
4. 把同一映射套到人类文本（HC3 人类回答 / 新闻）上，测出激进口径的误报，
   写进 meta 随报告披露。

为什么不用 Q-Q 直接匹配分位数：zhv3 对短句的分数是双峰（多数接近 0、少量接近 1），
句子级 Q-Q 会把档位线压到 0.03 这种离谱位置，人类文本 86% 被判疑似。
"""
from __future__ import annotations

import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402
from aidetect import site as site_mod  # noqa: E402
from aidetect.engines.classifier import Zhv3ClassifierEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANX_CSV = ROOT / "data/anx/test.csv"
ANCHOR_DOC = Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx")
OUT_PATH = ROOT / "aidetect/resources/site_map.json"

BANDS = {"high": 70.0, "mid_high": 60.0, "mid": 50.0}
WINDOW_CHARS = 240

SITE_META = {
    "site": "PaperPass（免费版 AIGC 检测报告）",
    "anchor_doc": "生物医学工程专业导论论文_刘颐刚.docx",
    "site_score": 54.98,
    "site_suspected_ratio": 66.49,
    "site_high_ratio": 48.4,
    "site_mid_ratio": 13.72,
    "site_low_ratio": 4.37,
}


def doc_windows(path: Path) -> list[site_mod._Win]:
    text, _ = textio.load(str(path))
    doc = segment.build_doc(text, path.stem)
    items = site_mod._body_sentences(doc)
    wins, _ids = site_mod._make_windows(items, WINDOW_CHARS)
    return wins


def window_scores(eng, wins) -> list[tuple[float, int]]:
    if not wins:
        return []
    res = eng.score(wins)
    return [(float(s), len(w.text)) for s, w in zip(res.scores, wins)]


def line_for_share(pairs: list[tuple[float, int]], share: float) -> float:
    """找一条线，使「分数 >= 线」的字数占比 ≈ share（从高往低累加）。"""
    rows = sorted(pairs, key=lambda t: t[0], reverse=True)
    total = sum(w for _, w in rows) or 1
    target = max(0.0, min(1.0, share)) * total
    acc = 0.0
    for value, weight in rows:
        acc += weight
        if acc >= target:
            return float(value)
    return 0.0


def share_at(pairs: list[tuple[float, int]], line: float) -> float:
    total = sum(w for _, w in pairs) or 1
    return round(sum(w for s, w in pairs if s >= line) / total, 4)


def map_score(p: float, anchors: list[list[float]]) -> float:
    if not anchors:
        return max(0.0, min(100.0, p * 100.0))
    if p <= anchors[0][0]:
        return float(anchors[0][1])
    if p >= anchors[-1][0]:
        return float(anchors[-1][1])
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= p <= x1:
            if x1 - x0 <= 1e-9:
                return float(y1)
            return float(y0 + (p - x0) * (y1 - y0) / (x1 - x0))
    return float(anchors[-1][1])


def ratios(pairs: list[tuple[float, int]], anchors: list[list[float]]) -> tuple[dict, float]:
    total = sum(w for _, w in pairs) or 1
    buckets = {"high": 0.0, "mid_high": 0.0, "mid": 0.0, "none": 0.0}
    weighted = 0.0
    for p, w in pairs:
        score = map_score(p, anchors)
        weighted += score * w
        if score >= BANDS["high"]:
            buckets["high"] += w
        elif score >= BANDS["mid_high"]:
            buckets["mid_high"] += w
        elif score >= BANDS["mid"]:
            buckets["mid"] += w
        else:
            buckets["none"] += w
    out = {k: round(v / total, 4) for k, v in buckets.items()}
    out["suspected"] = round(min(1.0, out["high"] + out["mid_high"] + out["mid"]), 4)
    return out, round(weighted / total, 2)


def load_anx(source: str, label: str, per: int) -> list[str]:
    texts: list[str] = []
    with ANX_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("source") or "") != source or (row.get("label") or "") != label:
                continue
            t = (row.get("text") or "").strip()
            if len(t) >= 300:
                texts.append(t)
    random.Random(20241025).shuffle(texts)
    return texts[:per]


def human_reference(eng, anchors, source: str, label: str, per: int, name: str) -> dict:
    susp_chars = 0.0
    high_chars = 0.0
    total_chars = 0.0
    doc_susp: list[float] = []
    for text in load_anx(source, label, per):
        doc = segment.build_doc(text, name)
        items = site_mod._body_sentences(doc)
        wins, _ids = site_mod._make_windows(items, WINDOW_CHARS)
        pairs = window_scores(eng, wins)
        if not pairs:
            continue
        r, _head = ratios(pairs, anchors)
        n = sum(w for _, w in pairs)
        total_chars += n
        susp_chars += r["suspected"] * n
        high_chars += r["high"] * n
        doc_susp.append(r["suspected"])
    if total_chars <= 0:
        return {}
    return {
        "docs": len(doc_susp),
        "chars": int(total_chars),
        # 语料级：被判疑似/高危的字数占比（更能代表「误报面积」）
        "corpus_suspected_share": round(susp_chars / total_chars, 4),
        "corpus_high_share": round(high_chars / total_chars, 4),
        # 文档级均值（短文本容易 0/1 极化，仅供对照）
        "doc_mean_suspected": round(sum(doc_susp) / len(doc_susp), 4),
    }


def main() -> int:
    eng = Zhv3ClassifierEngine()
    ok, why = eng.available()
    if not ok:
        print("zhv3 不可用：", why)
        return 3

    anchor = window_scores(eng, doc_windows(ANCHOR_DOC))
    if not anchor:
        print("锚点文档没有可用窗口")
        return 4
    total_w = sum(w for _, w in anchor) or 1
    print(f"锚点文档窗口数 {len(anchor)}，字数和 {total_w}", flush=True)

    # 站点公布的档位占比 → 三个累计占比 → 反解档位线
    high_share = SITE_META["site_high_ratio"] / 100.0
    cum_mid_high = high_share + SITE_META["site_mid_ratio"] / 100.0
    cum_mid = cum_mid_high + SITE_META["site_low_ratio"] / 100.0
    p_high = line_for_share(anchor, high_share)
    p_mid_high = line_for_share(anchor, cum_mid_high)
    p_mid = line_for_share(anchor, cum_mid)
    p_high = max(p_high, 0.0)
    p_mid_high = min(max(p_mid_high, 0.0), max(0.0, p_high - 1e-6))
    p_mid = min(max(p_mid, 0.0), max(0.0, p_mid_high - 1e-6))
    print(
        f"档位线（zhv3 原始分）：high {p_high:.6f} / mid_high {p_mid_high:.6f} / mid {p_mid:.6f}",
        flush=True,
    )

    def mean_for(y_top: float) -> float:
        anchors = [[0.0, 0.0], [p_mid, 50.0], [p_mid_high, 60.0], [p_high, 70.0], [1.0, y_top]]
        return sum(map_score(p, anchors) * w for p, w in anchor) / total_w

    target_head = float(SITE_META["site_score"])
    m70, m100 = mean_for(70.0), mean_for(100.0)
    y_top = 70.0 + (target_head - m70) / (m100 - m70) * 30.0 if abs(m100 - m70) > 1e-9 else 100.0
    y_top = round(max(70.0, min(100.0, y_top)), 2)
    anchors: list[list[float]] = [
        [0.0, 0.0],
        [round(p_mid, 6), 50.0],
        [round(p_mid_high, 6), 60.0],
        [round(p_high, 6), 70.0],
        [1.0, y_top],
    ]

    got, head = ratios(anchor, anchors)
    print("锚点文档映射后：", json.dumps(got, ensure_ascii=False), f"综合分 {head}", flush=True)
    print(
        "站点报告：      "
        '{"high": 0.484, "mid_high": 0.1372, "mid": 0.0437, "suspected": 0.6649, "score": 54.98}',
        flush=True,
    )

    human_ref: dict[str, dict] = {}
    for name, source, label in (("hc3_human", "hc3_human", "0"), ("thucnews_human", "thucnews", "0")):
        ref = human_reference(eng, anchors, source, label, 30, name)
        if ref:
            human_ref[name] = ref
            print(f"人类参考 {name}: {json.dumps(ref, ensure_ascii=False)}", flush=True)

    payload = {
        "version": 1,
        "method": (
            "anchor-ratio fitting：按站点公布的档位字数占比反解 zhv3 档位线，"
            "再解顶端锚点使综合分对齐站点；窗口 240 字，与 aidetect/site.py 同一切法"
        ),
        "engine": "zhv3",
        "bands": BANDS,
        "anchors": anchors,
        "band_lines_p": {
            "high": round(p_high, 6),
            "mid_high": round(p_mid_high, 6),
            "mid": round(p_mid, 6),
        },
        "meta": {
            **SITE_META,
            "source": "PaperPass 免费版 AIGC 检测报告（锚点文档：生物医学工程专业导论论文_刘颐刚.docx）",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "anchor_windows": len(anchor),
            "anchor_chars": total_w,
            "anchor_after_map": {**got, "headline": head},
            "human_reference": human_ref,
            "caveat": (
                "站点对照档是与商业网站对齐的激进口径：档位线由「站点在锚点文档上公布的占比」反解而来，"
                "对人类正式文本（新闻等）的误报显著高于保守口径；不要用它作为学术不端证据。"
            ),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
