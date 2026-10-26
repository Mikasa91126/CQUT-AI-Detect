"""只读统计脚本：HC3-Chinese 各域样本量 + 各 --limit 候选的 dev/test 单元数。

不加载任何模型、不联网，只复现 calibrate.py 里「选样 → 合并 → 切分」这段纯文本逻辑，
用来在正式标定前推算 --limit 候选（120/200/300）对应的实际单元数 n_dev / n_test，
再结合吞吐实测（tools/smoke.py）估 wall-clock。

用法：
    python -u -m eval.hc3_stats            # 在 ai-detect 目录下运行
    python -u eval/hc3_stats.py
输出：
    eval/hc3_stats.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aidetect import calibrate, config  # noqa: E402
from aidetect.segment import build_doc, make_units, split_sentences  # noqa: E402

LIMITS = (120, 150, 200, 300, None)
CLF_WINDOW = 300      # classifier spec window_chars
PPL_WINDOW = 240      # PerplexityEngine.MAX_WINDOW


def raw_counts() -> tuple[dict, dict]:
    """统计 all.jsonl 原始字段量（问题数、每域 human/AI 回答数、≥120 字回答数）。"""
    path = config.DATA_DIR / "hc3zh" / "all.jsonl"
    per_dom: dict[str, dict] = {}
    totals = {"questions": 0, "human": 0, "ai": 0, "human_keep": 0, "ai_keep": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dom = str(rec.get("source", "unknown"))
            d = per_dom.setdefault(dom, {"questions": 0, "human": 0, "ai": 0,
                                         "human_keep": 0, "ai_keep": 0,
                                         "human_chars": 0, "ai_chars": 0})
            d["questions"] += 1
            totals["questions"] += 1
            for label, key, side in ((0, "human_answers", "human"), (1, "chatgpt_answers", "ai")):
                for ans in rec.get(key) or []:
                    text = (ans or "").strip()
                    d[side] += 1
                    totals[side] += 1
                    if len(text) >= 120:
                        d[side + "_keep"] += 1
                        totals[side + "_keep"] += 1
                        d[side + "_chars"] += len(text)
    return per_dom, totals


def unit_plan(samples, domains, limit, rng):
    """复现 calibrate.select + collect 的切分口径，统计文档数/单元数/推理窗口数。"""
    picked = calibrate.select(samples, tuple(domains), limit, rng)
    docs = 0
    units = 0
    chars = []
    one_unit_docs = 0
    empty_docs = 0
    clf_windows: dict[str, int] = {}      # 文本 -> 出现次数（用于去重后推理次数）
    ppl_windows: dict[str, int] = {}
    sent_jobs = 0                          # burstiness 的句级窗口总数（≤24 句/单元）
    per_dom_docs: dict[str, int] = {}
    per_dom_units: dict[str, int] = {}
    per_dom_label_units: dict[str, dict[str, int]] = {}
    label_units = {"human": 0, "ai": 0}
    for s in picked:
        try:
            doc = build_doc(s.text, f"hc3-{s.qid}-{s.label}")
        except ValueError:
            empty_docs += 1
            continue
        us = make_units(doc)
        if not us:
            empty_docs += 1
            continue
        docs += 1
        per_dom_docs[s.domain] = per_dom_docs.get(s.domain, 0) + 1
        if len(us) == 1:
            one_unit_docs += 1
        units += len(us)
        per_dom_units[s.domain] = per_dom_units.get(s.domain, 0) + len(us)
        side = "ai" if s.label == 1 else "human"
        label_units[side] += len(us)
        per_dom_label_units.setdefault(s.domain, {"human": 0, "ai": 0})[side] += len(us)
        for u in us:
            chars.append(u.chars)
            for w in _windows(u.text, CLF_WINDOW):
                clf_windows[w] = clf_windows.get(w, 0) + 1
            pw = _windows(u.text, PPL_WINDOW)
            for w in pw:
                ppl_windows[w] = ppl_windows.get(w, 0) + 1
            nsents = len([x for x in split_sentences(u.text) if len(x.text.strip()) >= 6])
            sent_jobs += min(nsents, 24)
    chars.sort()

    def pct(p: float) -> int:
        if not chars:
            return 0
        return chars[min(len(chars) - 1, int(p * len(chars)))]

    return {
        "limit": limit,
        "n_docs": docs,
        "n_units": units,
        "n_docs_one_unit": one_unit_docs,
        "n_docs_empty": empty_docs,
        "unit_chars_mean": round(sum(chars) / len(chars), 1) if chars else 0,
        "unit_chars_p50": pct(0.50),
        "unit_chars_p90": pct(0.90),
        "unit_chars_max": chars[-1] if chars else 0,
        "per_domain_docs": per_dom_docs,
        "per_domain_units": per_dom_units,
        "per_domain_label_units": per_dom_label_units,
        "label_units": label_units,
        "n_human_units": label_units["human"],
        "n_ai_units": label_units["ai"],
        # 引擎实际推理次数（TextCache 按窗口文本去重）
        "clf_forwards_unique": len(clf_windows),
        "clf_forwards_total": sum(clf_windows.values()),
        "ppl_forwards_unique": len(ppl_windows),
        "ppl_forwards_total": sum(ppl_windows.values()),
        "ppl_sentence_forwards": sent_jobs,
        "ppl_forwards_est": len(ppl_windows) + sent_jobs,
    }


def _windows(text: str, max_chars: int) -> list[str]:
    from aidetect.segment import window_texts
    return window_texts(text, max_chars)


def main() -> int:
    t0 = time.time()
    path = config.DATA_DIR / "hc3zh" / "all.jsonl"
    print(f"数据文件：{path}（{path.stat().st_size/1e6:.1f} MB）", flush=True)
    per_dom, totals = raw_counts()
    print("原始字段统计：", flush=True)
    for dom, d in sorted(per_dom.items()):
        print(f"  {dom:<12} 问题 {d['questions']:>5}  human {d['human']:>6} (≥120字 {d['human_keep']:>5})"
              f"  AI {d['ai']:>6} (≥120字 {d['ai_keep']:>5})", flush=True)
    print(f"  合计 问题 {totals['questions']} human {totals['human']} AI {totals['ai']}", flush=True)

    samples = calibrate.load_hc3()
    print(f"load_hc3 后样本 {len(samples)} 条（len≥120 且截断到 900 字）", flush=True)

    results = {"raw": {"per_domain": per_dom, "totals": totals},
               "n_samples": len(samples), "limits": {}}
    for limit in LIMITS:
        # 必须和 calibrate.run() 完全一致的 rng 调用顺序：先 dev 后 test，同一个 rng
        rng = random.Random(calibrate.SEED)
        dev = unit_plan(samples, calibrate.DEV_DOMAINS, limit, rng)
        test = unit_plan(samples, calibrate.TEST_DOMAINS, limit, rng)
        results["limits"][str(limit)] = {"dev": dev, "test": test}
        print(f"\n--limit {limit}：n_dev={dev['n_units']} 单元 / {dev['n_docs']} 篇；"
              f"n_test={test['n_units']} 单元 / {test['n_docs']} 篇", flush=True)
        print(f"  dev  推理：clf 去重窗口 {dev['clf_forwards_unique']}，"
              f"ppl 窗口 {dev['ppl_forwards_unique']} + 句级 {dev['ppl_sentence_forwards']}", flush=True)
        print(f"  test 推理：clf 去重窗口 {test['clf_forwards_unique']}，"
              f"ppl 窗口 {test['ppl_forwards_unique']} + 句级 {test['ppl_sentence_forwards']}", flush=True)
        print(f"  dev 单元字数 p50={dev['unit_chars_p50']} p90={dev['unit_chars_p90']} "
              f"mean={dev['unit_chars_mean']}", flush=True)
        print(f"  dev 人类单元 {dev['n_human_units']} / AI 单元 {dev['n_ai_units']}；"
              f"test 人类单元 {test['n_human_units']} / AI 单元 {test['n_ai_units']}", flush=True)

    out = Path(__file__).with_name("hc3_stats.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {out}（{time.time()-t0:.1f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
