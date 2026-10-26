"""真实模型 smoke：加载三引擎、三档文本实跑、吞吐与标定耗时预估。

这是**开发脚本**，不属于 ``aidetect`` 包，也不会被 import。用途只有一个：
把「代码写完了但模型从没真跑过」这件事验证掉，并把 CPU 吞吐量记下来。

用法（在项目根目录 ai-detect 下）::

    python tools\\smoke.py

产物：
- stdout：引擎可用性/加载耗时/三档（120 / 900 / 3000 字）AI 与人类文本的逐单元分数、
  AI vs 人类 gap、3000 字单次 report 的引擎耗时、HC3 全量标定的 wall-clock 预估；
- ``out/smoke.json``：上面所有数字的结构化版本（版本、耗时、分数、gap、吞吐、预估）。

验收口径：三档 fixture 全部跑通无异常，且 clf 分数（AI 均值 - 人类均值）≥ 0.3。
"""
from __future__ import annotations

import gc
import importlib
import json
import math
import os
import platform
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许 `python tools\smoke.py` 直接跑，无需安装
    sys.path.insert(0, str(ROOT))

# 全程离线：权重已经在 models\ 下，禁止任何联网探测；顺带压掉加载进度条噪音。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from aidetect import __version__, aggregate, calibrate, config  # noqa: E402
from aidetect.engines.stylometry import FEATURE_KEYS as STYLO_KEYS  # noqa: E402
from aidetect.segment import (  # noqa: E402
    build_doc,
    char_count,
    make_units,
    split_sentences,
    window_texts,
)

SIZES = (120, 900, 3000)
CLF_KEYS = ("clf_mean", "clf_max", "clf_min", "clf_std", "clf_nwin")
PPL_KEYS = ("ppl", "meanlprob", "top10", "top100", "tail", "maxprob", "entropy",
            "lp_std", "lp_p10", "burst")
FEATURE_KEYS = {"clf": CLF_KEYS, "ppl": PPL_KEYS, "stylo": STYLO_KEYS}
# HC3 应答里偶尔以列表项/参考文献样式开头，会被 segment 当成标题剔掉，构造 fixture 时跳过。
HEADISH = re.compile(
    r"^\s*(?:\[\s*\d+\s*\]|参\s*考\s*文\s*献|摘\s*要|关\s*键\s*词|关键字|"
    r"结\s*论|引\s*言|前\s*言|讨\s*论)"
)


# --------------------------------------------------------------------------- 工具
def _num(x, nd: int = 4):
    """JSON 安全数字：NaN/inf → None，其余保留 nd 位。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _mean(values) -> float:
    vals = [v for v in values if v is not None and v == v]
    return st.fmean(vals) if vals else float("nan")


def _hr(title: str) -> None:
    print(f"\n== {title}", flush=True)


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            out[mod] = str(getattr(importlib.import_module(mod), "__version__", "?"))
        except Exception:  # noqa: BLE001 - 只做信息展示
            out[mod] = None
    return out


# --------------------------------------------------------------------------- fixture
def _pair_groups(samples: list) -> list[tuple[int, list, list]]:
    """按 qid 配对「同题的人类回答 / ChatGPT 回答」，保证 AI 与人类 fixture 主题可比。"""
    by_qid: dict[int, dict[int, list]] = {}
    for s in samples:
        by_qid.setdefault(s.qid, {0: [], 1: []})[s.label].append(s)
    groups = []
    for qid in sorted(by_qid):
        slot = by_qid[qid]
        if slot[0] and slot[1]:
            groups.append((qid, slot[0], slot[1]))
    return groups


def _trim_to(text: str, budget: int) -> str:
    """在句子边界把一段回答裁到 budget 字左右；裁不动（首句就超）时原样返回。"""
    if char_count(text) <= budget * 1.06:
        return text
    parts: list[str] = []
    used = 0
    for sent in split_sentences(text):
        n = char_count(sent.text)
        if parts and used + n > budget * 1.06:
            break
        parts.append(sent.text)
        used += n
    trimmed = "".join(parts)
    return trimmed if char_count(trimmed) >= min(60, budget * 0.6) else text


def _compose(groups: list, label: int, target: int) -> str:
    """从成对分组里按顺序取同 side 的回答拼成 target 字左右的文本（每回答一段）。"""
    parts: list[str] = []
    chars = 0
    for _qid, humans, ais in groups:
        text = "\n\n".join(s.text for s in (ais if label == 1 else humans))
        if not text or HEADISH.match(text):
            continue
        piece = _trim_to(text, target - chars if parts else target)
        if char_count(piece) < 24:
            continue
        parts.append(piece)
        chars += char_count(piece)
        if chars >= target:
            break
    return "\n\n".join(parts)


def _make_fixtures(groups: list) -> dict:
    fixtures: dict = {}
    for size in SIZES:
        for variant, label in (("ai", 1), ("human", 0)):
            text = _compose(groups, label, size)
            doc = build_doc(text, f"smoke-{variant}-{size}.txt")
            units = make_units(doc)
            if not units:
                raise SystemExit(f"fixture {size} 字 / {variant} 切分后没有可判定单元")
            fixtures[(size, variant)] = {
                "size": size, "variant": variant, "text": text,
                "doc": doc, "units": units, "chars": char_count(text),
            }
    return fixtures


# --------------------------------------------------------------------------- 引擎
# 两段长度接近的预热文本：引擎懒加载，第一次 score 的耗时里混着权重加载；
# 用另一段文本的首次 score 作为「纯推理」基线，两者相减才是加载耗时。
WARM_TEXT_A = ("这是一段用于测量模型加载耗时的预热文本，长度刚好足够形成一个判定单元。"
               "它的内容不重要，只用来触发一次真实前向计算。")
WARM_TEXT_B = ("另一段长度接近的预热文本，用来区分冷启动耗时与稳态推理耗时，"
               "内容同样不重要，只用来触发一次真实前向计算。")


def _warm_units() -> tuple[list, list]:
    return (make_units(build_doc(WARM_TEXT_A, "warm-a.txt")),
            make_units(build_doc(WARM_TEXT_B, "warm-b.txt")))


def _engine_meta(engines: list) -> list[dict]:
    """available() + 冷/热两次试探：冷启动含惰性加载，另一段文本的首次打分≈纯推理。"""
    units_a, units_b = _warm_units()
    meta: list[dict] = []
    for eng in engines:
        ok, why = eng.available()
        item = {
            "name": eng.name, "label": eng.label, "available": bool(ok), "note": why,
            "path": str(getattr(eng, "model_dir", "") or ""),
            "load_est_sec": None, "cold_total_sec": None,
            "first_score_sec": None, "steady_score_sec": None,
        }
        if ok:
            t0 = time.perf_counter()
            res_cold = eng.score(units_a)  # 第一次：含 tokenizer/权重惰性加载
            cold_total = time.perf_counter() - t0
            t1 = time.perf_counter()
            eng.score(units_b)  # 第二次（新文本、缓存未命中）：纯推理基线
            steady = time.perf_counter() - t1
            item["cold_total_sec"] = round(cold_total, 3)
            item["first_score_sec"] = round(res_cold.elapsed, 4)
            item["steady_score_sec"] = round(steady, 4)
            item["load_est_sec"] = round(max(0.0, cold_total - steady), 3)
        meta.append(item)
    return meta


def _score_fixture(engines: list, units: list) -> dict:
    out: dict = {}
    for eng in engines:
        t0 = time.perf_counter()
        res = eng.score(units)
        wall = time.perf_counter() - t0
        keys = FEATURE_KEYS.get(eng.name, tuple(res.features))
        out[eng.name] = {
            "available": res.available,
            "elapsed_sec": round(res.elapsed, 4),
            "wall_sec": round(wall, 4),
            "note": res.note,
            "scores": [_num(s) for s in res.scores],
            "features": {k: [_num(v) for v in res.features[k]] for k in keys if k in res.features},
        }
    return out


def _report(engines: list, doc) -> dict:
    """走完整 report 路径（融合 + 置信区间 + 句级证据）；引擎窗口缓存此时已热。"""
    t0 = time.perf_counter()
    analysis = aggregate.analyze(doc, engines=engines)
    wall = time.perf_counter() - t0
    return {
        "wall_sec": round(wall, 3),
        "analysis_elapsed_sec": round(analysis.elapsed, 3),
        "ai_rate_chars": _num(analysis.ai_rate_chars),
        "ai_rate_paras": _num(analysis.ai_rate_paras),
        "threshold": _num(analysis.threshold),
        "calibrated": analysis.recipe.calibrated,
        "unit_p": [_num(us.p) for us in analysis.unit_scores],
        "engine_elapsed_sec": {e["name"]: e["elapsed_sec"] for e in analysis.engine_info},
        "warnings": analysis.warnings,
    }


def _gaps(fixtures: dict, scored: dict) -> dict:
    gaps: dict = {}
    for size in SIZES:
        ai = scored[(size, "ai")]
        hu = scored[(size, "human")]
        clf_ai = _mean(ai["clf"]["features"].get("clf_mean", ai["clf"]["scores"]))
        clf_hu = _mean(hu["clf"]["features"].get("clf_mean", hu["clf"]["scores"]))
        ppl_ai = _mean(ai["ppl"]["features"].get("ppl", []))
        ppl_hu = _mean(hu["ppl"]["features"].get("ppl", []))
        top10_ai = _mean(ai["ppl"]["features"].get("top10", []))
        top10_hu = _mean(hu["ppl"]["features"].get("top10", []))
        gaps[str(size)] = {
            "clf_ai_mean": _num(clf_ai),
            "clf_human_mean": _num(clf_hu),
            "clf_gap": _num(clf_ai - clf_hu),
            "ppl_ai": _num(ppl_ai),
            "ppl_human": _num(ppl_hu),
            "ppl_gap_human_minus_ai": _num(ppl_hu - ppl_ai),
            "top10_ai": _num(top10_ai),
            "top10_human": _num(top10_hu),
            "top10_gap": _num(top10_ai - top10_hu),
            "stylo_gap": _num(_mean(ai["stylo"]["scores"]) - _mean(hu["stylo"]["scores"])),
            "ppl_score_gap": _num(_mean(ai["ppl"]["scores"]) - _mean(hu["ppl"]["scores"])),
        }
    return gaps


def _throughput(fixtures: dict) -> dict:
    """3000 字单次 report：全新引擎（只预热过加载用的短文本）+ 冷窗口缓存。"""
    gc.collect()
    fresh = aggregate.build_engines()
    units_a, units_b = _warm_units()
    reload_sec: dict = {}
    for eng in fresh:
        t0 = time.perf_counter()
        eng.score(units_a)  # 冷启动（含惰性加载）
        cold_total = time.perf_counter() - t0
        t1 = time.perf_counter()
        eng.score(units_b)  # 纯推理基线
        steady = time.perf_counter() - t1
        reload_sec[eng.name] = round(max(0.0, cold_total - steady), 3)

    fx = fixtures[(3000, "ai")]
    doc, units, text = fx["doc"], fx["units"], fx["text"]
    n_units = len(units)
    chars = char_count(text)
    clf_windows = sum(len(window_texts(u.text, int(config.MODEL_SPECS["clf"]["window_chars"])))
                      for u in units)
    ppl_windows = sum(len(window_texts(u.text, 240)) for u in units)

    t0 = time.perf_counter()
    analysis = aggregate.analyze(doc, engines=fresh)
    report_total = time.perf_counter() - t0
    info = {e["name"]: e for e in analysis.engine_info}

    clf_sec = float(info.get("clf", {}).get("elapsed_sec") or 0.0)
    ppl_sec = float(info.get("ppl", {}).get("elapsed_sec") or 0.0)
    stylo_sec = float(info.get("stylo", {}).get("elapsed_sec") or 0.0)
    engine_total = clf_sec + ppl_sec + stylo_sec

    def _per(sec: float, denom: float) -> float | None:
        return round(sec / denom, 4) if denom else None

    throughput = {
        "doc": {"variant": "ai", "size": 3000, "chars": chars, "units": n_units,
                "clf_windows": clf_windows, "ppl_windows": ppl_windows},
        "cache": "引擎窗口缓存为冷（同一批引擎只预热过加载用的短文本）",
        "reload_sec": reload_sec,
        "engines": {
            "clf": {"sec": clf_sec, "per_unit_sec": _per(clf_sec, n_units),
                    "per_window_sec": _per(clf_sec, clf_windows),
                    "per_char_ms": _per(clf_sec * 1000.0, chars)},
            "ppl": {"sec": ppl_sec, "per_unit_sec": _per(ppl_sec, n_units),
                    "per_window_sec": _per(ppl_sec, ppl_windows),
                    "per_char_ms": _per(ppl_sec * 1000.0, chars)},
            "stylo": {"sec": stylo_sec, "per_unit_sec": _per(stylo_sec, n_units),
                      "per_char_ms": _per(stylo_sec * 1000.0, chars)},
        },
        "engine_total_sec": round(engine_total, 2),
        "report_total_sec": round(report_total, 2),
        "second_first_report_sec": round(sum(reload_sec.values()) + report_total, 2),
        "reload_note": "本进程内第一批引擎已把权重读进页缓存，故第二批引擎的加载只剩页缓存读取；"
                       "真正的磁盘冷加载见 engines[].load_est_sec（首次加载）。",
        "note": "engine sec 取自 analysis.engine_info（与 calibrate.collect 同口径）；"
                "report_total 额外含句级证据（clf.score_sentences）+ 融合 + bootstrap + 切分。",
        "analysis": {
            "ai_rate_chars": _num(analysis.ai_rate_chars),
            "ai_rate_paras": _num(analysis.ai_rate_paras),
            "unit_p": [_num(us.p) for us in analysis.unit_scores],
        },
    }
    del fresh, analysis, info
    gc.collect()
    return throughput


def _profile(samples: list, limit: int | None) -> dict:
    """按 calibrate.run 的抽样口径统计 dev/test 的文档数/单元数/字数（不跑模型）。"""
    rng = random.Random(calibrate.SEED)
    dev = calibrate.select(samples, calibrate.DEV_DOMAINS, limit, rng)
    test = calibrate.select(samples, calibrate.TEST_DOMAINS, limit, rng)

    def _one(samples_part) -> dict:
        docs = units = chars = 0
        for s in samples_part:
            us = make_units(build_doc(s.text, "profile"))
            if not us:
                continue
            docs += 1
            units += len(us)
            chars += sum(u.chars for u in us)
        return {"docs": docs, "units": units, "chars": chars,
                "avg_chars_per_unit": round(chars / units, 1) if units else None}

    return {"dev": _one(dev), "test": _one(test)}


def _calibration_estimate(samples: list, throughput: dict, scored: dict) -> dict:
    """全量标定的 wall-clock 预估。

    用两次独立实测（吞吐取样 / 三档 fixture 原始打分取样，均为同一张 3000 字 AI
    文档、11 单元 / 3010 字）给出每单元、每字的耗时，再按 calibrate.run 的抽样口径
    乘 HC3 dev+test 规模，取极值给区间。
    """
    doc = throughput["doc"]
    n_units = doc["units"]
    chars = doc["chars"]
    t3000 = scored[(3000, "ai")]
    runs = {
        "throughput_cold": {k: throughput["engines"][k]["sec"] for k in ("clf", "ppl", "stylo")},
        "fixture_raw": {k: float(t3000[k]["elapsed_sec"]) for k in ("clf", "ppl", "stylo")},
    }
    per_engine = {
        run: {name: {"sec": round(sec, 3),
                     "per_unit_sec": round(sec / n_units, 4),
                     "per_char_ms": round(sec * 1000.0 / chars, 4)}
              for name, sec in times.items()}
        for run, times in runs.items()
    }
    rates = {run: {"sec_per_unit": sum(v["per_unit_sec"] for v in vals.values()),
                   "sec_per_char": sum(v["per_char_ms"] for v in vals.values()) / 1000.0}
             for run, vals in per_engine.items()}

    def _est(profile: dict) -> dict:
        units = profile["dev"]["units"] + profile["test"]["units"]
        chars_all = profile["dev"]["chars"] + profile["test"]["chars"]
        candidates = [units * r["sec_per_unit"] for r in rates.values()]
        candidates += [chars_all * r["sec_per_char"] for r in rates.values()]
        return {
            "n_units": units,
            "n_chars": chars_all,
            "minutes_lo": round(min(candidates) / 60.0, 1),
            "minutes_hi": round(max(candidates) / 60.0, 1),
            "minutes_mid": round(st.fmean(candidates) / 60.0, 1),
        }

    full = _est(_profile(samples, None))
    quick = _est(_profile(samples, 300))
    return {
        "dataset": "HC3-Chinese（calibrate.run：dev=finance/law/medicine，"
                   "test=open_qa/baike/psychology/nlpcc_dbqa）",
        "basis": "同一张 3000 字 AI 文档（11 单元 / 3010 字 / clf 15 窗口 / ppl 19 窗口）"
                 "的两次独立实测：throughput_cold（新引擎、冷缓存）与 fixture_raw（模型已热）",
        "per_engine": per_engine,
        "rates": rates,
        "full": full,
        "limit_300": quick,
        "note": "区间 = 两次实测的每单元/每字口径 × HC3 规模 的极值；HC3 单元平均 185 字，"
                "比 fixture 单元短，故区间偏保守；实际 wall-clock 另含一次性模型加载约数秒"
                "与切分耗时约 1s，可忽略。",
    }


# --------------------------------------------------------------------------- 打印
def _print_engines(meta: list[dict]) -> None:
    _hr("引擎加载")
    for m in meta:
        state = "OK" if m["available"] else "FAIL"
        load = "-" if m["load_est_sec"] is None else f"{m['load_est_sec']:.2f}s"
        cold = "-" if m["cold_total_sec"] is None else f"{m['cold_total_sec']:.2f}s"
        print(f"  [{m['name']:5s}] {state:4s} {m['label']}  冷启动 {cold}"
              f"（其中加载≈{load}）  路径 {m['path'] or '-'}")
        if m["note"]:
            print(f"          {m['note']}")


def _print_fixture(fx: dict, res: dict, report: dict) -> None:
    tag = {"ai": "AI  ", "human": "人类"}[fx["variant"]]
    print(f"\n  -- {fx['size']} 字 / {tag}：{len(fx['doc'].paras)} 段 -> "
          f"{len(fx['units'])} 单元 / 正文 {fx['doc'].body_chars} 字")
    clf_feat = res["clf"]["features"]
    ppl_feat = res["ppl"]["features"]
    sty_feat = res["stylo"]["features"]
    for i, unit in enumerate(fx["units"]):
        clf_v = clf_feat.get("clf_mean", [])[i] if i < len(clf_feat.get("clf_mean", [])) else None
        ppl_v = ppl_feat.get("ppl", [])[i] if i < len(ppl_feat.get("ppl", [])) else None
        top10_v = ppl_feat.get("top10", [])[i] if i < len(ppl_feat.get("top10", [])) else None
        cv_v = sty_feat.get("st_cv_slen", [])[i] if i < len(sty_feat.get("st_cv_slen", [])) else None
        conn_v = sty_feat.get("st_conn100", [])[i] if i < len(sty_feat.get("st_conn100", [])) else None
        sty_v = res["stylo"]["scores"][i] if i < len(res["stylo"]["scores"]) else None

        def _f(x, spec=".4f"):
            return format(float(x), spec) if x is not None else "-"

        print(f"     u{i:02d} {unit.chars:4d}字 | clf={_f(clf_v)} ppl={_f(ppl_v, '.2f')} "
              f"top10={_f(top10_v, '.2f')} cv={_f(cv_v, '.2f')} conn={_f(conn_v, '.2f')} "
              f"stylo={_f(sty_v, '.3f')}")
    print(f"     引擎：clf {res['clf']['elapsed_sec']:.2f}s  ppl {res['ppl']['elapsed_sec']:.2f}s  "
          f"stylo {res['stylo']['elapsed_sec']:.3f}s")
    print(f"     特征均值：clf_mean={_mean(clf_feat.get('clf_mean', [])):.4f} "
          f"clf_max={_mean(clf_feat.get('clf_max', [])):.4f} "
          f"ppl={_mean(ppl_feat.get('ppl', [])):.2f} "
          f"meanlprob={_mean(ppl_feat.get('meanlprob', [])):.3f} "
          f"top10={_mean(ppl_feat.get('top10', [])):.3f} "
          f"st_cv_slen={_mean(sty_feat.get('st_cv_slen', [])):.3f}")
    print(f"     report：AI 率(字数) {report['ai_rate_chars']:.1%}  "
          f"单元 P {report['unit_p']}  "
          f"阈值 {report['threshold']:.2f}  耗时 {report['wall_sec']:.2f}s")
    if report["warnings"]:
        print(f"     护栏：{report['warnings']}")


def _print_gaps(gaps: dict) -> None:
    _hr("AI vs 人类 gap（同题 HC3 样本拼文本，clf 目标 gap ≥ 0.30）")
    print("  档位   clf(AI)  clf(人)  clf_gap  ppl(AI)  ppl(人)  ppl差(人-AI)  "
          "top10(AI)  top10(人)  stylo_gap")
    for size in SIZES:
        g = gaps[str(size)]
        print(f"  {size:5d}  {g['clf_ai_mean']:.4f}   {g['clf_human_mean']:.4f}   "
              f"{g['clf_gap']:+.4f}   {g['ppl_ai']:7.2f}   {g['ppl_human']:7.2f}   "
              f"{g['ppl_gap_human_minus_ai']:+8.2f}     {g['top10_ai']:.3f}      "
              f"{g['top10_human']:.3f}    {g['stylo_gap']:+.4f}")


def _print_throughput(tp: dict, cal: dict) -> None:
    _hr("吞吐：3000 字单次 report")
    d = tp["doc"]
    print(f"  文档 {d['chars']} 字 / {d['units']} 单元 / clf 窗口 {d['clf_windows']} / "
          f"ppl 窗口 {d['ppl_windows']}（缓存：冷）")
    for name, label in (("clf", "判别模型"), ("ppl", "困惑度"), ("stylo", "文体统计")):
        e = tp["engines"][name]
        extra = ""
        if e.get("per_window_sec") is not None:
            extra = f"  每窗口 {e['per_window_sec']*1000:.0f}ms"
        print(f"  {label:6s} {e['sec']:7.2f}s  每单元 {e['per_unit_sec']:.3f}s  "
              f"每字 {e['per_char_ms']:.2f}ms{extra}")
    print(f"  引擎合计 {tp['engine_total_sec']:.2f}s；完整 report {tp['report_total_sec']:.2f}s"
          f"（含句级证据 + 融合 + bootstrap）")
    print(f"  冷机首次 report（磁盘冷加载≈{tp['cold_load_estimate_sec']:.2f}s + 报告 "
          f"{tp['report_total_sec']:.2f}s）：{tp['first_report_cold_sec']:.2f}s")
    print(f"  同进程二次 report（页缓存热，加载 {sum(tp['reload_sec'].values()):.2f}s）："
          f"{tp['second_first_report_sec']:.2f}s")

    _hr("全量标定 wall-clock 预估")
    f, q = cal["full"], cal["limit_300"]
    print(f"  全量（dev+test）：{f['n_units']} 单元 / {f['n_chars']} 字")
    print(f"    预估 {f['minutes_lo']:.1f} ~ {f['minutes_hi']:.1f} 分钟"
          f"（约 {f['minutes_lo'] / 60:.1f} ~ {f['minutes_hi'] / 60:.1f} 小时，"
          f"中位 {f['minutes_mid']:.1f} 分钟）")
    print(f"  快速版 --limit 300：{q['n_units']} 单元 / {q['n_chars']} 字")
    print(f"    预估 {q['minutes_lo']:.1f} ~ {q['minutes_hi']:.1f} 分钟")
    for run, vals in cal["rates"].items():
        print(f"    [{run}] 每单元 {vals['sec_per_unit']:.4f}s、"
              f"每字 {vals['sec_per_char'] * 1000:.3f}ms")


# --------------------------------------------------------------------------- 主流程
def main() -> int:
    t_start = time.perf_counter()
    print("ai-detect 真实模型 smoke")
    print(f"  项目根 {ROOT}")
    print(f"  平台 {platform.platform()} / {os.cpu_count()} 逻辑核")

    versions = _versions()
    print(f"  版本 {json.dumps(versions, ensure_ascii=False)}")

    # --- fixture：HC3 同题人类/ChatGPT 回答拼成 120 / 900 / 3000 字 -----------
    samples = calibrate.load_hc3()
    groups = _pair_groups(samples)
    print(f"  HC3 样本 {len(samples)} 条；同题成对分组 {len(groups)} 组")
    fixtures = _make_fixtures(groups)
    _hr("fixture 概览（HC3 同题样本拼接）")
    for (size, variant), fx in fixtures.items():
        tag = {"ai": "AI  ", "human": "人类"}[variant]
        print(f"  {size:5d} 字 / {tag}：正文 {fx['doc'].body_chars} 字 / "
              f"{len(fx['units'])} 单元 / {char_count(fx['text'])} 字")

    # --- 引擎加载 -------------------------------------------------------------
    engines = aggregate.build_engines()
    meta = _engine_meta(engines)
    _print_engines(meta)
    bad = [m["name"] for m in meta if not m["available"]]
    if bad:
        print(f"\nFAIL：引擎不可用 {bad}，先跑 python -m aidetect doctor")
        return 1
    try:
        import torch

        cpu_info = {"logical_cores": os.cpu_count(), "torch_threads": int(torch.get_num_threads())}
        print(f"\n  torch 线程 {cpu_info['torch_threads']}（AIDETECT_THREADS 可覆盖）")
    except Exception:  # noqa: BLE001
        cpu_info = {"logical_cores": os.cpu_count(), "torch_threads": None}

    # --- 三档 × AI/人类 实跑 --------------------------------------------------
    _hr("三档文本实跑（逐单元分数与关键特征）")
    scored: dict = {}
    reports: dict = {}
    for size in SIZES:
        for variant in ("ai", "human"):
            fx = fixtures[(size, variant)]
            res = _score_fixture(engines, fx["units"])
            rep = _report(engines, fx["doc"])
            scored[(size, variant)] = res
            reports[(size, variant)] = rep
            _print_fixture(fx, res, rep)

    gaps = _gaps(fixtures, scored)
    _print_gaps(gaps)
    gap_ok = all(gaps[str(s)]["clf_gap"] is not None and gaps[str(s)]["clf_gap"] >= 0.3
                 for s in SIZES)
    print(f"\n  验收 clf gap ≥ 0.30：{'通过' if gap_ok else '未通过'}")

    # --- 吞吐 + 标定预估 ------------------------------------------------------
    throughput = _throughput(fixtures)
    cold_load = sum((m.get("load_est_sec") or 0.0) for m in meta)
    throughput["cold_load_estimate_sec"] = round(cold_load, 2)
    throughput["first_report_cold_sec"] = round(cold_load + throughput["report_total_sec"], 2)
    calibration_estimate = _calibration_estimate(samples, throughput, scored)
    _print_throughput(throughput, calibration_estimate)

    # --- 落盘 ----------------------------------------------------------------
    fixture_rows = []
    for size in SIZES:
        for variant in ("ai", "human"):
            fx = fixtures[(size, variant)]
            fixture_rows.append({
                "size": size,
                "variant": variant,
                "chars": fx["chars"],
                "body_chars": fx["doc"].body_chars,
                "paragraphs": len(fx["doc"].paras),
                "units": len(fx["units"]),
                "unit_chars": [u.chars for u in fx["units"]],
                "engines": scored[(size, variant)],
                "report": reports[(size, variant)],
            })

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": "ai-detect",
        "tool_version": __version__,
        "script": "tools/smoke.py",
        "platform": platform.platform(),
        "versions": versions,
        "offline": {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE")},
        "cpu": cpu_info,
        "engines": meta,
        "fixtures": fixture_rows,
        "gaps": gaps,
        "acceptance": {"clf_gap_ge_0_3": bool(gap_ok),
                       "all_fixtures_ran": True,
                       "elapsed_sec": round(time.perf_counter() - t_start, 1)},
        "throughput": throughput,
        "calibration_estimate": calibration_estimate,
    }
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUT_DIR / "smoke.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                        encoding="utf-8")
    _hr("产物")
    print(f"  {out_path}")
    print(f"  smoke 总耗时 {payload['acceptance']['elapsed_sec']}s")
    print("\nSMOKE OK" if gap_ok else "\nSMOKE FAIL（clf gap 未达 0.30）")
    return 0 if gap_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
