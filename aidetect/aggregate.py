"""汇总层：把单元级 P(AI) 变成文档级结论（AI 率、置信区间、可疑句、风险提示）。

口径说明（报告里也会写）：
- **AI 生成占比（字数口径）** = Σ(单元字数 × P(AI)) / Σ正文。连续值，不是「判为 AI 的字数」。
- **AI 段落占比** = 判定为 AI（P ≥ 当前模式阈值）的单元数 / 单元总数。
- 置信区间用单元层面的 bootstrap（1000 次，固定种子，可复现），因为不确定性来自「哪些段落像 AI」。
"""
from __future__ import annotations

import random
import statistics as st
import time
from dataclasses import dataclass, field

from . import config, fusion
from .engines.base import EngineResult
from .engines.classifier import ClassifierEngine
from .engines.perplexity import PerplexityEngine
from .engines.stylometry import StylometryEngine, explain as stylo_explain
from .segment import Doc, Para, Unit, make_units, stats as doc_stats


@dataclass
class UnitScore:
    unit: Unit
    p: float
    grade: str
    engine_scores: dict[str, float] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)

    @property
    def is_ai(self) -> bool:
        return self.grade in ("high", "mid_high")


@dataclass
class ParagraphRow:
    index: int
    text: str
    chars: int
    kind: str
    p: float | None
    grade: str
    unit_index: int | None
    merged: bool
    signals: list[str] = field(default_factory=list)


@dataclass
class SentenceRow:
    text: str
    p: float
    unit_index: int
    para_index: int


@dataclass
class Analysis:
    doc: Doc
    stats: dict
    units: list[Unit]
    unit_scores: list[UnitScore]
    paragraphs: list[ParagraphRow]
    sentences: list[SentenceRow]
    ai_rate_chars: float
    ai_rate_paras: float
    ai_rate_ci: tuple[float, float] | None
    mode: str
    threshold: float
    grade_counts: dict[str, int]
    engine_info: list[dict]
    recipe: fusion.Recipe
    warnings: list[str]
    elapsed: float
    cutoffs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": "ai-detect",
            "version": __import__("aidetect").__version__,
            "document": self.stats,
            "mode": self.mode,
            "threshold": round(self.threshold, 4),
            "grade_cutoffs": {k: round(float(v), 4) for k, v in self.cutoffs.items()},
            "ai_rate_chars": round(self.ai_rate_chars, 4),
            "ai_rate_paras": round(self.ai_rate_paras, 4),
            "ai_rate_ci": [round(self.ai_rate_ci[0], 4), round(self.ai_rate_ci[1], 4)]
            if self.ai_rate_ci else None,
            "grade_counts": self.grade_counts,
            "warnings": self.warnings,
            "calibration": {
                "calibrated": self.recipe.calibrated,
                "version": self.recipe.version,
                "note": self.recipe.note,
                "mode_thresholds": self.recipe.mode_thresholds,
                "evaluation": self.recipe.evaluation,
            },
            "engines": self.engine_info,
            "paragraphs": [
                {
                    "index": r.index,
                    "chars": r.chars,
                    "kind": r.kind,
                    "p_ai": None if r.p is None else round(r.p, 4),
                    "grade": r.grade,
                    "unit": r.unit_index,
                    "merged": r.merged,
                    "signals": r.signals,
                    "text": r.text,
                }
                for r in self.paragraphs
            ],
            "suspicious_sentences": [
                {"text": s.text, "p_ai": round(s.p, 4), "para": s.para_index}
                for s in self.sentences
            ],
            "elapsed_sec": round(self.elapsed, 2),
        }


def build_engines() -> list:
    return [ClassifierEngine(), PerplexityEngine(), StylometryEngine()]


def analyze(doc: Doc, mode: str = config.DEFAULT_MODE, engines: list | None = None,
            max_sentences: int = 20, seed: int = 20241025) -> Analysis:
    t0 = time.time()
    engines = engines if engines is not None else build_engines()
    units = make_units(doc)
    results: list[EngineResult] = [eng.score(units) for eng in engines]
    recipe = fusion.load_recipe()
    probs, rows = fusion.fuse(units, results, recipe)

    mode = mode if mode in recipe.mode_thresholds else config.DEFAULT_MODE
    threshold = float(recipe.mode_thresholds.get(mode, 0.5))
    cutoffs = fusion.grade_cutoffs_for(threshold)

    unit_scores: list[UnitScore] = []
    for i, (unit, p) in enumerate(zip(units, probs)):
        es = {res.name: res.scores[i] for res in results if len(res.scores) == len(units)}
        signals = _signals(rows[i], p, threshold)
        unit_scores.append(UnitScore(unit=unit, p=p, grade=fusion.grade_of(p, cutoffs),
                                     engine_scores=es, signals=signals))

    # 段落行（含标题/被剔除区，便于用户定位）
    paras: list[ParagraphRow] = []
    for para in doc.paras:
        us = next((u for u in unit_scores if para.index in u.unit.para_indexes), None)
        if para.is_scorable and us is not None:
            paras.append(ParagraphRow(
                index=para.index, text=para.text, chars=para.chars, kind=para.kind,
                p=us.p, grade=us.grade, unit_index=us.unit.index,
                merged=len(us.unit.paras) > 1, signals=us.signals,
            ))
        else:
            paras.append(ParagraphRow(
                index=para.index, text=para.text, chars=para.chars, kind=para.kind,
                p=None, grade="excluded", unit_index=None, merged=False,
            ))

    # 文档级 AI 率（字数加权）
    total = sum(us.unit.chars for us in unit_scores) or 1
    ai_rate_chars = sum(us.p * us.unit.chars for us in unit_scores) / total
    ai_units = [us for us in unit_scores if us.p >= threshold]
    ai_rate_paras = len(ai_units) / len(unit_scores) if unit_scores else 0.0
    ci = _bootstrap_ci(unit_scores, rounds=config.BOOTSTRAP_ROUNDS, seed=seed)

    grade_counts = {k: 0 for k in ("high", "mid_high", "mid", "low")}
    for us in unit_scores:
        grade_counts[us.grade] = grade_counts.get(us.grade, 0) + 1

    # 可疑句：只对已判为 AI 的单元做句级细化
    sentences = _suspicious_sentences(unit_scores, threshold, engines, max_sentences)

    engine_info = [
        {
            "name": res.name,
            "label": res.label,
            "available": res.available,
            "note": res.note,
            "mean_score": round(res.mean_score(), 4) if res.available else None,
            "elapsed_sec": round(res.elapsed, 2),
        }
        for res in results
    ]
    warnings = _warnings(doc, unit_scores, recipe, results, threshold)
    stats = doc_stats(doc)
    stats["units"] = len(units)

    return Analysis(
        doc=doc, stats=stats, units=units, unit_scores=unit_scores,
        paragraphs=paras, sentences=sentences,
        ai_rate_chars=ai_rate_chars, ai_rate_paras=ai_rate_paras, ai_rate_ci=ci,
        mode=mode, threshold=threshold, grade_counts=grade_counts,
        engine_info=engine_info, recipe=recipe, warnings=warnings,
        elapsed=time.time() - t0, cutoffs=cutoffs,
    )


def _signals(row: dict, p: float, threshold: float) -> list[str]:
    """给每个单元生成「为什么像/不像 AI」的一句话解释。"""
    out: list[str] = []
    stylo_row = {k: v for k, v in row.items() if k.startswith("st_")}
    if stylo_row:
        out.extend(stylo_explain(stylo_row))
    cmax = row.get("clf_max")
    cmean = row.get("clf_mean")
    if cmax is not None and cmean is not None and cmax - cmean > 0.30:
        out.append(f"段落内窗口分歧大（最高 {cmax:.2f} / 均值 {cmean:.2f}），可能有拼接或人工改写")
    rel = row.get("ppl_rel")
    if rel is not None and rel > 1.0:
        out.append("流畅度明显高于全文平均（更像模型生成）")
    elif rel is not None and rel < -1.0:
        out.append("流畅度明显低于全文平均（更像人工写作或专业表述）")
    if p < threshold and not out:
        out.append("未触发明显 AI 特征")
    return out[:4]


def _bootstrap_ci(unit_scores: list[UnitScore], rounds: int = 1000,
                  seed: int = 0) -> tuple[float, float] | None:
    if len(unit_scores) < 3:
        return None
    rng = random.Random(seed)
    n = len(unit_scores)
    chars = [u.unit.chars for u in unit_scores]
    ps = [u.p for u in unit_scores]
    rates: list[float] = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        tot = sum(chars[i] for i in idx) or 1
        rates.append(sum(ps[i] * chars[i] for i in idx) / tot)
    rates.sort()
    lo = rates[int(0.025 * rounds)]
    hi = rates[min(rounds - 1, int(0.975 * rounds))]
    return (lo, hi)


def _suspicious_sentences(unit_scores: list[UnitScore], threshold: float,
                          engines: list, limit: int) -> list[SentenceRow]:
    clf = next((e for e in engines if isinstance(e, ClassifierEngine)), None)
    if clf is not None and not clf.available()[0]:
        clf = None
    rows: list[SentenceRow] = []
    for us in unit_scores:
        if us.p < threshold:
            continue
        sents = [s.text for p in us.unit.paras for s in p.sentences]
        if not sents:
            continue
        if clf is not None:
            probs = clf.score_sentences(sents)
        else:
            probs = [us.p] * len(sents)
        offset = 0
        for para in us.unit.paras:
            para_sents = [s.text for s in para.sentences]
            chunk = probs[offset:offset + len(para_sents)]
            offset += len(para_sents)
            for sent, p in zip(para_sents, chunk):
                if p >= max(0.70, threshold):
                    rows.append(SentenceRow(text=sent, p=p, unit_index=us.unit.index,
                                            para_index=para.index))
        if len(rows) >= limit * 4:
            break
    rows.sort(key=lambda r: -r.p)
    return rows[:limit]


def _script_ratio_chinese(doc: Doc) -> tuple[float, int]:
    """返回 (汉字占「汉字+拉丁字母」的比例, 计数总量)。

    用于非中文文本护栏：模型与标定只覆盖中文，非中文正文的分数没有意义。
    """
    text = doc.raw_text or ""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + latin
    if total <= 0:
        return 0.0, 0
    return cjk / total, total


def _warnings(doc: Doc, unit_scores: list[UnitScore], recipe: fusion.Recipe,
              results: list[EngineResult], threshold: float) -> list[str]:
    warns: list[str] = []
    cjk_ratio, script_total = _script_ratio_chinese(doc)
    if script_total >= 50 and cjk_ratio < 0.5:
        warns.append(
            f"正文中汉字只占 {cjk_ratio:.0%}（非中文内容为主）：本工具的模型与标定只针对中文，"
            "非中文/多语混排文本的分数不可用，请勿引用。"
        )
    if doc.body_chars < config.MIN_RELIABLE_CHARS:
        warns.append(
            f"正文仅 {doc.body_chars} 字，低于 {config.MIN_RELIABLE_CHARS} 字的可靠性下限，"
            "判定波动很大，建议补充文本后再测。"
        )
    if not recipe.calibrated:
        warns.append(
            "当前未加载标定参数（calibration.json），分数来自启发式加权，仅可用于相对比较，"
            "不应作为正式结论；请先运行 python -m aidetect calibrate。"
        )
    else:
        warns.append(
            "判定阈值是在 HC3 的财经/法律/医学三种人类文本上标定的。若待检文本文体不同"
            "（如问答、百科、口语化写作），人类文本被误判的比例会升高（域外实测误报率高于目标值）。"
            "重要场合请用 --mode strict 复核，并以逐段结果和可疑句为准，不要只看整篇百分比。"
        )
    missing = [r.label for r in results if not r.available]
    if missing:
        warns.append(f"以下引擎不可用，已降级运行：{'、'.join(missing)}。结果可靠性下降。")
    if doc.excluded_chars and doc.total_chars:
        ratio = doc.excluded_chars / max(doc.total_chars, 1)
        if ratio > 0.25:
            warns.append(
                f"已剔除的参考文献/元信息占全文 {ratio:.0%}，正文样本偏少，结论代表性有限。"
            )
    if len(unit_scores) >= 5:
        sd = st.pstdev([u.p for u in unit_scores])
        if sd > 0.28:
            warns.append(
                f"段落间判定分歧较大（标准差 {sd:.2f}），可能存在混合创作（人工 + AI）或风格差异，"
                "建议看逐段结果而不是只看整篇百分比。"
            )
    return warns
