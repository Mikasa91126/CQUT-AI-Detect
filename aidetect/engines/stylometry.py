"""文体计量引擎：不依赖任何模型，纯统计特征。三个作用：

1. 模型缺失时的降级引擎（仍能给方向性判断，报告会明确标注「未用神经模型」）；
2. 给融合层提供与判别模型互补的信号（判别模型看措辞，这里看结构均匀度）；
3. 完全可解释 —— 每个子分都能用一句话讲清楚，便于在报告里向用户交代。

启发式分由四个子分加权而成（权重在代码里写死并记录在报告「方法」区块）：

    score = 0.35*u1 + 0.25*u2 + 0.25*u3 + 0.15*u4
    u1 句长均匀度   u2 连接词密度   u3 模板短语命中   u4 三字重复率

这些数字**不是**从数据拟合出来的，标定后会被 fusion 的权重取代；此处只做兜底展示。
"""
from __future__ import annotations

import re
import statistics as st
import time

from ..segment import Unit, char_count, split_sentences
from .base import EngineResult, clamp01

FUNCTION_WORDS = set("的了是在和与并且而或其该此这那就都也还又很更最从对把被让给向于为以")
CONNECTIVES = [
    "首先", "其次", "再次", "此外", "然而", "因此", "所以", "综上", "总之", "同时",
    "与此同时", "一方面", "另一方面", "不仅", "而且", "由此可见", "换言之", "具体而言",
    "值得注意的是", "总的来说", "综上所述", "总而言之",
]
TEMPLATES = [
    "在当今社会", "随着", "的发展", "起着重要", "具有重要意义", "有着重要影响",
    "不可忽视", "至关重要", "奠定了基础", "提供了参考", "具有广阔", "应用前景",
    "越来越", "日益", "各种领域", "方方面面", "探索和", "深入研究",
]
_CN_PUNCT = "，。、；：？！“”‘’（）《》…—～·"
_ASCII_PUNCT = set(",.;:?!\"'()<>[]{}")

FEATURE_KEYS = (
    "st_nsent", "st_mean_slen", "st_cv_slen", "st_short_ratio", "st_comma_per_sent",
    "st_semicolon100", "st_fw_ratio", "st_conn100", "st_tmpl100", "st_repeat3",
    "st_mixed_en", "st_digit_ratio", "st_punct_ratio",
)


def _ngrams(text: str, n: int) -> list[str]:
    t = re.sub(r"\s", "", text)
    return [t[i:i + n] for i in range(max(0, len(t) - n + 1))]


def features(text: str) -> dict[str, float]:
    chars = char_count(text)
    sents = [s.text for s in split_sentences(text)] or [text]
    slens = [max(0, char_count(s)) for s in sents]
    mean_slen = st.fmean(slens) if slens else 0.0
    cv = (st.pstdev(slens) / mean_slen) if len(slens) > 1 and mean_slen else 0.0
    short_ratio = sum(1 for x in slens if x < 8) / len(slens) if slens else 0.0
    commas = text.count("，") + text.count(",")
    semis = text.count("；") + text.count(";")
    fw = sum(1 for ch in text if ch in FUNCTION_WORDS)
    conn = sum(text.count(w) for w in CONNECTIVES)
    tmpl = sum(text.count(w) for w in TEMPLATES)
    grams = _ngrams(text, 3)
    rep3 = 1.0 - (len(set(grams)) / len(grams)) if grams else 0.0
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    digits = sum(1 for ch in text if ch.isdigit())
    punct = sum(1 for ch in text if ch in _CN_PUNCT) + sum(1 for ch in text if ch in _ASCII_PUNCT)
    denom = max(chars, 1)
    return {
        "st_nsent": float(len(sents)),
        "st_mean_slen": mean_slen,
        "st_cv_slen": cv,
        "st_short_ratio": short_ratio,
        "st_comma_per_sent": commas / max(len(sents), 1),
        "st_semicolon100": semis / denom * 100,
        "st_fw_ratio": fw / denom,
        "st_conn100": conn / denom * 100,
        "st_tmpl100": tmpl / denom * 100,
        "st_repeat3": rep3,
        "st_mixed_en": latin / denom,
        "st_digit_ratio": digits / denom,
        "st_punct_ratio": punct / denom,
    }


def heuristic_score(f: dict[str, float]) -> float:
    u1 = clamp01((0.55 - f["st_cv_slen"]) / 0.35)          # 句长越均匀越像 AI
    u2 = clamp01(f["st_conn100"] / 1.2)                     # 连接词越密越像 AI
    u3 = clamp01(f["st_tmpl100"] / 1.0)                     # 模板短语越多越像 AI
    u4 = clamp01(f["st_repeat3"] / 0.20)                    # 三字重复越多越像 AI
    return clamp01(0.35 * u1 + 0.25 * u2 + 0.25 * u3 + 0.15 * u4)


def explain(f: dict[str, float]) -> list[str]:
    """给报告用的中文解释（只列触发项）。"""
    out: list[str] = []
    if f["st_cv_slen"] < 0.40:
        out.append(f"句式长度高度均匀（变异系数 {f['st_cv_slen']:.2f}）")
    if f["st_conn100"] >= 0.8:
        out.append(f"连接词密度偏高（{f['st_conn100']:.2f}/百字）")
    if f["st_tmpl100"] >= 0.5:
        out.append(f"模板化短语命中（{f['st_tmpl100']:.2f}/百字）")
    if f["st_repeat3"] >= 0.12:
        out.append(f"三字片段重复率偏高（{f['st_repeat3']:.0%}）")
    return out


class StylometryEngine:
    name = "stylo"
    label = "文体计量"

    def available(self) -> tuple[bool, str]:
        return True, "就绪（无需模型）"

    def score(self, units: list[Unit]) -> EngineResult:
        t0 = time.time()
        feats: dict[str, list[float]] = {k: [] for k in FEATURE_KEYS}
        scores: list[float] = []
        for unit in units:
            f = features(unit.text)
            for k in FEATURE_KEYS:
                feats[k].append(f[k])
            scores.append(heuristic_score(f))
        return EngineResult(
            name=self.name, label=self.label, available=True,
            scores=scores, features=feats, note="纯统计特征，无需下载模型",
            elapsed=time.time() - t0,
        )
