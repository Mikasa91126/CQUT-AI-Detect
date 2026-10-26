"""融合层：把三个引擎的输出合成为单元级 P(AI)。

两种模式，报告里会明写是哪一种：

- **已标定**：读取 ``resources/calibration.json``，用其中的特征标准化参数 + 逻辑回归系数
  算 logit，再用温度缩放得到概率。特征顺序、系数、样本量、判据都可追溯。
- **未标定**（标定文件缺失）：退化为「引擎启发式分加权」，权重 0.70/0.20/0.10 写死在
  ``DEFAULT_ENGINE_WEIGHTS``，仅在可用引擎间重新归一化。报告会打橙色标记提醒不可全信。

无论哪种模式，特征都先做「文档内相对化」：同一篇文档里 PPL/burst 的绝对水平受领域影响，
相对水平才可比。
"""
from __future__ import annotations

import json
import statistics as st
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .engines.base import EngineResult, sigmoid
from .segment import Unit

CALIBRATION_PATH = config.RESOURCES_DIR / "calibration.json"

# 融合层使用的特征（顺序即标定参数顺序；改动必须同步 bump 标定文件版本）
CURATED_FEATURES = [
    "clf_mean",        # 分类器段落均值分
    "clf_max",         # 分类器最可疑窗口
    "clf_std",         # 分类器窗口内分歧（越大越像拼接/混合）
    "ppl_rel",         # 文档内相对流畅度（越高越像 AI）
    "top10",           # GLTR top-10 桶占比
    "burst_rel",       # 文档内相对 burstiness（越低越像 AI，取负号后越高越像）
    "st_cv_slen",      # 句长变异系数
    "st_conn100",      # 连接词密度
    "st_tmpl100",      # 模板短语密度
    "st_repeat3",      # 三字重复率
]

DEFAULT_ENGINE_WEIGHTS = {"clf": 0.70, "ppl": 0.20, "stylo": 0.10}
DEFAULT_TEMPERATURE = 1.0


@dataclass
class Recipe:
    """一套可复现的打分参数。"""

    calibrated: bool = False
    version: str = "uncalibrated"
    features: list[str] = field(default_factory=lambda: list(CURATED_FEATURES))
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    coef: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    temperature: float = DEFAULT_TEMPERATURE
    thresholds: dict[str, float] = field(default_factory=lambda: dict(config.DEFAULT_THRESHOLDS))
    mode_thresholds: dict[str, float] = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "calibrated": self.calibrated,
            "features": self.features,
            "mean": self.mean,
            "std": self.std,
            "coef": self.coef,
            "intercept": self.intercept,
            "temperature": self.temperature,
            "thresholds": self.thresholds,
            "mode_thresholds": self.mode_thresholds,
            "evaluation": self.evaluation,
            "dataset": self.dataset,
            "note": self.note,
        }


def _doc_relative(values: list[float]) -> list[float]:
    """文档内 z 值；单元太少或方差过小时全部置 0，避免放大噪声。"""
    vals = [v for v in values if v == v]  # 过滤 NaN
    if len(vals) < 3:
        return [0.0] * len(values)
    mu = st.fmean(vals)
    sd = st.pstdev(vals)
    if not sd or sd < 1e-6:
        return [0.0] * len(values)
    return [0.0 if v != v else (v - mu) / sd for v in values]


def build_features(units: list[Unit], results: list[EngineResult]) -> list[dict[str, float]]:
    """把各引擎特征按单元对齐，并补齐派生特征。缺失引擎的特征留 NaN。"""
    n = len(units)
    rows: list[dict[str, float]] = [{} for _ in range(n)]
    for res in results:
        if not res.available:
            continue
        for key, values in res.features.items():
            if len(values) != n:
                continue
            for i, v in enumerate(values):
                rows[i][key] = float(v)

    # 派生：文档内相对化
    lp = [r.get("meanlprob", float("nan")) for r in rows]
    lp_rel = _doc_relative(lp)
    burst = [r.get("burst", float("nan")) for r in rows]
    burst_rel = _doc_relative(burst)
    for i, row in enumerate(rows):
        row["ppl_rel"] = lp_rel[i]
        # burstiness 越低越像 AI（单调、平稳）→ 取负号，让「越高越像 AI」
        row["burst_rel"] = -burst_rel[i] if burst_rel[i] == burst_rel[i] else 0.0
    return rows


def fuse(units: list[Unit], results: list[EngineResult], recipe: Recipe) -> tuple[list[float], list[dict]]:
    """返回 (每单元 P(AI), 每单元特征字典)。"""
    rows = build_features(units, results)
    probs: list[float] = []

    if recipe.calibrated and recipe.coef:
        for row in rows:
            z = 0.0
            for name in recipe.features:
                x = row.get(name, float("nan"))
                if x != x:  # NaN → 用均值代替，等价于该特征贡献 0
                    x = recipe.mean.get(name, 0.0)
                mu = recipe.mean.get(name, 0.0)
                sd = recipe.std.get(name, 1.0) or 1.0
                z += recipe.coef.get(name, 0.0) * ((x - mu) / sd)
            z += recipe.intercept
            p = sigmoid(z / max(recipe.temperature, 1e-3))
            probs.append(p)
        return probs, rows

    # --- 未标定：引擎启发式分加权 ---
    engine_scores: dict[str, list[float]] = {}
    for res in results:
        if res.available and len(res.scores) == len(units):
            engine_scores[res.name] = res.scores
    total_w = sum(DEFAULT_ENGINE_WEIGHTS.get(k, 0.0) for k in engine_scores) or 1.0
    for i in range(len(units)):
        acc = 0.0
        for name, scores in engine_scores.items():
            acc += DEFAULT_ENGINE_WEIGHTS.get(name, 0.0) * scores[i]
        probs.append(acc / total_w)
    return probs, rows


def grade_of(p: float, thresholds: dict[str, float]) -> str:
    if p >= thresholds.get("high", 0.80):
        return "high"
    if p >= thresholds.get("mid_high", 0.60):
        return "mid_high"
    if p >= thresholds.get("mid", 0.40):
        return "mid"
    return "low"


def grade_cutoffs_for(mode_threshold: float) -> dict[str, float]:
    """由模式阈值推出四档边界，保证 high > mid_high(≈判定阈值) > mid > 0。

    未标定时阈值在 0.4~0.8 区间，沿用「+0.25 / +0.02 / -0.15」的线性偏移；
    标定后的概率尺度可能非常小（例如 0.009），线性偏移会让 mid(0.10) 反超
    mid_high(0.029) 造成分级倒挂，此时改为按比例缩放。
    """
    t = max(float(mode_threshold), 1e-6)
    high = min(0.95, t + 0.25)
    mid_high = min(0.92, t + 0.02)
    mid = max(0.10, t - 0.15)
    if mid >= mid_high:  # 阈值过小 → 线性偏移失效，改按比例缩放
        high = min(0.95, t * 4.0)
        mid_high = t
        mid = t * 0.5
    return {"high": high, "mid_high": mid_high, "mid": mid}


def _fallback_mode_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
    base = thresholds.get("mid_high", 0.60)
    return {
        "strict": min(0.95, base + 0.15),
        "balanced": base,
        "lenient": max(0.05, base - 0.15),
    }


def load_recipe(path: Path | None = None) -> Recipe:
    path = path or CALIBRATION_PATH
    if not path.exists():
        return Recipe(
            calibrated=False,
            note="未找到 calibration.json，使用未标定的启发式融合",
            thresholds=dict(config.DEFAULT_THRESHOLDS),
            mode_thresholds=_fallback_mode_thresholds(config.DEFAULT_THRESHOLDS),
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Recipe(
            calibrated=False,
            note=f"calibration.json 读取失败：{exc}",
            thresholds=dict(config.DEFAULT_THRESHOLDS),
            mode_thresholds=_fallback_mode_thresholds(config.DEFAULT_THRESHOLDS),
        )
    recipe = Recipe(
        calibrated=bool(raw.get("calibrated")),
        version=str(raw.get("version", "unknown")),
        features=list(raw.get("features") or CURATED_FEATURES),
        mean={k: float(v) for k, v in (raw.get("mean") or {}).items()},
        std={k: float(v) for k, v in (raw.get("std") or {}).items()},
        coef={k: float(v) for k, v in (raw.get("coef") or {}).items()},
        intercept=float(raw.get("intercept", 0.0)),
        temperature=float(raw.get("temperature", 1.0) or 1.0),
        thresholds={k: float(v) for k, v in (raw.get("thresholds") or {}).items()}
        or dict(config.DEFAULT_THRESHOLDS),
        mode_thresholds={k: float(v) for k, v in (raw.get("mode_thresholds") or {}).items()},
        evaluation=raw.get("evaluation") or {},
        dataset=raw.get("dataset") or {},
        note=str(raw.get("note", "")),
    )
    if not recipe.mode_thresholds:
        recipe.mode_thresholds = _fallback_mode_thresholds(recipe.thresholds)
    if not recipe.calibrated and not recipe.note:
        recipe.note = "calibration.json 标记为未标定"
    return recipe
