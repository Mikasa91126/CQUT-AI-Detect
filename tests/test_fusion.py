"""融合层：特征相对化、未标定/已标定两条打分路径、等级边界。"""
from __future__ import annotations

import json

import pytest

from aidetect import fusion
from aidetect.fusion import (
    Recipe, _doc_relative, build_features, fuse, grade_cutoffs_for, grade_of,
    load_recipe,
)
from aidetect.segment import make_units
from conftest import fake_results, make_doc


def test_doc_relative_requires_enough_units():
    assert _doc_relative([1.0, 2.0]) == [0.0, 0.0]
    assert _doc_relative([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]  # 方差过小不放大
    z = _doc_relative([1.0, 2.0, 3.0])
    assert z[0] < 0 < z[2]
    assert sum(z) == pytest.approx(0.0, abs=1e-9)


def test_build_features_aligns_and_derives_relative_terms():
    doc = make_doc(4)
    units = make_units(doc)
    rows = build_features(units, fake_results(len(units)))
    assert len(rows) == len(units)
    assert all("ppl_rel" in r and "burst_rel" in r and "clf_mean" in r for r in rows)
    # burst_rel 是 burst 的负数 z：恒定 burst → 0
    assert all(r["burst_rel"] == 0.0 for r in rows)


def test_uncalibrated_fuse_uses_engine_weights():
    units = make_units(make_doc(2))
    probs, _ = fuse(units, fake_results(len(units), clf=0.9, ppl=0.1, stylo=0.1), Recipe())
    w = fusion.DEFAULT_ENGINE_WEIGHTS
    total = w["clf"] + w["ppl"] + w["stylo"]  # cmj 未参与 → 权重重归一化
    expect = (w["clf"] * 0.9 + w["ppl"] * 0.1 + w["stylo"] * 0.1) / total
    assert probs == pytest.approx([expect, expect])


def test_uncalibrated_fuse_renormalizes_missing_engine():
    units = make_units(make_doc(2))
    only_stylo = [r for r in fake_results(len(units)) if r.name == "stylo"]
    probs, _ = fuse(units, only_stylo, Recipe())
    assert probs == pytest.approx([0.1, 0.1])


def test_calibrated_fuse_maps_features_monotonically():
    units = make_units(make_doc(2))
    recipe = Recipe(
        calibrated=True, version="t",
        features=["clf_mean"], mean={"clf_mean": 0.5}, std={"clf_mean": 0.1},
        coef={"clf_mean": 2.0}, intercept=0.0, temperature=1.0,
    )
    low, _ = fuse(units, fake_results(len(units), clf=0.4), recipe)
    high, _ = fuse(units, fake_results(len(units), clf=0.9), recipe)
    assert 0.0 < low[0] < high[0] < 1.0
    assert high[0] == pytest.approx(1 / (1 + 2.718281828 ** (-8.0)), rel=1e-4)


def test_grade_of_and_cutoffs():
    cut = grade_cutoffs_for(0.60)
    assert cut == pytest.approx({"high": 0.85, "mid_high": 0.62, "mid": 0.45})
    assert grade_of(0.9, cut) == "high"
    assert grade_of(0.62, cut) == "mid_high"
    assert grade_of(0.5, cut) == "mid"
    assert grade_of(0.1, cut) == "low"
    assert grade_cutoffs_for(0.90) == pytest.approx(
        {"high": 0.95, "mid_high": 0.92, "mid": 0.75})


def test_load_recipe_missing_file_is_uncalibrated(tmp_path):
    rc = load_recipe(tmp_path / "nope.json")
    assert rc.calibrated is False
    assert set(rc.mode_thresholds) == {"strict", "balanced", "lenient"}
    assert rc.mode_thresholds["strict"] > rc.mode_thresholds["balanced"] > rc.mode_thresholds["lenient"]
    assert rc.note


def test_load_recipe_reads_calibration_json(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "version": "1.0.0", "calibrated": True,
        "features": ["clf_mean"], "mean": {"clf_mean": 0.5}, "std": {"clf_mean": 0.2},
        "coef": {"clf_mean": 1.5}, "intercept": -0.2, "temperature": 0.9,
        "mode_thresholds": {"strict": 0.8, "balanced": 0.6, "lenient": 0.4},
        "evaluation": {"balanced": {"test": {"acc": 0.9}}},
    }, ensure_ascii=False), encoding="utf-8")
    rc = load_recipe(path)
    assert rc.calibrated and rc.version == "1.0.0"
    assert rc.coef["clf_mean"] == 1.5
    assert rc.mode_thresholds["balanced"] == 0.6
    assert rc.evaluation["balanced"]["test"]["acc"] == 0.9


def test_load_recipe_broken_file_falls_back(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{ not json", encoding="utf-8")
    rc = load_recipe(path)
    assert rc.calibrated is False
    assert "读取失败" in rc.note


def test_curated_features_match_recipe_contract():
    assert fusion.CURATED_FEATURES == [
        "clf_mean", "clf_max", "clf_std", "cmj_mean", "cmj_max", "cmj_std",
        "ppl_rel", "top10", "burst_rel",
        "st_cv_slen", "st_conn100", "st_tmpl100", "st_repeat3",
    ]

def test_grade_cutoffs_stay_monotonic_for_tiny_calibrated_threshold():
    """标定后阈值可能只有 0.009：四档边界必须 still 单调，不能 mid 反超 mid_high。"""
    cuts = fusion.grade_cutoffs_for(0.0087)
    assert cuts["high"] > cuts["mid_high"] > cuts["mid"] > 0
    assert cuts["mid_high"] == pytest.approx(0.0087)
    assert cuts["high"] <= 0.95
    # 未标定的大阈值行为保持不变（README 记录的 0.85/0.62/0.45）
    big = fusion.grade_cutoffs_for(0.60)
    assert (big["high"], big["mid_high"], big["mid"]) == (pytest.approx(0.85), pytest.approx(0.62), pytest.approx(0.45))
