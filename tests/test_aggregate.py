"""汇总层：AI 率口径、置信区间、等级计数、护栏提示、JSON 结构。"""
from __future__ import annotations

import json

import pytest

from aidetect import aggregate, config, fusion
from aidetect.segment import build_doc
from conftest import AI_PARA, HUMAN_PARA, fake_engines, make_doc

# 未标定加权：clf 0.9 / ppl 0.1 / stylo 0.1；cmj 未参与 → 权重按剩余三引擎重归一化
_W = fusion.DEFAULT_ENGINE_WEIGHTS
EXPECT_P = (_W["clf"] * 0.9 + _W["ppl"] * 0.1 + _W["stylo"] * 0.1) / (
    _W["clf"] + _W["ppl"] + _W["stylo"])


def test_ai_rate_is_char_weighted_probability(uncalibrated):
    a = aggregate.analyze(make_doc(4), engines=fake_engines())
    assert a.ai_rate_chars == pytest.approx(EXPECT_P)
    assert a.threshold == pytest.approx(0.60)
    # v2 权重下 clf=0.9 被 cmj 缺席与 ppl/stylo 的低分摊薄 → 融合分 0.39 落在「低」档
    assert a.ai_rate_paras == 0.0
    assert a.grade_counts == {"high": 0, "mid_high": 0, "mid": 0, "low": 4}
    assert a.stats["units"] == 4


def test_suspect_rate_is_independent_of_calibrated_threshold(uncalibrated):
    """疑似口径走 v2 判别器原始分 + 固定疑似线，不受 FPR 标定阈值影响。"""
    a = aggregate.analyze(make_doc(4), engines=fake_engines(cmj=0.9))
    # 疑似占比 = 被判疑似段落的字数占比（与商业网站「AI 疑似度」口径一致）
    assert a.ai_rate_suspect_chars == pytest.approx(1.0)
    assert a.ai_rate_suspect_paras == pytest.approx(1.0)
    assert a.grade_suspect in ("high", "mid_high")
    assert a.grade_counts_suspect["high"] + a.grade_counts_suspect["mid_high"] == 4
    # 保守口径仍按标定阈值判，未被疑似线污染
    assert a.threshold == pytest.approx(0.60)


def test_confidence_interval_brackets_rate(uncalibrated):
    a = aggregate.analyze(make_doc(6), engines=fake_engines())
    assert a.ai_rate_ci is not None
    lo, hi = a.ai_rate_ci
    assert 0.0 <= lo <= a.ai_rate_chars <= hi <= 1.0


def test_ci_none_when_too_few_units(uncalibrated):
    a = aggregate.analyze(make_doc(1), engines=fake_engines())
    assert a.ai_rate_ci is None


def test_low_scores_give_zero_rate(uncalibrated):
    a = aggregate.analyze(make_doc(3), engines=fake_engines(clf=0.0, ppl=0.0, stylo=0.0))
    assert a.ai_rate_chars == pytest.approx(0.0)
    assert a.ai_rate_paras == 0.0
    assert a.grade_counts["low"] == 3
    assert a.sentences == []


def test_suspicious_sentences_when_scores_are_high(uncalibrated):
    a = aggregate.analyze(make_doc(2), engines=fake_engines(clf=1.0, ppl=1.0, stylo=1.0))
    assert a.ai_rate_chars == pytest.approx(1.0)
    assert a.sentences, "满分单元应产出句级证据"
    assert all(s.p >= 0.70 for s in a.sentences)


def test_headers_are_excluded_from_scoring(uncalibrated):
    doc = build_doc("研究背景\n\n" + AI_PARA + "\n\n结论\n\n" + HUMAN_PARA, "t.txt")
    a = aggregate.analyze(doc, engines=fake_engines())
    heading_rows = [r for r in a.paragraphs if r.kind != "body"]
    assert len(heading_rows) == 2
    assert all(r.p is None and r.grade == "excluded" for r in heading_rows)
    assert a.stats["units"] == 2


def test_warnings_for_short_text_and_missing_engines(uncalibrated):
    short = build_doc("就这一句。", "s.txt")
    a = aggregate.analyze(short, engines=fake_engines())
    assert any("可靠性下限" in w for w in a.warnings)

    engines = fake_engines()
    engines[1].ok = False
    engines[2].ok = False
    b = aggregate.analyze(make_doc(3), engines=engines)
    assert any("降级运行" in w for w in b.warnings)
    assert any("未加载标定参数" in w for w in b.warnings)


def test_warning_for_non_chinese_text(uncalibrated):
    english = (
        "This paragraph is written in English and talks about machine learning systems. "
        "Another sentence follows here to give the unit enough characters to score."
    )
    doc = build_doc(english + "\n\n" + english, "en.txt")
    a = aggregate.analyze(doc, engines=fake_engines())
    assert any("非中文" in w for w in a.warnings)
    assert any("汉字只占" in w for w in a.warnings)


def test_no_non_chinese_warning_for_chinese_text(uncalibrated):
    a = aggregate.analyze(make_doc(2), engines=fake_engines())
    assert not any("非中文" in w for w in a.warnings)


def test_to_dict_is_json_serializable(uncalibrated):
    a = aggregate.analyze(make_doc(3), engines=fake_engines())
    d = a.to_dict()
    text = json.dumps(d, ensure_ascii=False)
    assert len(text) > 100
    assert d["tool"] == "ai-detect"
    assert d["mode"] == config.DEFAULT_MODE
    assert len(d["paragraphs"]) == 3
    assert d["ai_rate_ci"] is not None
    assert d["calibration"]["calibrated"] is False
    assert len(d["engines"]) == 3


def test_suspect_track_fields(uncalibrated):
    """v2「疑似度」轨道：固定疑似线 + 段落占比口径，供与商业网站对照。"""
    a = aggregate.analyze(make_doc(4), engines=fake_engines())
    assert a.suspect_threshold == pytest.approx(config.SUSPECT_BY_MODE[config.DEFAULT_MODE])
    cuts = a.suspect_cutoffs
    assert cuts["high"] > cuts["mid_high"] > cuts["mid"] > 0
    assert all(0.0 <= us.p_suspect <= 1.0 for us in a.unit_scores)
    assert sum(a.grade_counts_suspect.values()) == len(a.unit_scores)
    assert 0.0 <= a.ai_rate_suspect_chars <= 1.0
    assert 0.0 <= a.ai_rate_suspect_paras <= 1.0
    assert a.suspect_engine  # cmj 缺失时退化为融合分，仍要写明来源

    d = a.to_dict()
    for key in ("suspect_threshold", "suspect_engine", "ai_rate_chars_suspect",
                "ai_rate_paras_suspect", "ai_rate_suspect_ci", "grade_suspect",
                "grade_counts_suspect", "suspect_cutoffs", "share_grades"):
        assert key in d, f"to_dict 缺 {key}"
    assert all("p_suspect" in p for p in d["paragraphs"])
    assert json.dumps(d, ensure_ascii=False)  # 仍然可序列化
