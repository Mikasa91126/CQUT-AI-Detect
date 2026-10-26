"""汇总层：AI 率口径、置信区间、等级计数、护栏提示、JSON 结构。"""
from __future__ import annotations

import json

import pytest

from aidetect import aggregate, config
from aidetect.segment import build_doc
from conftest import AI_PARA, HUMAN_PARA, fake_engines, make_doc

EXPECT_P = 0.70 * 0.9 + 0.20 * 0.1 + 0.10 * 0.1  # clf 0.9 / ppl 0.1 / stylo 0.1


def test_ai_rate_is_char_weighted_probability(uncalibrated):
    a = aggregate.analyze(make_doc(4), engines=fake_engines())
    assert a.ai_rate_chars == pytest.approx(EXPECT_P)
    assert a.ai_rate_paras == 1.0
    assert a.threshold == pytest.approx(0.60)
    assert a.grade_counts == {"high": 0, "mid_high": 4, "mid": 0, "low": 0}
    assert a.stats["units"] == 4


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
