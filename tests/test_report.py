"""报告渲染：单文件、离线、关键信息齐全。"""
from __future__ import annotations

import re

import pytest

from aidetect import aggregate, report
from conftest import fake_engines, make_doc


@pytest.fixture
def analysis(uncalibrated):
    return aggregate.analyze(make_doc(4), engines=fake_engines())


def test_risk_key_levels():
    # v2：首屏按「疑似占比」分档（60/40/20 三档）
    assert report.risk_key(0.90, 0.60) == "high"
    assert report.risk_key(0.70, 0.60) == "high"
    assert report.risk_key(0.45, 0.60) == "mid_high"
    assert report.risk_key(0.30, 0.60) == "mid"
    assert report.risk_key(0.10, 0.60) == "low"


def test_render_is_self_contained_html(analysis):
    html = report.render(analysis)
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # 离线：不引用任何外部资源
    assert not re.search(r"<link[^>]+href=", html)
    assert "src=\"http" not in html
    assert "@import" not in html
    assert "https://" not in html and "http://" not in html


def test_render_contains_conclusion_first_sections(analysis):
    html = report.render(analysis)
    assert "疑似 AI 占比" in html
    assert "疑似段落占比" in html
    assert "保守口径 AI 生成占比" in html
    assert "可疑句" in html
    assert "逐段明细" in html
    assert "未标定" in html  # 未标定时必须显眼提示
    assert analysis.stats["name"] in html


def test_render_prints_scores_and_signals(analysis):
    html = report.render(analysis)
    assert f"{analysis.ai_rate_suspect_chars * 100:.1f}%" in html or "疑似 AI 占比" in html
    assert "<svg" in html  # 段落分布图内联
    assert "style=" in html


def test_render_escapes_untrusted_text(uncalibrated):
    from aidetect.segment import build_doc

    evil = build_doc("<script>alert('x')</script>" + "正常内容。" * 20, "evil.txt")
    html = report.render(aggregate.analyze(evil, engines=fake_engines()))
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_write_creates_file(analysis, tmp_path):
    path = tmp_path / "out" / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    report.write(analysis, path)
    text = path.read_text(encoding="utf-8")
    assert path.exists() and len(text) > 2000
    assert text.startswith("<!DOCTYPE html>")
