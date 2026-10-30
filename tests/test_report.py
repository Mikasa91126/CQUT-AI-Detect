"""报告渲染：单文件、离线、关键信息齐全。"""
from __future__ import annotations

import re
from types import SimpleNamespace

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


# ---------- 站点对照（PaperPass 口径）区块 ----------

def _fake_site(**overrides):
    """按接口契约伪造站点数据（不依赖真实 site 模块，也不跑模型）。"""
    site = SimpleNamespace(
        engine="cmj",
        headline=54.98,
        suspected_ratio=0.6649,
        ratios={"high": 0.484, "mid_high": 0.1372, "mid": 0.0437,
                "none": 0.1457, "suspected": 0.6649},
        meta={"source": "PaperPass 锚点",
              "human_reference": {
                  "hc3_human": {"corpus_suspected_share": 0.6359},
                  "thucnews_human": {"corpus_suspected_share": 0.6926},
              }},
        sentences=[
            SimpleNamespace(index=0, para_index=0, text="第一句按高疑似着色。",
                            p=0.92, score=82.4, band="high"),
            SimpleNamespace(index=1, para_index=0,
                            text="第二句含 <script>alert(1)</script> 的注入文本。",
                            p=0.64, score=64.0, band="mid_high"),
            SimpleNamespace(index=2, para_index=1, text="第三句是中低疑似。",
                            p=0.53, score=53.5, band="mid"),
            SimpleNamespace(index=3, para_index=1, text="第四句没有 AI 疑似。",
                            p=0.12, score=12.0, band="none"),
        ],
    )
    for key, value in overrides.items():
        setattr(site, key, value)
    return site


def test_site_panel_renders_bands_and_highlight(analysis):
    analysis.site = _fake_site()
    html = report.render(analysis)

    assert "站点对照" in html
    assert "站点对照 · 全文按句着色" in html
    assert "疑似文字占比" in html
    assert "66.5%" in html            # suspected_ratio 一位小数
    assert "55.0" in html             # headline 54.98 → 55.0
    assert "高疑似" in html and "无 AI 疑似" in html
    assert "PaperPass 锚点" in html   # 口径来源
    for cls in ("bandhigh", "bandmidhigh", "bandmid", "bandnone"):
        assert f'class="{cls}"' in html
    # 句子文本里的注入内容必须转义，不能形成可执行脚本
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "句子分数明细" in html
    # 首屏改用站点对照口径：大字标签 + 保守口径小字 + 人类误报参考一句话披露
    assert "疑似文字占比 · 站点对照" in html
    assert "保守口径（本机标定）AI 生成占比" in html
    assert "人类正式文本误报参考" in html
    assert "63.6%" in html and "69.3%" in html
    # 站点对照区块提到「护栏」之后、「段落分布」之前
    assert html.index('id="sec-guard"') < html.index('id="sec-site"')
    assert html.index('id="sec-site"') < html.index('id="sec-dist"')


def test_site_panel_absent_when_no_site(analysis):
    analysis.site = None
    html = report.render(analysis)
    assert "站点对照" not in html
    assert 'id="sec-site"' not in html
    # 首屏保持保守口径，不出现站点大字标签与保守口径小字
    assert "疑似 AI 占比" in html
    assert "疑似文字占比 · 站点对照" not in html
    assert "保守口径（本机标定）" not in html


def test_site_panel_tolerates_partial_site(analysis):
    # 只有引擎名和疑似占比：没有 sentences / ratios / meta 也不能抛异常
    analysis.site = SimpleNamespace(engine="cmj", suspected_ratio=0.5)
    html = report.render(analysis)
    assert "站点对照" in html
    assert "50.0%" in html
    assert "疑似文字占比 · 站点对照" in html  # 首屏仍切到站点口径
    assert "人类正式文本误报参考" not in html  # 没有 meta 时不硬编数字
    assert html.startswith("<!DOCTYPE html>")


def _section_numbers(html):
    return re.findall(r'<span class="sec-no">([^<]+)</span>', html)


def test_site_section_order_and_numbers(analysis):
    # 有站点对照时：00 护栏 / 01 站点对照 / 02 分布 / 03 可疑句 / 04 逐段 / 05 引擎，且不重复
    analysis.site = _fake_site()
    html = report.render(analysis)
    assert _section_numbers(html) == ["00", "01", "02", "03", "04", "05"]

    # 站点缺失时回到原编号：00 / 01 / 02 / 03 / 04
    analysis.site = None
    html = report.render(analysis)
    assert _section_numbers(html) == ["00", "01", "02", "03", "04"]
