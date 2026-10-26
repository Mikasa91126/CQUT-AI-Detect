"""文档切分：段落/句子/判定单元/参考文献剔除。"""
from __future__ import annotations

import pytest

from aidetect import config
from aidetect.segment import (
    build_doc, char_count, clean_for_ppl, make_units, normalize,
    split_sentences, stats, window_texts,
)
from conftest import AI_PARA, HUMAN_PARA


def test_char_count_ignores_whitespace():
    assert char_count("中 文\n\t测试") == 4
    assert char_count("") == 0


def test_normalize_cleans_zero_width_and_blank_lines():
    assert normalize("中\u200b文 \u3000测试\n\n\n\n下一段") == "中文 测试\n\n下一段"
    assert normalize("\r\n第一行\r\n第二行") == "第一行\n第二行"


def test_split_sentences_keeps_closing_quotes():
    sents = [s.text for s in split_sentences("他说：“走吧。”然后走了。")]
    assert sents == ["他说：“走吧。”", "然后走了。"]
    assert len(split_sentences("没有句号的句子")) == 1
    assert len(split_sentences("第一句。第二句！第三句？")) == 3


def test_build_doc_classifies_heading_and_body():
    doc = build_doc("研究背景\n\n" + AI_PARA, "t.txt")
    kinds = [p.kind for p in doc.paras]
    assert kinds[0] == "heading"
    assert "body" in kinds
    assert doc.body_chars > 0
    assert doc.ref_cut is False


def test_build_doc_cuts_references():
    tail = "[1] 张三. 一篇论文[J]. 期刊, 2020."
    doc = build_doc("研究背景\n\n" + AI_PARA + "\n\n参考文献\n\n" + tail, "t.txt")
    assert doc.ref_cut is True
    assert doc.excluded_chars == char_count("参考文献") + char_count(tail)
    assert all(p.kind == "meta" for p in doc.paras if p.text.startswith("["))
    assert doc.body_chars == char_count(AI_PARA)


def test_build_doc_rejects_blank_text():
    with pytest.raises(ValueError):
        build_doc("   \n\n  \n")


def test_make_units_merges_short_paragraphs():
    doc = build_doc("\n\n".join(["这是很短的一句。"] * 5), "t.txt")
    units = make_units(doc)
    assert len(units) == 2
    assert [len(u.paras) for u in units] == [3, 2]
    assert all(u.chars >= config.MIN_UNIT_CHARS for u in units[:-1])
    assert units[0].para_indexes == [0, 1, 2]


def test_long_paragraph_is_split_at_sentence_bounds():
    long_para = "这是一句用于测试的长句子，包含足够多的字符以便累积超过阈值。" * 30
    doc = build_doc(long_para, "t.txt")
    assert len(doc.paras) >= 2
    assert all(p.chars <= 900 for p in doc.paras)
    assert all(p.kind == "body" for p in doc.paras)


def test_window_texts_covers_whole_text():
    text = "这是一个句子。" * 100
    windows = window_texts(text, max_chars=300)
    assert len(windows) > 1
    assert "".join(windows) == text
    assert all(char_count(w) <= 300 for w in windows)


def test_clean_for_ppl_strips_citations_and_urls():
    out = clean_for_ppl("见文献[1-3]与（张三, 2020）所述，详见 https://example.com/a 。")
    assert "[1-3]" not in out
    assert "https://" not in out
    assert "2020" not in out


def test_stats_reports_body_and_total():
    doc = build_doc("研究背景\n\n" + AI_PARA + "\n\n" + HUMAN_PARA, "t.txt")
    s = stats(doc)
    assert s["name"] == "t.txt"
    assert s["body_chars"] == char_count(AI_PARA) + char_count(HUMAN_PARA)
    assert s["total_chars"] >= s["body_chars"]
    assert s["body_paragraphs"] == 2


def test_split_long_para_never_exceeds_limit():
    """超长段必须切到 <= MAX_PARA_CHARS，即使整段没有句号（单句超长）。"""
    from aidetect.segment import MAX_PARA_CHARS, _split_long_para

    no_punct = "字" * 1700
    pieces = _split_long_para(no_punct)
    assert len(pieces) > 1
    assert "".join(pieces) == no_punct
    assert all(len(p) <= MAX_PARA_CHARS for p in pieces)

    long_sents = "这是一句很长的话但没有那么长。" * 60
    pieces = _split_long_para(long_sents)
    assert "".join(pieces) == long_sents
    assert all(len(p) <= MAX_PARA_CHARS for p in pieces)


def test_window_texts_hard_splits_oversized_sentence():
    """单句超长时窗口也必须 <= max_chars，否则 512 token 会被静默截断。"""
    text = "无标点的超长片段" * 120  # 960 字，无句末标点
    windows = window_texts(text, max_chars=300)
    assert len(windows) > 1
    assert "".join(windows) == text
    assert all(len(w) <= 300 for w in windows)


def test_markdown_heading_is_not_body():
    doc = build_doc("# 一、引言\n\n" + AI_PARA, "t.md")
    assert doc.paras[0].kind == "heading"
    assert doc.paras[0].chars > 0


def test_reference_heading_sharing_block_with_first_entry():
    """「参考文献」和 [1] 挤在一行（无空行）时也要剔除，不能当正文计分。"""
    doc = build_doc("正文第一段足够长以便建模。" * 12 + "\n\n参考文献 [1] 张三. 2020. AI 检测研究.")
    assert doc.ref_cut is True
    assert doc.excluded_chars > 0
    assert doc.body_chars == char_count("正文第一段足够长以便建模。" * 12)

def test_markdown_heading_requires_space_after_hash():
    """`#1号样本的观测结果…` 是正文，不是标题（R-21）。"""
    doc = build_doc("#1号样本的观测结果如下所示，" + AI_PARA, "t.md")
    assert doc.paras[0].kind == "body"
    assert build_doc("## 方法\n\n" + AI_PARA, "t.md").paras[0].kind == "heading"
