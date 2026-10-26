"""文本读取：编码回退、docx、错误信息。"""
from __future__ import annotations

import pytest

from aidetect import textio


def test_read_utf8(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("中文内容，English text.", encoding="utf-8")
    assert textio.read_text_file(p) == "中文内容，English text."


def test_read_utf8_bom(tmp_path):
    p = tmp_path / "bom.txt"
    p.write_bytes(b"\xef\xbb\xbf" + "带 BOM 的中文".encode("utf-8"))
    assert textio.read_text_file(p).strip() == "带 BOM 的中文"


def test_read_gb18030(tmp_path):
    p = tmp_path / "gbk.txt"
    p.write_bytes("这是 GBK 编码的中文。".encode("gb18030"))
    assert textio.read_text_file(p) == "这是 GBK 编码的中文。"


def test_markdown_suffix_supported(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# 标题\n\n正文段落。", encoding="utf-8")
    assert "正文段落" in textio.read_text_file(p)


def test_load_returns_display_name(tmp_path):
    p = tmp_path / "论文.txt"
    p.write_text("内容", encoding="utf-8")
    text, name = textio.load(str(p))
    assert text == "内容"
    assert name == "论文.txt"


def test_missing_file_raises_read_error(tmp_path):
    with pytest.raises(textio.ReadError, match="文件不存在"):
        textio.load(str(tmp_path / "nope.txt"))


def test_directory_raises_read_error(tmp_path):
    with pytest.raises(textio.ReadError, match="目录"):
        textio.load(str(tmp_path))


def test_empty_file_raises_read_error(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n\n ", encoding="utf-8")
    with pytest.raises(textio.ReadError, match="没有可读文本"):
        textio.load(str(p))


def test_docx_paragraphs_and_tables(tmp_path):
    docx = pytest.importorskip("docx", reason="未安装 python-docx")
    d = docx.Document()
    d.add_paragraph("第一段中文内容。")
    d.add_paragraph("第二段内容。")
    table = d.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "指标"
    table.cell(0, 1).text = "数值"
    p = tmp_path / "doc.docx"
    d.save(str(p))

    text = textio.read_text_file(p)
    assert "第一段中文内容。" in text
    assert "第二段内容。" in text
    assert "指标 | 数值" in text

    loaded, name = textio.load(str(p))
    assert loaded and name == "doc.docx"


def test_binary_file_is_rejected(tmp_path):
    p = tmp_path / "blob.txt"
    p.write_bytes(bytes(range(256)) * 16)
    with pytest.raises(textio.ReadError, match="二进制"):
        textio.read_text_file(p)


def test_unknown_suffix_binary_is_rejected(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    with pytest.raises(textio.ReadError, match="二进制"):
        textio.read_text_file(p)


def test_ansi_colored_log_is_still_text(tmp_path):
    p = tmp_path / "app.log"
    line = b"\x1b[31mERROR\x1b[0m something happened at 12:00:00\n"
    p.write_bytes(line * 60)
    text = textio.read_text_file(p)
    assert "ERROR" in text and "something happened" in text


def test_pure_ansi_sequences_are_rejected_as_no_text(tmp_path):
    p = tmp_path / "noise.txt"
    p.write_bytes(b"\x1b[31m" * 200)
    with pytest.raises(textio.ReadError, match="没有可读文本"):
        textio.read_text_file(p)


def test_utf16_text_is_not_treated_as_binary(tmp_path):
    p = tmp_path / "u16.txt"
    p.write_bytes("十六位编码的中文段落。".encode("utf-16"))
    assert textio.read_text_file(p).strip() == "十六位编码的中文段落。"


def test_read_stdin(monkeypatch):
    class FakeStdin:
        buffer = type("B", (), {"read": staticmethod(lambda: "标准输入的中文".encode("utf-8"))})()

    monkeypatch.setattr(textio.sys, "stdin", FakeStdin())
    assert textio.load("-") == ("标准输入的中文", "<stdin>")
