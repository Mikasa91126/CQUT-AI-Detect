"""命令行：参数解析、退出码、报告落盘路径。"""
from __future__ import annotations

import io
import json
import sys

import pytest

from aidetect import cli, config


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "ai-detect" in capsys.readouterr().out


def test_invalid_mode_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["report", "x.txt", "--mode", "bogus"])
    assert excinfo.value.code == 2


def test_report_without_models_returns_resource_error(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_models_ready", lambda: (False, ["clf"]))
    assert cli.main(["report", "whatever.txt"]) == cli.EXIT_RESOURCE
    assert "缺少模型" in capsys.readouterr().err


def test_report_with_bad_input_returns_input_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "_models_ready", lambda: (True, []))
    assert cli.main(["report", str(tmp_path / "nope.txt")]) == cli.EXIT_INPUT
    assert "读取失败" in capsys.readouterr().err


def test_report_writes_html_and_json(monkeypatch, capsys, tmp_path, fake_analysis):
    monkeypatch.setattr(cli, "_models_ready", lambda: (True, []))
    monkeypatch.setattr(cli.aggregate, "analyze",
                        lambda doc, mode=None, max_sentences=20: fake_analysis)
    src = tmp_path / "论文.txt"
    src.write_text("正文内容。" * 50, encoding="utf-8")
    out_dir = tmp_path / "out"

    code = cli.main(["report", str(src), "--out-dir", str(out_dir)])
    assert code == cli.EXIT_OK

    html = out_dir / "论文_report.html"
    js = out_dir / "论文_report.json"
    assert html.exists() and js.exists()
    assert html.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["tool"] == "ai-detect"
    assert "AI 生成占比" in capsys.readouterr().out


def test_json_subcommand_prints_payload(monkeypatch, capsys, tmp_path, fake_analysis):
    monkeypatch.setattr(cli, "_models_ready", lambda: (True, []))
    monkeypatch.setattr(cli.aggregate, "analyze",
                        lambda doc, mode=None, max_sentences=20: fake_analysis)
    src = tmp_path / "a.txt"
    src.write_text("正文内容。" * 50, encoding="utf-8")
    assert cli.main(["json", str(src)]) == cli.EXIT_OK
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert payload["tool"] == "ai-detect"


def test_resolve_out_defaults_to_out_dir():
    path = cli._resolve_out(None, "论文.docx", "report.html")
    assert path == config.OUT_DIR / "论文_report.html"
    assert cli._resolve_out("custom", "a/b.md", "report.json").name == "b_report.json"


def test_subcommands_are_registered():
    parser = cli.build_parser()
    for cmd in ("report", "json", "doctor", "fetch", "calibrate", "bench"):
        with pytest.raises(SystemExit):
            parser.parse_args([cmd, "--help"])

def test_safe_stem_handles_stdin_and_illegal_chars():
    """stdin 的展示名是 '<stdin>'，直接拼文件名在 Windows 会 OSError；必须清洗。"""
    assert cli._safe_stem("<stdin>") == "stdin"
    out = cli._resolve_out(None, "<stdin>", "report.html")
    assert out.name == "stdin_report.html"
    assert not set('<>:"/\\|?*') & set(out.name)
    assert cli._safe_stem("a|b?.docx") == "a_b"
    assert cli._safe_stem("..") == "document"


def test_bench_parser_accepts_limit():
    parser = cli.build_parser()
    ns = parser.parse_args(["bench", "--limit", "150"])
    assert ns.limit == 150
    assert cli.build_parser().parse_args(["bench"]).limit is None


def test_launch_parser_accepts_files():
    parser = cli.build_parser()
    assert parser.parse_args(["launch", "a.docx", "b.pdf"]).files == ["a.docx", "b.pdf"]
    assert parser.parse_args(["launch"]).files == []


def test_main_tolerates_gbk_console(monkeypatch):
    """Windows 传统 GBK 控制台装不下 ✓/✗：要降级成可替换字符，而不是内部错误 rc=4。"""
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    assert cli.main(["doctor"]) in (0, 1)
