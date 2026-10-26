"""交互式启动器：输入解析、文件夹展开、缺模型提示、直接检测与交互循环。"""
from __future__ import annotations

from pathlib import Path

import pytest

from aidetect import config, launch

FIXTURE = Path(__file__).parent / "fixtures" / "mixed_hc3.txt"


# --------------------------------------------------------------------------- 输入解析
@pytest.mark.parametrize("raw", ["", "   ", "q", "Q", "quit", "exit", ":q", "退出"])
def test_parse_input_quits(raw):
    assert launch.parse_input(raw).kind == "quit"


@pytest.mark.parametrize(
    "raw,mode",
    [("1", "balanced"), ("2", "strict"), ("3", "lenient"),
     ("balanced", "balanced"), ("strict", "strict"), ("宽松", "lenient")],
)
def test_parse_input_switches_mode(raw, mode):
    command = launch.parse_input(raw)
    assert command.kind == "mode"
    assert command.value == mode


@pytest.mark.parametrize("raw", ["h", "help", "?", "菜单"])
def test_parse_input_help(raw):
    assert launch.parse_input(raw).kind == "help"


def test_parse_input_strips_drag_and_drop_quotes():
    command = launch.parse_input('"C:\\Users\\mikasa\\论文 终稿.docx"')
    assert command.kind == "files"
    assert command.value == "C:\\Users\\mikasa\\论文 终稿.docx"


# --------------------------------------------------------------------------- 路径展开
def test_expand_folder_keeps_only_documents(tmp_path):
    (tmp_path / "a.docx").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert [p.name for p in launch.expand([str(tmp_path)])] == ["a.docx", "b.md"]


def test_expand_dedupes_and_skips_missing(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("x", encoding="utf-8")
    found = launch.expand([str(doc), str(doc), str(tmp_path / "nope.docx")])
    assert [p.name for p in found] == ["a.txt"]


# --------------------------------------------------------------------------- 直接检测
def test_main_reports_missing_models(monkeypatch, capsys):
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: False)
    assert launch.main(["x.docx"]) == 3
    out = capsys.readouterr().out
    assert "缺少模型" in out
    assert "fetch all" in out


def test_main_without_files_needing_models(monkeypatch, capsys):
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert launch.main([]) == 0
    assert "已退出" in capsys.readouterr().out


def test_main_returns_input_error_for_unusable_path(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: True)
    assert launch.main([str(tmp_path / "nope.docx")]) == 2
    assert "没有找到可检测的文件" in capsys.readouterr().out


def _fake_engine_patch(monkeypatch):
    from conftest import fake_engines

    monkeypatch.setattr(launch.aggregate, "build_engines", lambda: fake_engines())


def test_main_detects_file_with_fake_engines(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AIDETECT_NO_OPEN", "1")
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: True)
    _fake_engine_patch(monkeypatch)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)

    assert launch.main([str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "检测" in out
    assert (tmp_path / "mixed_hc3_report.html").exists()
    assert (tmp_path / "mixed_hc3_report.json").exists()
    payload = (tmp_path / "mixed_hc3_report.json").read_text(encoding="utf-8")
    assert '"ai_rate_chars"' in payload


# --------------------------------------------------------------------------- 交互循环
def test_interactive_mode_switch_and_detect(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AIDETECT_NO_OPEN", "1")
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: True)
    _fake_engine_patch(monkeypatch)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    answers = iter(["2", "h", str(FIXTURE), "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert launch.main([]) == 0
    out = capsys.readouterr().out
    assert "模式已切换为 strict" in out
    assert "自查工具，结果不能当作学术不端证据" in out
    assert (tmp_path / "mixed_hc3_report.html").exists()


def test_interactive_mode_switches_for_whole_session(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("AIDETECT_NO_OPEN", "1")
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: True)
    _fake_engine_patch(monkeypatch)
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    answers = iter(["strict", str(FIXTURE), "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert launch.main([]) == 0
    payload = (tmp_path / "mixed_hc3_report.json").read_text(encoding="utf-8")
    assert '"mode": "strict"' in payload


def test_interactive_unknown_path_is_not_fatal(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(launch.fetch, "is_ready", lambda name: True)
    answers = iter(["c:\\nope\\nothing.docx", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    assert launch.main([]) == 0
    out = capsys.readouterr().out
    assert "没有找到可检测的文件" in out
    assert "已退出" in out


def test_detect_reports_whitespace_only_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AIDETECT_NO_OPEN", "1")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ", encoding="utf-8")
    assert launch.detect(empty, [], config.DEFAULT_MODE) is False
    assert "没有可读文本" in capsys.readouterr().out


def test_detect_skips_document_without_body(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AIDETECT_NO_OPEN", "1")
    monkeypatch.setattr(config, "OUT_DIR", tmp_path)
    only_refs = tmp_path / "only_refs.txt"
    only_refs.write_text(
        "参考文献\n\n[1] 张三. 深度学习综述[J]. 计算机学报, 2020, 43(1): 1-20.\n"
        "[2] 李四. 自然语言处理[M]. 北京: 科学出版社, 2019.\n",
        encoding="utf-8",
    )
    assert launch.detect(only_refs, [], config.DEFAULT_MODE) is False
    assert "正文为空" in capsys.readouterr().out
