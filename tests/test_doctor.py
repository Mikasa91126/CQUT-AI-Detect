"""自检命令：输出格式、退出码（各检查项均被替换为桩，不依赖本机状态）。"""
from __future__ import annotations

from aidetect import doctor


def _stub(monkeypatch, *, models=True, calibration=True, deps=True):
    monkeypatch.setattr(doctor, "_check_python", lambda: (True, [doctor._row("✓", "python")]))
    monkeypatch.setattr(doctor, "_check_deps", lambda: (deps, [doctor._row("✓", "deps")]))
    monkeypatch.setattr(doctor, "_check_models", lambda: (models, [doctor._row("✓", "models")]))
    monkeypatch.setattr(doctor, "_check_calibration",
                        lambda: (calibration, [doctor._row("✓", "calibration")]))
    monkeypatch.setattr(doctor, "_check_data", lambda: (True, [doctor._row("✓", "data")]))
    monkeypatch.setattr(doctor, "_check_runtime", lambda: [doctor._row("✓", "runtime")])


def test_row_formatting():
    assert doctor._row("✓", "标题") == " ✓ 标题"
    assert doctor._row("✗", "标题", "细节") == " ✗ 标题\n     细节"


def test_run_passes_when_everything_present(monkeypatch, capsys):
    _stub(monkeypatch)
    assert doctor.run() == 0
    out = capsys.readouterr().out
    assert "自检通过" in out
    assert "[模型]" in out and "[标定]" in out


def test_run_fails_with_missing_model(monkeypatch, capsys):
    _stub(monkeypatch, models=False)
    assert doctor.run() == 1
    assert "自检发现缺失项" in capsys.readouterr().out


def test_run_warns_without_calibration(monkeypatch, capsys):
    _stub(monkeypatch, calibration=False)
    assert doctor.run() == 1
    assert "自检发现缺失项" in capsys.readouterr().out


def test_symbols_fall_back_on_gbk_console():
    """GBK 装不下 ✓/✗：符号要退化成 ASCII，不能让自检崩在打印上。"""
    assert doctor.symbols_for("gbk") == ("[OK]", "[!]", "[X]")
    assert doctor.symbols_for("cp936") == ("[OK]", "[!]", "[X]")
    assert doctor.symbols_for("utf-8") == ("✓", "!", "✗")
    assert doctor.symbols_for(None) == ("✓", "!", "✗")
