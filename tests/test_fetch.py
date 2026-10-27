"""下载器：URL 形状、分片布局、单分片/多分片合并、断点跳过（全程离线）。"""
from __future__ import annotations

import json

import pytest

from aidetect import fetch

BASE = "https://huggingface.co"


def test_url_shapes_for_models_and_datasets():
    assert fetch._url("a/b", "model", "f.bin", BASE) == f"{BASE}/a/b/resolve/main/f.bin"
    assert fetch._url("a/b", "dataset", "all.jsonl", BASE) == \
        f"{BASE}/datasets/a/b/resolve/main/all.jsonl"


def test_plan_parts_small_file_is_single_part(tmp_path):
    dest = tmp_path / "s.bin"
    parts, spans = fetch._plan_parts(dest, 500, workers=8, chunk_bytes=16 * 1024 * 1024)
    assert len(parts) == 1 and len(spans) == 1
    assert parts[0].name == "s.bin.part"
    assert spans == [(0, 499)]
    assert json.loads((tmp_path / "s.bin.layout").read_text())["n"] == 1


def test_plan_parts_large_file_covers_whole_range(tmp_path):
    size = 100 * 1024 * 1024
    parts, spans = fetch._plan_parts(tmp_path / "big.bin", size, workers=8,
                                    chunk_bytes=16 * 1024 * 1024)
    assert len(parts) == len(spans) > 1
    assert spans[0][0] == 0 and spans[-1][1] == size - 1
    for (_, end), (nxt, _) in zip(spans, spans[1:]):
        assert end + 1 == nxt
    assert all(p.name.startswith("big.bin.p") for p in parts)


def test_plan_parts_discards_stale_layout(tmp_path):
    dest = tmp_path / "x.bin"
    stale = dest.with_name("x.bin.p007")
    stale.write_bytes(b"old-part")
    fetch._plan_parts(dest, 400, workers=8, chunk_bytes=16 * 1024 * 1024)
    assert not stale.exists()


def _patch_transfer(monkeypatch, data: bytes, fail_before: int = 0):
    calls = {"n": 0}

    def head(urls, attempts=6):
        return len(data)

    def chunk(urls, path, start, end, progress, retries=200):
        calls["n"] += 1
        if calls["n"] <= fail_before:
            raise RuntimeError("模拟断流")
        buf = data[start:end + 1]
        with open(path, "ab") as f:
            f.write(buf)
        progress.add(len(buf))

    monkeypatch.setattr(fetch, "_head_size", head)
    monkeypatch.setattr(fetch, "_fetch_chunk", chunk)
    monkeypatch.setattr(fetch, "_endpoints",
                        lambda mirror=None: ["https://example.invalid"])
    return calls


def test_download_single_part_file(monkeypatch, tmp_path):
    data = b"y" * 500
    _patch_transfer(monkeypatch, data)
    dest = tmp_path / "small.bin"
    fetch.download_file("r/m", "model", "small.bin", dest)
    assert dest.read_bytes() == data
    assert not (tmp_path / "small.bin.part").exists()


def test_download_multi_part_file(monkeypatch, tmp_path):
    data = (b"0123456789abcdef" * 156_250)  # 2.5 MB，>1 MB 阈值
    _patch_transfer(monkeypatch, data)
    dest = tmp_path / "big.bin"
    fetch.download_file("r/m", "model", "big.bin", dest, chunk_bytes=1024 * 1024)
    assert dest.read_bytes() == data
    assert not list(tmp_path.glob("big.bin.p*"))


def test_download_writes_provenance(monkeypatch, tmp_path):
    data = b"z" * 300
    _patch_transfer(monkeypatch, data)
    dest = tmp_path / "p.bin"
    fetch.download_file("owner/repo", "model", "p.bin", dest)
    prov = json.loads((tmp_path / ".provenance.json").read_text(encoding="utf-8"))
    assert prov["p.bin"]["repo"] == "owner/repo"
    assert prov["p.bin"]["bytes"] == len(data)
    assert prov["p.bin"]["fetched_at"]


def test_download_skips_complete_file(monkeypatch, tmp_path):
    data = b"w" * 200
    calls = _patch_transfer(monkeypatch, data)
    dest = tmp_path / "done.bin"
    dest.write_bytes(data)
    fetch.download_file("r/m", "model", "done.bin", dest)
    assert calls["n"] == 0


def test_is_ready_for_unknown_resource():
    assert fetch.is_ready("no-such-resource") is False


def test_endpoints_prefers_configured_env(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://my-mirror.example/")
    eps = fetch._endpoints()
    assert eps[0] == "https://my-mirror.example"
    assert "https://hf-mirror.com" in eps
