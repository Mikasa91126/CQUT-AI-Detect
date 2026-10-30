"""站点对照轨道：分数映射、四档边界、映射文件回退、窗口拼接与 evaluate 降级。

全部用 conftest 里的假引擎，不加载真实模型、不联网。
"""
from __future__ import annotations

import json

import pytest

from aidetect import config, site
from aidetect.engines.base import EngineResult
from aidetect.segment import build_doc
from conftest import AI_PARA, FakeEngine, make_doc


# --------------------------------------------------------------------------- 假引擎
class RecordingEngine(FakeEngine):
    """记录每次打分收到的窗口文本（验证窗口拼接 / _Win 对象）。"""

    def __init__(self, name="zhv3", value=0.9):
        super().__init__(name=name, value=value)
        self.windows: list[list[str]] = []

    def score(self, units):
        for u in units:
            assert hasattr(u, "text") and hasattr(u, "chars")
            assert u.chars == len(u.text)
        self.windows.append([u.text for u in units])
        return super().score(units)


class SteppedEngine(FakeEngine):
    """按窗口序号循环给出不同原始分（用来制造多个档位）。"""

    def __init__(self, name="zhv3", values=(0.9, 0.5)):
        super().__init__(name=name, value=float(values[0]))
        self.values = list(values)

    def score(self, units):
        scores = [float(self.values[i % len(self.values)]) for i in range(len(units))]
        return EngineResult(name=self.name, label=self.label, available=True, scores=scores)


class WrongLenEngine(FakeEngine):
    """分数长度故意和窗口数不一致。"""

    def score(self, units):
        return EngineResult(name=self.name, label=self.label, available=True,
                            scores=[0.5] * (len(units) + 1))


def _site_doc():
    """正文 + 标题 + 参考文献区（只有 body 段落该被统计）。"""
    text = ("引言\n\n"
            + AI_PARA + "\n\n"
            + "这是一句完整的话。好。再来一句完整的话。\n\n"
            + "参考文献\n\n[1] 张三. 示例论文. 2020.")
    return build_doc(text, "site.txt")


def _first_engine_name() -> str:
    return str(config.SITE_TRACK["engines"][0])


# --------------------------------------------------------------------------- 映射
def test_score_of_piecewise_linear():
    anchors = [(0.0, 0.0), (0.5, 50.0), (1.0, 100.0)]
    assert site.score_of(0.0, anchors) == pytest.approx(0.0)
    assert site.score_of(0.5, anchors) == pytest.approx(50.0)
    assert site.score_of(1.0, anchors) == pytest.approx(100.0)
    assert site.score_of(0.25, anchors) == pytest.approx(25.0)
    assert site.score_of(0.75, anchors) == pytest.approx(75.0)


def test_score_of_nonlinear_and_outliers():
    anchors = [(0.2, 10.0), (0.6, 40.0), (0.9, 100.0)]
    assert site.score_of(0.4, anchors) == pytest.approx(25.0)    # 10 + 30 * 0.5
    assert site.score_of(0.75, anchors) == pytest.approx(70.0)   # 40 + 60 * (0.15/0.3)
    # 离群输入：低于首锚 / 高于末锚 → 取端点（不外推）
    assert site.score_of(-0.5, anchors) == pytest.approx(10.0)
    assert site.score_of(1.0, anchors) == pytest.approx(100.0)
    assert site.score_of(2.0, anchors) == pytest.approx(100.0)
    # 结果必须 clamp 到 0-100
    clamp_anchors = [(0.0, -20.0), (1.0, 130.0)]
    assert site.score_of(0.0, clamp_anchors) == pytest.approx(0.0)
    assert site.score_of(1.0, clamp_anchors) == pytest.approx(100.0)


def test_band_of_boundaries():
    assert site.band_of(100.0) == "high"
    assert site.band_of(70.0) == "high"
    assert site.band_of(69.99) == "mid_high"
    assert site.band_of(60.0) == "mid_high"
    assert site.band_of(59.99) == "mid"
    assert site.band_of(50.0) == "mid"
    assert site.band_of(49.99) == "none"
    assert site.band_of(0.0) == "none"


# --------------------------------------------------------------------------- 映射文件
def test_load_map_missing_file_falls_back_to_linear(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    anchors, meta = site.load_map()
    assert anchors == [(0.0, 0.0), (1.0, 100.0)]
    assert meta == {"source": "linear-fallback"}


def test_load_map_corrupt_file_falls_back(monkeypatch, tmp_path):
    (tmp_path / str(config.SITE_TRACK["map_file"])).write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    anchors, meta = site.load_map()
    assert anchors == [(0.0, 0.0), (1.0, 100.0)]
    assert meta == {"source": "linear-fallback"}


def test_load_map_reads_anchors_and_sorts(monkeypatch, tmp_path):
    payload = {"version": 1,
               "anchors": [[0.8, 85.0], [0.1, 15.0], [0.45, 40.0]],
               "meta": {"source": "qq"}}
    (tmp_path / str(config.SITE_TRACK["map_file"])).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    anchors, meta = site.load_map()
    assert anchors == [(0.1, 15.0), (0.45, 40.0), (0.8, 85.0)]
    assert meta == {"source": "qq"}


# --------------------------------------------------------------------------- evaluate
def test_evaluate_only_body_and_skips_short_sentences(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)   # 无映射文件 → 线性
    doc = _site_doc()
    eng = RecordingEngine(name=_first_engine_name(), value=0.6)
    rep = site.evaluate(doc, engines=[eng])
    assert rep is not None
    assert rep.engine == _first_engine_name()

    body_indexes = {p.index for p in doc.paras if p.kind == "body"}
    assert all(s.para_index in body_indexes for s in rep.sentences)
    expected = [s.text.strip()
                for p in doc.paras if p.kind == "body"
                for s in p.sentences if len(s.text.strip()) >= 6]
    assert [s.text for s in rep.sentences] == expected
    assert "引言" not in [s.text for s in rep.sentences]       # 标题段不参与
    assert "好。" not in [s.text for s in rep.sentences]       # 短句跳过
    assert [s.index for s in rep.sentences] == list(range(len(expected)))
    # 短句都在正文里但被跳过：判定句数 < 正文句数
    body_sent_count = sum(len(p.sentences) for p in doc.paras if p.kind == "body")
    assert len(rep.sentences) < body_sent_count


def test_evaluate_merges_windows_and_keeps_long_sentence(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    long_sent = "长" * 300
    doc = build_doc(AI_PARA + "\n\n" + long_sent, "win.txt")
    eng = RecordingEngine(name=_first_engine_name(), value=0.5)
    rep = site.evaluate(doc, engines=[eng])
    assert rep is not None
    wins = eng.windows[0]
    assert long_sent in wins                       # 单句超长 → 整句单独成窗
    assert len(wins) < len(rep.sentences)          # 普通短句被拼进同一个窗口
    assert all(len(w) <= config.SITE_TRACK["window_chars"]
               for w in wins if w != long_sent)


def test_evaluate_uses_map_file(monkeypatch, tmp_path):
    payload = {"version": 1, "anchors": [[0.0, 10.0], [1.0, 90.0]],
               "meta": {"source": "test-map"}}
    (tmp_path / str(config.SITE_TRACK["map_file"])).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    rep = site.evaluate(make_doc(1, text=AI_PARA),
                        engines=[FakeEngine(_first_engine_name(), value=0.5)])
    assert rep is not None
    assert rep.headline == pytest.approx(50.0)      # 10 + 80 * 0.5
    assert all(abs(s.score - 50.0) < 1e-9 for s in rep.sentences)
    assert all(s.band == "mid" for s in rep.sentences)   # 50 分 → 低疑似
    assert rep.meta["source"] == "test-map"
    assert rep.meta["requested_engine"] == list(config.SITE_TRACK["engines"])


def test_evaluate_prefers_configured_engine_order(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    cands = list(config.SITE_TRACK["engines"])
    assert len(cands) >= 2
    fallback = FakeEngine(cands[1], value=0.9)
    preferred = FakeEngine(cands[0], value=0.2)
    # 两个都在列表里：按配置顺序取第一个
    rep = site.evaluate(make_doc(1, text=AI_PARA), engines=[fallback, preferred])
    assert rep is not None
    assert rep.engine == cands[0]
    assert rep.headline == pytest.approx(20.0)
    # 首选不可用：回退到第二个候选
    rep2 = site.evaluate(make_doc(1, text=AI_PARA),
                         engines=[FakeEngine(cands[0], ok=False), fallback])
    assert rep2 is not None
    assert rep2.engine == cands[1]


def test_evaluate_ratios_headline_and_to_dict(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    monkeypatch.setitem(config.SITE_TRACK, "window_chars", 1)   # 每句一窗，分数可控
    text = "".join(f"测试句子内容样本{ch}。" for ch in "甲乙丙丁戊己")
    doc = build_doc(text, "bands.txt")
    eng = SteppedEngine(_first_engine_name(),
                        values=(0.9, 0.55, 0.2, 0.65, 0.49, 0.71))
    rep = site.evaluate(doc, engines=[eng])
    assert rep is not None
    assert [s.band for s in rep.sentences] == ["high", "mid", "none",
                                               "mid_high", "none", "high"]
    # 六个句子字数相同 → 每个档位占比 = 句数 / 6
    assert rep.ratios["high"] == pytest.approx(2 / 6)
    assert rep.ratios["mid"] == pytest.approx(1 / 6)
    assert rep.ratios["mid_high"] == pytest.approx(1 / 6)
    assert rep.ratios["none"] == pytest.approx(2 / 6)
    assert rep.ratios["suspected"] == pytest.approx(4 / 6)
    assert rep.suspected_ratio == pytest.approx(rep.ratios["suspected"])
    assert sum(rep.ratios[b] for b in site.BAND_ORDER) == pytest.approx(1.0)
    # 线性映射下 headline = 各句 100*p 的均值（字数相同）
    assert 0.0 <= rep.headline <= 100.0
    assert rep.headline == pytest.approx((90 + 55 + 20 + 65 + 49 + 71) / 6, abs=1e-6)

    d = rep.to_dict()
    assert set(d) == {"engine", "headline", "suspected_ratio", "ratios",
                      "bands", "meta", "sentences"}
    assert set(d["ratios"]) == {"high", "mid_high", "mid", "none", "suspected"}
    assert d["bands"] == dict(config.SITE_TRACK["bands"])
    assert d["engine"] == _first_engine_name()
    assert d["meta"]["requested_engine"] == list(config.SITE_TRACK["engines"])
    assert 0.0 <= d["headline"] <= 100.0
    for row in d["sentences"]:
        assert set(row) == {"index", "para_index", "text", "p", "score", "band"}
        assert row["band"] in site.BAND_ORDER
        assert 0.0 <= row["p"] <= 1.0
        assert 0.0 <= row["score"] <= 100.0
    json.dumps(d, ensure_ascii=False)              # JSON 可序列化


def test_evaluate_without_usable_engine_returns_none():
    doc = make_doc(1, text=AI_PARA)
    # 传了引擎列表但没有配置里的候选 → None（且不能去构造真实模型）
    assert site.evaluate(doc, engines=[FakeEngine("clf", value=0.9)]) is None
    assert site.evaluate(doc, engines=[]) is None
    # 候选都存在但不可用 → None
    bad = [FakeEngine(name, ok=False) for name in config.SITE_TRACK["engines"]]
    assert site.evaluate(doc, engines=bad) is None


def test_evaluate_no_body_sentences_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    doc = build_doc("引言\n\n参考文献\n\n[1] 张三. 示例论文. 2020.", "empty.txt")
    assert all(p.kind != "body" for p in doc.paras)
    assert site.evaluate(doc, engines=[FakeEngine(_first_engine_name())]) is None


def test_evaluate_scores_length_mismatch_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESOURCES_DIR", tmp_path)
    eng = WrongLenEngine(_first_engine_name())
    assert site.evaluate(make_doc(2, text=AI_PARA), engines=[eng]) is None
