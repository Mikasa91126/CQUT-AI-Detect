"""测试公共夹具：全部离线、不加载真实模型、不联网。

真实模型推理只在 ``tools/smoke.py`` 里跑一次，单元测试用假引擎验证逻辑。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许不 pip install 直接 pytest
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from aidetect.engines.base import EngineResult  # noqa: E402
from aidetect.segment import build_doc  # noqa: E402

AI_PARA = (
    "随着人工智能技术的快速发展，教育领域正在经历深刻的变革。"
    "在线学习平台通过数据驱动的方式为学生提供个性化学习路径，同时也为教师提供了更加精准的"
    "教学反馈与决策支持，从而显著提升了整体教学效率与质量。"
)
HUMAN_PARA = (
    "我上学期选修的那门课，老师讲得挺快的，一节课能讲三章。期末的时候大家都有点慌，"
    "我们宿舍几个人熬了两晚上才把实验报告凑出来，结果成绩出来还行，老师说主要是看过程。"
)


class FakeEngine:
    """满足引擎协议的最小实现：恒定分数 + 可控特征。"""

    def __init__(self, name="clf", label="判别器", ok=True, value=0.9,
                 features=None, note=""):
        self.name = name
        self.label = label
        self.ok = ok
        self.value = value
        self.features = features or {}
        self.note = note

    def available(self):
        return (self.ok, self.note if not self.ok else "")

    def score(self, units):
        n = len(units)
        return EngineResult(
            name=self.name, label=self.label, available=self.ok,
            scores=[float(self.value)] * n,
            features={k: [float(v)] * n for k, v in self.features.items()},
            note=self.note, elapsed=0.0,
        )

    def score_sentences(self, sents):
        return [float(self.value)] * len(sents)


def fake_engines(clf=0.9, ppl=0.1, stylo=0.1):
    return [
        FakeEngine("clf", "判别器", True, clf,
                   {"clf_mean": clf, "clf_max": clf, "clf_std": 0.05,
                    "meanlprob": -2.0, "burst": 0.5, "top10": 0.3}),
        FakeEngine("ppl", "困惑度", True, ppl,
                   {"meanlprob": -1.8, "burst": 0.4, "top10": 0.5, "ppl": 20.0}),
        FakeEngine("stylo", "文体统计", True, stylo,
                   {"st_cv_slen": 0.3, "st_conn100": 2.0, "st_tmpl100": 1.0,
                    "st_repeat3": 0.1, "st_nsent": 10}),
    ]


def fake_results(n=4, clf=0.9, ppl=0.1, stylo=0.1):
    """融合层要的是 EngineResult 列表，不是引擎对象。"""
    return [e.score([None] * n) for e in fake_engines(clf=clf, ppl=ppl, stylo=stylo)]


def make_doc(paras: int = 4, text: str = AI_PARA, name: str = "test.txt"):
    return build_doc("\n\n".join([text] * paras), name)


@pytest.fixture
def uncalibrated(monkeypatch, tmp_path):
    """强制走「未标定」融合分支，避免本机 calibration.json 影响测试结果。"""
    from aidetect import fusion

    monkeypatch.setattr(fusion, "CALIBRATION_PATH", tmp_path / "no_calibration.json")
    return tmp_path


@pytest.fixture
def fake_analysis(uncalibrated):
    """一份用假引擎算出来的完整 Analysis，供报告/CLI 测试复用。"""
    from aidetect import aggregate

    return aggregate.analyze(make_doc(4), engines=fake_engines())
