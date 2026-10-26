"""标定与评测工具函数（不加载模型、不读 HC3）。"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from aidetect import calibrate, config, fusion
from aidetect.calibrate import Sample, _merge_same_question, metrics, select, threshold_for_fpr


# --------------------------------------------------------------------------- 工具函数单测
def test_threshold_for_fpr_uses_human_quantile():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    labels = np.array([0, 0, 0, 0, 1, 1])
    assert threshold_for_fpr(scores, labels, 0.25) == pytest.approx(0.3)
    # 人类样本只有一条：退化为该条分数
    assert threshold_for_fpr(np.array([0.42]), np.array([0]), 0.05) == pytest.approx(0.42)


def test_metrics_perfect_separation():
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    m = metrics(scores, labels, 0.5)
    assert m["n"] == 4
    assert m["acc"] == 1.0 and m["auc"] == 1.0
    assert m["fpr"] == 0.0 and m["fnr"] == 0.0 and m["precision"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.01 + 0.0025) / 4, abs=1e-4)
    assert m["threshold"] == 0.5


def test_metrics_all_negative_labels_has_nan_auc():
    m = metrics(np.array([0.1, 0.2]), np.array([0, 0]), 0.5)
    assert m["acc"] == 1.0
    assert m["auc"] != m["auc"]  # NaN
    assert m["fnr"] == 0.0


def test_merge_same_question_joins_answers():
    samples = [
        Sample("回答一。", 1, "open_qa", 7),
        Sample("回答二。", 1, "open_qa", 7),
        Sample("人类答案。", 0, "open_qa", 7),
    ]
    merged = _merge_same_question(samples)
    assert len(merged) == 2
    ai = next(s for s in merged if s.label == 1)
    assert ai.text == "回答一。\n\n回答二。"
    assert ai.qid == 7


def test_select_is_domain_and_label_stratified():
    samples = []
    for dom in ("finance", "law"):
        for label in (0, 1):
            for i in range(4):
                samples.append(Sample(f"{dom}-{label}-{i}", label, dom, qid=i))
    picked = select(samples, ("finance",), limit=2, rng=random.Random(0))
    assert len(picked) == 4  # 2 label × 2 条，且同 qid 合并后仍是 2+2
    assert {s.label for s in picked} == {0, 1}
    assert all(s.domain == "finance" for s in picked)


def test_dev_and_test_domains_are_disjoint():
    assert set(calibrate.DEV_DOMAINS).isdisjoint(calibrate.TEST_DOMAINS)
    assert calibrate.TEST_DOMAINS == ("open_qa", "baike", "psychology", "nlpcc_dbqa")


# --------------------------------------------------------------------------- calibration.json 契约
# 这些用例不加载模型、不读 HC3：只读 resources/calibration.json 做 schema 校验。
# 文件不存在（标定还没跑）时 skip，不阻塞其它 73 个单元测试。
METRIC_KEYS = ("n", "acc", "auc", "fpr", "fnr", "precision", "brier", "threshold")
MODES = ("strict", "balanced", "lenient")


def _load_calibration() -> tuple[dict, object]:
    path = fusion.CALIBRATION_PATH
    if not path.exists():
        pytest.skip(f"尚未生成标定文件 {path}（等 python -m aidetect calibrate 跑完）")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw, path


def _num(value) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), f"不是数字：{value!r}"
    return float(value)


def _assert_metrics(m: dict, where: str) -> None:
    assert isinstance(m, dict), f"{where} 不是对象"
    for key in METRIC_KEYS:
        assert key in m, f"{where} 缺指标 {key}"
    assert isinstance(m["n"], int) and m["n"] > 0, f"{where}.n 应为正整数"
    for key in METRIC_KEYS[1:]:
        v = _num(m[key])
        if key == "auc" and not math.isfinite(v):
            continue  # 单类域 AUC 可能是 NaN，schema 上允许
        assert math.isfinite(v), f"{where}.{key} 不是有限数"
    for key in ("acc", "auc", "fpr", "fnr", "precision"):
        v = _num(m[key])
        if math.isfinite(v):
            assert 0.0 <= v <= 1.0, f"{where}.{key}={v} 越界"
    brier = _num(m["brier"])
    if math.isfinite(brier):
        assert 0.0 <= brier <= 1.0, f"{where}.brier={brier} 越界"


def test_calibration_schema_when_present():
    raw, path = _load_calibration()
    required = ("version", "calibrated", "generated_at", "features", "mean", "std", "coef",
                "intercept", "temperature", "thresholds", "mode_thresholds", "evaluation",
                "per_domain", "dataset", "note")
    for key in required:
        assert key in raw, f"{path} 缺字段 {key}"
    assert isinstance(raw["version"], str) and raw["version"]
    assert raw["calibrated"] is True, "标定文件必须标记 calibrated=True"
    assert isinstance(raw["generated_at"], str) and raw["generated_at"]
    assert isinstance(raw["note"], str) and raw["note"]

    feats = raw["features"]
    assert isinstance(feats, list) and feats and all(isinstance(f, str) for f in feats)
    assert feats == list(fusion.CURATED_FEATURES), "特征顺序必须与 fusion.CURATED_FEATURES 一致"
    for key in ("mean", "std", "coef"):
        assert isinstance(raw[key], dict), f"{key} 不是对象"
        assert set(raw[key]) == set(feats), f"{key} 的键必须与 features 完全一致"
        for name, value in raw[key].items():
            _num(value)
    for name, sd in raw["std"].items():
        assert _num(sd) > 0, f"std[{name}] 必须为正"
    _num(raw["intercept"])
    temp = _num(raw["temperature"])
    assert 0.05 <= temp <= 20.0, f"temperature={temp} 不在合理范围"

    thr = raw["thresholds"]
    assert isinstance(thr, dict) and {"high", "mid_high", "mid"} <= set(thr)
    high, mid_high, mid = (_num(thr[k]) for k in ("high", "mid_high", "mid"))
    assert high > mid_high > mid, f"四档阈值应递减：{high} > {mid_high} > {mid}"
    assert 0.0 <= mid and high <= 1.0

    modes = raw["mode_thresholds"]
    assert isinstance(modes, dict)
    assert set(modes) == set(MODES), f"mode_thresholds 必须且只能有三键 {MODES}"
    for mode in MODES:
        _num(modes[mode])
    assert modes["strict"] > modes["balanced"] > modes["lenient"], (
        f"阈值必须 strict > balanced > lenient，实际 {modes}"
    )
    assert all(0.0 <= v <= 1.0 for v in modes.values())


def test_calibration_mode_thresholds_order_and_target_fpr():
    raw, _ = _load_calibration()
    modes = raw["mode_thresholds"]
    assert modes["strict"] > modes["balanced"] > modes["lenient"]
    ev = raw["evaluation"]
    assert isinstance(ev, dict) and set(ev) == set(MODES), "evaluation 必须覆盖三档模式"
    for mode in MODES:
        block = ev[mode]
        assert isinstance(block, dict), f"evaluation[{mode}] 不是对象"
        assert "target_fpr" in block
        assert _num(block["target_fpr"]) == pytest.approx(config.MODE_TARGET_FPR[mode])
        assert isinstance(block.get("dev"), dict), f"evaluation[{mode}] 缺 dev 指标"
        assert isinstance(block.get("test"), dict), f"evaluation[{mode}] 缺 test 指标"
        _assert_metrics(block["dev"], f"evaluation[{mode}].dev")
        _assert_metrics(block["test"], f"evaluation[{mode}].test")
        # 阈值是在 dev 上按目标 FPR 取的：dev 实测 FPR 应贴近目标，test 允许域偏移
        dev_fpr = _num(block["dev"]["fpr"])
        assert dev_fpr <= config.MODE_TARGET_FPR[mode] + 0.05, (
            f"{mode} 的 dev FPR={dev_fpr} 明显高于目标 {config.MODE_TARGET_FPR[mode]}"
        )


def test_calibration_dataset_and_test_domains():
    raw, _ = _load_calibration()
    ds = raw["dataset"]
    assert isinstance(ds, dict)
    for key in ("name", "dev_domains", "test_domains", "n_dev", "n_test", "note"):
        assert key in ds, f"dataset 缺字段 {key}"
    assert isinstance(ds["name"], str) and ds["name"]
    assert isinstance(ds["note"], str) and ds["note"]
    assert isinstance(ds["dev_domains"], list) and isinstance(ds["test_domains"], list)
    dev_domains = tuple(ds["dev_domains"])
    test_domains = tuple(ds["test_domains"])
    # 域外留出协议：dev/test 必须仍然分离，且与代码常量一致
    assert dev_domains == calibrate.DEV_DOMAINS, f"dev 域被改动：{dev_domains}"
    assert test_domains == calibrate.TEST_DOMAINS, f"test 域被改动：{test_domains}"
    assert set(dev_domains).isdisjoint(test_domains), "dev/test 域发生混用"
    assert isinstance(ds["n_dev"], int) and ds["n_dev"] > 0
    assert isinstance(ds["n_test"], int) and ds["n_test"] > 0
    # 门槛：标定集 ≥600 单元、域外测试集 ≥1200 单元（T2 硬约束）
    assert ds["n_dev"] >= 600, f"n_dev={ds['n_dev']} < 600"
    assert ds["n_test"] >= 1200, f"n_test={ds['n_test']} < 1200"


def test_calibration_per_domain():
    raw, _ = _load_calibration()
    per_domain = raw["per_domain"]
    assert isinstance(per_domain, dict) and per_domain, "per_domain 不应为空"
    assert set(per_domain) <= set(calibrate.TEST_DOMAINS), f"per_domain 混入了非测试域：{set(per_domain)}"
    for dom, m in per_domain.items():
        _assert_metrics(m, f"per_domain[{dom}]")


def test_load_recipe_roundtrips_calibration():
    raw, _ = _load_calibration()
    recipe = fusion.load_recipe()
    assert recipe.calibrated is True
    assert recipe.version == str(raw["version"])
    assert recipe.features == list(raw["features"])
    for name in recipe.features:
        assert recipe.mean[name] == pytest.approx(float(raw["mean"][name]))
        assert recipe.std[name] == pytest.approx(float(raw["std"][name]))
        assert recipe.coef[name] == pytest.approx(float(raw["coef"][name]))
    assert recipe.intercept == pytest.approx(float(raw["intercept"]))
    assert recipe.temperature == pytest.approx(float(raw["temperature"]))
    assert set(recipe.mode_thresholds) == set(MODES)
    # 融合层实际判定用的阈值 = 标定文件里的 balanced 阈值
    assert recipe.mode_thresholds["balanced"] == pytest.approx(float(raw["mode_thresholds"]["balanced"]))


# --------------------------------------------------------------------------- bench/报告辅助（不加载模型）
def _synthetic_samples() -> list[Sample]:
    samples: list[Sample] = []
    qid = 1
    for dom in calibrate.DEV_DOMAINS + calibrate.TEST_DOMAINS:
        for label in (0, 1):
            for i in range(3):
                samples.append(Sample(f"{dom}-{label}-{i} 的回答文本。", label, dom, qid=qid + i))
            qid += 10
    return samples


def test_select_sets_advances_rng_like_run():
    samples = _synthetic_samples()
    dev, test = calibrate._select_sets(
        samples, calibrate.DEV_DOMAINS, calibrate.TEST_DOMAINS, 2)
    # 手动复现 run() 的固定顺序：先 dev 再 test，共用同一个 rng
    rng = random.Random(calibrate.SEED)
    manual_dev = select(samples, calibrate.DEV_DOMAINS, 2, rng)
    manual_test = select(samples, calibrate.TEST_DOMAINS, 2, rng)
    key = [(s.qid, s.label, s.domain) for s in dev]
    assert key == [(s.qid, s.label, s.domain) for s in manual_dev]
    assert [(s.qid, s.label, s.domain) for s in test] == \
           [(s.qid, s.label, s.domain) for s in manual_test]
    # 同一 test 子集可重复得到：bench --limit 靠这个复评与 calibrate 相同的样本
    _, test_again = calibrate._select_sets(
        samples, calibrate.DEV_DOMAINS, calibrate.TEST_DOMAINS, 2)
    assert test_again == test
    # limit=None 时不截断：每域每类全部 3 条
    dev_all, test_all = calibrate._select_sets(
        samples, calibrate.DEV_DOMAINS, calibrate.TEST_DOMAINS, None)
    assert len(dev_all) == 3 * 2 * 3
    assert len(test_all) == 4 * 2 * 3


def test_update_run_meta_merges_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "RUN_META_PATH", tmp_path / "run_meta.json")
    calibrate._update_run_meta(limit=200, n_dev=1504, n_test=1802, wall_clock_sec=601.5)
    calibrate._update_run_meta(bench={"limit": 200, "n_test": 1802})
    meta = calibrate._read_run_meta()
    assert meta["limit"] == 200 and meta["n_dev"] == 1504 and meta["n_test"] == 1802
    assert meta["wall_clock_sec"] == 601.5
    assert meta["bench"] == {"limit": 200, "n_test": 1802}


def test_run_meta_section_reports_scale_and_bench(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "RUN_META_PATH", tmp_path / "missing.json")
    assert "未记录" in calibrate._run_meta_section({})
    txt = calibrate._run_meta_section({
        "limit": 200, "n_dev": 1504, "n_test": 1802, "wall_clock_sec": 600.0,
        "bench": {"limit": 200, "n_test": 1802, "wall_clock_sec": 231.0,
                  "modes": {"balanced": {"max_abs_diff": 0.0, "reproduced": True}}},
    })
    assert "--limit 200" in txt
    assert "1504" in txt and "1802" in txt and "600" in txt and "10.0 分钟" in txt
    assert "bench --limit 200" in txt and "复现一致" in txt
    failed = calibrate._run_meta_section({
        "limit": 200, "bench": {"limit": 200, "n_test": 10,
                                "modes": {"balanced": {"max_abs_diff": 0.05,
                                                       "reproduced": False}}},
    })
    assert "存在差异" in failed
