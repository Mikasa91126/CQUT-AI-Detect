"""标定后诊断（只读 eval/features.npz，不加载模型、不联网）。

用途：当域外 test 门槛不达标时，按任务要求的顺序做诊断，全部复算，不改 calibration.json：
  0) 复算一致性：用 calibration.json 里存的 mean/std/coef/intercept/temperature/thresholds
     对 dumped 原始特征重新算概率，核对与 evaluation 里的 test 指标一致（验证序列化链路）。
  1) 特征消融：去掉 ppl_rel / burst_rel 重新拟合。
  2) C 网格：LogisticRegression C=0.1 / 0.5 / 1.0。
  3) 域分离检查：dev ⊆ {finance,law,medicine}、test ⊆ {open_qa,baike,psychology,nlpcc_dbqa}，两者不相交。
另附各域 AUC（balanced 阈值）。

用法（ai-detect 目录下，先跑过 calibrate --limit N --dump-features）：
    python -u eval/diagnose_features.py
输出：控制台表格 + eval/diagnose.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aidetect import calibrate, config, fusion  # noqa: E402

TARGET_FPR = config.MODE_TARGET_FPR[config.DEFAULT_MODE]
DIAG_PATH = ROOT / "eval" / "diagnose.json"


def _rows(X: np.ndarray, feats: list[str]) -> list[dict]:
    return [dict(zip(feats, row)) for row in X]


def _fit(rows_dev, y_dev, rows_test, feats: list[str], C: float = 0.5):
    from sklearn.linear_model import LogisticRegression

    mean = {f: float(np.nanmean([r.get(f, np.nan) for r in rows_dev])) for f in feats}
    std = {f: float(np.nanstd([r.get(f, np.nan) for r in rows_dev])) or 1.0 for f in feats}
    Xd = calibrate.matrix(rows_dev, feats, mean, std)
    Xt = calibrate.matrix(rows_test, feats, mean, std)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    clf.fit(Xd, y_dev)
    z_d = Xd @ clf.coef_[0] + clf.intercept_[0]
    z_t = Xt @ clf.coef_[0] + clf.intercept_[0]
    temp = calibrate._fit_temperature(z_d, y_dev)
    p_d = 1.0 / (1.0 + np.exp(-z_d / temp))
    p_t = 1.0 / (1.0 + np.exp(-z_t / temp))
    thr = calibrate.threshold_for_fpr(p_d, y_dev, TARGET_FPR)
    return {
        "C": C, "temperature": round(float(temp), 4), "threshold": round(float(thr), 4),
        "coef": {f: round(float(c), 4) for f, c in zip(feats, clf.coef_[0])},
        "dev": calibrate.metrics(p_d, y_dev, thr),
        "test": calibrate.metrics(p_t, y_test_global, thr),
        "p_test": p_t,
    }


def _gate_status(m: dict) -> list[str]:
    fails = []
    tgt = TARGET_FPR
    if m["auc"] < 0.85:
        fails.append(f"AUC {m['auc']} < 0.85")
    if m["acc"] < 0.80:
        fails.append(f"Acc {m['acc']} < 0.80")
    if not (tgt - 0.02 <= m["fpr"] <= tgt + 0.03):
        fails.append(f"FPR {m['fpr']} 不在 [{tgt - 0.02:.2f}, {tgt + 0.03:.2f}]")
    if m["brier"] > 0.22:
        fails.append(f"Brier {m['brier']} > 0.22")
    return fails


def _per_domain(probs, dom, y, feats: list[str]) -> dict:
    out = {}
    for d in sorted(set(dom.tolist())):
        sel = dom == d
        m = calibrate.metrics(probs[sel], y[sel], 0.5)
        out[d] = m
    return out


def main() -> int:
    calib = json.loads(fusion.CALIBRATION_PATH.read_text(encoding="utf-8"))
    data = np.load(ROOT / "eval" / "features.npz", allow_pickle=False)
    feats = [str(x) for x in data["features"]]
    X_dev, y_dev, dom_dev = data["X_dev"], data["y_dev"], data["dom_dev"]
    X_test, y_test, dom_test = data["X_test"], data["y_test"], data["dom_test"]
    global y_test_global
    y_test_global = y_test
    rows_dev, rows_test = _rows(X_dev, feats), _rows(X_test, feats)

    result: dict = {"limit_units": {"dev": int(len(y_dev)), "test": int(len(y_test))},
                    "target_fpr": TARGET_FPR}

    # ---- 0) 复算一致性：直接用 calibration.json 的序列化参数
    mean, std = calib["mean"], calib["std"]
    Xd = calibrate.matrix(rows_dev, feats, mean, std)
    Xt = calibrate.matrix(rows_test, feats, mean, std)
    coef = np.array([calib["coef"][f] for f in feats])
    z = Xt @ coef + calib["intercept"]
    p = 1.0 / (1.0 + np.exp(-z / calib["temperature"]))
    thr = float(calib["mode_thresholds"][config.DEFAULT_MODE])
    ok_metrics = calibrate.metrics(p, y_test, thr)
    expected = (calib["evaluation"][config.DEFAULT_MODE] or {}).get("test") or {}
    diffs = {k: round(float(ok_metrics[k]) - float(expected[k]), 6)
             for k in ("acc", "auc", "fpr", "fnr", "precision", "brier")
             if k in expected}
    result["recompute_check"] = {"max_abs_diff": max(abs(v) for v in diffs.values()) if diffs else None,
                                 "diffs": diffs, "test_metrics": ok_metrics}

    # ---- 1) 特征消融
    drop = ["ppl_rel", "burst_rel"]
    feats_no = [f for f in feats if f not in drop]
    abl = _fit(rows_dev, y_dev, rows_test, feats_no)
    result["ablation_no_ppl_rel_burst_rel"] = {k: abl[k] for k in ("C", "temperature", "threshold", "dev", "test")}

    # ---- 2) C 网格
    grid = {}
    for C in (0.1, 0.5, 1.0):
        g = _fit(rows_dev, y_dev, rows_test, feats, C)
        grid[str(C)] = {k: g[k] for k in ("C", "temperature", "threshold", "dev", "test")}
    result["c_grid"] = grid

    # ---- 3) 域分离检查
    dev_domains = sorted(set(dom_dev.tolist()))
    test_domains = sorted(set(dom_test.tolist()))
    overlap = sorted(set(dev_domains) & set(test_domains))
    result["domain_check"] = {
        "dev_domains": dev_domains, "test_domains": test_domains, "overlap": overlap,
        "dev_ok": set(dev_domains) <= set(calibrate.DEV_DOMAINS),
        "test_ok": set(test_domains) <= set(calibrate.TEST_DOMAINS),
        "units_per_test_domain": {d: int((dom_test == d).sum()) for d in test_domains},
        "labels_per_test_domain": {d: {"human": int(((dom_test == d) & (y_test == 0)).sum()),
                                       "ai": int(((dom_test == d) & (y_test == 1)).sum())}
                                   for d in test_domains},
    }

    # ---- 各域 AUC（用 calibration.json 序列化后的概率）
    result["per_domain_auc_serialized"] = _per_domain(p, dom_test, y_test, feats)

    # ---- 门槛判定
    base_test = result["recompute_check"]["test_metrics"]
    result["gates"] = {
        "balanced_test": _gate_status(base_test),
        "balanced_ok": not _gate_status(base_test),
        "per_domain_auc_ok": all(v["auc"] >= 0.75 for v in result["per_domain_auc_serialized"].values()),
        "per_domain_auc_bad": {d: v["auc"] for d, v in result["per_domain_auc_serialized"].items()
                               if v["auc"] < 0.75},
    }

    DIAG_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(name: str, m: dict) -> str:
        return (f"{name:<34} n={m['n']:>4} Acc={m['acc']:.4f} AUC={m['auc']:.4f} "
                f"FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} Prec={m['precision']:.4f} Brier={m['brier']:.4f}")

    print("== 0) 用 calibration.json 参数复算 test（序列化一致性）==")
    print(line("recompute(test)", ok_metrics))
    print(f"   max_abs_diff vs calibration.json = {result['recompute_check']['max_abs_diff']}")
    print("\n== 1) 消融（去 ppl_rel / burst_rel，C=0.5）==")
    print(line("ablation dev", abl["dev"]))
    print(line("ablation test", abl["test"]))
    print(f"   temperature={abl['temperature']} threshold={abl['threshold']}")
    print("\n== 2) C 网格（全特征）==")
    for c, g in grid.items():
        print(line(f"C={c} test", g["test"]))
    print("\n== 3) 域分离检查 ==")
    print(f"   dev={dev_domains} test={test_domains} overlap={overlap or '无'} "
          f"dev_ok={result['domain_check']['dev_ok']} test_ok={result['domain_check']['test_ok']}")
    print("\n== 各域 AUC（序列化概率，balanced 阈值）==")
    for d, m in result["per_domain_auc_serialized"].items():
        print(line(d, m))
    print("\n== 门槛 ==")
    print(f"   balanced test: {'通过' if result['gates']['balanced_ok'] else '未通过 ' + '; '.join(result['gates']['balanced_test'])}")
    print(f"   各域 AUC≥0.75: {'通过' if result['gates']['per_domain_auc_ok'] else '未通过 ' + str(result['gates']['per_domain_auc_bad'])}")
    print(f"\n结果 -> {DIAG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
