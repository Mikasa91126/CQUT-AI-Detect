"""T2 产出自动核对（只读）：calibration.json ↔ eval/report.md ↔ eval/run_meta.json。

用法（项目根执行）：
    $env:PYTHONIOENCODING='utf-8'
    python -B eval/check_t2.py
退出码：0 = 全部通过；1 = 存在失败项（会逐条打印）。

复核清单（对应 task-6 第 4 条 + lead 指定的 R-14/R-19 检查）：
  1. calibration.json 必需字段齐全、calibrated=True、features 与 fusion.CURATED_FEATURES 一致；
  2. thresholds / mode_thresholds / temperature 均 ≤4 位小数（R-14）；
  3. mode_thresholds 三键齐全且 strict > balanced > lenient；
  4. thresholds 四档 high > mid_high > mid（四档语义，test_calibrate.py 同款断言）；
  5. evaluation.* 数字与 eval/report.md 三、结果表逐项一致（容差 0，报告按 3 位小数四舍五入）；
  6. per_domain.* 与报告「各域表现」表一致；coef 与「融合特征权重」表一致；
  7. dataset.n_dev / n_test / limit 与 eval/run_meta.json 一致；
  8. report.md 复现命令：同时含 CLI 正式路径 `aidetect bench --limit` 与模块入口 `aidetect.calibrate bench --limit`（lead m00327 裁定），且不再出现「CLI 的 bench 子命令不接受 --limit」过时说明（R-19）；
  9. 门槛判定与诊断文案一致（R-24：未全部达标须如实写、不得声称「一次达标」）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "aidetect" / "resources" / "calibration.json"
REPORT = ROOT / "eval" / "report.md"
META = ROOT / "eval" / "run_meta.json"

fails: list[str] = []
passes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        passes.append(name)
    else:
        fails.append(f"{name}{(' — ' + detail) if detail else ''}")


def num(v) -> float:
    return float(v)


def fmt3(v) -> str:
    return f"{num(v):.3f}"


def decimals(v) -> int:
    s = repr(float(v))
    return len(s.split(".")[1]) if "." in s else 0


def table_rows(md: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def main() -> int:
    raw = json.loads(CAL.read_text(encoding="utf-8"))
    md = REPORT.read_text(encoding="utf-8")
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    rows = table_rows(md)

    # 1. schema
    required = ("version", "calibrated", "generated_at", "features", "mean", "std", "coef",
                "intercept", "temperature", "thresholds", "mode_thresholds", "evaluation",
                "per_domain", "dataset", "note")
    missing = [k for k in required if k not in raw]
    check("schema 必需字段齐全", not missing, f"缺 {missing}")
    check("calibrated=True", raw.get("calibrated") is True, repr(raw.get("calibrated")))
    try:
        sys.path.insert(0, str(ROOT))
        from aidetect import fusion  # 不加载 torch，只读常量
        check("features 与 CURATED_FEATURES 一致",
              list(raw.get("features", [])) == list(fusion.CURATED_FEATURES),
              f"{raw.get('features')} != {list(fusion.CURATED_FEATURES)}")
    except Exception as exc:  # noqa: BLE001
        check("features 与 CURATED_FEATURES 一致", False, f"导入 fusion 失败: {exc}")

    thr = raw.get("thresholds", {})
    mt = raw.get("mode_thresholds", {})
    temp = raw.get("temperature")

    # 2. decimals <= 4 (R-14)
    for label, group in (("thresholds", thr), ("mode_thresholds", mt), ("temperature", {"T": temp})):
        bad = {k: v for k, v in group.items() if round(float(v), 4) != float(v)}
        detail = f"超 4 位小数的键: {bad}" if bad else ""
        check(f"{label} 均 ≤4 位小数 (R-14)", not bad, detail)
        for k, v in group.items():
            check(f"  {label}[{k}] 小数位 ≤4", decimals(v) <= 4, f"{v} 有 {decimals(v)} 位")

    # 3. mode_thresholds 三键 + 单调
    check("mode_thresholds 三键齐全", set(mt) >= {"strict", "balanced", "lenient"}, str(sorted(mt)))
    if {"strict", "balanced", "lenient"} <= set(mt):
        check("mode_thresholds: strict > balanced > lenient",
              mt["strict"] > mt["balanced"] > mt["lenient"],
              f"{mt['strict']} > {mt['balanced']} > {mt['lenient']} 不成立")

    # 4. thresholds 四档单调（四档语义 / test_calibrate.py 同款断言）
    check("thresholds 三键齐全", {"high", "mid_high", "mid"} <= set(thr), str(sorted(thr)))
    if {"high", "mid_high", "mid"} <= set(thr):
        check("thresholds: high > mid_high > mid",
              thr["high"] > thr["mid_high"] > thr["mid"],
              f"{thr['high']} > {thr['mid_high']} > {thr['mid']} 不成立")

    # 5. evaluation 与 report.md 表格一致（3 位小数、容差 0）
    metrics = ("n", "acc", "auc", "fpr", "fnr", "precision", "brier")

    def data_start(cells: list[str]) -> int:
        """新版报告表在「数据」列后多了「阈值」「目标FPR」两列，自动识别列偏移。"""
        start = 1
        if len(cells) > 1 and re.fullmatch(r"\d+\.\d{4}", cells[1].strip()):
            start = 2
        if len(cells) > start and re.fullmatch(r"\d+%", cells[start].strip()):
            start += 1
        return start

    labels = {"strict": "strict", "balanced": "balanced", "lenient": "lenient"}
    found_eval = 0
    found_eval_thr = 0
    found_eval_tgt = 0
    for cells in rows:
        if len(cells) >= 8 and "·" in cells[0]:
            head = cells[0]
            mode = next((m for m in labels if head.startswith(m)), None)
            which = "dev" if "标定集" in head else ("test" if "测试集" in head else None)
            if not mode or not which:
                continue
            data = raw.get("evaluation", {}).get(mode, {}).get(which, {})
            if not data:
                check(f"evaluation.{mode}.{which} 存在", False, "JSON 缺该节")
                continue
            found_eval += 1
            start = data_start(cells)
            if start >= 2:
                found_eval_thr += 1
                check(f"evaluation.{mode}.{which}.threshold 与报告阈值列一致",
                      cells[1].strip() == f"{num(data['threshold']):.4f}",
                      f"{cells[1]} != {num(data['threshold']):.4f}")
            if start >= 3:
                tgt = raw.get("evaluation", {}).get(mode, {}).get("target_fpr")
                if isinstance(tgt, (int, float)):
                    found_eval_tgt += 1
                    check(f"evaluation.{mode}.{which}.目标FPR 与 JSON target_fpr 一致",
                          cells[2].strip() == f"{float(tgt):.0%}",
                          f"{cells[2]} != {float(tgt):.0%}")
            for i, key in enumerate(metrics, start=start):
                cell = cells[i].replace(",", "").strip()
                exp_num = int(round(num(data[key]))) if key == "n" else None
                if key == "n":
                    check(f"evaluation.{mode}.{which}.n 与报告一致",
                          cell == str(exp_num), f"{cell} != {exp_num}")
                else:
                    check(f"evaluation.{mode}.{which}.{key} 与报告一致",
                          cell == fmt3(data[key]), f"{cell} != {fmt3(data[key])} (json {data[key]})")
    check("报告含 dev/test 三模式共 6 行评测表", found_eval == 6, f"只匹配到 {found_eval} 行")
    check("6 行评测表的阈值列均与 JSON 逐项核对（防列偏移静默跳过）",
          found_eval_thr == 6, f"只核对了 {found_eval_thr} 行")
    check("6 行评测表的目标FPR列均与 JSON 逐项核对（防列偏移静默跳过）",
          found_eval_tgt == 6, f"只核对了 {found_eval_tgt} 行")

    # 6a. per_domain 表
    found_dom = 0
    found_dom_thr = 0
    found_dom_tgt = 0
    dom_tgt = raw.get("evaluation", {}).get("balanced", {}).get("target_fpr")
    for cells in rows:
        if len(cells) >= 8 and cells[0] in raw.get("per_domain", {}):
            dom = cells[0]
            data = raw["per_domain"][dom]
            found_dom += 1
            start = data_start(cells)
            if start >= 2:
                found_dom_thr += 1
                check(f"per_domain.{dom}.threshold 与报告阈值列一致",
                      cells[1].strip() == f"{num(data['threshold']):.4f}",
                      f"{cells[1]} != {num(data['threshold']):.4f}")
            if start >= 3 and isinstance(dom_tgt, (int, float)):
                found_dom_tgt += 1
                check(f"per_domain.{dom}.目标FPR 与 JSON target_fpr 一致",
                      cells[2].strip() == f"{float(dom_tgt):.0%}",
                      f"{cells[2]} != {float(dom_tgt):.0%}")
            for i, key in enumerate(metrics, start=start):
                cell = cells[i].replace(",", "").strip()
                if key == "n":
                    check(f"per_domain.{dom}.n 与报告一致", cell == str(int(round(num(data[key])))),
                          f"{cell} != {data[key]}")
                else:
                    check(f"per_domain.{dom}.{key} 与报告一致",
                          cell == fmt3(data[key]), f"{cell} != {fmt3(data[key])}")
    check("报告含全部 per_domain 行", found_dom == len(raw.get("per_domain", {})),
          f"{found_dom} != {len(raw.get('per_domain', {}))}")
    check("per_domain 行的阈值列均与 JSON 逐项核对（防列偏移静默跳过）",
          found_dom_thr == len(raw.get("per_domain", {})), f"只核对了 {found_dom_thr} 行")
    check("per_domain 行的目标FPR列均与 JSON 逐项核对（防列偏移静默跳过）",
          found_dom_tgt == len(raw.get("per_domain", {})), f"只核对了 {found_dom_tgt} 行")

    # 6b. 权重表
    found_w = 0
    for cells in rows:
        if len(cells) == 2 and cells[0] in raw.get("coef", {}):
            found_w += 1
            cell = cells[1].lstrip("+")
            check(f"coef[{cells[0]}] 与报告一致",
                  cell.replace("+", "") == f"{num(raw['coef'][cells[0]]):+.3f}".lstrip("+"),
                  f"{cells[1]} vs {raw['coef'][cells[0]]:+.3f}")
    check("报告含全部特征权重行", found_w == len(raw.get("coef", {})),
          f"{found_w} != {len(raw.get('coef', {}))}")

    # 7. dataset / run_meta 一致
    ds = raw.get("dataset", {})
    check("dataset.n_dev == run_meta.n_dev",
          ds.get("n_dev") == meta.get("n_dev"), f"{ds.get('n_dev')} != {meta.get('n_dev')}")
    check("dataset.n_test == run_meta.n_test",
          ds.get("n_test") == meta.get("n_test"), f"{ds.get('n_test')} != {meta.get('n_test')}")
    check("run_meta.limit 与报告一致",
          meta.get("limit") is not None and f"--limit {meta['limit']}" in md,
          f"limit={meta.get('limit')}")
    check("run_meta 含 wall_clock_sec", isinstance(meta.get("wall_clock_sec"), (int, float)),
          str(meta.get("wall_clock_sec")))
    if isinstance(meta.get("wall_clock_sec"), (int, float)):
        check("报告 wall-clock 秒数与 run_meta 一致（取整）",
              f"{int(round(meta['wall_clock_sec']))} 秒" in md,
              f"run_meta={meta['wall_clock_sec']}")

    # 8. R-19 复现命令（按 lead m00327 裁定：报告同时给出 CLI 正式路径与模块入口脚本路径）
    check("report.md 无「CLI 的 bench 子命令不接受 --limit」", "不接受 `--limit`" not in md,
          "仍出现过时说明")
    check("report.md 有 CLI 形式复现命令 `aidetect bench --limit`", "aidetect bench --limit" in md,
          "复现命令未更新为 CLI 形式")
    check("report.md 有模块入口脚本路径 `aidetect.calibrate bench --limit`（lead 要求两种路径）",
          "aidetect.calibrate bench --limit" in md,
          "缺脚本路径复现命令")

    # 9. 门槛判定与诊断文案一致（R-24，按 lead 裁定：域外 FPR 不达标须如实写）
    try:
        bal_ev = raw["evaluation"]["balanced"]
        fpr = float(bal_ev["test"]["fpr"])
        tgt = float(bal_ev.get("target_fpr", 0.05))
        lo, hi = tgt - 0.02, tgt + 0.03
        gate_ok = lo <= fpr <= hi
        check("balanced test FPR 已复核（信息项）", True,
              f"FPR={fpr:.4f}，门槛 [{lo:.2f},{hi:.2f}]，{'达标' if gate_ok else '未达标'}")
        if not gate_ok:
            check("报告如实标注「未全部达标」（R-24）", "未全部达标" in md,
                  "报告未出现门槛判定")
            check("报告未声称「一次达标」（R-24）", "一次达标" not in md,
                  "与未达标事实矛盾")
            check("报告 FPR 处有「未达标」标注（R-24）", "未达标" in md,
                  "缺未达标说明")
            check("报告给出可操作建议（strict/重标定/不作唯一证据）",
                  any(k in md for k in ("strict", "重标定", "唯一证据")),
                  "缺可操作建议")
        else:
            check("报告出现「全部达标」", "全部达标" in md)
    except (KeyError, TypeError) as exc:
        check("R-24 门槛判定可核对", False, f"缺字段: {exc}")

    # 10. §七.1 门槛表逐项与 JSON evaluation.test 交叉核对（防止表格改写不被发现）
    gate_hdr = "| 模式 | 目标 FPR | 域外实测 FPR |"
    gate_rows: list[list[str]] = []
    in_gate = False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith(gate_hdr):
            in_gate = True
            continue
        if in_gate:
            if not s.startswith("|"):
                break
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set(cells[0]) <= set("-: "):
                continue
            gate_rows.append(cells)
    found_gate = 0
    for cells in gate_rows:
        if len(cells) < 7 or cells[0] not in ("strict", "balanced", "lenient"):
            continue
        mode = cells[0]
        t = raw["evaluation"][mode]["test"]
        tgt = float(raw["evaluation"][mode].get("target_fpr", 0.05))
        found_gate += 1
        check(f"gate {mode} 目标FPR 列与 JSON 一致", cells[1] == f"{tgt:.0%}", f"{cells[1]} != {tgt:.0%}")
        check(f"gate {mode} 域外实测FPR 列与 JSON 一致", cells[2] == f"{float(t['fpr']):.2%}",
              f"{cells[2]} != {float(t['fpr']):.2%}")
        for idx, key, label in ((3, "acc", "Acc"), (4, "auc", "AUC"), (5, "brier", "Brier")):
            check(f"gate {mode} {label} 列与 JSON 一致", cells[idx] == f"{float(t[key]):.4f}",
                  f"{cells[idx]} != {float(t[key]):.4f}")
        lo, hi = tgt - 0.02, tgt + 0.03
        exp_verdict = "未达标" if not (lo <= float(t["fpr"]) <= hi) else "达标"
        check(f"gate {mode} 判定列与阈值协议一致", exp_verdict in cells[6],
              f"{cells[6]!r} 不含 {exp_verdict!r}")
    check("§七.1 门槛表 3 行均与 JSON 逐项核对", found_gate == 3, f"只核对了 {found_gate} 行")

    print(f"== T2 产出核对：{len(passes)} 项通过, {len(fails)} 项失败 ==")
    for name in passes:
        print(f"  [PASS] {name}")
    for name in fails:
        print(f"  [FAIL] {name}")
    if fails:
        print("\n结论：存在失败项，T2 产出不可直接签字。")
        return 1
    print("\n结论：全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
