"""从 aidetect/resources/calibration.json 打印标定结果表（只读）。

用法（ai-detect 目录下）：
    python -u eval/print_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aidetect import config, fusion  # noqa: E402

MODES = ["strict", "balanced", "lenient"]
FIELDS = ["threshold", "n", "acc", "auc", "fpr", "fnr", "precision", "brier"]


def main() -> int:
    calib = json.loads(fusion.CALIBRATION_PATH.read_text(encoding="utf-8"))
    ds = calib["dataset"]
    print(f"calibration: version={calib['version']} calibrated={calib['calibrated']} "
          f"generated_at={calib['generated_at']} temperature={calib['temperature']} limit={ds.get('note')}")
    print(f"features: {', '.join(calib['features'])}")
    print(f"dataset: {ds['name']} dev={ds['dev_domains']} test={ds['test_domains']} "
          f"n_dev={ds['n_dev']} n_test={ds['n_test']}")
    print("\n=== 三模式 dev/test ===")
    hdr = "mode       set    " + " ".join(f"{f:>9}" for f in FIELDS)
    print(hdr)
    for mode in MODES:
        ev = calib["evaluation"][mode]
        for split in ("dev", "test"):
            m = ev.get(split) or {}
            if not m:
                print(f"{mode:<10} {split:<6} (缺失)")
                continue
            vals = []
            for f in FIELDS:
                v = m.get(f)
                vals.append("--" if v is None else f"{v:9.4f}" if f != "n" else f"{int(v):9d}")
            print(f"{mode:<10} {split:<6} " + " ".join(vals))
        print(f"           target_fpr={ev.get('target_fpr')}")

    print("\n=== 稳健阈值（grade_cutoffs, 基于 balanced）===")
    print(calib["thresholds"])
    print("\n=== 各域（test 集）===")
    print(f"{'domain':<14} " + " ".join(f"{f:>9}" for f in FIELDS))
    for d, m in calib["per_domain"].items():
        vals = []
        for f in FIELDS:
            v = m.get(f)
            vals.append("--" if v is None else f"{v:9.4f}" if f != "n" else f"{int(v):9d}")
        print(f"{d:<14} " + " ".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
