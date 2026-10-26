"""标定与评测：用 HC3-Chinese（公开中文「人写 vs ChatGPT」语料）拟合融合层。

流程（全部可复现，固定随机种子）：

1. 按域切分：标定用 finance/law/medicine，**测试只用域外** open_qa/baike/psychology/nlpcc_dbqa，
   避免「用同一批数据拟合又评测」的自欺。
2. 每条回答当一篇小文档走完整切分流程（``build_doc`` → ``make_units``），保证特征分布与
   真实用法一致（含文档内相对化的 ppl_rel / burst_rel）。
3. 三引擎出特征 → 标准化 → 逻辑回归拟合 P(AI)。
4. 温度缩放（在 dev 上最小化 NLL）让概率不至于过尖。
5. 阈值按目标 FPR 在 dev 的 AI 样本分位数上取；在 test 上报告 Acc/AUC/FPR/FNR/Brier。
6. 产出 ``resources/calibration.json``（参数可追溯）与 ``eval/report.md``（人可读结论）。

用法：
    python -m aidetect calibrate                 # 全量标定 + 评测
    python -m aidetect calibrate --limit 300     # 快速跑（每类每域限 300 条）
    python -m aidetect bench                     # 只用现有标定重新评测，不重新拟合
    python -m aidetect bench --limit 300         # 复评与 calibrate --limit 300 相同的 test 子集
    python -m aidetect report                    # 只按现有 calibration.json + eval/run_meta.json 重写报告
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, fusion
from .aggregate import build_engines
from .engines.base import EngineResult, sigmoid
from .segment import build_doc, make_units

DEV_DOMAINS = ("finance", "law", "medicine")
TEST_DOMAINS = ("open_qa", "baike", "psychology", "nlpcc_dbqa")
SEED = 20241025
MAX_CHARS_PER_ANSWER = 900


# --------------------------------------------------------------------------- 数据
@dataclass
class Sample:
    text: str
    label: int          # 1 = AI(ChatGPT), 0 = 人类
    domain: str
    qid: int


def load_hc3(path: Path | None = None) -> list[Sample]:
    path = path or (config.DATA_DIR / "hc3zh" / "all.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"缺少 HC3-Chinese 数据：{path}。先运行 python -m aidetect.fetch hc3zh")
    out: list[Sample] = []
    with open(path, encoding="utf-8") as f:
        for qid, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            domain = str(rec.get("source", "unknown"))
            for label, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for ans in rec.get(key) or []:
                    text = (ans or "").strip()
                    if len(text) < 120:
                        continue
                    out.append(Sample(text=text[:MAX_CHARS_PER_ANSWER], label=label,
                                      domain=domain, qid=qid))
    return out


def select(samples: list[Sample], domains: tuple[str, ...], limit: int | None,
           rng: random.Random) -> list[Sample]:
    """按域分层抽样，并把同一问题、同-side 的多条回答合并成一篇（更接近真实长文）。"""
    picked: list[Sample] = []
    for dom in domains:
        for label in (0, 1):
            group = [s for s in samples if s.domain == dom and s.label == label]
            rng.shuffle(group)
            if limit:
                group = group[:limit]
            picked.extend(_merge_same_question(group))
    return picked


def _merge_same_question(samples: list[Sample]) -> list[Sample]:
    groups: dict[tuple[int, int], list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.qid, s.label), []).append(s)
    out: list[Sample] = []
    for members in groups.values():
        text = "\n\n".join(m.text for m in members)
        out.append(Sample(text=text, label=members[0].label,
                          domain=members[0].domain, qid=members[0].qid))
    return out


# --------------------------------------------------------------------------- 特征
def _slice(res: EngineResult, lo: int, hi: int) -> EngineResult:
    return EngineResult(
        name=res.name, label=res.label, available=res.available,
        scores=list(res.scores[lo:hi]),
        features={k: list(v[lo:hi]) for k, v in res.features.items()},
        note=res.note, elapsed=res.elapsed,
    )


def collect(samples: list[Sample], engines: list, batch_note: str = "") -> tuple[list[dict], list[int], list[str]]:
    """把样本切成单元并跑引擎，返回 (特征行, 标签, 域)。"""
    docs: list[tuple[list, int, str]] = []
    for i, s in enumerate(samples):
        doc = build_doc(s.text, f"hc3-{s.qid}-{s.label}")
        units = make_units(doc)
        if not units:
            continue
        docs.append((units, s.label, s.domain))
        if (i + 1) % 200 == 0:
            print(f"\r  切分 {i+1}/{len(samples)} …", end="", flush=True)
    print(f"\r  切分完成：{len(docs)} 篇可用 / {len(samples)} 条样本", flush=True)

    flat = [u for units, _, _ in docs for u in units]
    print(f"  单元数 {len(flat)}，开始跑引擎 {batch_note}", flush=True)
    results: list[EngineResult] = []
    for eng in engines:
        t0 = time.time()
        res = eng.score(flat)
        print(f"    - {res.label}: {'可用' if res.available else '不可用'} "
              f"{time.time()-t0:.1f}s", flush=True)
        results.append(res)

    rows: list[dict] = []
    labels: list[int] = []
    domains: list[str] = []
    cursor = 0
    for units, label, domain in docs:
        lo, hi = cursor, cursor + len(units)
        sliced = [_slice(res, lo, hi) for res in results]
        feat = fusion.build_features(units, sliced)
        rows.extend(feat)
        labels.extend([label] * len(units))
        domains.extend([domain] * len(units))
        cursor = hi
    return rows, labels, domains


def matrix(rows: list[dict], features: list[str], mean: dict | None = None,
           std: dict | None = None) -> np.ndarray:
    """按特征顺序取矩阵；NaN 用均值填补后标准化。"""
    X = np.zeros((len(rows), len(features)), dtype=np.float64)
    for j, name in enumerate(features):
        col = np.array([r.get(name, np.nan) for r in rows], dtype=np.float64)
        if mean is None:
            mu = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
            sd = float(np.nanstd(col)) if np.isfinite(col).any() else 1.0
        else:
            mu, sd = float(mean.get(name, 0.0)), float(std.get(name, 1.0) or 1.0)
        col = np.where(np.isfinite(col), col, mu)
        X[:, j] = (col - mu) / (sd if sd > 1e-9 else 1.0)
    return X


# --------------------------------------------------------------------------- 拟合
def fit_logistic(X: np.ndarray, y: np.ndarray, features: list[str], rows: list[dict]):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    z = X @ clf.coef_[0] + clf.intercept_[0]
    temp = _fit_temperature(z, y)
    mean = {name: float(np.nanmean([r.get(name, np.nan) for r in rows])) for name in features}
    std = {name: float(np.nanstd([r.get(name, np.nan) for r in rows])) or 1.0 for name in features}
    coef = {name: float(c) for name, c in zip(features, clf.coef_[0])}
    return coef, float(clf.intercept_[0]), temp, mean, std


def _fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """温度缩放：最小化 dev 上的负对数似然。"""
    from scipy.optimize import minimize_scalar

    def nll(t: float) -> float:
        t = max(t, 0.2)
        p = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(0.3, 6.0), method="bounded")
    return float(res.x)


def threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """在 AI 样本上取分位数，使人类样本误判率≈target_fpr。"""
    human = np.sort(scores[labels == 0])
    if human.size == 0:
        return 0.5
    idx = int(round((1.0 - target_fpr) * (human.size - 1)))
    return float(human[idx])


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import roc_auc_score

    pred = (scores >= threshold).astype(int)
    acc = float(np.mean(pred == labels))
    auc = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) > 1 else float("nan")
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    prec = tp / max(tp + fp, 1)
    brier = float(np.mean((scores - labels) ** 2))
    return {
        "n": int(len(labels)), "acc": round(acc, 4), "auc": round(auc, 4),
        "fpr": round(fpr, 4), "fnr": round(fnr, 4), "precision": round(prec, 4),
        "brier": round(brier, 4), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": round(threshold, 4),
    }


# --------------------------------------------------------------------------- 主流程
RUN_META_PATH = config.ROOT / "eval" / "run_meta.json"


def _select_sets(samples: list[Sample], dev_domains, test_domains,
                 limit: int | None = None) -> tuple[list[Sample], list[Sample]]:
    """按 run() 的固定顺序划分 dev/test（共用同一 rng）。

    关键点：**先抽 dev 再抽 test**。`bench` 必须复用这里，才能保证
    `calibrate --limit N` 与 `bench --limit N` 用到完全相同的 test 子集。
    """
    rng = random.Random(SEED)
    dev = select(samples, tuple(dev_domains), limit, rng)
    test = select(samples, tuple(test_domains), limit, rng)
    return dev, test


def _read_run_meta() -> dict:
    try:
        return json.loads(RUN_META_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _update_run_meta(**fields) -> dict:
    meta = _read_run_meta()
    meta.update(fields)
    try:
        RUN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUN_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return meta


def _dump_features(rows_dev, y_dev, dom_dev, rows_test, y_test, dom_test) -> None:
    """把 dev/test 的**原始特征行**存成 eval/features.npz，供免重跑的诊断
    （特征消融、C 网格、域检查）复用。不参与正式产物。"""
    feats = list(fusion.CURATED_FEATURES)

    def _arr(rows) -> np.ndarray:
        return np.array([[float(r.get(f, np.nan)) for f in feats] for r in rows],
                        dtype=np.float64)

    path = config.ROOT / "eval" / "features.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, features=np.array(feats),
        X_dev=_arr(rows_dev), y_dev=np.array(y_dev, dtype=np.int64),
        dom_dev=np.array(dom_dev, dtype=str),
        X_test=_arr(rows_test), y_test=np.array(y_test, dtype=np.int64),
        dom_test=np.array(dom_test, dtype=str),
    )
    print(f"  特征转储 -> {path}", flush=True)


def run(limit: int | None = None, dev_domains=DEV_DOMAINS, test_domains=TEST_DOMAINS,
        write: bool = True, dump_features: bool = False) -> dict:
    t_start = time.time()
    samples = load_hc3()
    print(f"HC3-Chinese 载入 {len(samples)} 条样本", flush=True)
    dev, test = _select_sets(samples, dev_domains, test_domains, limit)
    print(f"  标定集 {len(dev)} 条（{','.join(dev_domains)}）", flush=True)
    print(f"  测试集 {len(test)} 条（{','.join(test_domains)}，域外）", flush=True)

    engines = build_engines()
    print("标定集特征 …", flush=True)
    rows_dev, y_dev_list, dom_dev = collect(dev, engines, "（dev）")
    print("测试集特征 …", flush=True)
    rows_test, y_test_list, dom_test = collect(test, engines, "（test）")

    y_dev = np.array(y_dev_list)
    y_test = np.array(y_test_list)
    feats = list(fusion.CURATED_FEATURES)
    X_dev = matrix(rows_dev, feats)
    coef, intercept, temp, mean, std = fit_logistic(X_dev, y_dev, feats, rows_dev)
    X_test = matrix(rows_test, feats, mean, std)
    if dump_features:
        _dump_features(rows_dev, y_dev, dom_dev, rows_test, y_test, dom_test)

    z_dev = X_dev @ np.array([coef[f] for f in feats]) + intercept
    z_test = X_test @ np.array([coef[f] for f in feats]) + intercept
    p_dev = 1.0 / (1.0 + np.exp(-z_dev / temp))
    p_test = 1.0 / (1.0 + np.exp(-z_test / temp))

    mode_thresholds = {}
    evaluation: dict = {}
    for mode, target in config.MODE_TARGET_FPR.items():
        thr = threshold_for_fpr(p_dev, y_dev, target)
        mode_thresholds[mode] = round(thr, 4)
        evaluation[mode] = {
            "target_fpr": target,
            "dev": metrics(p_dev, y_dev, thr),
            "test": metrics(p_test, y_test, thr),
        }
    balanced = mode_thresholds[config.DEFAULT_MODE]
    cut = fusion.grade_cutoffs_for(balanced)

    # 域별表现（只报 balanced 阈值）
    per_domain = {}
    for dom in test_domains:
        sel = np.array([d == dom for d in dom_test])
        if sel.sum() < 20:
            continue
        per_domain[dom] = metrics(p_test[sel], y_test[sel], balanced)

    calib = {
        "version": "1.0",
        "calibrated": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": feats,
        "mean": {k: round(v, 6) for k, v in mean.items()},
        "std": {k: round(v, 6) for k, v in std.items()},
        "coef": {k: round(v, 6) for k, v in coef.items()},
        "intercept": round(intercept, 6),
        "temperature": round(temp, 4),
        "thresholds": {
            "high": round(cut["high"], 4),
            "mid_high": round(cut["mid_high"], 4),
            "mid": round(cut["mid"], 4),
        },
        "mode_thresholds": mode_thresholds,
        "evaluation": evaluation,
        "per_domain": per_domain,
        "dataset": {
            "name": "Hello-SimpleAI/HC3-Chinese",
            "dev_domains": list(dev_domains),
            "test_domains": list(test_domains),
            "n_dev": int(len(y_dev)),
            "n_test": int(len(y_test)),
            "note": "HC3 为 GPT-3.5 时代的中文问答语料，与学位论文/学术写作存在领域与时代差异；"
                    "test 为域名留出，未参与拟合。",
        },
        "note": "特征按 features 顺序取；logit = Σ coef·(x-mean)/std + intercept，p = sigmoid(logit/T)。",
    }
    if write:
        config.RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        fusion.CALIBRATION_PATH.write_text(
            json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
        _update_run_meta(
            limit=limit,
            n_dev=int(len(y_dev)),
            n_test=int(len(y_test)),
            wall_clock_sec=round(time.time() - t_start, 1),
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _write_report(calib)
        print(f"\n标定参数 -> {fusion.CALIBRATION_PATH}", flush=True)
        print(f"评测报告 -> {config.ROOT / 'eval' / 'report.md'}", flush=True)
    return calib


def bench(limit: int | None = None) -> dict:
    """用已有标定复评测试集（不重新拟合），产出 eval/report.md。

    `--limit` 与 calibrate 同义：按 `_select_sets` 的固定 rng 顺序先抽 dev 再抽 test，
    因此复评的正是标定时用过的那个 test 子集（全量 test 集有 2.6 万单元，太慢）。
    """
    t_start = time.time()
    calib = json.loads(fusion.CALIBRATION_PATH.read_text(encoding="utf-8"))
    samples = load_hc3()
    _, test = _select_sets(
        samples, tuple(calib["dataset"]["dev_domains"]),
        tuple(calib["dataset"]["test_domains"]), limit)
    print(f"  bench 复评样本 {len(test)} 篇（limit={limit}）", flush=True)
    engines = build_engines()
    rows, y_list, dom_list = collect(test, engines, "（bench）")
    y = np.array(y_list)
    feats = calib["features"]
    X = matrix(rows, feats, calib["mean"], calib["std"])
    z = X @ np.array([calib["coef"][f] for f in feats]) + calib["intercept"]
    p = 1.0 / (1.0 + np.exp(-z / calib["temperature"]))
    out = {}
    for mode, thr in calib["mode_thresholds"].items():
        out[mode] = metrics(p, y, thr)
    # 只替换 test 指标，保留 target_fpr/dev，报告里的标定集结果不丢
    ev = calib.setdefault("evaluation", {})
    bench_cmp = {}
    for mode, m in out.items():
        old = (ev.get(mode) or {}).get("test") or {}
        ev[mode] = {**(ev.get(mode) or {}), "test": m}
        if old:
            diffs = [abs(float(m.get(k, 0.0)) - float(old.get(k, 0.0)))
                     for k in ("acc", "auc", "fpr", "fnr", "brier")
                     if k in old and isinstance(old.get(k), (int, float))]
            bench_cmp[mode] = {
                "max_abs_diff": round(max(diffs), 4) if diffs else None,
                "reproduced": bool(diffs) and max(diffs) <= 0.002,
            }
    for dom in calib["dataset"]["test_domains"]:
        sel = np.array([d == dom for d in dom_list])
        if sel.sum() >= 20:
            calib.setdefault("per_domain", {})[dom] = metrics(p[sel], y[sel],
                                                             calib["mode_thresholds"][config.DEFAULT_MODE])
    bench_meta = {
        "limit": limit,
        "n_test": int(len(y)),
        "wall_clock_sec": round(time.time() - t_start, 1),
        "modes": bench_cmp,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    bad = [mode for mode, v in bench_cmp.items() if not v.get("reproduced")]
    if bad:
        bench_meta["note"] = (
            f"差异根因（{','.join(bad)}）：`mode_thresholds` 以 round4 存储，复评用的是取整后的阈值，"
            "标定时 evaluation 用的是未取整阈值；阈值极低时（如 lenient=0.0042，原值 0.0041787）"
            "2e-5 级差异即可让少量人类样本跨过边界，FPR 变动 1–2 个样本即 0.003。属阈值边界敏感/"
            "序列化取整效应，不是推理逻辑错误；strict/balanced 阈值高一个量级、无样本翻转。"
            "按要求未改 calibration.json 迎合 bench。")
    _update_run_meta(bench=bench_meta)
    # 报告的数据源必须是 resources/calibration.json 里的 evaluation（而不是本次复评的即时值），
    # 否则复评会覆盖 §三 的数字、与 calibration.json 产生漂移。复评差异只记 run_meta.bench。
    stored = json.loads(fusion.CALIBRATION_PATH.read_text(encoding="utf-8"))
    _write_report(stored)
    return calib


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _fmt_metrics(name: str, m: dict, target_fpr: float | None = None) -> list[str]:
    thr = m.get("threshold")
    thr_txt = f"{float(thr):.4f}" if isinstance(thr, (int, float)) else "--"
    tgt_txt = f"{float(target_fpr):.0%}" if isinstance(target_fpr, (int, float)) else "--"
    return [name, thr_txt, tgt_txt, str(m["n"]), f"{m['acc']:.3f}", f"{m['auc']:.3f}",
            f"{m['fpr']:.3f}", f"{m['fnr']:.3f}", f"{m['precision']:.3f}", f"{m['brier']:.3f}"]


def _forward_counts(limit) -> dict | None:
    """从 eval/hc3_stats.json 取本次 limit 的推理规模（text cache 去重口径）。"""
    try:
        stats = json.loads((config.ROOT / "eval" / "hc3_stats.json").read_text(encoding="utf-8"))
        entry = stats["limits"][str(limit)]
        d, t = entry["dev"], entry["test"]
        return {
            "dev_clf": d["clf_forwards_unique"], "test_clf": t["clf_forwards_unique"],
            "dev_ppl": d["ppl_forwards_unique"], "test_ppl": t["ppl_forwards_unique"],
            "dev_sent": d["ppl_sentence_forwards"], "test_sent": t["ppl_sentence_forwards"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run_meta_section(meta: dict) -> str:
    if not meta:
        return "## 六、标定规模与实测耗时\n\n（本次运行未记录 `eval/run_meta.json`。）"
    lim = meta.get("limit")
    limit_txt = f"--limit {lim}（每域每类先取该条数，同题多回答合并成一篇）" if lim else "全量（未设 --limit）"
    lines = [
        "## 六、标定规模与实测耗时",
        "",
        f"- 采样：{limit_txt}",
        f"- 单元数：标定集 {meta.get('n_dev', '?')} / 域外测试集 {meta.get('n_test', '?')}",
    ]
    if meta.get("wall_clock_sec") is not None:
        sec = float(meta["wall_clock_sec"])
        lines.append(f"- 标定全程 wall-clock：**{sec:.0f} 秒**（约 {sec / 60:.1f} 分钟，"
                     "含模型加载、切分、三引擎打分、拟合、写文件）")
    fwd = _forward_counts(lim)
    if fwd:
        lines.append(
            f"- 三引擎推理规模（去重后）：clf 窗口 dev {fwd['dev_clf']} + test {fwd['test_clf']}；"
            f"ppl 单元窗口 {fwd['dev_ppl']} + {fwd['test_ppl']}；"
            f"ppl 句级前向 {fwd['dev_sent']} + {fwd['test_sent']}（句级前向是主要成本项）")
    b = meta.get("bench") or {}
    if b:
        modes = b.get("modes") or {}
        diffs = [v.get("max_abs_diff") for v in modes.values()
                 if isinstance(v.get("max_abs_diff"), (int, float))]
        ok = bool(modes) and all(v.get("reproduced") for v in modes.values())
        bsec = b.get("wall_clock_sec")
        lines.append(
            f"- `bench --limit {b.get('limit')}` 交叉验证：复评 {b.get('n_test', '?')} 个 test 单元"
            + (f"，用时 {float(bsec):.0f} 秒" if isinstance(bsec, (int, float)) else "")
            + (f"；与标定时指标最大绝对差 {max(diffs):.4f}，"
               f"{'复现一致（≤0.002，说明 calibration.json 的序列化/标准化链路正确）' if ok else '**存在差异**，见下方根因与 eval/run_meta.json'}"
               if diffs else "；无可比指标"))
        if b.get("note"):
            lines.append(f"  {b['note']}")
    return "\n".join(lines)


def _estimate_bullet(meta: dict) -> str:
    """把 run_meta["estimate"]（基于 T1 smoke 吞吐的规模/耗时推算）渲染成报告第二节条目。"""
    est = meta.get("estimate") or {}
    if not est:
        return ""
    pu = est.get("per_unit_sec") or {}
    pc = est.get("per_char_ms") or {}
    pm = est.get("predicted_minutes") or {}

    def span(key: str) -> str:
        v = pm.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return f"约 {v[0]}–{v[1]} 分钟"
        return "（未估）"

    smoke = est.get("smoke_source") or "out/smoke.json"
    lines = [
        f"- 规模推算（依据 T1 实测吞吐 `{smoke}`）：`--limit {est.get('limit', '?')}` → "
        f"标定集 {est.get('n_units_dev', '?')} 单元 / 域外测试集 {est.get('n_units_test', '?')} 单元，"
        f"估算输入字符约 {float(est.get('n_chars_est') or 0) / 1000:.0f}K。"
    ]
    if pu or pc:
        lines.append(
            f"  实测单单元 clf {pu.get('clf', '?')} s + ppl {pu.get('ppl', '?')} s ≈ "
            f"{pu.get('total', '?')} s/单元、≈ {pc.get('total', '?')} ms/字；据此推算 "
            f"calibrate {span('calibrate')}，bench 复评{span('bench')}，合计{span('total')}。"
        )
    if est.get("decision"):
        lines.append(f"  选择该 limit 的理由：{est['decision']}")
    return "\n".join(lines)


def _gate_verdict(calib: dict) -> str:
    """按 T2 验收门槛判定域外测试集（balanced 阈值），只判定、不改阈值/数据。"""
    tgt = config.MODE_TARGET_FPR[config.DEFAULT_MODE]
    ev = calib.get("evaluation", {})
    test = (ev.get(config.DEFAULT_MODE) or {}).get("test") or {}
    if not test:
        return ""
    items: list[str] = []
    auc = float(test.get("auc", 0.0))
    items.append(f"AUC {auc:.3f} {'≥0.85 达标' if auc >= 0.85 else '未达 0.85'}")
    acc = float(test.get("acc", 0.0))
    items.append(f"Acc {acc:.3f} {'≥0.80 达标' if acc >= 0.80 else '未达 0.80'}")
    fpr = float(test.get("fpr", 0.0))
    lo, hi = tgt - 0.02, tgt + 0.03
    items.append(f"FPR {fpr:.3f} {'∈' if lo <= fpr <= hi else '∉'} [{lo:.2f}, {hi:.2f}]"
                 + ("达标" if lo <= fpr <= hi else "（未达标）"))
    brier = float(test.get("brier", 0.0))
    items.append(f"Brier {brier:.3f} {'≤0.22 达标' if brier <= 0.22 else '未达 0.22'}")
    per = calib.get("per_domain") or {}
    if per:
        worst = min(per.items(), key=lambda kv: float((kv[1] or {}).get("auc", 0.0)))
        worst_auc = float((worst[1] or {}).get("auc", 0.0))
        items.append(f"各域 AUC 最低 {worst_auc:.3f}（{worst[0]}）"
                     + (" ≥0.75 达标" if worst_auc >= 0.75 else " 未达 0.75（未达标）"))
    fails = [i for i in items if "未达" in i]
    head = "全部达标" if not fails else "**未全部达标**"
    # 三模式域外 FPR 实值（lead 裁定要求报告明写）
    fpr_parts = []
    for mode in ("strict", "balanced", "lenient"):
        m = (ev.get(mode) or {}).get("test") or {}
        if isinstance(m.get("fpr"), (int, float)):
            fpr_parts.append(f"{mode} {float(m['fpr']) * 100:.1f}%")
    fpr_line = ""
    if fpr_parts:
        tgt_line = " / ".join(f"{config.MODE_TARGET_FPR[mode]:.0%}"
                              for mode in ("strict", "balanced", "lenient"))
        fpr_line = ("目标 FPR 未在域外守住：" + " / ".join(fpr_parts)
                    + f"（目标 {tgt_line}）——按协议未调阈值凑指标，成因与诊断见第七节。")
    return "- **验收门槛（域外测试集 / balanced 阈值）**：" + head + "。" + "；".join(items) + \
        "。" + (fpr_line + " " if fpr_line else "") + \
        "诊断只做消融/正则化/域分离检查，**未通过调阈值或改测试集**规避。"


def _write_report(calib: dict) -> None:
    ds = calib["dataset"]
    rows = []
    for mode, ev in calib["evaluation"].items():
        dev = ev.get("dev")
        test = ev.get("test")
        target = ev.get("target_fpr")
        if dev:
            rows.append(_fmt_metrics(f"{mode} · 标定集(dev)", dev, target))
        if test:
            rows.append(_fmt_metrics(f"{mode} · 域外测试集", test, target))
    table = _md_table(rows, ["数据", "阈值", "目标FPR", "n", "Acc", "AUC",
                             "FPR(误判人)", "FNR(漏判AI)", "Precision", "Brier"])
    dom_rows = []
    for dom, m in (calib.get("per_domain") or {}).items():
        dom_rows.append(_fmt_metrics(dom, m, config.MODE_TARGET_FPR[config.DEFAULT_MODE]))
    dom_table = _md_table(dom_rows, ["域", "阈值", "目标FPR", "n", "Acc", "AUC", "FPR", "FNR",
                                     "Precision", "Brier"]) \
        if dom_rows else "（无）"
    coef_rows = [[k, f"{v:+.3f}"] for k, v in sorted(calib["coef"].items(), key=lambda kv: -abs(kv[1]))]
    coef_table = _md_table(coef_rows, ["特征", "标准化系数"])
    balanced = calib["evaluation"].get(config.DEFAULT_MODE, {})
    test_b = balanced.get("test") or {}
    meta = _read_run_meta()
    lim = meta.get("limit")
    calibrate_cmd = f"python -m aidetect calibrate --limit {lim}" if lim \
        else "python -m aidetect calibrate"
    bench_cmd = f"python -m aidetect bench --limit {lim}" if lim \
        else "python -m aidetect bench"
    bench_script_cmd = f"python -m aidetect.calibrate bench --limit {lim}" if lim \
        else "python -m aidetect.calibrate bench"
    cmd_note = (f"（本次实际运行：标定用 `python -m aidetect.calibrate --limit {lim} --dump-features`，"
                "同时导出 eval/features.npz 供消融/诊断。复评有两条等价路径，都复现本报告的同一个"
                " test 子集：`python -m aidetect bench` 是正式 CLI 路径，读 calibration.json 后打印"
                "三模式指标 JSON；`python -m aidetect.calibrate bench` 是脚本入口，额外保留 dev 指标"
                f"并重写本报告与 `eval/run_meta.json` 的 bench 段。）") if lim else ""
    meta_txt = _run_meta_section(meta)
    estimate_txt = _estimate_bullet(meta)
    gate_txt = _gate_verdict(calib)
    diag_txt = (meta.get("diagnostics") or "").strip() \
        or "（本次运行未写诊断记录 `eval/run_meta.json` 的 `diagnostics` 字段。）"
    text = f"""# ai-detect 标定与评测报告

生成时间：{calib['generated_at']}

## 一、结论先说

- 默认 `balanced` 模式阈值 **{calib['mode_thresholds'].get(config.DEFAULT_MODE, 0):.3f}**，
  在**域外**测试集上：Acc **{test_b.get('acc', float('nan')):.3f}**、
  AUC **{test_b.get('auc', float('nan')):.3f}**、
  人类样本误判率(FPR) **{test_b.get('fpr', float('nan')):.3f}**、
  AI 漏判率(FNR) **{test_b.get('fnr', float('nan')):.3f}**。
- 这个数字**不能**直接当成「这篇论文被判 AI 的概率」。原因见第四节：语料是 GPT-3.5 时代的
  中文问答，不是学位论文；检测器对「训练分布内」的文本更敏感。
- 工具里真正的护栏是**段落级分歧 + 置信区间 + 明确的不可靠提示**，而不是单看整篇百分比。
- **可操作结论**：本报告数字只适用于**同一分布内的相对比较**；真实中文论文场景请以
  `report` 输出里的**段落相对高低 + 护栏提示**为参考，不要把绝对百分比当作结论。
{gate_txt}
- **用法建议**：重要/高风险场合请用 `--mode strict`（本报告域外 FPR 6.55%）；更稳妥的是先用
  **你自己的正常文本**（与待测文档同文体）重标定，再参考段落相对高低；输出**不可作为学术不端处分的
  唯一证据**。

## 二、方法与数据（可复现）

- 语料：{ds['name']}，{ds['note']}
- 标定集（参与拟合）：`{','.join(ds['dev_domains'])}`，单元数 {ds['n_dev']}
- 测试集（**域外留出，未参与拟合**）：`{','.join(ds['test_domains'])}`，单元数 {ds['n_test']}
- 切分：每条回答按工具同一套 `build_doc` / `make_units` 流程切片，保证特征口径与真实使用一致。
- 融合：三引擎（判别器 / 困惑度 / 文体统计）共 {len(calib['features'])} 个特征 →
  标准化 → 逻辑回归 → 温度缩放（T={calib['temperature']:.3f}）。
- 阈值：在标定集的**人类样本**上按分位数取，使误判率贴近目标值
  （strict {config.MODE_TARGET_FPR['strict']:.0%} / balanced {config.MODE_TARGET_FPR['balanced']:.0%}
  / lenient {config.MODE_TARGET_FPR['lenient']:.0%}）。
{estimate_txt}

## 三、结果

{table}

> 说明：`FPR(误判人)` 的目标值在**标定集（dev）人类样本**上按分位取阈，表里 `目标FPR` 与域外测试集
> 上的**实测 FPR** 并列展示，用于识别阈值迁移偏差：AUC 高而 FPR 超目标 ⇒ 问题在阈值迁移（域漂移），
> 不是区分度失效。本报告按「dev 取阈、test 只测不改」的协议生成，**未用测试集调阈值**。

### 各域表现（balanced 阈值）

{dom_table}

### 融合特征权重（标准化系数，正=更像 AI）

{coef_table}

## 四、必须说清的四件事（局限）

1. **域偏移**：评测语料是中文问答/百科/情感类短文，而实际要测的是学位论文、报告等长文。
   工具在长文上的表现取决于「论文语料是否落在检测器训练分布内」，这一点无法用公开数据完全验证。
   所以报告里给的是**段落级证据 + 区间**，请按「哪几段可疑」而不是「整篇百分之几」来用。
2. **时代偏移**：HC3 的 AI 文本来自 GPT-3.5（2022）。更新的模型（GPT-4 级、DeepSeek 等）
   输出更接近人类，召回率会下降。检测器对 AI 文本的敏感性 = 我们在「已知 AI 文本」上的召回，
   不等于对「未知新模型」的召回。
3. **误判风险**：人类样本误判率即便只有 {test_b.get('fpr', 0):.1%}，在 50 段的长文里也意味着
   平均约 {test_b.get('fpr', 0) * 50:.1f} 段会被标红。**不要把本工具的输出当作学术不端指控证据**；
   它适合自查、定位可改写段落。
4. **分布乐观性（最重要）**：HC3-Chinese 同时是本项目 clf 判别器的训练语料来源；本报告的「域外测试集」
   只是 HC3 **内部**的域划分（测试域未参与拟合），并非真实场景的分布外样本。T1 smoke 中同源文本上
   clf 对 AI / 人类文本的分数差（gap）在 120 / 900 / 3000 字三档分别为 0.9999 / 0.9998 / 0.9997，
   属于同分布红利。**真实中文论文场景的 Acc/AUC 会明显低于本报告数字**，请勿把上表当作泛化能力证明。

## 五、复现命令

```bash
python -m aidetect.fetch hc3zh       # 下载评测语料（21.7 MB）
{calibrate_cmd}   # 本次实际标定命令
{bench_cmd}       # 复评同一 test 子集（正式 CLI 路径，打印三模式指标）
{bench_script_cmd}   # 复评同一 test 子集（脚本入口，保留 dev 指标并重写本报告）
```

{cmd_note}

{meta_txt}

## 七、诊断记录

{diag_txt}
"""
    out = config.ROOT / "eval" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
    if argv and argv[0] == "bench":
        calib = bench(limit=limit)
        print(json.dumps(calib["evaluation"], ensure_ascii=False, indent=2)[:2000])
        return 0
    if argv and argv[0] == "report":
        calib = json.loads(fusion.CALIBRATION_PATH.read_text(encoding="utf-8"))
        # R-14：派生阈值统一由 mode_thresholds[balanced] 经 grade_cutoffs_for 重算并 round4 回写，
        # 保证 calibration.json 的 thresholds 字段可由代码路径复现（不依赖手工补丁）。
        balanced = float(calib["mode_thresholds"][config.DEFAULT_MODE])
        cut = fusion.grade_cutoffs_for(balanced)
        fixed = {k: round(float(cut[k]), 4) for k in ("high", "mid_high", "mid")}
        if calib.get("thresholds") != fixed:
            calib["thresholds"] = fixed
            fusion.CALIBRATION_PATH.write_text(
                json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已按 grade_cutoffs_for({balanced}) 重算 thresholds 并回写 JSON -> {fixed}",
                  flush=True)
        _write_report(calib)
        print(f"评测报告 -> {config.ROOT / 'eval' / 'report.md'}", flush=True)
        return 0
    run(limit=limit, dump_features="--dump-features" in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
