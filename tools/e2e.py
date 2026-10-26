"""ai-detect 端到端验收脚本（开发工具，不被包导入、不参与打包）。

真实跑 CLI（不是单测里的假引擎）：

    python -m aidetect report tests/fixtures/mixed_hc3.txt --out-dir out

默认验收 fixture 是 ``tests/fixtures/mixed_hc3.txt``：由本地 HC3-Chinese 语料
（``data/hc3zh/all.jsonl``，离线、不联网）里 4 条真实 chatgpt 回答 + 4 条真实 human
回答交替拼接而成，来源与选择规则写在 ``mixed_hc3.meta.json`` 里（可审计、可重建：
``python tools/e2e.py --build-mixed-fixture``）。断言因此建立在段落级强证据上：

* AI 段 ``p_ai`` 中位数 > 0.5，人类段 ``p_ai`` 中位数 < 0.1；
* 整篇 ``ai_rate_chars`` 落在 ``(0.30, 0.85)``——下界来自「8 段里 4 段是真 AI 文本、
  字数权重约一半」的保守估计，上界防止「全判 AI」的退化；不用 (0.15,0.85) 那种
  拍脑袋区间（旧区间的原因见 out/ 下的历史日志与 T4 汇报）。

本脚本负责：计时、校验 HTML/JSON 产物、校验混排文档的段落级分数与 AI 率区间、
校验降级路径、（可选）doctor 自检，并打印一张逐条断言表。可重复运行：每次覆盖
out/ 下的产物，并按 mtime 校验产物确实是本次新生成的，避免拿到上一次的旧结果。

用法（在项目根目录执行）：

    python tools/e2e.py                    # 完整验收：doctor + report + 降级
    python tools/e2e.py --repeat 2         # 重复测耗时 + 校验 AI 率可复现
    python tools/e2e.py --skip-doctor      # 标定还没就绪时，只跑 report + 降级
    python tools/e2e.py --threads 8        # 给子进程设 AIDETECT_THREADS
    python tools/e2e.py --json-out out/e2e_result.json
    python tools/e2e.py --build-mixed-fixture   # 由 HC3 语料重建默认 fixture（含 meta）
    python tools/e2e.py --record-only --fixture tests/fixtures/essay_3000.txt --out-dir out/human_balanced
    python tools/e2e.py --compare-human    # 额外记录人类单篇 fixture 的 balanced/strict 结果

退出码：0 = 全部断言通过；1 = 有断言失败；2 = 环境/参数问题（fixture 缺失等）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "mixed_hc3.txt"
HUMAN_FIXTURE_PATH = ROOT / "tests" / "fixtures" / "essay_3000.txt"
HC3_DATASET = ROOT / "data" / "hc3zh" / "all.jsonl"
DEFAULT_OUT_DIR = ROOT / "out"
TIME_BUDGET_SEC = 60.0
# 默认按「混排」验收：8 段里 4 段真实 AI 文本（字数权重约一半），故下界取 0.30；
# 上界 0.85 防止「全判 AI」的退化。真正的事实来源是 fixture meta 的 assertions 字段。
DEFAULT_AI_RATE_RANGE = (0.30, 0.85)
HTML_MIN_BYTES = 20 * 1024
MIXED_TRIM_LIMIT = 400
MIXED_META_SUFFIX = ".meta.json"

# 混排 fixture 的固定选择规格：严格按行号 + 答案索引，重建结果逐字节可复现。
# 每两条一组 = (label, dataset 行号, 字段 chatgpt|human, 答案索引, source 域名)；
# AI 侧 = 4 个不同域名/问题的 chatgpt_answers[0]；人类侧 = 同 4 个域名、不同问题的 human_answers[0]。
# 选择规则与合法性见混合 fixture 的 meta（选 220–430 字、超过 400 字按句边界截断）。
MIXED_SPEC = (
    ("ai", 1273, "chatgpt", 0, "open_qa"),
    ("human", 1310, "human", 0, "open_qa"),
    ("ai", 3309, "chatgpt", 0, "baike"),
    ("human", 3300, "human", 0, "baike"),
    ("ai", 9622, "chatgpt", 0, "finance"),
    ("human", 9634, "human", 0, "finance"),
    ("ai", 10334, "chatgpt", 0, "medicine"),
    ("human", 10310, "human", 0, "medicine"),
)
SENT_END = "。！？!?；;…"
CLOSERS = "\"'”’」』）)"

# 任务书里点名要有的 JSON 字段；实际 to_dict() 用了一些更精确的键名，
# 这里用显式别名映射，既满足验收口径，又不改动源码字段命名。
JSON_KEY_ALIASES = {
    "stats": "document",
    "grade": "grade_counts",
    "sentences": "suspicious_sentences",
}
REQUIRED_JSON_KEYS = (
    "ai_rate_chars",
    "ai_rate_paras",
    "grade",
    "paragraphs",
    "sentences",
    "engines",
    "calibration",
    "stats",
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str = "", required: bool = True) -> None:
        self.checks.append(Check(name=name, ok=bool(ok), detail=detail, required=required))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]


# --------------------------------------------------------------- 混排 fixture 构建
def _norm_answer(text: str) -> str:
    """HC3 答案规范化：CRLF→LF、换行→空格、连续空白折叠、首尾去空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _trim_answer(text: str, limit: int) -> tuple[str, bool]:
    """超过 limit 时按句边界截断（保留句末标点及其后的闭合引号/括号）。"""
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    cut = max((i for i, ch in enumerate(head) if ch in SENT_END), default=-1)
    if cut >= limit // 2:
        end = cut + 1
        while end < len(text) and end < limit and text[end] in CLOSERS:
            end += 1
        return text[:end], True
    return head, True


def _read_hc3_record(line_no: int, dataset: Path = HC3_DATASET) -> dict:
    """按 1-based 行号流式读取 HC3 数据集的一行（不整文件载入内存）。"""
    last = 0
    with dataset.open("r", encoding="utf-8") as fh:
        for last, line in enumerate(fh, 1):
            if last == line_no:
                return json.loads(line)
    raise ValueError(f"HC3 数据集只有 {last} 行，取不到第 {line_no} 行：{dataset}")


def _answer_of(record: dict, field: str, index: int) -> str:
    key = "chatgpt_answers" if field == "chatgpt" else "human_answers"
    # HC3 里 human_answers 可能混入 None，必须过滤后取第 index 条可用答案。
    answers = [a for a in (record.get(key) or []) if isinstance(a, str) and a.strip()]
    if index >= len(answers):
        raise ValueError(f"字段 {key} 只有 {len(answers)} 条可用答案，取不到索引 {index}")
    return answers[index]


def _dataset_sha256(dataset: Path = HC3_DATASET) -> str:
    digest = hashlib.sha256()
    with dataset.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _char_count(text: str) -> int:
    """与 aidetect.segment.char_count 同口径：去掉所有空白后的字符数。"""
    return len(re.sub(r"\s", "", text))


def mixed_meta_path(fixture: Path = FIXTURE_PATH) -> Path:
    return fixture.parent / (fixture.stem + MIXED_META_SUFFIX)


def build_mixed_fixture(
    fixture: Path = FIXTURE_PATH,
    limit: int = MIXED_TRIM_LIMIT,
    dataset: Path = HC3_DATASET,
) -> dict:
    """按 MIXED_SPEC 从本地 HC3 语料重建混排 fixture 与同名 .meta.json（返回 meta）。

    fixture 只含正文（空行分隔段落）；来源、行号、问题、字符数、截断标记、断言口径
    全部写进 meta，保证可复现、可审计。
    """
    dataset_hash = _dataset_sha256(dataset)
    texts: list[str] = []
    sections: list[dict] = []
    for paragraph, (label, line_no, field, answer_index, source) in enumerate(MIXED_SPEC):
        record = _read_hc3_record(line_no, dataset)
        trimmed, was_trimmed = _trim_answer(_norm_answer(_answer_of(record, field, answer_index)), limit)
        texts.append(trimmed)
        sections.append(
            {
                "paragraph": paragraph,
                "label": label,
                "source": source,
                "line_no": line_no,
                "field": field,
                "answer_index": answer_index,
                "question": record.get("question", ""),
                "chars": _char_count(trimmed),
                "truncated": was_trimmed,
            }
        )
    body = "\n\n".join(texts) + "\n"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(body, encoding="utf-8")

    lo, hi = DEFAULT_AI_RATE_RANGE
    meta = {
        "dataset": "Hello-SimpleAI/HC3-Chinese",
        "dataset_path": str(dataset),
        "dataset_sha256": dataset_hash,
        "fixture": fixture.name,
        "size_bytes": len(body.encode("utf-8")),
        "total_chars": _char_count(body),
        "normalization": "CRLF→LF；换行→空格；连续空白折叠；首尾去空白",
        "selection": {
            "rule": (
                "AI 侧取 4 条真实 chatgpt_answers[0]（4 个不同域名/问题），人类侧取 4 条真实 "
                "human_answers[0]（同 4 个域名、不同问题），按 ai,human 交替拼接；"
                f"单段上限 {limit} 字，超出按句边界截断"
            ),
            "trim_limit": limit,
            "spec": [
                {
                    "paragraph": i,
                    "label": label,
                    "line_no": line_no,
                    "field": field,
                    "answer_index": answer_index,
                    "source": source,
                }
                for i, (label, line_no, field, answer_index, source) in enumerate(MIXED_SPEC)
            ],
        },
        "sections": sections,
        "order": [s["label"] for s in sections],
        "assertions": {
            "ai_rate_chars_range": [lo, hi],
            "rationale": (
                "8 段中 4 段为真实 chatgpt 回答、4 段为真实 human 回答，字数权重约各一半；"
                f"整篇 AI 率下界取 {lo}（宽松但非平凡），上界 {hi} 防止「全判 AI」的退化。"
                "段落级断言更严格：AI 段 p_ai 中位数 > 0.5、人类段 p_ai 中位数 < 0.1。"
                "这是测试设计选择（用真实 AI 文本做证据），不是对产品指标的期望值。"
            ),
            "ai_label_median_min": 0.5,
            "human_label_median_max": 0.1,
        },
        "generator": "python tools/e2e.py --build-mixed-fixture",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    mixed_meta_path(fixture).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def load_fixture_meta(fixture: Path) -> dict | None:
    """读取 fixture 同名 .meta.json；没有就当普通文档验收（返回 None）。"""
    meta_path = mixed_meta_path(fixture)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def check_mixed_paragraph_scores(payload: dict, meta: dict) -> list[Check]:
    """混排文档的段落级强证据：AI 段的 p_ai 显著高于人类段（不是整篇拍区间）。"""
    sections = meta.get("sections") or []
    assertions = meta.get("assertions") or {}
    ai_median_min = float(assertions.get("ai_label_median_min", 0.5))
    human_median_max = float(assertions.get("human_label_median_max", 0.1))

    rows = payload.get("paragraphs") or []
    by_index = {r.get("index"): r for r in rows if isinstance(r, dict)}
    scored: list[tuple[str, float, dict]] = []
    missing: list[int] = []
    for sec in sections:
        idx = sec.get("paragraph")
        row = by_index.get(idx)
        p_ai = _as_float(row.get("p_ai")) if isinstance(row, dict) else None
        if p_ai is None:
            missing.append(idx)
            continue
        scored.append((str(sec.get("label")), p_ai, row))

    checks = [
        Check(
            "混排 fixture 可评分段落数等于 meta 段数（每段都能定位到 p_ai）",
            len(scored) == len(sections),
            f"meta {len(sections)} 段，可评分 {len(scored)} 段"
            + (f"，未评分段 {missing}" if missing else ""),
        )
    ]

    ai_scores = [p for label, p, _ in scored if label == "ai"]
    human_scores = [p for label, p, _ in scored if label == "human"]
    ai_median = statistics.median(ai_scores) if ai_scores else None
    human_median = statistics.median(human_scores) if human_scores else None
    per_para = "，".join(
        f"P{sec['paragraph']}({sec['label']})={_as_float((by_index.get(sec['paragraph']) or {}).get('p_ai'))}"
        for sec in sections
    )
    checks.append(
        Check(
            f"AI 段 p_ai 中位数 > {ai_median_min}（段落级强证据）",
            ai_median is not None and ai_median > ai_median_min,
            f"AI 段中位数 {ai_median}；{per_para}",
        )
    )
    checks.append(
        Check(
            f"人类段 p_ai 中位数 < {human_median_max}（段落级强证据）",
            human_median is not None and human_median < human_median_max,
            f"人类段中位数 {human_median}；{per_para}",
        )
    )
    threshold = _as_float(payload.get("threshold"))
    if threshold is not None:
        flagged = [
            (label, row.get("index"), p_ai)
            for label, p_ai, row in scored
            if p_ai > threshold
        ]
        checks.append(
            Check(
                "被判为 AI 的段落（p_ai > threshold）恰好都是 AI 段（INFO，仅参考）",
                all(label == "ai" for label, _idx, _p in flagged),
                f"被判 AI 的段：{flagged}；threshold={threshold}",
                required=False,
            )
        )
    return checks


# --------------------------------------------------------------------- 纯校验函数
def resolve_json_key(payload: dict, requested: str) -> str | None:
    """把验收口径里的字段名解析成 JSON 里实际存在的键（含别名）。"""
    if requested in payload:
        return requested
    alias = JSON_KEY_ALIASES.get(requested)
    if alias and alias in payload:
        return alias
    return None


def check_html_self_contained(html: str) -> list[Check]:
    """HTML 必须离线自包含：无 http/https 外链、无外部样式表/字体/图片，JS 内联。"""
    low = html.lower()
    external = [
        pat for pat in ("http://", "https://", "<link", "@import", "url(", "<img", "<iframe")
        if pat in low
    ]
    checks = [
        Check("HTML 自包含：不含 http://", "http://" not in low),
        Check("HTML 自包含：不含 https://", "https://" not in low),
        Check("HTML 自包含：不含外部样式表/字体（<link）", "<link" not in low),
        Check(
            "HTML 自包含：无外链资源（src= / @import / url( / <img / <iframe）",
            "src=" not in low and "@import" not in low and "url(" not in low
            and "<img" not in low and "<iframe" not in low,
            f"命中：{external}" if external else "",
        ),
        Check("HTML 自包含：不含 <script src=", "<script src=" not in low),
        Check("HTML 含内联 <script>（交互表格依赖内联 JS）", "<script" in low),
        Check(
            f"HTML 体积 > {HTML_MIN_BYTES // 1024}KB",
            len(html.encode("utf-8")) > HTML_MIN_BYTES,
            f"{len(html.encode('utf-8')) / 1024:.1f} KB",
            required=False,
        ),
    ]
    return checks


def _as_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def check_json_payload(
    payload: dict,
    ai_rate_range: tuple[float, float] | None = DEFAULT_AI_RATE_RANGE,
) -> list[Check]:
    """校验 report JSON 的字段、口径与混排文档的 AI 率区间。

    ai_rate_range=None 表示 record-only（例如人类单篇 fixture / 域外文本）：只记录
    实测值，不判定通过与否——避免把「产品表现」当成测试通过条件。
    """
    checks: list[Check] = []
    if not isinstance(payload, dict):
        checks.append(Check("JSON 顶层为对象", False, f"实际 {type(payload).__name__}"))
        return checks
    checks.append(Check("JSON 顶层为对象", True))

    missing = [k for k in REQUIRED_JSON_KEYS if resolve_json_key(payload, k) is None]
    mapped = {k: resolve_json_key(payload, k) for k in REQUIRED_JSON_KEYS}
    alias_note = "，".join(f"{k}→{v}" for k, v in mapped.items() if v and v != k)
    checks.append(
        Check(
            "JSON 含全部必需字段",
            not missing,
            (f"缺失：{missing}" if missing else "") + (f"（别名映射：{alias_note}）" if alias_note else ""),
        )
    )

    rate = _as_float(payload.get("ai_rate_chars"))
    if ai_rate_range is None:
        checks.append(
            Check("ai_rate_chars（record-only：仅记录，不作通过/失败判定）", True, f"实测 {rate}", required=False)
        )
    else:
        lo, hi = ai_rate_range
        checks.append(
            Check(
                f"混排文档 ai_rate_chars ∈ ({lo}, {hi})",
                rate is not None and lo < rate < hi,
                f"实测 {rate}",
            )
        )

    paras_rate = _as_float(payload.get("ai_rate_paras"))
    checks.append(
        Check("ai_rate_paras ∈ [0, 1]", paras_rate is not None and 0.0 <= paras_rate <= 1.0, f"实测 {paras_rate}")
    )

    doc = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    checks.append(
        Check(
            "document（=stats）含 total_chars/body_chars/units/ref_cut",
            isinstance(doc, dict)
            and all(k in doc for k in ("total_chars", "body_chars", "units", "ref_cut")),
            "缺字段：" + str([k for k in ("total_chars", "body_chars", "units", "ref_cut") if k not in doc]),
        )
    )

    paragraphs = payload.get("paragraphs")
    checks.append(Check("paragraphs 为列表且非空", isinstance(paragraphs, list) and bool(paragraphs),
                        f"{len(paragraphs) if isinstance(paragraphs, list) else 'N/A'} 段"))

    if isinstance(paragraphs, list) and isinstance(doc, dict):
        n_scorable = sum(1 for row in paragraphs if isinstance(row, dict) and row.get("p_ai") is not None)
        unit_rows = [row for row in paragraphs if isinstance(row, dict) and row.get("unit") is not None]
        units = doc.get("units")
        checks.append(
            Check(
                "可评分段落数等于 document.units",
                isinstance(units, int) and n_scorable == units,
                f"units={units}，可评分行={n_scorable}",
            )
        )
        checks.append(
            Check(
                "paragraphs 覆盖全部段落（含标题/被剔除区）",
                isinstance(doc.get("paragraphs"), int) and len(paragraphs) == doc.get("paragraphs"),
                f"doc.paragraphs={doc.get('paragraphs')}，JSON 行数={len(paragraphs)}",
            )
        )
        kinds = {row.get("kind") for row in paragraphs if isinstance(row, dict)}
        valid_kinds = {"heading", "body", "meta"}
        structure_ok = bool(kinds) and kinds <= valid_kinds and "body" in kinds
        if doc.get("ref_cut") is True:
            structure_ok = structure_ok and "meta" in kinds
        checks.append(
            Check(
                "段落结构自洽（kind 合法、含 body；ref_cut 时含 meta）",
                structure_ok,
                f"实际 kind={sorted(k for k in kinds if k)}，ref_cut={doc.get('ref_cut')}",
            )
        )

    ref_cut = doc.get("ref_cut") if isinstance(doc, dict) else None
    excluded = doc.get("excluded_chars") if isinstance(doc, dict) else None
    consistent_ref = (
        isinstance(ref_cut, bool)
        and isinstance(excluded, int)
        and ((not ref_cut and excluded == 0) or (ref_cut and excluded > 0))
    )
    checks.append(
        Check(
            "ref_cut 与 excluded_chars 一致（有引用区则 >0，无引用区则 =0）",
            consistent_ref,
            f"ref_cut={ref_cut}，excluded_chars={excluded}",
        )
    )

    engines = payload.get("engines")
    ok_engines = isinstance(engines, list) and bool(engines) and all(
        isinstance(e, dict) and "name" in e and "available" in e for e in engines
    )
    checks.append(
        Check(
            "engines 为非空列表且每项含 name/available",
            ok_engines,
            f"实际 {[e.get('name') for e in engines] if isinstance(engines, list) else engines}",
        )
    )

    calib = payload.get("calibration")
    checks.append(
        Check(
            "calibration 含 calibrated/mode_thresholds",
            isinstance(calib, dict) and "calibrated" in calib and "mode_thresholds" in calib,
            f"calibrated={calib.get('calibrated') if isinstance(calib, dict) else 'N/A'}",
        )
    )

    grades = payload.get("grade_counts")
    grade_keys = ("high", "mid_high", "mid", "low")
    checks.append(
        Check(
            "grade_counts 覆盖 high/mid_high/mid/low",
            isinstance(grades, dict) and all(k in grades for k in grade_keys),
            f"实际 {sorted(grades) if isinstance(grades, dict) else grades}",
        )
    )

    sents = payload.get("suspicious_sentences")
    checks.append(
        Check(
            "suspicious_sentences（=sentences）为列表且元素含 text/p_ai",
            isinstance(sents, list) and all(isinstance(s, dict) and "text" in s and "p_ai" in s for s in sents),
            f"{len(sents) if isinstance(sents, list) else 'N/A'} 句",
            required=False,
        )
    )

    mode = payload.get("mode")
    threshold = _as_float(payload.get("threshold"))
    checks.append(
        Check(
            "mode/threshold 合法",
            isinstance(mode, str) and threshold is not None and 0.0 <= threshold <= 1.0,
            f"mode={mode}，threshold={threshold}",
            required=False,
        )
    )
    return checks


def check_degraded_run(returncode: int, stderr: str, stdout: str = "") -> list[Check]:
    """缺模型时必须是「干净降级」：退出码 3、提示 fetch 命令、无崩栈/内部错误。"""
    text = f"{stderr or ''}\n{stdout or ''}"
    return [
        Check("降级路径退出码为 3（EXIT_RESOURCE）", returncode == 3, f"实际 {returncode}"),
        Check(
            "降级提示含 'python -m aidetect fetch'",
            "python -m aidetect fetch" in text,
        ),
        Check(
            "降级无崩栈（无 Traceback、无内部错误）",
            "Traceback (most recent call last)" not in text and "内部错误" not in text,
        ),
    ]


def check_timing(seconds: list[float], budget: float = TIME_BUDGET_SEC) -> Check:
    bad = [s for s in seconds if s >= budget]
    detail = f"实测 {[round(s, 2) for s in seconds]}s"
    if len(seconds) > 1:
        detail += f"（min/median/max = {min(seconds):.2f}/{statistics.median(seconds):.2f}/{max(seconds):.2f}s）"
    return Check(f"真实 report 耗时 < {budget:g}s", not bad, detail)


# --------------------------------------------------------------------- 运行时
def build_env(threads: int | None = None, extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("AIDETECT_DEBUG", None)  # 统一走「用户视角」，不让调试开关把堆栈抛出来
    if threads:
        env["AIDETECT_THREADS"] = str(threads)
    if extra:
        env.update(extra)
    return env


def run_cmd(args: list[str], env: dict | None = None, timeout: float | None = None):
    """跑一个子命令，返回 (CompletedProcess, 秒)。"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return proc, time.perf_counter() - t0


def _short(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …"


def _print_checks(report: Report) -> None:
    width = max((len(c.name) for c in report.checks), default=10)
    print("\n== 断言明细 ==")
    for c in report.checks:
        if not c.required:
            tag = "INFO"  # 非必需项只展示，不计入失败
        else:
            tag = "PASS" if c.ok else "FAIL"
        line = f"[{tag}] {c.name.ljust(width)}"
        if c.detail:
            line += f"  {c.detail}"
        print(line)


def _print_metrics(report: Report) -> None:
    m = report.metrics
    if not m:
        return
    print("\n== 摘要 ==")
    for key, value in m.items():
        print(f"  {key}: {value}")


def run_human_comparison(args: argparse.Namespace, py: str, env: dict, report: Report) -> None:
    """人类单篇 fixture 的 balanced/strict 记录（仅记录，不判定 AI 率）。

    用途：把「人类文本在标定后的域外误报表现」如实留在 e2e 产物里（lead 要求），
    不把产品表现当成测试通过条件。
    """
    fixture = Path(args.human_fixture)
    if not fixture.is_absolute():
        fixture = (ROOT / fixture).resolve()
    if not fixture.exists():
        report.add(f"人类单篇 fixture 存在：{fixture.name}", False, str(fixture))
        return
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = (ROOT / out_root).resolve()
    for mode in ("balanced", "strict"):
        out_dir = out_root / "human" / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{fixture.stem}_report.json"
        try:
            proc, dt = run_cmd(
                [py, "-m", "aidetect", "report", str(fixture), "--out-dir", str(out_dir), "--mode", mode],
                env=env,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            report.add(f"人类单篇 fixture {mode}：超时（>{args.timeout:g}s）", False)
            continue
        payload = {}
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
            except (json.JSONDecodeError, OSError):
                payload = {}
        ok_run = proc.returncode == 0 and bool(payload)
        report.add(
            f"人类单篇 fixture {mode} 退出码 0 且产物可解析",
            ok_run,
            "" if ok_run else f"rc={proc.returncode}，stderr：{_short(proc.stderr, 300)}",
        )
        paras = [
            _as_float(p.get("p_ai"))
            for p in (payload.get("paragraphs") or [])
            if isinstance(p, dict) and _as_float(p.get("p_ai")) is not None
        ]
        detail = (
            f"ai_rate_chars={payload.get('ai_rate_chars')}，"
            f"ai_rate_paras={payload.get('ai_rate_paras')}，"
            f"threshold={payload.get('threshold')}，耗时 {dt:.1f}s"
        )
        if paras:
            detail += f"，段落 p_ai 中位数 {statistics.median(paras):.4f}"
        report.add(f"人类单篇 fixture {mode}（record-only：仅记录，不判定 AI 率）", True, detail, required=False)
        report.metrics[f"人类单篇 fixture / {mode}"] = {
            "ai_rate_chars": payload.get("ai_rate_chars"),
            "ai_rate_paras": payload.get("ai_rate_paras"),
            "threshold": payload.get("threshold"),
            "grade_counts": payload.get("grade_counts"),
            "elapsed_sec": payload.get("elapsed_sec"),
            "wall_sec": round(dt, 2),
        }


def run(args: argparse.Namespace) -> int:
    fixture = Path(args.fixture)
    if not fixture.is_absolute():
        fixture = (ROOT / fixture).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    meta = load_fixture_meta(fixture)
    rate_range: tuple[float, float] | None = DEFAULT_AI_RATE_RANGE
    if meta:
        meta_range = (meta.get("assertions") or {}).get("ai_rate_chars_range")
        if isinstance(meta_range, (list, tuple)) and len(meta_range) == 2:
            rate_range = (float(meta_range[0]), float(meta_range[1]))
    if args.record_only:
        rate_range = None
    kind = "混排（HC3 拼接）" if meta else "普通文档"
    rate_note = f"({rate_range[0]}, {rate_range[1]})" if rate_range else "record-only（不判定）"

    print("== ai-detect 端到端验收 ==")
    print(f"  python    : {sys.version.split()[0]} ({sys.executable})")
    print(f"  fixture   : {fixture}   类型: {kind}")
    print(f"  out-dir   : {out_dir}")
    print(f"  mode      : {args.mode}   threads={args.threads or '默认'}   repeat={max(1, args.repeat)}")
    print(f"  耗时预算  : < {args.budget:g}s/轮   AI 率区间: {rate_note}")
    if meta:
        print(f"  fixture meta: {mixed_meta_path(fixture).name}（dataset={meta.get('dataset')}，"
              f"total_chars={meta.get('total_chars')}，{len(meta.get('sections') or [])} 段）")

    report = Report()
    if not fixture.exists():
        print(f"\n[错误] fixture 不存在：{fixture}", file=sys.stderr)
        return 2
    fixture_text = fixture.read_text(encoding="utf-8")
    report.metrics["fixture 原始字符数（含空白）"] = len(fixture_text)
    report.metrics["fixture 去空白字符数"] = _char_count(fixture_text)

    py = sys.executable
    env = build_env(threads=args.threads)

    # 1) doctor（可选：标定/语料没就绪时用 --skip-doctor 跳过）
    if args.skip_doctor:
        report.add("doctor 自检（已跳过 --skip-doctor）", True, "跳过", required=False)
    else:
        proc, dt = run_cmd([py, "-m", "aidetect", "doctor"], env=env, timeout=args.timeout)
        report.add(
            "doctor 退出码 0（环境/模型/标定/语料就绪）",
            proc.returncode == 0,
            f"实际 {proc.returncode}，耗时 {dt:.1f}s" + (f"，末行：{_short(proc.stdout.splitlines()[-1] if proc.stdout.splitlines() else '', 120)}"),
        )

    # 2) 真实 report（可多轮测耗时）
    html_path = out_dir / f"{fixture.stem}_report.html"
    json_path = out_dir / f"{fixture.stem}_report.json"
    started_at = time.time()
    times: list[float] = []
    last_stdout = ""
    repeat_runs: list[dict] = []
    for i in range(max(1, args.repeat)):
        cmd = [py, "-m", "aidetect", "report", str(fixture), "--out-dir", str(out_dir), "--mode", args.mode]
        try:
            proc, dt = run_cmd(cmd, env=env, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            report.add(f"report 第 {i + 1} 轮：超时（>{args.timeout:g}s）", False)
            break
        times.append(dt)
        last_stdout = proc.stdout
        tag = f"第 {i + 1} 轮" if args.repeat > 1 else ""
        report.add(f"report {tag}退出码 0".strip(), proc.returncode == 0,
                   "" if proc.returncode == 0 else f"stderr：{_short(proc.stderr)}")
        if proc.returncode != 0:
            print(f"\n[stderr]\n{_short(proc.stderr, 4000)}")
            break
        # 记下本轮产物里的 AI 率，用于验证 --repeat 的可复现性（固定 seed 应完全一致）
        if json_path.exists():
            try:
                run_payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
            except (json.JSONDecodeError, OSError):
                run_payload = {}
            repeat_runs.append(
                {
                    "ai_rate_chars": run_payload.get("ai_rate_chars"),
                    "ai_rate_paras": run_payload.get("ai_rate_paras"),
                    "grade_counts": run_payload.get("grade_counts"),
                    "elapsed_sec": run_payload.get("elapsed_sec"),
                }
            )

    if times:
        t = check_timing(times, args.budget)
        report.checks.append(t)
        report.metrics["report 实测耗时(s)"] = [round(x, 2) for x in times]
        print(f"\n  实测耗时：{[round(x, 2) for x in times]}s")

    if len(repeat_runs) > 1:
        sigs = {
            (
                r.get("ai_rate_chars"),
                r.get("ai_rate_paras"),
                json.dumps(r.get("grade_counts"), sort_keys=True),
            )
            for r in repeat_runs
        }
        report.add(
            f"--repeat {len(repeat_runs)} 轮 AI 率/分段完全一致（固定 seed 可复现）",
            len(sigs) == 1,
            f"逐轮 ai_rate_chars/ai_rate_paras："
            f"{[(r.get('ai_rate_chars'), r.get('ai_rate_paras')) for r in repeat_runs]}",
        )
        report.metrics["repeat 每轮（ai_rate_chars / ai_rate_paras / 内部 elapsed_sec）"] = [
            {k: r.get(k) for k in ("ai_rate_chars", "ai_rate_paras", "elapsed_sec")}
            for r in repeat_runs
        ]

    # 3) 产物：存在 + 是本次新生成
    for path, label in ((html_path, "HTML"), (json_path, "JSON")):
        exists = path.exists()
        report.add(f"{label} 产物存在：{path.name}", exists, str(path))
        if exists:
            fresh = path.stat().st_mtime >= started_at - 2
            report.add(f"{label} 产物为本次新生成（mtime 校验）", fresh,
                       f"mtime={time.strftime('%H:%M:%S', time.localtime(path.stat().st_mtime))}")

    # 4) HTML 自包含
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="replace")
        for c in check_html_self_contained(html):
            report.checks.append(c)
        if "HTML 报告" in last_stdout and html_path.name in last_stdout:
            report.add("CLI 打印的 HTML 路径与 --out-dir 一致", True)
        else:
            report.add("CLI 打印的 HTML 路径与 --out-dir 一致", False, _short(last_stdout, 300))

    # 5) JSON：可解析 + 字段/口径/区间
    payload = None
    if json_path.exists():
        raw = json_path.read_text(encoding="utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            report.add("JSON 可 json.load 解析", True, f"{len(raw) / 1024:.1f} KB")
        except json.JSONDecodeError as exc:
            report.add("JSON 可 json.load 解析", False, f"{exc}")
    if payload is not None:
        for c in check_json_payload(payload, ai_rate_range=rate_range):
            report.checks.append(c)
        if meta:
            if args.record_only:
                report.add("段落级断言（record-only 已跳过）", True, "跳过", required=False)
            else:
                for c in check_mixed_paragraph_scores(payload, meta):
                    report.checks.append(c)
                report.metrics["混排逐段 p_ai（段号/标签/分数）"] = [
                    {
                        "paragraph": sec["paragraph"],
                        "label": sec["label"],
                        "source": sec.get("source"),
                        "chars": sec.get("chars"),
                        "p_ai": next(
                            (
                                p.get("p_ai")
                                for p in (payload.get("paragraphs") or [])
                                if isinstance(p, dict) and p.get("index") == sec["paragraph"]
                            ),
                            None,
                        ),
                    }
                    for sec in (meta.get("sections") or [])
                ]
        doc = payload.get("document") or {}
        calib = payload.get("calibration") or {}
        report.metrics.update(
            {
                "ai_rate_chars": payload.get("ai_rate_chars"),
                "ai_rate_paras": payload.get("ai_rate_paras"),
                "ai_rate_ci": payload.get("ai_rate_ci"),
                "grade_counts": payload.get("grade_counts"),
                "document": {k: doc.get(k) for k in ("total_chars", "body_chars", "excluded_chars", "paragraphs", "units", "ref_cut")},
                "calibration.calibrated": calib.get("calibrated"),
                "mode/threshold": f"{payload.get('mode')} / {payload.get('threshold')}",
                "engines": [(e.get("name"), e.get("available")) for e in (payload.get("engines") or [])],
            }
        )

    # 6) 降级路径：AIDETECT_HOME 指向空临时目录
    tmp_home = Path(tempfile.mkdtemp(prefix="aidetect_e2e_home_"))
    try:
        degrade_env = build_env(threads=args.threads)
        degrade_env["AIDETECT_HOME"] = str(tmp_home)
        try:
            proc, dt = run_cmd(
                [py, "-m", "aidetect", "report", str(fixture), "--out-dir", str(tmp_home / "out")],
                env=degrade_env,
                timeout=args.timeout,
            )
            report.checks.extend(check_degraded_run(proc.returncode, proc.stderr, proc.stdout))
            stray = list((tmp_home / "out").glob("*")) if (tmp_home / "out").exists() else []
            report.add("降级路径不产出报告文件", not stray, f"意外产物：{stray}" if stray else "")
            report.metrics["降级路径耗时(s)"] = round(dt, 2)
        except subprocess.TimeoutExpired:
            report.add("降级路径：超时（预期应立刻返回 3）", False, f">{args.timeout:g}s")
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)

    # 6.5) 人类单篇 fixture 对照（记录用；不参与通过/失败）
    if args.compare_human:
        run_human_comparison(args, py, env, report)

    # 7) 汇总
    _print_checks(report)
    _print_metrics(report)

    total = sum(1 for c in report.checks if c.required)
    failed = report.failed
    print(f"\n结论：{'全部通过' if not failed else f'{len(failed)} 项失败'}（必需断言 {total} 条）")
    print(f"产物：HTML {html_path}；JSON {json_path}")

    if args.json_out:
        dest = Path(args.json_out)
        if not dest.is_absolute():
            dest = ROOT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload_out = {
            "ok": not failed,
            "failed": [c.name for c in failed],
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, "required": c.required} for c in report.checks],
            "metrics": report.metrics,
            "html": str(html_path),
            "json": str(json_path),
        }
        dest.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入：{dest}")

    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ai-detect 端到端验收（开发脚本）")
    ap.add_argument("--fixture", default=str(FIXTURE_PATH), help="用于验收的输入文档（默认混排 fixture）")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="产物目录（默认 out/）")
    ap.add_argument("--mode", default="balanced", choices=["strict", "balanced", "lenient"])
    ap.add_argument("--threads", type=int, default=None, help="设 AIDETECT_THREADS")
    ap.add_argument("--repeat", type=int, default=1, help="report 重复轮数（测耗时 + 校验可复现）")
    ap.add_argument("--budget", type=float, default=TIME_BUDGET_SEC, help="单轮耗时上限（秒）")
    ap.add_argument("--timeout", type=float, default=180.0, help="单个子进程的硬超时（秒）")
    ap.add_argument("--skip-doctor", action="store_true", help="跳过 doctor（标定未就绪时用）")
    ap.add_argument("--json-out", default=None, help="把验收结果写到 JSON 文件")
    ap.add_argument(
        "--record-only",
        action="store_true",
        help="只记录 AI 率、不判定区间/段落级断言（用于人类文本、域外文本）",
    )
    ap.add_argument(
        "--compare-human",
        action="store_true",
        help=f"额外记录人类单篇 fixture（{HUMAN_FIXTURE_PATH.name}）的 balanced/strict 结果",
    )
    ap.add_argument("--human-fixture", default=str(HUMAN_FIXTURE_PATH), help="人类单篇对照 fixture 路径")
    ap.add_argument(
        "--build-mixed-fixture",
        action="store_true",
        help="按 MIXED_SPEC 从本地 HC3 语料重建混排 fixture 与 .meta.json，然后退出",
    )
    args = ap.parse_args(argv)

    if args.build_mixed_fixture:
        if not HC3_DATASET.is_file():
            print(f"[错误] 找不到本地 HC3 数据集：{HC3_DATASET}", file=sys.stderr)
            return 2
        fixture = Path(args.fixture)
        if not fixture.is_absolute():
            fixture = (ROOT / fixture).resolve()
        meta = build_mixed_fixture(fixture)
        print(f"已重建混排 fixture：{fixture}")
        print(f"  来源：{meta['dataset']}（sha256 {meta['dataset_sha256'][:12]}…，{meta['dataset_path']}）")
        print(f"  总字数：{meta['total_chars']}（去空白），{len(meta['sections'])} 段")
        for sec in meta["sections"]:
            print(
                f"  P{sec['paragraph']} {sec['label']:5} {sec['source']:9} line={sec['line_no']:6} "
                f"chars={sec['chars']:4} truncated={sec['truncated']} question={sec['question'][:28]}"
            )
        print(f"  meta：{mixed_meta_path(fixture)}")
        print(
            f"  断言口径：ai_rate_chars ∈ {tuple(meta['assertions']['ai_rate_chars_range'])}；"
            f"AI 段 p_ai 中位数 > {meta['assertions']['ai_label_median_min']}、"
            f"人类段 < {meta['assertions']['human_label_median_max']}"
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
