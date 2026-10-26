"""端到端验收相关测试（无模型也能跑；真实模型路径按需 skip）。

覆盖四块：
1. 默认验收 fixture ``tests/fixtures/mixed_hc3.txt``（真实 HC3-Chinese 语料拼接：
   4 条 chatgpt 回答 + 4 条 human 回答交替）与其 ``mixed_hc3.meta.json``：
   fixture 与 meta 一致、可由本地语料逐字节重建（可审计、可复现）、8 段都能被
   build_doc/make_units 识别为独立单元、无空单元、无 >800 字单元；
2. 人类单篇 fixture ``tests/fixtures/essay_3000.txt`` 的切分契约：标题/正文/引用区
   都被识别、参考文献被剔除（ref_cut）、超长段强制切开且首块恰好 800 字；
   ``_split_long_para`` 保证每块 ≤MAX_PARA_CHARS（多句累积 + 单句超长两条路径）；
3. ``tools/e2e.py`` 的纯校验函数（HTML 自包含、JSON 字段、AI 率区间/record-only、
   段落级强证据、降级输出、耗时预算），全部用合成数据测，不加载模型；
4. 产物层：``out/mixed_hc3_report.*`` 存在时校验，否则 skip；真实 report 子进程默认
   skip，需要模型就绪 + 环境变量 ``AIDETECT_E2E=1`` 才跑（避免单测意外加载 400MB 模型）。

断言的依据见 ``tools/e2e.py`` 顶部 docstring 与 ``mixed_hc3.meta.json`` 的 assertions：
混排 fixture 的 4 段是真实 AI 文本，段落级 p_ai 的分离度（AI 段中位数 > 0.5、
人类段 < 0.1）比整篇区间更硬；整篇区间 (0.30, 0.85) 只是宽松但非平凡的下界 + 防退化上界。
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIXED_FIXTURE = ROOT / "tests" / "fixtures" / "mixed_hc3.txt"
MIXED_META = ROOT / "tests" / "fixtures" / "mixed_hc3.meta.json"
ESSAY_FIXTURE = ROOT / "tests" / "fixtures" / "essay_3000.txt"
OUT_DIR = ROOT / "out"
REPORT_HTML = OUT_DIR / "mixed_hc3_report.html"
REPORT_JSON = OUT_DIR / "mixed_hc3_report.json"

from aidetect import fetch  # noqa: E402
from aidetect.segment import MAX_PARA_CHARS, build_doc, char_count, make_units, stats  # noqa: E402


def _load_e2e_module():
    """载入开发脚本 tools/e2e.py（它不属于 aidetect 包）。"""
    path = ROOT / "tools" / "e2e.py"
    assert path.exists(), f"缺少工具脚本：{path}"
    spec = importlib.util.spec_from_file_location("aidetect_tools_e2e", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclass 在 exec 期间需要能按模块名查回 sys.modules（e2e.py 用了 from __future__ import annotations）
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


e2e = _load_e2e_module()


def _failed(checks):
    return [c for c in checks if c.required and not c.ok]


# --------------------------------------------------------------------- 1. 混排 fixture
@pytest.fixture(scope="module")
def mixed_meta() -> dict:
    assert MIXED_META.exists(), f"缺少 meta：{MIXED_META}（运行 python tools/e2e.py --build-mixed-fixture）"
    return json.loads(MIXED_META.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mixed_text() -> str:
    assert MIXED_FIXTURE.exists(), f"缺少 fixture：{MIXED_FIXTURE}"
    return MIXED_FIXTURE.read_text(encoding="utf-8")


def test_mixed_fixture_and_meta_exist(mixed_text, mixed_meta):
    assert mixed_meta["dataset"].startswith("Hello-SimpleAI/HC3")
    assert mixed_meta["fixture"] == MIXED_FIXTURE.name
    assert len(mixed_meta["dataset_sha256"]) == 64
    assert mixed_meta["total_chars"] == char_count(mixed_text)
    sections = mixed_meta["sections"]
    assert len(sections) == 8, f"期望 8 段（4 AI + 4 human），实际 {len(sections)}"
    assert mixed_meta["order"] == ["ai", "human"] * 4


def test_mixed_fixture_is_auditable(mixed_text, mixed_meta):
    """fixture 与 meta 逐段一致，且来源/行号/规则齐全（可审计、可复现）。"""
    sections = mixed_meta["sections"]
    blocks = [b.strip() for b in mixed_text.split("\n\n") if b.strip()]
    assert len(blocks) == len(sections)
    for block, sec in zip(blocks, sections):
        assert char_count(block) == sec["chars"], f"P{sec['paragraph']} 字数与 meta 不一致"
        assert 200 <= sec["chars"] <= 430, f"P{sec['paragraph']} 字数 {sec['chars']} 不在 200–430"
        assert sec["label"] in ("ai", "human")
        assert isinstance(sec["line_no"], int) and sec["line_no"] > 0
        assert sec["field"] in ("chatgpt", "human")
        assert sec["source"], "缺少 source 域名"
        assert isinstance(sec["truncated"], bool)
    assert {sec["source"] for sec in sections} == {"open_qa", "baike", "finance", "medicine"}
    assert 1500 <= mixed_meta["total_chars"] <= 2500, f"fixture 总字数 {mixed_meta['total_chars']} 不在 1500–2500"
    assertions = mixed_meta["assertions"]
    assert tuple(assertions["ai_rate_chars_range"]) == (0.30, 0.85)
    assert assertions["ai_label_median_min"] == 0.5
    assert assertions["human_label_median_max"] == 0.1
    assert "rationale" in assertions and assertions["rationale"]


@pytest.mark.skipif(not (ROOT / "data" / "hc3zh" / "all.jsonl").is_file(), reason="本机没有 HC3 语料，无法重建")
def test_mixed_fixture_is_reproducible(tmp_path, mixed_text):
    """用本地 HC3 语料按 MIXED_SPEC 重建，结果应与仓库里的 fixture 逐字节一致。"""
    rebuilt = tmp_path / "mixed_hc3.txt"
    meta = e2e.build_mixed_fixture(rebuilt)
    assert rebuilt.read_text(encoding="utf-8") == mixed_text
    assert meta["total_chars"] == char_count(mixed_text)
    assert len(meta["sections"]) == 8


def test_mixed_fixture_segmentation(mixed_text):
    doc = build_doc(mixed_text, MIXED_FIXTURE.name)
    st = stats(doc)
    assert st["body_paragraphs"] == 8, f"正文段数 {st['body_paragraphs']}，期望 8"
    assert st["body_chars"] == st["total_chars"] == char_count(mixed_text)
    assert st["ref_cut"] is False, "混排 fixture 没有参考文献段"
    assert st["excluded_chars"] == 0

    units = make_units(doc)
    assert len(units) == 8, f"判定单元数 {len(units)}，期望 8"
    assert all(u.chars > 0 and u.text.strip() for u in units), "存在空判定单元"
    assert max(u.chars for u in units) <= MAX_PARA_CHARS
    assert sum(u.chars for u in units) == st["body_chars"]


# ------------------------------------------------------- 2. 人类单篇 fixture 切分契约
@pytest.fixture(scope="module")
def essay_text() -> str:
    return ESSAY_FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def essay_doc(essay_text):
    return build_doc(essay_text, ESSAY_FIXTURE.name)


def test_essay_fixture_exists_and_is_mixed(essay_text):
    assert ESSAY_FIXTURE.exists()
    assert "随着人工智能技术的快速发展" in essay_text  # AI 模板腔
    assert "说实话" in essay_text  # 人类腔（个人经历）
    assert "参考文献" in essay_text  # 引用区，用来触发 ref_cut


def test_fixture_segmentation_contract(essay_doc):
    st = stats(essay_doc)
    assert 2700 <= st["total_chars"] <= 3400, f"fixture 字数 {st['total_chars']} 不在 2700–3400"
    assert st["body_chars"] >= 2000
    assert st["body_paragraphs"] >= 6
    assert st["ref_cut"] is True, "参考文献段没有被识别（ref_cut=False）"
    assert st["excluded_chars"] > 0, "参考文献没有计入 excluded_chars"
    assert st["paragraphs"] > st["body_paragraphs"], "标题/引用区段落没有单独成段"

    kinds = {p.kind for p in essay_doc.paras}
    assert {"heading", "body", "meta"} <= kinds, f"段落类型不全：{sorted(kinds)}"

    units = make_units(essay_doc)
    assert len(units) >= 7, f"判定单元太少：{len(units)}"
    assert all(u.chars > 0 and u.text.strip() for u in units), "存在空判定单元"
    assert max(u.chars for u in units) <= MAX_PARA_CHARS, (
        f"存在 >{MAX_PARA_CHARS} 字单元：{[u.chars for u in units]}"
    )
    assert any(u.chars == MAX_PARA_CHARS for u in units), "fixture 没有触发 800 字句边界切分"
    assert sum(u.chars for u in units) == st["body_chars"]


def test_oversized_paragraph_forced_split(essay_text):
    """fixture 里恰好有一个 >800 字的原始段，它被切成 800 + 余量两块。"""
    blocks = [b.strip() for b in essay_text.split("\n\n") if b.strip()]
    long_blocks = [b for b in blocks if char_count(b) > MAX_PARA_CHARS]
    assert len(long_blocks) == 1, f"期望恰好 1 个超长段，实际 {[char_count(b) for b in blocks]}"
    sub = build_doc(long_blocks[0], "long_para.txt")
    body = [p for p in sub.paras if p.kind == "body"]
    assert len(body) >= 2, "超长段没有被切开"
    assert body[0].chars == MAX_PARA_CHARS, f"首块 {body[0].chars} 字，未按 fixture 约定对齐 800"
    assert all(p.chars <= MAX_PARA_CHARS for p in body)
    units = make_units(sub)
    assert len(units) >= 2
    assert max(u.chars for u in units) <= MAX_PARA_CHARS


def test_split_upper_bound_is_guaranteed():
    """_split_long_para 必须保证每块 ≤MAX_PARA_CHARS（多句累积 + 单句超长两条路径）。"""
    sent = "这是一句用于测试超长段落切分行为的完整中文句子，长度大约四十个字左右。"
    assert len(sent) < MAX_PARA_CHARS

    # 路径 1：多句累积（30 句 ≈ 1170 字）→ 在句边界切开，每块 ≤ 800
    doc = build_doc(sent * 30, "synthetic_multi.txt")
    body = [p for p in doc.paras if p.kind == "body"]
    assert len(body) >= 2, "多句超长段没有被切开"
    assert all(p.chars <= MAX_PARA_CHARS for p in body), (
        f"多句累积路径存在 >{MAX_PARA_CHARS} 字块：{[p.chars for p in body]}"
    )

    # 路径 2：单句本身超过上限（无标点长串）→ 按字符硬切，每块 ≤ 800
    long_sent = "长" * (MAX_PARA_CHARS * 2 + 137) + "。"
    doc2 = build_doc(long_sent, "synthetic_single.txt")
    body2 = [p for p in doc2.paras if p.kind == "body"]
    assert body2, "单句超长段没有被切开"
    assert all(p.chars <= MAX_PARA_CHARS for p in body2), (
        f"单句超长路径存在 >{MAX_PARA_CHARS} 字块：{[p.chars for p in body2]}"
    )


# --------------------------------------------------------------------- 3. e2e 纯校验函数
def _fake_payload(**overrides) -> dict:
    paragraphs = [
        {"index": 0, "chars": 4, "kind": "heading", "p_ai": None, "grade": "excluded", "unit": None},
        *[
            {"index": i + 1, "chars": 300, "kind": "body", "p_ai": 0.9, "grade": "high", "unit": i}
            for i in range(8)
        ],
        {"index": 9, "chars": 30, "kind": "meta", "p_ai": None, "grade": "excluded", "unit": None},
    ]
    payload = {
        "tool": "ai-detect",
        "version": "0.0.0",
        "document": {
            "name": "essay_3000.txt",
            "total_chars": 2971,
            "body_chars": 2841,
            "excluded_chars": 126,
            "paragraphs": len(paragraphs),
            "body_paragraphs": 8,
            "units": 8,
            "ref_cut": True,
            "avg_para_chars": 355.1,
        },
        "mode": "balanced",
        "threshold": 0.6,
        "ai_rate_chars": 0.55,
        "ai_rate_paras": 0.5,
        "ai_rate_ci": [0.3, 0.8],
        "grade_counts": {"high": 3, "mid_high": 1, "mid": 2, "low": 2},
        "warnings": [],
        "calibration": {
            "calibrated": True,
            "version": "test",
            "note": "",
            "mode_thresholds": {"strict": 0.75, "balanced": 0.6, "lenient": 0.45},
            "evaluation": {},
        },
        "engines": [
            {"name": "clf", "label": "判别器", "available": True, "note": ""},
            {"name": "ppl", "label": "困惑度", "available": True, "note": ""},
            {"name": "stylo", "label": "文体统计", "available": True, "note": ""},
        ],
        "paragraphs": paragraphs,
        "suspicious_sentences": [{"text": "示例句。", "p_ai": 0.9, "para": 1}],
        "elapsed_sec": 1.0,
    }
    payload.update(overrides)
    return payload


def _fake_mixed_payload(ai_p=0.9, human_p=0.01) -> tuple[dict, dict]:
    """构造与 meta sections 一致的 8 段 payload（4 AI + 4 human 交替）。"""
    meta = {
        "sections": [
            {
                "paragraph": i,
                "label": "ai" if i % 2 == 0 else "human",
                "source": "open_qa",
                "line_no": 100 + i,
                "field": "chatgpt" if i % 2 == 0 else "human",
                "answer_index": 0,
                "question": "q",
                "chars": 250,
                "truncated": False,
            }
            for i in range(8)
        ],
        "assertions": {"ai_label_median_min": 0.5, "human_label_median_max": 0.1},
    }
    paragraphs = [
        {
            "index": sec["paragraph"],
            "chars": sec["chars"],
            "kind": "body",
            "p_ai": ai_p if sec["label"] == "ai" else human_p,
            "grade": "high" if sec["label"] == "ai" else "low",
            "unit": sec["paragraph"],
        }
        for sec in meta["sections"]
    ]
    payload = _fake_payload(
        paragraphs=paragraphs,
        threshold=0.15,
        document={
            "name": "mixed_hc3.txt",
            "total_chars": 2000,
            "body_chars": 2000,
            "excluded_chars": 0,
            "paragraphs": 8,
            "body_paragraphs": 8,
            "units": 8,
            "ref_cut": False,
            "avg_para_chars": 250.0,
        },
        ai_rate_chars=0.55,
    )
    return payload, meta


def test_e2e_module_points_at_project_root():
    assert Path(e2e.ROOT).resolve() == ROOT
    assert Path(e2e.FIXTURE_PATH).resolve() == MIXED_FIXTURE.resolve()
    assert Path(e2e.HUMAN_FIXTURE_PATH).resolve() == ESSAY_FIXTURE.resolve()


def test_default_ai_rate_range_matches_meta(mixed_meta):
    assert tuple(e2e.DEFAULT_AI_RATE_RANGE) == tuple(mixed_meta["assertions"]["ai_rate_chars_range"])


def test_check_json_payload_passes_and_uses_aliases():
    payload = _fake_payload()
    assert _failed(e2e.check_json_payload(payload)) == []
    # 任务书口径 stats/grade/sentences 映射到实际键名 document/grade_counts/suspicious_sentences
    assert e2e.resolve_json_key(payload, "stats") == "document"
    assert e2e.resolve_json_key(payload, "grade") == "grade_counts"
    assert e2e.resolve_json_key(payload, "sentences") == "suspicious_sentences"


def test_check_json_payload_flags_missing_required_field():
    payload = _fake_payload()
    del payload["document"]
    failed = _failed(e2e.check_json_payload(payload))
    assert any("必需字段" in c.name for c in failed), [(c.name, c.detail) for c in failed]


def test_check_json_payload_flags_rate_out_of_mixed_range():
    failed = _failed(e2e.check_json_payload(_fake_payload(ai_rate_chars=0.05)))
    assert any("ai_rate_chars" in c.name for c in failed), [(c.name, c.detail) for c in failed]


def test_check_json_payload_record_only_does_not_judge_rate():
    payload = _fake_payload(ai_rate_chars=0.05)
    checks = e2e.check_json_payload(payload, ai_rate_range=None)
    assert _failed(checks) == []  # record-only 不把 AI 率当通过条件
    assert any("record-only" in c.name and not c.required for c in checks)


def test_check_json_payload_rejects_non_dict():
    failed = _failed(e2e.check_json_payload(["not", "a", "dict"]))  # type: ignore[arg-type]
    assert failed and "顶层为对象" in failed[0].name


def test_check_mixed_paragraph_scores_accepts_separated_labels():
    payload, meta = _fake_mixed_payload(ai_p=0.9, human_p=0.01)
    assert _failed(e2e.check_mixed_paragraph_scores(payload, meta)) == []


def test_check_mixed_paragraph_scores_flags_indistinguishable_labels():
    payload, meta = _fake_mixed_payload(ai_p=0.2, human_p=0.15)
    failed = _failed(e2e.check_mixed_paragraph_scores(payload, meta))
    names = [c.name for c in failed]
    assert any("AI 段 p_ai 中位数" in n for n in names), names
    assert any("人类段 p_ai 中位数" in n for n in names), names


def test_check_mixed_paragraph_scores_flags_missing_paragraph():
    payload, meta = _fake_mixed_payload()
    payload["paragraphs"] = payload["paragraphs"][:-1]
    failed = _failed(e2e.check_mixed_paragraph_scores(payload, meta))
    assert any("可评分段落数" in c.name for c in failed), [(c.name, c.detail) for c in failed]


def test_check_html_self_contained_accepts_inline_only():
    html = "<!doctype html><html><head><style>body{color:#333}</style></head>" \
           "<body><script>var a=1;</script></body></html>"
    assert _failed(e2e.check_html_self_contained(html)) == []


def test_check_html_self_contained_flags_external_resources():
    bad_http = "<html><body><a href=\"http://example.com\">x</a></body></html>"
    assert any("http://" in c.name for c in _failed(e2e.check_html_self_contained(bad_http)))
    bad_https = "<html><body><a href=\"https://example.com/x\">x</a></body></html>"
    assert any("https://" in c.name for c in _failed(e2e.check_html_self_contained(bad_https)))
    bad_src = "<html><body><script src=\"app.js\"></script></body></html>"
    assert any("<script src=" in c.name for c in _failed(e2e.check_html_self_contained(bad_src)))
    bad_link = "<html><head><link rel=\"stylesheet\" href=\"site.css\"></head></html>"
    assert any("<link" in c.name for c in _failed(e2e.check_html_self_contained(bad_link)))
    bad_font = "<html><head><style>@import url('font.woff2');</style></head><body></body></html>"
    assert any("外链资源" in c.name for c in _failed(e2e.check_html_self_contained(bad_font)))


def test_check_degraded_run_accepts_clean_failure():
    stderr = "缺少模型：clf, ppl。先运行 python -m aidetect fetch clf ppl\n"
    assert _failed(e2e.check_degraded_run(3, stderr)) == []


def test_check_degraded_run_flags_wrong_exit_and_stack():
    assert any("退出码" in c.name for c in _failed(e2e.check_degraded_run(0, "")))
    assert any("崩栈" in c.name for c in _failed(e2e.check_degraded_run(3, "Traceback (most recent call last):")))


def test_check_timing_budget():
    assert e2e.check_timing([12.0]).ok
    assert not e2e.check_timing([61.0]).ok
    multi = e2e.check_timing([10.0, 20.0, 30.0])
    assert multi.ok and "min/median/max" in multi.detail


# --------------------------------------------------------------------- 4. 产物层
@pytest.mark.skipif(not REPORT_JSON.exists(), reason="真实报告尚未产出：模型就绪后运行 python tools/e2e.py")
def test_real_report_json_satisfies_acceptance_checks(mixed_meta):
    payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    failed = _failed(e2e.check_json_payload(payload))
    failed += _failed(e2e.check_mixed_paragraph_scores(payload, mixed_meta))
    assert not failed, [(c.name, c.detail) for c in failed]


@pytest.mark.skipif(not REPORT_HTML.exists(), reason="真实报告尚未产出：模型就绪后运行 python tools/e2e.py")
def test_real_report_html_is_self_contained():
    html = REPORT_HTML.read_text(encoding="utf-8", errors="replace")
    failed = _failed(e2e.check_html_self_contained(html))
    assert not failed, [(c.name, c.detail) for c in failed]


def _models_ready() -> bool:
    return all(fetch.is_ready(name) for name in ("clf", "ppl"))


@pytest.mark.skipif(not _models_ready(), reason="本机缺少 clf/ppl 模型文件，跳过真实报告端到端")
@pytest.mark.skipif(
    os.environ.get("AIDETECT_E2E") != "1",
    reason="真实 report 会加载模型、耗时较长；设 AIDETECT_E2E=1 显式开启（正式验收用 python tools/e2e.py）",
)
def test_real_report_end_to_end(tmp_path, mixed_meta):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "aidetect", "report", str(MIXED_FIXTURE), "--out-dir", str(tmp_path)]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "总耗时" in proc.stdout
    html_path = tmp_path / f"{MIXED_FIXTURE.stem}_report.html"
    json_path = tmp_path / f"{MIXED_FIXTURE.stem}_report.json"
    assert html_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    failed = _failed(e2e.check_json_payload(payload))
    failed += _failed(e2e.check_mixed_paragraph_scores(payload, mixed_meta))
    assert not failed, [(c.name, c.detail) for c in failed]
    html_failed = _failed(e2e.check_html_self_contained(html_path.read_text(encoding="utf-8", errors="replace")))
    assert not html_failed, [(c.name, c.detail) for c in html_failed]
