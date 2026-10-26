"""单文件 HTML 报告：不依赖任何外部资源（离线可用），可直接发给别人看。

设计原则：
- 结论先行：整篇 AI 率 + 风险档 + 置信区间 + 护栏提示放在第一屏；
- 证据可查：逐段表给 P(AI)、各引擎分与「为什么」；
- 口径透明：每个数字都能追到定义、阈值和标定来源；
- 打印/存档友好：print CSS 隐藏交互元素，A4 分页。
"""
from __future__ import annotations

import html
import json
import time

from .aggregate import Analysis

CSS = """
:root{
  --ink:#1a1a1a; --ink-2:#4a4a4a; --ink-3:#767676; --line:#e3e0d8;
  --paper:#fbfaf6; --card:#ffffff; --accent:#7c4dff; --accent-2:#0f766e;
  --hi:#b42318; --hi-bg:#fee4e2; --mhi:#b54708; --mhi-bg:#fef0c7;
  --mid:#846200; --mid-bg:#fdf6d8; --low:#027a48; --low-bg:#dcfae6;
  --shadow:0 1px 2px rgba(20,18,12,.06),0 8px 24px rgba(20,18,12,.06);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.7 "Source Han Sans SC","Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 80px}
a{color:var(--accent)}
h1,h2,h3{font-family:"Source Han Serif SC","Noto Serif SC","Songti SC",Georgia,serif;
  line-height:1.25;margin:0}
header.top{border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:26px}
.eyebrow{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3)}
h1{font-size:30px;margin:8px 0 6px;letter-spacing:-.01em}
.meta{color:var(--ink-3);font-size:13px;display:flex;flex-wrap:wrap;gap:6px 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px;box-shadow:var(--shadow);margin:0 0 20px}
.verdict{display:grid;grid-template-columns:minmax(240px,1.1fr) 2fr;gap:26px;align-items:center}
@media(max-width:760px){.verdict{grid-template-columns:1fr}}
.big{font-size:56px;font-weight:700;line-height:1;letter-spacing:-.02em}
.big small{font-size:22px;font-weight:600;margin-left:2px}
.pill{display:inline-flex;align-items:center;gap:8px;border-radius:999px;
  padding:5px 12px;font-size:13px;font-weight:600;border:1px solid transparent}
.pill.high{background:var(--hi-bg);color:var(--hi);border-color:#f9c6c0}
.pill.mid_high{background:var(--mhi-bg);color:var(--mhi);border-color:#f7d9a0}
.pill.mid{background:var(--mid-bg);color:var(--mid);border-color:#ecdfa6}
.pill.low{background:var(--low-bg);color:var(--low);border-color:#a9e8c4}
.pill.neutral{background:#f1f0ea;color:var(--ink-2);border-color:var(--line)}
.sub{color:var(--ink-3);font-size:13px;margin-top:10px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;margin-top:4px}
.kv div{background:#faf9f4;border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kv b{display:block;font-size:19px;font-variant-numeric:tabular-nums}
.kv span{font-size:12px;color:var(--ink-3)}
.bar{height:12px;border-radius:999px;background:#eeebe1;overflow:hidden;margin:14px 0 6px;position:relative}
.bar>i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,#0f766e,#b54708,#b42318)}
.ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--ink-3)}
h2{font-size:20px;margin:0 0 6px}
h2 .n{color:var(--ink-3);font-weight:400;font-size:14px;margin-right:8px;font-family:inherit}
.sec-lead{color:var(--ink-2);font-size:14px;margin:0 0 16px}
.warn{background:#fff8e6;border:1px solid #f2d59b;border-left:4px solid #d99a1b;
  border-radius:10px;padding:12px 16px;margin:0 0 10px;font-size:14px;color:#5c4410}
.warn b{color:#7a4b00}
.ok{background:#effaf3;border:1px solid #b7e4c7;border-left:4px solid #2f9e63;
  border-radius:10px;padding:12px 16px;font-size:14px;color:#14532d}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;letter-spacing:.04em;color:var(--ink-3);font-weight:600;text-transform:uppercase}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tbody tr:hover{background:#fcfbf7}
.g{display:inline-block;min-width:44px;text-align:center;border-radius:6px;padding:1px 6px;
  font-size:12px;font-weight:600}
.g.high{background:var(--hi-bg);color:var(--hi)}
.g.mid_high{background:var(--mhi-bg);color:var(--mhi)}
.g.mid{background:var(--mid-bg);color:var(--mid)}
.g.low{background:var(--low-bg);color:var(--low)}
.g.excluded{background:#f1f0ea;color:var(--ink-3)}
td.pcell{position:relative}
td.pcell::before{content:"";position:absolute;inset:0;opacity:.55;
  background:linear-gradient(90deg,var(--heat) var(--w),transparent var(--w))}
td.pcell span{position:relative}
.ptext{color:var(--ink-2);font-size:13px}
details summary{cursor:pointer;color:var(--accent);font-size:12.5px;list-style:none}
details summary::-webkit-details-marker{display:none}
details[open] summary{margin-bottom:6px}
details .full{white-space:pre-wrap;background:#faf9f4;border:1px solid var(--line);
  border-radius:8px;padding:10px;font-size:13px;color:var(--ink-2)}
.sig{color:var(--ink-3);font-size:12px;margin-top:4px}
.sig li{margin:1px 0}
.chart{width:100%;height:120px;display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink-3);margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
ul.plain{margin:6px 0 0;padding-left:18px;color:var(--ink-2);font-size:13.5px}
ul.plain li{margin:4px 0}
code{background:#f1f0ea;border-radius:4px;padding:1px 5px;font-size:12.5px}
footer{color:var(--ink-3);font-size:12.5px;border-top:1px solid var(--line);padding-top:16px;margin-top:36px}
.grade-chip{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--ink-2)}
@media print{
  body{background:#fff}
  .wrap{max-width:none;padding:0}
  .card{box-shadow:none;border-color:#ddd;break-inside:avoid}
  details{display:none}
  .noprint{display:none}
}
"""

JS = """
(function(){
  var btns=document.querySelectorAll('[data-filter]');
  btns.forEach(function(b){b.addEventListener('click',function(){
    var k=b.getAttribute('data-filter');
    btns.forEach(function(x){x.classList.remove('on')});b.classList.add('on');
    document.querySelectorAll('tr[data-grade]').forEach(function(tr){
      tr.style.display=(k==='all'||tr.getAttribute('data-grade')===k)?'':'none';
    });
  });});
})();
"""

RISK = {
    "low": ("结论：AI 生成痕迹不明显", "AI 生成占比低于当前阈值的 1/2，文本整体更像人工写作。"),
    "mid": ("结论：存在少量 AI 特征", "AI 生成占比低于判定阈值但并非接近 0，可能有局部段落由模型辅助或经过润色。"),
    "mid_high": ("结论：AI 特征明显，建议逐段自查", "AI 生成占比已过判定阈值，建议按逐段表定位并改写可疑段落。"),
    "high": ("结论：AI 特征强，风险高", "AI 生成占比显著高于阈值，多段落被判为 AI 生成，建议整体重写后再交付。"),
}
GRADE_LABEL = {"high": "极高", "mid_high": "偏高", "mid": "中等", "low": "低", "excluded": "未纳入"}
GRADE_HEX = {"high": "#f6bdb6", "mid_high": "#fbd9a2", "mid": "#f4e7ad", "low": "#bfe8d0"}


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def risk_key(rate: float, threshold: float) -> str:
    if rate >= threshold * 1.25:
        return "high"
    if rate >= threshold:
        return "mid_high"
    if rate >= threshold * 0.5:
        return "mid"
    return "low"


def _grade_cutoffs(analysis: Analysis) -> dict[str, float]:
    """段落分级阈值：优先用本次分析实际使用的分级线（随 mode 变化），保证图例与判定一致。"""
    cuts = getattr(analysis, "cutoffs", None) or {}
    if all(k in cuts for k in ("high", "mid_high", "mid")):
        return {k: float(cuts[k]) for k in ("high", "mid_high", "mid")}
    t = float(getattr(analysis, "threshold", 0.6) or 0.6)
    return {
        "high": min(0.95, t + 0.25),
        "mid_high": min(0.92, t + 0.02),
        "mid": max(0.10, t - 0.15),
    }


def _pct(x: float, digits: int = 1) -> str:
    return f"{x*100:.{digits}f}%"


def _spark_chart(analysis: Analysis) -> str:
    """单元分条带图：每根柱子 = 一个单元，颜色按等级，高度按 P(AI)。"""
    scores = analysis.unit_scores
    if not scores:
        return ""
    w, h = 1000, 120
    n = len(scores)
    gap = 1.5 if n <= 80 else 0.6
    bar_w = max((w - gap * (n - 1)) / n, 1.2)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
             f'role="img" aria-label="段落 AI 概率分布">']
    parts.append(f'<line x1="0" y1="{h-30}" x2="{w}" y2="{h-30}" stroke="#d8d4c8" stroke-width="1"/>')
    thr_y = h - 30 - (h - 42) * analysis.threshold
    parts.append(f'<line x1="0" y1="{thr_y:.1f}" x2="{w}" y2="{thr_y:.1f}" '
                 f'stroke="#b42318" stroke-width="1" stroke-dasharray="4 4" opacity=".7"/>')
    for i, us in enumerate(scores):
        x = i * (bar_w + gap)
        bh = max(1.5, (h - 42) * us.p)
        y = h - 30 - bh
        color = GRADE_HEX.get(us.grade, "#cfd8dc")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                     f'rx="{min(2, bar_w/2):.1f}" fill="{color}">'
                     f'<title>第 {us.unit.index+1} 段 P(AI)={us.p:.2f}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def _engine_rows(analysis: Analysis) -> str:
    rows = []
    for e in analysis.engine_info:
        state = ('<span class="g low">可用</span>' if e["available"]
                 else '<span class="g excluded">不可用</span>')
        mean = "—" if e["mean_score"] is None else f"{e['mean_score']:.3f}"
        rows.append(
            f"<tr><td>{_esc(e['label'])}</td><td>{state}</td>"
            f"<td class='num'>{mean}</td><td class='num'>{e['elapsed_sec']:.1f}s</td>"
            f"<td class='ptext'>{_esc(e['note'] or '')}</td></tr>")
    return "".join(rows)


def _para_rows(analysis: Analysis) -> str:
    rows = []
    for r in analysis.paragraphs:
        if r.p is None:
            rows.append(
                f"<tr data-grade='excluded'><td class='num'>{r.index+1}</td>"
                f"<td>{_esc(r.kind)}</td><td class='num'>{r.chars}</td><td class='num'>—</td>"
                f"<td><span class='g excluded'>未纳入</span></td>"
                f"<td class='ptext'>{_esc(r.text[:90])}</td></tr>")
            continue
        w = f"{max(0.0, min(1.0, r.p))*100:.0f}%"
        heat = GRADE_HEX.get(r.grade, "#cfd8dc")
        sigs = "".join(f"<li>{_esc(s)}</li>" for s in r.signals) or "<li>未触发明显特征</li>"
        merged = " <span class='sub'>(合并段)</span>" if r.merged else ""
        rows.append(
            f"<tr data-grade='{r.grade}'>"
            f"<td class='num'>{r.index+1}</td>"
            f"<td>{_esc(r.kind)}{merged}</td>"
            f"<td class='num'>{r.chars}</td>"
            f"<td class='num pcell' style='--heat:{heat};--w:{w}'><span>{r.p:.2f}</span></td>"
            f"<td><span class='g {r.grade}'>{GRADE_LABEL.get(r.grade, r.grade)}</span></td>"
            f"<td class='ptext'>{_esc(r.text[:90])}"
            f"<details><summary>展开全文 / 判定依据</summary>"
            f"<div class='full'>{_esc(r.text)}</div>"
            f"<ul class='plain sig'>{sigs}</ul></details></td></tr>")
    return "".join(rows)


def _sentence_rows(analysis: Analysis) -> str:
    if not analysis.sentences:
        return ("<tr><td colspan='4' class='ptext'>没有句子超过句级阈值。"
                "（也可能判别引擎不可用，此时仅按段落分推定。）</td></tr>")
    rows = []
    for s in analysis.sentences:
        rows.append(
            f"<tr><td class='num'>{s.para_index+1}</td>"
            f"<td class='num'>{s.p:.2f}</td>"
            f"<td class='ptext'>{_esc(s.text)}</td></tr>")
    return "".join(rows)


def _calibration_block(analysis: Analysis) -> str:
    rc = analysis.recipe
    if rc.calibrated:
        test = (rc.evaluation.get("test") or {})
        m = test.get("test") if isinstance(test, dict) and "test" in test else test
        acc = m.get("acc") if isinstance(m, dict) else None
        auc = m.get("auc") if isinstance(m, dict) else None
        line = (f"标定版本 <code>{_esc(rc.version)}</code>"
                + (f"，域外测试集 Acc {acc:.3f} / AUC {auc:.3f}" if acc is not None else ""))
        return (f"<div class='ok'><b>已加载标定参数。</b>{line}。"
                f"阈值（{analysis.mode}）：{analysis.threshold:.3f}。</div>")
    return ("<div class='warn'><b>当前为未标定模式。</b>"
            f"{_esc(rc.note)}。分数仅可做相对比较，"
            "请先运行 <code>python -m aidetect calibrate</code> 生成标定参数。</div>")


def render(analysis: Analysis, generated_at: str | None = None) -> str:
    st = analysis.stats
    rate = analysis.ai_rate_chars
    rk = risk_key(rate, analysis.threshold)
    title, desc = RISK[rk]
    ci = analysis.ai_rate_ci
    ci_txt = (f"{_pct(ci[0])} – {_pct(ci[1])}（95%，bootstrap）"
              if ci else "样本段数太少，无法给出区间")
    generated_at = generated_at or time.strftime("%Y-%m-%d %H:%M:%S")
    calib = analysis.recipe
    ev = calib.evaluation or {}
    cuts = _grade_cutoffs(analysis)

    warns = "".join(f"<div class='warn'>{_esc(w)}</div>" for w in analysis.warnings)
    if not warns:
        warns = "<div class='ok'>未触发护栏提示：文本长度、引擎、标定状态均正常。</div>"

    fingerprint = json.dumps({
        "tool": "ai-detect",
        "calibration": {"version": calib.version, "calibrated": calib.calibrated},
        "threshold": analysis.threshold, "mode": analysis.mode,
        "ai_rate_chars": round(rate, 4),
    }, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 生成率检测报告 · {_esc(st.get('name', '文档'))}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header class="top">
  <div class="eyebrow">ai-detect · 本地离线检测</div>
  <h1>AI 生成率检测报告</h1>
  <div class="meta">
    <span>文档：<b>{_esc(st.get('name', ''))}</b></span>
    <span>正文 {st.get('body_chars', 0)} 字 / {st.get('units', 0)} 段</span>
    <span>模式：{_esc(analysis.mode)}（阈值 {analysis.threshold:.3f}）</span>
    <span>生成于 {_esc(generated_at)}</span>
    <span>耗时 {analysis.elapsed:.1f}s</span>
  </div>
</header>

<section class="card">
  <div class="verdict">
    <div>
      <div class="pill {rk}">{_esc(title)}</div>
      <div class="big" style="margin-top:14px">{rate*100:.1f}<small>%</small></div>
      <div class="sub">AI 生成占比（字数口径，置信区间 {ci_txt}）</div>
    </div>
    <div>
      <div class="bar"><i style="width:{max(1.0, min(100.0, rate*100)):.1f}%"></i></div>
      <div class="ticks"><span>0%</span><span>判定阈值 {analysis.threshold*100:.0f}%</span><span>100%</span></div>
      <div class="kv" style="margin-top:18px">
        <div><b>{analysis.ai_rate_paras*100:.0f}%</b><span>AI 段落占比（{analysis.grade_counts.get('high',0)+analysis.grade_counts.get('mid_high',0)}/{st.get('units',0)} 段）</span></div>
        <div><b>{analysis.grade_counts.get('high',0)}</b><span>极高风险段</span></div>
        <div><b>{analysis.grade_counts.get('mid_high',0)}</b><span>偏高风险段</span></div>
        <div><b>{st.get('excluded_chars',0)}</b><span>已剔除字数（参考文献等）</span></div>
      </div>
      <div class="sub" style="margin-top:14px">{_esc(desc)}</div>
    </div>
  </div>
</section>

<section class="card">
  <h2>先说风险：这些结论不能怎么用</h2>
  <p class="sec-lead">机器检测没有「学术不端判决」效力。本报告给的是特征证据与区间，请按证据用。</p>
  {warns}
  <ul class="plain">
    <li><b>不要只看整篇百分比。</b>真正的信息在逐段表：哪几段触发了什么特征。</li>
    <li><b>人类写作也会被误判</b>（术语密集、句式工整、模板化公文腔，都会让分数升高）。</li>
    <li><b>最新模型更难检出。</b>检测器对旧模型更敏感；「没测出来」不等于「不是 AI 写的」。</li>
    <li>本工具全部本地运行，文本不出本机；除首次下载模型外不联网。</li>
  </ul>
</section>

<section class="card">
  <h2><span class="n">01</span>段落分布</h2>
  <p class="sec-lead">每根柱子是一个段落（或合并段）的 P(AI)；虚线是当前模式的判定阈值。</p>
  {_spark_chart(analysis)}
  <div class="legend">
    <span><i style="background:{GRADE_HEX['high']}"></i>极高（≥{cuts['high']:.2f}）</span>
    <span><i style="background:{GRADE_HEX['mid_high']}"></i>偏高（≥{cuts['mid_high']:.2f}）</span>
    <span><i style="background:{GRADE_HEX['mid']}"></i>中等（≥{cuts['mid']:.2f}）</span>
    <span><i style="background:{GRADE_HEX['low']}"></i>低</span>
  </div>
</section>

<section class="card">
  <h2><span class="n">02</span>可疑句（句级细化）</h2>
  <p class="sec-lead">只对已判为 AI 的段落做句级打分，最多列 20 条；这些是最值得优先改写的位置。</p>
  <table>
    <thead><tr><th class="num">段</th><th class="num">P(AI)</th><th>句子</th></tr></thead>
    <tbody>{_sentence_rows(analysis)}</tbody>
  </table>
</section>

<section class="card">
  <h2><span class="n">03</span>逐段明细</h2>
  <p class="sec-lead">按段落给出分数、等级与触发特征。点「展开全文 / 判定依据」看原文与解释。</p>
  <table>
    <thead><tr>
      <th class="num">#</th><th>类型</th><th class="num">字数</th>
      <th class="num">P(AI)</th><th>等级</th><th>段落 / 依据</th>
    </tr></thead>
    <tbody>{_para_rows(analysis)}</tbody>
  </table>
</section>

<section class="card">
  <h2><span class="n">04</span>引擎与标定</h2>
  <p class="sec-lead">三个独立信号融合：判别器看「像不像模型写的」，困惑度看「有多可预测」，文体统计看「结构是否过分工整」。</p>
  {_calibration_block(analysis)}
  <table style="margin-top:14px">
    <thead><tr><th>引擎</th><th>状态</th><th class="num">平均分</th><th class="num">耗时</th><th>说明</th></tr></thead>
    <tbody>{_engine_rows(analysis)}</tbody>
  </table>
  <div class="grid2" style="margin-top:18px">
    <div>
      <h3 style="font-size:15px">口径定义</h3>
      <ul class="plain">
        <li><b>AI 生成占比（字数口径）</b>：Σ(段落字数 × P(AI)) ÷ 正文字数。连续值，不是「判定为 AI 的字数」。</li>
        <li><b>AI 段落占比</b>：P(AI) ≥ 阈值 的段落数 ÷ 总段落数。</li>
        <li><b>置信区间</b>：段落级 bootstrap 1000 次（固定种子，可复现），反映「段落间分歧」带来的不确定性。</li>
        <li><b>正文</b>：已剔除参考文献、页眉页脚等元信息后的可评分数。</li>
      </ul>
    </div>
    <div>
      <h3 style="font-size:15px">怎么改</h3>
      <ul class="plain">
        <li>先改「极高」段落，再看「偏高」段落；改完重跑对比数字。</li>
        <li>不要只做同义词替换——检测器主要看句式节奏与可预测性，同义替换无效。</li>
        <li>把过分工整的三段式/排比句改成有具体信息量的短句，加入你自己的判断和例证。</li>
        <li>保留专有名词与数据；删掉「综上所述」「值得注意的是」这类模板连接。</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  <p>复现：<code>python -m aidetect report 你的文档.docx --mode {_esc(analysis.mode)}</code> ·
  指纹 <code>{_esc(fingerprint)}</code></p>
  <p>ai-detect 是自查工具，检测结论不等同于任何机构（知网/维普等）的正式检测结果。它们的模型、语料、
  阈值都不公开，且彼此结论也常有分歧。请把本报告当作「哪些段落值得重写」的线索。</p>
</footer>
</div>
<script>{JS}</script>
</body>
</html>
"""


def write(analysis: Analysis, path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render(analysis)
    path.write_text(content, encoding="utf-8")
    return str(path)
