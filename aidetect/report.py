"""单文件 HTML 报告：不依赖任何外部资源（离线可用），可直接发给别人看。

设计原则：
- 结论先行：整篇 AI 率 + 风险档 + 置信区间 + 护栏提示放在第一屏；
- 证据可查：逐段表给 P(AI)、等级与「为什么」，可筛选、可跳转；
- 口径透明：每个数字都能追到定义、阈值和标定来源；
- 打印/存档友好：print CSS 隐藏交互元素，A4 分页；
- 离线自包含：所有样式、脚本、图形内联，没有任何 http(s) 引用与 url() 资源。
"""
from __future__ import annotations

import html
import json
import math
import time

from .aggregate import Analysis, UnitScore

CSS = """
:root{
  --bg:#f1efe9; --bg-2:#e9e6dd; --surface:#fffefa; --surface-2:#f7f5ee;
  --ink:#17181d; --ink-2:#4a4c56; --ink-3:#787a87; --ink-4:#a3a4ad;
  --line:#e2ded3; --line-2:#cfc9ba;
  --brand:#1b5e6b; --brand-soft:#e3eff0; --link:#1b5e6b;
  --high:#a83226; --high-bg:#fbe8e4; --high-line:#eec2b9;
  --mid-high:#a75d17; --mid-high-bg:#fbeee0; --mid-high-line:#f0d3ad;
  --mid:#7d6410; --mid-bg:#f8f2d8; --mid-line:#e8dca4;
  --low:#2c7350; --low-bg:#e5f2e9; --low-line:#b6ddc3;
  --excluded:#6c6f79; --excluded-bg:#efece4;
  --radius:18px; --radius-sm:12px;
  --shadow-1:0 1px 2px rgba(23,24,29,.05),0 10px 30px -12px rgba(23,24,29,.14);
  --shadow-2:0 1px 1px rgba(23,24,29,.05),0 2px 6px rgba(23,24,29,.05);
  --sans:"Noto Sans SC","Source Han Sans SC","PingFang SC","Hiragino Sans GB",
         "Microsoft YaHei",-apple-system,"Segoe UI",Roboto,system-ui,sans-serif;
  --serif:"Noto Serif SC","Source Han Serif SC","Songti SC","STSong",Georgia,serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Mono",Menlo,Consolas,monospace;
  --gauge-track:#e8e4d9;
  color-scheme:light;
}
html[data-theme="dark"]{
  --bg:#111318; --bg-2:#0c0e12; --surface:#191c23; --surface-2:#20242c;
  --ink:#eef0f4; --ink-2:#c0c4cf; --ink-3:#8d92a0; --ink-4:#6a6f7d;
  --line:#2a2f39; --line-2:#3a404c;
  --brand:#6fc2cf; --brand-soft:#16333a; --link:#8fd3dd;
  --high:#ff8d7d; --high-bg:#3a1d1a; --high-line:#5b2b26;
  --mid-high:#f0b166; --mid-high-bg:#3a2a17; --mid-high-line:#5a3f22;
  --mid:#e3ca74; --mid-bg:#332c14; --mid-line:#4e431f;
  --low:#7dd3a0; --low-bg:#14301f; --low-line:#245033;
  --excluded:#9ba0ad; --excluded-bg:#24272f;
  --shadow-1:0 1px 2px rgba(0,0,0,.4),0 12px 32px -16px rgba(0,0,0,.7);
  --shadow-2:0 1px 1px rgba(0,0,0,.3);
  --gauge-track:#2b303a;
  color-scheme:dark;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;overflow-x:hidden}
body{
  margin:0;background:
    radial-gradient(1200px 520px at 12% -8%,rgba(27,94,107,.07),transparent 60%),
    radial-gradient(900px 420px at 92% 0%,rgba(168,50,38,.05),transparent 55%),
    var(--bg);
  color:var(--ink);font:15px/1.72 var(--sans);
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
}
a{color:var(--link);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--link) 35%,transparent)}
a:hover{border-bottom-color:var(--link)}
code,.mono{font-family:var(--mono);font-size:.88em;font-variant-numeric:tabular-nums}
code{background:var(--surface-2);border:1px solid var(--line);border-radius:6px;padding:1px 6px}
.shell{max-width:1120px;margin:0 auto;padding:36px 24px 96px}
svg{display:block}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum" 1}

/* ---------- masthead ---------- */
.masthead{margin:0 0 26px}
.brandline{display:flex;align-items:center;gap:12px;font-size:13px;color:var(--ink-3);
  letter-spacing:.02em;padding-bottom:16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.brandmark{width:26px;height:26px;border-radius:8px;background:var(--brand);color:#fff;
  display:grid;place-items:center;box-shadow:var(--shadow-2)}
.brandmark svg{width:15px;height:15px}
.brandtext b{color:var(--ink);font-weight:700;letter-spacing:.01em}
.brandtext em{font-style:normal;color:var(--ink-3);margin-left:8px}
.spacer{flex:1}
.ghostbtn{font:inherit;font-size:12.5px;color:var(--ink-2);background:var(--surface);
  border:1px solid var(--line);border-radius:999px;padding:6px 14px;cursor:pointer;
  transition:background .15s,border-color .15s,transform .15s}
.ghostbtn:hover{background:var(--surface-2);border-color:var(--line-2)}
.ghostbtn:active{transform:translateY(1px)}
.titles{margin:26px 0 20px}
.eyebrow{font-size:11.5px;letter-spacing:.22em;text-transform:uppercase;color:var(--brand);
  font-weight:700}
h1{font-family:var(--serif);font-size:clamp(28px,4.6vw,42px);line-height:1.16;
  margin:10px 0 0;letter-spacing:-.015em;font-weight:700}
.subtitle{margin:12px 0 0;max-width:70ch;color:var(--ink-2);font-size:15px}
.metarow{display:flex;flex-wrap:wrap;margin:22px 0 0;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--radius-sm);overflow:hidden}
.metarow>div{flex:1 1 150px;background:var(--surface);padding:11px 15px;
  border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.metarow dt{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-4);margin:0 0 3px}
.metarow dd{margin:0;font-size:13.5px;color:var(--ink);font-weight:600}

/* ---------- cards & sections ---------- */
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow-1);padding:26px 28px;margin:0 0 20px}
@media(max-width:640px){.card{padding:20px 18px;border-radius:15px}}
.sec-head{display:flex;align-items:baseline;gap:12px;margin:0 0 4px}
.sec-no{font-family:var(--mono);font-size:12px;color:var(--brand);background:var(--brand-soft);
  border-radius:6px;padding:2px 7px;font-weight:700;letter-spacing:.06em}
h2{font-family:var(--serif);font-size:21px;margin:0;font-weight:700;letter-spacing:-.01em}
.sec-lead{color:var(--ink-2);font-size:13.8px;margin:8px 0 20px;max-width:78ch}
h3{font-size:14.5px;margin:0 0 8px;font-family:var(--sans);font-weight:700}

/* ---------- verdict / scoreboard ---------- */
.verdict{display:grid;grid-template-columns:300px 1fr;gap:34px;align-items:start}
@media(max-width:840px){.verdict{grid-template-columns:1fr;gap:22px}}
.gauge-wrap{display:flex;flex-direction:column;align-items:center;gap:14px}
.gauge{position:relative;width:236px;max-width:100%}
.gauge svg{width:100%;height:auto;overflow:visible}
.gauge-center{position:absolute;inset:0;display:grid;place-content:center;text-align:center;padding-top:6px}
.gauge-num{font-family:var(--serif);font-size:46px;line-height:1;font-weight:700;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums}
.gauge-num small{font-size:20px;font-weight:600;color:var(--ink-3);margin-left:1px}
.gauge-cap{font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin-top:8px}
.pill{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:6px 14px;
  font-size:13px;font-weight:700;border:1px solid transparent;line-height:1.35}
.pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;flex:none}
.pill.high{background:var(--high-bg);color:var(--high);border-color:var(--high-line)}
.pill.mid_high{background:var(--mid-high-bg);color:var(--mid-high);border-color:var(--mid-high-line)}
.pill.mid{background:var(--mid-bg);color:var(--mid);border-color:var(--mid-line)}
.pill.low{background:var(--low-bg);color:var(--low);border-color:var(--low-line)}
.verdict-body>p{margin:0 0 18px;color:var(--ink-2);font-size:14.5px;max-width:60ch}
.statgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(122px,1fr));gap:10px;margin:0 0 20px}
.stat{background:var(--surface-2);border:1px solid var(--line);border-radius:var(--radius-sm);
  padding:12px 14px}
.stat b{display:block;font-size:22px;line-height:1.15;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat b small{font-size:13px;color:var(--ink-3);font-weight:600;margin-left:2px}
.stat span{display:block;font-size:11.5px;color:var(--ink-3);margin-top:4px;letter-spacing:.02em}
.range{margin:0 0 6px}
.range-head{display:flex;justify-content:space-between;font-size:11.5px;color:var(--ink-3);
  letter-spacing:.04em;margin-bottom:6px}
.range-track{position:relative;height:9px;border-radius:999px;background:var(--bg-2);overflow:visible}
html[data-theme="dark"] .range-track{background:#252a33}
.range-band{position:absolute;top:0;bottom:0;border-radius:999px;
  background:color-mix(in srgb,var(--brand) 26%,transparent)}
.range-dot{position:absolute;top:50%;width:13px;height:13px;border-radius:50%;background:var(--ink);
  border:2.5px solid var(--surface);transform:translate(-50%,-50%);box-shadow:var(--shadow-2)}
.range-thr{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--high);opacity:.85}
.range-foot{font-size:11.5px;color:var(--ink-3);margin-top:7px}
.gradebar{display:flex;height:10px;border-radius:999px;overflow:hidden;background:var(--bg-2);margin:16px 0 10px}
.gradebar i{display:block;height:100%}
.gradechips{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;color:var(--ink-2)}
.gradechips span{display:inline-flex;align-items:center;gap:6px}
.gradechips i{width:9px;height:9px;border-radius:2.5px;display:inline-block}
.tops{margin:18px 0 0;padding:14px 16px;background:var(--surface-2);border:1px solid var(--line);
  border-radius:var(--radius-sm)}
.tops h3{margin:0 0 6px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.tops ol{margin:0;padding:0;list-style:none;display:grid;gap:6px}
.tops li{display:flex;gap:10px;align-items:baseline;font-size:13px;color:var(--ink-2)}
.tops li b{color:var(--ink);font-variant-numeric:tabular-nums}
.tops li .tag{font-size:11.5px;color:var(--ink-3)}

/* ---------- warnings ---------- */
.warn{display:grid;grid-template-columns:22px 1fr;gap:12px;background:var(--mid-bg);
  border:1px solid var(--mid-line);border-radius:var(--radius-sm);padding:13px 16px;margin:0 0 10px;
  font-size:13.6px;color:var(--ink-2)}
.warn svg{width:17px;height:17px;margin-top:3px;color:var(--mid)}
.warn b{color:var(--ink)}
.ok{display:grid;grid-template-columns:22px 1fr;gap:12px;background:var(--low-bg);
  border:1px solid var(--low-line);border-radius:var(--radius-sm);padding:13px 16px;
  font-size:13.6px;color:var(--ink-2)}
.ok svg{width:17px;height:17px;margin-top:3px;color:var(--low)}
.guard-list{display:grid;grid-template-columns:1fr 1fr;gap:14px 30px;margin:18px 0 0;padding:0;
  list-style:none}
@media(max-width:760px){.guard-list{grid-template-columns:1fr}}
.guard-list li{position:relative;padding-left:26px;font-size:13.6px;color:var(--ink-2)}
.guard-list li::before{content:"";position:absolute;left:8px;top:9px;width:6px;height:6px;
  border-radius:50%;background:var(--brand);opacity:.7}
.guard-list b{color:var(--ink)}

/* ---------- chart ---------- */
.chart-wrap{margin:0 0 10px;position:relative}
.chart{width:100%;height:auto}
.chart text{font-family:var(--mono);font-size:10.5px;fill:var(--ink-3)}
.chart .vlabel{font-size:9.5px;fill:var(--ink-2);font-weight:700}
.chart .axis{stroke:var(--line-2);stroke-width:1}
.chart .grid{stroke:var(--line);stroke-width:1;stroke-dasharray:2 4}
.chart .thr{stroke:var(--high);stroke-width:1.4;stroke-dasharray:5 4}
.chart .thr-label{fill:var(--high);font-weight:700}
.chart rect{transition:opacity .12s}
.chart rect:hover{opacity:.78}
.legend{display:flex;flex-wrap:wrap;gap:10px 20px;font-size:12px;color:var(--ink-2);margin-top:14px}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend i{width:10px;height:10px;border-radius:3px;display:inline-block}
/* ---------- tables ---------- */
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 -4px}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:13.4px;min-width:640px}
thead th{position:sticky;top:0;z-index:1;background:var(--surface);font-size:11px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-4);font-weight:700;
  padding:0 12px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tbody tr{scroll-margin-top:96px}
tbody tr:hover td{background:var(--surface-2)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pnum{font-family:var(--mono);color:var(--ink-3)}
.heat{min-width:104px}
.heat .pv{font-variant-numeric:tabular-nums;font-weight:600}
.heat .hbar{display:block;width:66px;height:3px;border-radius:2px;background:var(--line);
  margin:6px 0 0 auto;overflow:hidden}
.heat .hbar i{display:block;height:100%;border-radius:2px}
.g{display:inline-block;min-width:52px;text-align:center;border-radius:7px;padding:2px 8px;
  font-size:12px;font-weight:700;border:1px solid transparent;white-space:nowrap}
.g.high{background:var(--high-bg);color:var(--high);border-color:var(--high-line)}
.g.mid_high{background:var(--mid-high-bg);color:var(--mid-high);border-color:var(--mid-high-line)}
.g.mid{background:var(--mid-bg);color:var(--mid);border-color:var(--mid-line)}
.g.low{background:var(--low-bg);color:var(--low);border-color:var(--low-line)}
.g.excluded{background:var(--excluded-bg);color:var(--excluded)}
.cell-text{color:var(--ink-2);max-width:52ch}
details{margin-top:8px}
details summary{cursor:pointer;color:var(--link);font-size:12.3px;list-style:none;
  display:inline-flex;align-items:center;gap:5px}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"▸";font-size:10px;transition:transform .15s}
details[open] summary::before{transform:rotate(90deg)}
details .full{white-space:pre-wrap;background:var(--surface-2);border:1px solid var(--line);
  border-radius:10px;padding:12px 14px;font-size:13px;color:var(--ink-2);margin-top:8px;
  max-height:340px;overflow:auto;line-height:1.75}
.sig{margin:10px 0 0;padding:0;list-style:none;display:grid;gap:4px}
.sig li{position:relative;padding-left:15px;font-size:12.4px;color:var(--ink-3)}
.sig li::before{content:"";position:absolute;left:2px;top:8px;width:5px;height:5px;border-radius:50%;
  background:var(--brand);opacity:.55}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}
.chipbtn{font:inherit;font-size:12.5px;color:var(--ink-2);background:var(--surface-2);
  border:1px solid var(--line);border-radius:999px;padding:5px 12px;cursor:pointer}
.chipbtn:hover{border-color:var(--line-2)}
.chipbtn.on{background:var(--ink);color:var(--surface);border-color:var(--ink)}

/* ---------- sentences ---------- */
.sent-list{display:grid;gap:10px;margin:0;padding:0;list-style:none}
.sent{display:grid;grid-template-columns:56px 1fr;gap:14px;align-items:start;
  background:var(--surface-2);border:1px solid var(--line);border-radius:var(--radius-sm);
  padding:12px 15px}
.sent .sp{font-family:var(--mono);font-size:12px;color:var(--ink-3);text-align:right;padding-top:2px}
.sent .sp b{display:block;font-size:14px;color:var(--ink);font-variant-numeric:tabular-nums}
.sent p{margin:0;font-size:13.6px;color:var(--ink-2);line-height:1.7}
.sent p::before{content:"「"}
.sent p::after{content:"」"}
.empty{color:var(--ink-3);font-size:13.5px;margin:0}
details.more{margin-top:14px}
details.more>summary{display:inline-flex;font-size:12.5px;font-weight:600;padding:7px 14px;
  background:var(--surface-2);border:1px solid var(--line);border-radius:999px;color:var(--ink-2)}
details.more>summary::before{content:"＋";font-size:12px}
details.more[open]>summary::before{content:"－"}

/* ---------- engine / calibration ---------- */
.calib{display:grid;grid-template-columns:22px 1fr;gap:12px;border-radius:var(--radius-sm);
  padding:13px 16px;font-size:13.6px;color:var(--ink-2);margin:0 0 18px;
  background:var(--low-bg);border:1px solid var(--low-line)}
.calib svg{width:17px;height:17px;margin-top:3px;color:var(--low)}
.calib.uncal{background:var(--mid-bg);border-color:var(--mid-line)}
.calib.uncal svg{color:var(--mid)}
.dot{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600}
.dot::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.dot.on{color:var(--low)}
.dot.off{color:var(--excluded)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-top:22px}
@media(max-width:800px){.cols{grid-template-columns:1fr;gap:18px}}
ul.plain{margin:0;padding:0;list-style:none;display:grid;gap:8px}
ul.plain li{position:relative;padding-left:18px;font-size:13.3px;color:var(--ink-2)}
ul.plain li::before{content:"";position:absolute;left:4px;top:8px;width:5px;height:5px;
  border-radius:50%;background:var(--brand);opacity:.6}
ul.plain b{color:var(--ink)}

/* ---------- footer ---------- */
footer{border-top:1px solid var(--line);margin:34px 0 0;padding:22px 0 0;color:var(--ink-3);
  font-size:12.6px;display:grid;gap:10px}
footer code{font-size:11.8px}

/* ---------- sticky summary ---------- */
.stickybar{position:fixed;left:0;right:0;top:0;z-index:50;background:
  color-mix(in srgb,var(--surface) 88%,transparent);backdrop-filter:blur(10px);
  border-bottom:1px solid var(--line);transform:translateY(-104%);transition:transform .22s ease;
  box-shadow:var(--shadow-2)}
.stickybar.show{transform:none}
.stickybar .inner{max-width:1120px;margin:0 auto;padding:9px 24px;display:flex;align-items:center;
  gap:14px;font-size:12.8px;color:var(--ink-2)}
.stickybar b{color:var(--ink);font-variant-numeric:tabular-nums;font-size:14px}
.stickybar .risk{font-size:11.5px;font-weight:700;padding:2px 9px;border-radius:999px}

/* ---------- print ---------- */
@media print{
  @page{size:A4;margin:14mm 12mm}
  html,body{background:#fff !important;color:#000}
  body{font-size:11px;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .shell{max-width:none;padding:0}
  .card{box-shadow:none;border:1px solid #ccc;break-inside:avoid;page-break-inside:avoid;
    padding:14px 16px;margin-bottom:10px;border-radius:8px}
  .noprint,.stickybar,.ghostbtn,.filters{display:none !important}
  details{display:none}
  h1{font-size:22px}
  .verdict{grid-template-columns:200px 1fr;gap:16px}
  .gauge{width:170px}
  thead th{position:static}
  footer{border-top:1px solid #ccc}
}
@media(max-width:560px){
  .shell{padding:20px 14px 72px}
  .card{padding:18px 15px}
  .metarow>div{flex-basis:132px}
  .sent{grid-template-columns:44px 1fr;gap:10px}
  h2{font-size:19px}
  .statgrid{grid-template-columns:repeat(auto-fit,minmax(104px,1fr))}
  .stickybar .inner{padding:8px 14px;gap:9px;font-size:12px}
  .stickybar .inner>span:first-child{max-width:38%;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *{transition:none !important;animation:none !important}
}
"""

JS = """
(function(){
  var root=document.documentElement;

  /* 深浅色切换：默认浅色（报告/打印为主），记住手动选择 */
  var saved=null;
  try{saved=localStorage.getItem('aidetect-theme')}catch(e){}
  var theme=(saved==='dark'||saved==='light')?saved:'light';
  var toggle=document.getElementById('theme-toggle');
  function apply(t){
    root.setAttribute('data-theme',t);
    if(toggle){toggle.textContent=(t==='dark'?'浅色':'深色');toggle.setAttribute('aria-label','切换到'+(t==='dark'?'浅色':'深色')+'主题')}
  }
  apply(theme);
  if(toggle){toggle.addEventListener('click',function(){
    theme=(root.getAttribute('data-theme')==='dark')?'light':'dark';
    apply(theme);
    try{localStorage.setItem('aidetect-theme',theme)}catch(e){}
  });}

  /* 段落筛选 */
  var btns=[].slice.call(document.querySelectorAll('[data-filter]'));
  var total=document.querySelectorAll('tr[data-grade]').length;
  btns.forEach(function(b){b.addEventListener('click',function(){
    var k=b.getAttribute('data-filter');
    btns.forEach(function(x){x.classList.remove('on')});b.classList.add('on');
    [].forEach.call(document.querySelectorAll('tr[data-grade]'),function(tr){
      tr.style.display=(k==='all'||tr.getAttribute('data-grade')===k)?'':'none';
    });
    var n=[].filter.call(document.querySelectorAll('tr[data-grade]'),function(tr){
      return tr.style.display!=='none';
    }).length;
    var tag=document.getElementById('filter-count');
    if(tag){tag.textContent='显示 '+n+' / 共 '+total+' 段'};
  });});

  /* 滚动到正文后浮出摘要条 */
  var bar=document.getElementById('stickybar');
  if(bar){
    var onScroll=function(){
      if(window.scrollY>420){bar.classList.add('show')}else{bar.classList.remove('show')}
    };
    window.addEventListener('scroll',onScroll,{passive:true});onScroll();
  }
  var printBtn=document.getElementById('print-btn');
  if(printBtn){printBtn.addEventListener('click',function(){window.print()});}
})();
"""

RISK = {
    "low": ("结论：AI 生成痕迹不明显", "AI 生成占比低于当前判定阈值的一半，文本整体更像人工写作。"),
    "mid": ("结论：存在少量 AI 特征",
            "AI 生成占比低于判定阈值但并非接近 0，可能有局部段落由模型辅助或经过润色。"),
    "mid_high": ("结论：AI 特征明显，建议逐段自查",
                 "AI 生成占比已过判定阈值，建议按逐段表定位可疑段落并改写。"),
    "high": ("结论：AI 特征强，风险高",
             "AI 生成占比显著高于阈值，多个段落被判为 AI 生成。建议按逐段明细找到"
             "「极高」段落，逐段核对后用自己的表达改写，再重跑对比。"),
}
GRADE_LABEL = {"high": "极高", "mid_high": "偏高", "mid": "中等", "low": "低", "excluded": "未纳入"}
KIND_LABEL = {"body": "正文", "heading": "标题", "meta": "元信息"}
GRADE_HEX = {"high": "#a83226", "mid_high": "#a75d17", "mid": "#7d6410", "low": "#2c7350",
             "excluded": "#8a8d96"}
_RISK_VAR = {"high": "var(--high)", "mid_high": "var(--mid-high)", "mid": "var(--mid)",
             "low": "var(--low)"}

_ICON_SHIELD = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
                '<path d="M12 3l7 3v6c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6l7-3z"/>'
                '<path d="M9 12l2 2 4-4"/></svg>')
_ICON_ALERT = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
               'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
               '<path d="M12 9v5"/><path d="M12 17h.01"/>'
               '<path d="M10.3 3.9 2.6 17a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
               '</svg>')
_ICON_CHECK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
               'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
               '<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/></svg>')


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


def _cut_txt(value: float) -> str:
    """分级线文案：标定后的阈值可能只有千分之几，按量级给合适位数。"""
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _gauge(rate: float, threshold: float, risk: str) -> str:
    """环形量表：进度环 = AI 率，刻度 = 判定阈值，颜色随风险档变化。"""
    size, center, radius, width = 236, 118, 92, 13
    circ = 2 * math.pi * radius
    dash = max(0.0, min(1.0, rate)) * circ
    angle = math.radians(-90 + 360 * max(0.0, min(1.0, threshold)))
    tick_in, tick_out = radius - width, radius + 11
    x1, y1 = center + tick_in * math.cos(angle), center + tick_in * math.sin(angle)
    x2, y2 = center + tick_out * math.cos(angle), center + tick_out * math.sin(angle)
    color = _RISK_VAR.get(risk, "var(--brand)")
    minor = []
    for frac in (0.25, 0.5, 0.75):
        a = math.radians(-90 + 360 * frac)
        minor.append(
            f'<line x1="{center + (radius + 4) * math.cos(a):.1f}" '
            f'y1="{center + (radius + 4) * math.sin(a):.1f}" '
            f'x2="{center + (radius + 8) * math.cos(a):.1f}" '
            f'y2="{center + (radius + 8) * math.sin(a):.1f}" '
            f'stroke="var(--line-2)" stroke-width="1.5" stroke-linecap="round"/>')
    return f"""<div class="gauge">
  <svg viewBox="0 0 {size} {size}" role="img" aria-label="AI 生成占比 {rate*100:.1f}%，判定阈值 {threshold*100:.2f}%">
    <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="var(--gauge-track)" stroke-width="{width}"/>
    <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="{width}"
      stroke-linecap="round" stroke-dasharray="{dash:.2f} {circ:.2f}" transform="rotate(-90 {center} {center})"/>
    {''.join(minor)}
    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--high)" stroke-width="2.4"
      stroke-linecap="round"/>
  </svg>
  <div class="gauge-center">
    <div class="gauge-num">{rate*100:.1f}<small>%</small></div>
    <div class="gauge-cap">AI 生成占比</div>
  </div>
</div>"""


def _range_bar(rate: float, ci, threshold: float) -> str:
    """0–100% 轨道：区间带 + 当前值圆点 + 阈值刻度线。"""
    lo, hi = (ci if ci else (rate, rate))
    lo_pct = max(0.0, min(100.0, lo * 100))
    hi_pct = max(0.0, min(100.0, hi * 100))
    width = max(hi_pct - lo_pct, 0.8)
    thr_pct = max(0.0, min(100.0, threshold * 100))
    return f"""<div class="range">
  <div class="range-head"><span>置信区间与阈值</span><span>0 — 100%</span></div>
  <div class="range-track">
    <span class="range-band" style="left:{lo_pct:.1f}%;width:{width:.1f}%"></span>
    <span class="range-thr" style="left:{thr_pct:.1f}%"></span>
    <span class="range-dot" style="left:{max(0.6, min(99.4, rate*100)):.1f}%"></span>
  </div>
  <div class="range-foot">圆点 = 本次 AI 率；竖线 = 判定阈值 {threshold*100:.2f}%；阴影带 = 95% 置信区间（段落级 bootstrap）。</div>
</div>"""


def _grade_bar(analysis: Analysis) -> str:
    counts = analysis.grade_counts or {}
    order = [("high", "极高"), ("mid_high", "偏高"), ("mid", "中等"), ("low", "低")]
    total = sum(counts.get(k, 0) for k, _ in order) or 1
    segs, chips = [], []
    for key, label in order:
        n = counts.get(key, 0)
        segs.append(f'<i style="width:{n / total * 100:.2f}%;background:{GRADE_HEX[key]}"></i>'
                    if n else "")
        chips.append(f'<span><i style="background:{GRADE_HEX[key]}"></i>{label} {n}</span>')
    excluded = counts.get("excluded", 0)
    if excluded:
        chips.append(f'<span><i style="background:{GRADE_HEX["excluded"]}"></i>未纳入 {excluded}</span>')
    return (f'<div class="gradebar">{"".join(segs) or "<i style=\"width:100%;background:var(--line)\"></i>"}'
            f'</div><div class="gradechips">{"".join(chips)}</div>')


def _unit_label(us: UnitScore) -> str:
    """单位（段落或合并段）对应的段号文案，与逐段明细的锚点一致。"""
    pis = list(getattr(us.unit, "para_indexes", ()) or ())
    if not pis:
        return f"第 {us.unit.index + 1} 段"
    if len(pis) == 1:
        return f"第 {pis[0] + 1} 段"
    return f"第 {pis[0] + 1}–{pis[-1] + 1} 段"


def _distribution_chart(analysis: Analysis) -> str:
    """单元概率分布柱状图：带坐标轴、阈值线、段号；柱高 = P(AI)，颜色 = 等级。"""
    scores = analysis.unit_scores
    if not scores:
        return '<p class="empty">没有可展示的段落（正文为空或全部被剔除）。</p>'
    width, height = 1000, 268
    left, right, top, bottom = 44, 16, 30, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(scores)
    slot = plot_w / n
    bar_w = min(58.0, max(3.0, slot * 0.62))
    show_vals = n <= 24
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="各段落 AI 概率分布柱状图，共 {n} 段">']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - frac)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 9}" y="{y + 3.5:.1f}" text-anchor="end">{frac*100:.0f}%</text>')
    thr_y = top + plot_h * (1 - max(0.0, min(1.0, analysis.threshold)))
    thr_label_y = thr_y - 7 if thr_y > top + 20 else thr_y + 15
    parts.append(f'<line class="thr" x1="{left}" y1="{thr_y:.1f}" x2="{width - right}" y2="{thr_y:.1f}"/>')
    parts.append(f'<text class="thr-label" x="{width - right}" y="{thr_label_y:.1f}" '
                 f'text-anchor="end">判定阈值 {analysis.threshold*100:.2f}%</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" '
                 f'x2="{width - right}" y2="{top + plot_h}"/>')
    step = max(1, n // 16)
    for i, us in enumerate(scores):
        x = left + i * slot + (slot - bar_w) / 2
        p = max(0.0, min(1.0, us.p))
        bar_h = max(plot_h * p, 3.0)
        y = top + plot_h - bar_h
        color = GRADE_HEX.get(us.grade, GRADE_HEX["excluded"])
        label = _unit_label(us)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2.5" '
            f'fill="{color}"><title>{label} · P(AI)={us.p:.3f} · '
            f'{GRADE_LABEL.get(us.grade, us.grade)} · {us.unit.chars} 字</title></rect>')
        if show_vals:
            parts.append(f'<text class="vlabel" x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" '
                         f'text-anchor="middle">{us.p:.2f}</text>')
        if i % step == 0 or i == n - 1:
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 17:.1f}" '
                         f'text-anchor="middle">{_unit_label(us).replace("第 ", "").replace(" 段", "")}</text>')
    parts.append(f'<text x="{left}" y="{height - 8}" text-anchor="start">段落序号（共 {n} 段；合并段显示为区间）</text>')
    parts.append("</svg>")
    return "".join(parts)


def _top_paragraphs(analysis: Analysis, limit: int = 3) -> str:
    rows = [r for r in analysis.paragraphs if r.p is not None]
    rows.sort(key=lambda r: r.p or 0.0, reverse=True)
    picks = rows[:limit]
    if not picks:
        return ""
    items = []
    for r in picks:
        items.append(
            f'<li><a href="#p{r.index + 1}">第 {r.index + 1} 段</a>'
            f'<b>{r.p:.2f}</b>'
            f'<span class="tag">{GRADE_LABEL.get(r.grade, r.grade)} · {r.chars} 字 · '
            f'{_esc(r.text[:26])}…</span></li>')
    return f'<div class="tops"><h3>优先自查</h3><ol>{"".join(items)}</ol></div>'


def _engine_rows(analysis: Analysis) -> str:
    rows = []
    for e in analysis.engine_info:
        state = ('<span class="dot on">可用</span>' if e["available"]
                 else '<span class="dot off">不可用</span>')
        mean = "—" if e["mean_score"] is None else f"{e['mean_score']:.3f}"
        rows.append(
            f"<tr><td><b>{_esc(e['label'])}</b></td><td>{state}</td>"
            f"<td class='num'>{mean}</td><td class='num'>{e['elapsed_sec']:.1f}s</td>"
            f"<td class='cell-text'>{_esc(e['note'] or '')}</td></tr>")
    return "".join(rows)


def _para_rows(analysis: Analysis) -> str:
    rows = []
    for r in analysis.paragraphs:
        kind = _esc(KIND_LABEL.get(r.kind, r.kind))
        if r.p is None:
            rows.append(
                f"<tr id='p{r.index + 1}' data-grade='excluded'><td class='num pnum'>{r.index + 1}</td>"
                f"<td>{kind}</td><td class='num'>{r.chars}</td>"
                f"<td class='num'>—</td>"
                f"<td><span class='g excluded'>未纳入</span></td>"
                f"<td class='cell-text'>{_esc(r.text[:90])}</td></tr>")
            continue
        p = max(0.0, min(1.0, r.p))
        heat = GRADE_HEX.get(r.grade, GRADE_HEX["excluded"])
        sigs = "".join(f"<li>{_esc(s)}</li>" for s in r.signals) or "<li>未触发明显特征</li>"
        merged = " <span class='pnum'>(合并段)</span>" if r.merged else ""
        rows.append(
            f"<tr id='p{r.index + 1}' data-grade='{r.grade}'>"
            f"<td class='num pnum'>{r.index + 1}</td>"
            f"<td>{kind}{merged}</td>"
            f"<td class='num'>{r.chars}</td>"
            f"<td class='num heat'><span class='pv'>{r.p:.2f}</span>"
            f"<span class='hbar'><i style='width:{p*100:.1f}%;background:{heat}'></i></span></td>"
            f"<td><span class='g {r.grade}'>{GRADE_LABEL.get(r.grade, r.grade)}</span></td>"
            f"<td class='cell-text'>{_esc(r.text[:90])}"
            f"<details><summary>展开全文 / 判定依据</summary>"
            f"<div class='full'>{_esc(r.text)}</div>"
            f"<ul class='sig'>{sigs}</ul></details></td></tr>")
    return "".join(rows)


def _sentence_rows(analysis: Analysis, visible: int = 8) -> str:
    if not analysis.sentences:
        return ("<p class='empty'>没有句子超过句级阈值（也可能判别引擎不可用，此时仅按段落分推定）。</p>")
    items = []
    for s in analysis.sentences:
        items.append(
            f"<li class='sent'><div class='sp'>第 {s.para_index + 1} 段<b>{s.p:.2f}</b></div>"
            f"<p>{_esc(s.text)}</p></li>")
    head, tail = items[:visible], items[visible:]
    html_out = f"<ul class='sent-list'>{''.join(head)}</ul>"
    if tail:
        html_out += (f"<details class='more'><summary>展开其余 {len(tail)} 条可疑句</summary>"
                     f"<ul class='sent-list' style='margin-top:10px'>{''.join(tail)}</ul></details>")
    return html_out


def _calibration_block(analysis: Analysis) -> str:
    rc = analysis.recipe
    if rc.calibrated:
        block = (rc.evaluation or {}).get(analysis.mode) or {}
        m = block.get("test") or {}
        acc, auc = m.get("acc"), m.get("auc")
        fpr = m.get("fpr")
        line = f"标定版本 <code>{_esc(rc.version)}</code>"
        if acc is not None:
            line += f" · 域外测试集 Acc <b>{acc:.3f}</b> / AUC <b>{auc:.3f}</b>"
            if fpr is not None:
                line += f" / FPR <b>{fpr*100:.1f}%</b>"
        return (f"<div class='calib'>{_ICON_CHECK}<div>已加载标定参数。{line}；"
                f"当前模式 <b>{_esc(analysis.mode)}</b>，判定阈值 "
                f"<b>{analysis.threshold*100:.2f}%</b>（{analysis.threshold:.4f}）。</div></div>")
    return (f"<div class='calib uncal'>{_ICON_ALERT}<div><b>当前为未标定模式。</b>"
            f"{_esc(rc.note)}。分数仅可做相对比较，请先运行 "
            f"<code>python -m aidetect calibrate</code> 生成标定参数。</div></div>")


def render(analysis: Analysis, generated_at: str | None = None) -> str:
    st = analysis.stats
    rate = analysis.ai_rate_chars
    rk = risk_key(rate, analysis.threshold)
    title, desc = RISK[rk]
    ci = analysis.ai_rate_ci
    ci_txt = (f"{_pct(ci[0])} – {_pct(ci[1])}" if ci else "样本段数太少，无法给出区间")
    generated_at = generated_at or time.strftime("%Y-%m-%d %H:%M:%S")
    calib = analysis.recipe
    cuts = _grade_cutoffs(analysis)
    name = st.get("name", "文档")
    units = st.get("units", len(analysis.unit_scores) or 0)
    high_n = analysis.grade_counts.get("high", 0)
    mid_high_n = analysis.grade_counts.get("mid_high", 0)
    ai_paras = analysis.grade_counts.get("high", 0) + analysis.grade_counts.get("mid_high", 0)
    ps = [us.p for us in analysis.unit_scores]
    spread = (sum((p - sum(ps) / len(ps)) ** 2 for p in ps) / len(ps)) ** 0.5 if len(ps) > 1 else 0.0
    spread_txt = "低" if spread < 0.15 else ("中" if spread < 0.3 else "高")

    warns = "".join(
        f"<div class='warn'>{_ICON_ALERT}<div>{_esc(w)}</div></div>" for w in analysis.warnings)
    if not warns:
        warns = f"<div class='ok'>{_ICON_CHECK}<div>未触发护栏提示：文本长度、引擎、标定状态均正常。</div></div>"

    fingerprint = json.dumps({
        "tool": "ai-detect",
        "calibration": {"version": calib.version, "calibrated": calib.calibrated},
        "threshold": analysis.threshold, "mode": analysis.mode,
        "ai_rate_chars": round(rate, 4),
    }, ensure_ascii=False)

    filter_defs = [("all", "全部"), ("high", "极高"), ("mid_high", "偏高"), ("mid", "中等"),
                   ("low", "低"), ("excluded", "未纳入")]
    filters = "".join(
        f"<button class='chipbtn{' on' if key == 'all' else ''}' type='button' "
        f"data-filter='{key}'>{label}</button>" for key, label in filter_defs)

    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>AI 生成率检测报告 · {_esc(name)}</title>
<style>{CSS}</style>
</head>
<body>

<div class="stickybar noprint" id="stickybar">
  <div class="inner">
    <span>{_esc(name)}</span>
    <b class="num">{rate*100:.1f}%</b>
    <span class="risk pill {rk}">{_esc(title.replace('结论：', ''))}</span>
    <span>模式 {_esc(analysis.mode)} · 阈值 {analysis.threshold*100:.2f}%</span>
  </div>
</div>

<div class="shell">

<header class="masthead">
  <div class="brandline">
    <span class="brandmark">{_ICON_SHIELD}</span>
    <span class="brandtext"><b>ai-detect</b><em>本地离线检测 · 文本不出本机</em></span>
    <span class="spacer"></span>
    <button class="ghostbtn noprint" id="theme-toggle" type="button">深色</button>
    <button class="ghostbtn noprint" id="print-btn" type="button">打印 / 存 PDF</button>
  </div>
  <div class="titles">
    <div class="eyebrow">Detection report</div>
    <h1>AI 生成率检测报告</h1>
    <p class="subtitle">{_esc(title.replace('结论：', ''))}。整篇 AI 生成占比 <b>{rate*100:.1f}%</b>
      （95% 置信区间 {ci_txt}），{ai_paras}/{units} 段被判为 AI；建议按「逐段明细」
      定位可疑段落并用你自己的表达改写。</p>
  </div>
  <dl class="metarow">
    <div><dt>文档</dt><dd>{_esc(name)}</dd></div>
    <div><dt>正文</dt><dd>{st.get('body_chars', 0)} 字 · {units} 段</dd></div>
    <div><dt>模式</dt><dd>{_esc(analysis.mode)} · 阈值 {analysis.threshold*100:.2f}%</dd></div>
    <div><dt>生成时间</dt><dd>{_esc(generated_at)}</dd></div>
    <div><dt>检测耗时</dt><dd>{analysis.elapsed:.1f}s</dd></div>
  </dl>
</header>

<section class="card">
  <div class="verdict">
    <div class="gauge-wrap">
      {_gauge(rate, analysis.threshold, rk)}
      <div class="pill {rk}">{_esc(title.replace('结论：', ''))}</div>
    </div>
    <div class="verdict-body">
      <p>{_esc(desc)}</p>
      <div class="statgrid">
        <div class="stat"><b>{analysis.ai_rate_paras*100:.0f}<small>%</small></b><span>AI 段落占比（{ai_paras}/{units} 段）</span></div>
        <div class="stat"><b>{high_n}<small> 段</small></b><span>极高风险段</span></div>
        <div class="stat"><b>{mid_high_n}<small> 段</small></b><span>偏高风险段</span></div>
        <div class="stat"><b>{spread:.2f}</b><span>段落分歧（标准差·{spread_txt}）</span></div>
        <div class="stat"><b>{st.get('excluded_chars', 0)}<small> 字</small></b><span>已剔除（参考文献等）</span></div>
      </div>
      {_range_bar(rate, ci, analysis.threshold)}
      {_grade_bar(analysis)}
      {_top_paragraphs(analysis)}
    </div>
  </div>
</section>

<section class="card" id="sec-guard">
  <div class="sec-head"><span class="sec-no">00</span><h2>先说风险：这些结论不能怎么用</h2></div>
  <p class="sec-lead">机器检测没有「学术不端判决」效力。本报告给的是<b>特征证据与区间</b>，请按证据用，
    不要把它当结论用。</p>
  {warns}
  <ul class="guard-list">
    <li><b>不要只看整篇百分比。</b>真正的信息在逐段表：哪几段触发了什么特征。</li>
    <li><b>人类写作也会被误判。</b>术语密集、句式工整、模板化公文腔，都会让分数升高。</li>
    <li><b>最新模型更难检出。</b>检测器对旧模型更敏感；「没测出来」不等于「不是 AI 写的」。</li>
    <li><b>全部本地运行。</b>文本不出本机；除首次下载模型外不联网。</li>
  </ul>
</section>

<section class="card" id="sec-dist">
  <div class="sec-head"><span class="sec-no">01</span><h2>段落分布</h2></div>
  <p class="sec-lead">每根柱子是一个段落（或合并段）的 P(AI)；虚线是当前模式的判定阈值，
    柱色表示风险档。鼠标悬停可看段号、精确概率与字数。</p>
  <div class="chart-wrap">{_distribution_chart(analysis)}</div>
  <div class="legend">
    <span><i style="background:{GRADE_HEX['high']}"></i>极高（≥{_cut_txt(cuts['high'])}）</span>
    <span><i style="background:{GRADE_HEX['mid_high']}"></i>偏高（≥{_cut_txt(cuts['mid_high'])}）</span>
    <span><i style="background:{GRADE_HEX['mid']}"></i>中等（≥{_cut_txt(cuts['mid'])}）</span>
    <span><i style="background:{GRADE_HEX['low']}"></i>低（低于中等线）</span>
    <span><i style="background:var(--high);height:2px;border-radius:1px"></i>判定阈值 {analysis.threshold*100:.2f}%</span>
  </div>
</section>

<section class="card" id="sec-sent">
  <div class="sec-head"><span class="sec-no">02</span><h2>可疑句（句级细化）</h2></div>
  <p class="sec-lead">只对已判为 AI 的段落做句级打分，最多列 20 条；这些是<b>最值得优先改写</b>的位置。</p>
  {_sentence_rows(analysis)}
</section>

<section class="card" id="sec-para">
  <div class="sec-head"><span class="sec-no">03</span><h2>逐段明细</h2></div>
  <p class="sec-lead">按段落给出分数、等级与触发特征。可按等级筛选；点「展开全文 / 判定依据」看原文与解释。</p>
  <div class="filters noprint">{filters}<span class="chipbtn" id="filter-count" style="cursor:default">共 {len(analysis.paragraphs)} 段</span></div>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th class="num">#</th><th>类型</th><th class="num">字数</th>
        <th class="num">P(AI)</th><th>等级</th><th>段落 / 依据</th>
      </tr></thead>
      <tbody>{_para_rows(analysis)}</tbody>
    </table>
  </div>
</section>

<section class="card" id="sec-engine">
  <div class="sec-head"><span class="sec-no">04</span><h2>引擎与标定</h2></div>
  <p class="sec-lead">三个独立信号融合：判别器看「像不像模型写的」，困惑度看「有多可预测」，
    文体统计看「结构是否过分工整」。</p>
  {_calibration_block(analysis)}
  <div class="tablewrap">
    <table>
      <thead><tr><th>引擎</th><th>状态</th><th class="num">平均分</th><th class="num">耗时</th><th>说明</th></tr></thead>
      <tbody>{_engine_rows(analysis)}</tbody>
    </table>
  </div>
  <div class="cols">
    <div>
      <h3>口径定义</h3>
      <ul class="plain">
        <li><b>AI 生成占比（字数口径）</b>：Σ(段落字数 × P(AI)) ÷ 正文字数。连续值，不是「判定为 AI 的字数」。</li>
        <li><b>AI 段落占比</b>：P(AI) ≥ 阈值的段落数 ÷ 总段落数。</li>
        <li><b>置信区间</b>：段落级 bootstrap 1000 次（固定种子，可复现），反映段落间分歧带来的不确定性。</li>
        <li><b>正文</b>：已剔除参考文献、页眉页脚等元信息后的可评分数。</li>
      </ul>
    </div>
    <div>
      <h3>怎么改</h3>
      <ul class="plain">
        <li>先改「极高」段落，再看「偏高」段落；改完重跑对比数字。</li>
        <li>不要只做同义词替换——检测器主要看句式节奏与可预测性，同义替换无效。</li>
        <li>把过分工整的三段式 / 排比句改成有具体信息量的短句，加入你自己的判断和例证。</li>
        <li>保留专有名词与数据；删掉「综上所述」「值得注意的是」这类模板连接。</li>
      </ul>
    </div>
  </div>
</section>

<footer>
  <p>复现：<code>python -m aidetect report 你的文档.docx --mode {_esc(analysis.mode)}</code></p>
  <p>指纹 <code>{_esc(fingerprint)}</code></p>
  <p>ai-detect 是自查工具，检测结论不等同于任何机构（知网 / 维普等）的正式检测结果。它们的模型、
  语料、阈值都不公开，且彼此结论也常有分歧。请把本报告当作「哪些段落值得重写」的线索。</p>
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
