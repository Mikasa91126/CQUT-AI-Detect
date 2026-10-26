# T4 端到端验收证据（engineer-e2e）

生成时间：2026-10-26 13:3x（Asia/Shanghai）；机器：Windows / 18 逻辑核 / CPU-only torch 2.14.0+cpu / transformers 5.20.2。

## 1. 验收命令与结论

```
cd ai-detect
$env:PYTHONIOENCODING='utf-8'
python tools\e2e.py --repeat 2 --compare-human --json-out out\e2e_result.json
```

- 退出码：**0**（全部通过）
- 必需断言：**39 条全部 PASS**（明细见 out\e2e_result.json 与终端输出）
- 结论行：「结论：全部通过（必需断言 39 条）」

## 2. fixture

| fixture | 字数（去空白） | 段数 | 说明 |
|---|---|---|---|
| `tests\fixtures\mixed_hc3.txt` | 2319 | 8（ai/human 交替） | 真实 HC3-Chinese 拼接；meta 记录 dataset/sha256/行号/答案索引；`--build-mixed-fixture` 可逐字节重现 |
| `tests\fixtures\essay_3000.txt` | 2971（body 2841） | 13（heading+8 body+4 meta） | 混排中文 essay；含标题、参考文献段（ref_cut=True，excluded=126）、1080 字超长段（切 800+280）；仅用于分段契约与人类单篇 record-only |

mixed_hc3 逐段：P0 ai open_qa 223 / P1 human open_qa 224 / P2 ai baike 304 / P3 human baike 235 /
P4 ai finance 363 / P5 human finance 321 / P6 ai medicine 366 / P7 human medicine 283（字，去空白）。
来源 dataset=Hello-SimpleAI/HC3-Chinese，dataset_sha256=425e7071bbd5…（完整值见 meta）。

## 3. 耗时

| 场景 | wall（含进程启动+模型加载） | 报告内部 elapsed_sec | 备注 |
|---|---|---|---|
| mixed_hc3 2319 字 balanced，repeat2 | 16.32s / 15.77s | 14.85 / 14.32 | 断言：<60s PASS；两次 AI 率完全一致 |
| essay_3000 2971 字 balanced | 14.19s / 14.48s（13:14 空闲时）；16.59–16.79s（13:29 计时实验） | 13.07/13.21；15.06–15.31 | 机器负载波动导致 ±2s |
| mixed_hc3 strict / lenient | 15.97s / 15.80s | 14.65 / 14.58 | — |
| doctor | 6.9s | — | 退出码 0 |
| 降级路径（空 AIDETECT_HOME） | 0.15s | — | 退出码 3 |

AIDETECT_THREADS 对照（essay_3000，每档 2 次，见 out\timing\timing.json）：

| threads | wall | 内部 elapsed | clf | ppl |
|---|---|---|---|---|
| 默认（auto=8） | 16.79 / 16.59 | 15.31 / 15.06 | 8.84 / 8.63 | 6.24 / 6.19 |
| 18 | 16.00 / 17.29 | 14.41 / 15.86 | 8.55 / 9.67 | 5.65 / 5.98 |
| 4 | 19.05 / 17.66 | 17.71 / 16.47 | 9.69 / 9.90 | 7.68 / 6.32 |

结论：默认 8 线程与 18 线程差异在噪声内（±1s），4 线程明显变慢；**无需调 AIDETECT_THREADS**，且远低于 60s 预算。
引擎固定成本：clf 15 窗口推理、ppl 100 次窗口推理（2132–2971 字规模），本次全部包含在 wall 内。

## 4. mixed_hc3 检测结果（balanced）

- ai_rate_chars = **0.5424**（区间 (0.30, 0.85) PASS）；ai_rate_paras = 0.5；threshold = 0.0087（balanced）
- ai_rate_ci = [0.1564, 0.8897]；grade_counts = {high: 4, mid_high: 0, mid: 1, low: 3}
- 逐段 p_ai：AI 段 0.9999 ×4；人类段 0.002 / 0.0052 / 0.0003 / 0.0006 ⇒ AI 段中位数 0.9999 > 0.5，人类段中位数 0.0013 < 0.1（段落级断言 PASS）
- 被判 AI 的段落恰为 4 个 AI 段（INFO 证实）
- 引擎：clf / ppl / stylo 全部 available；calibration.calibrated=True

模式对比（mixed_hc3）：
- strict：ai_rate_chars 0.5424、ai_rate_paras 0.5、threshold 0.0618、grades {high4, mid_high0, mid0, low4}
- lenient：ai_rate_chars 0.5424、ai_rate_paras 0.625、threshold 0.0042（P3 人类 baike 段 p=0.0052 被 lenient 判 mid_high）

人类单篇 essay_3000（record-only，不作断言）：balanced ai_rate_chars 0.1247（wall 16.93s），strict 同样 0.1247（wall 17.17s）；
原因：该文段落 p_ai 两极分化（0.9956 vs ≤0.0011），三档阈值 [0.0042, 0.0618] 落在空档，故模式间整篇 AI 率一致。

## 5. HTML 自包含

`out\mixed_hc3_report.html`：29.6 KB；http:// 0、https:// 0、`<link` 0、`<script src=` 0、`src=` 0、`@import` 0、`url(` 0、`<img` 0、`<iframe` 0；含内联 `<script>`（交互表格）。

## 6. JSON 结构

`out\mixed_hc3_report.json`：10.0 KB，可 json.load；必需字段（别名映射 stats→document、grade→grade_counts、sentences→suspicious_sentences）全部存在；
document = {total_chars:2319, body_chars:2319, excluded_chars:0, paragraphs:8, units:8, ref_cut:false}；engines 3 项；calibration.calibrated=true。

## 7. 降级路径

```
$tmp = 空临时目录; $env:AIDETECT_HOME=$tmp
python -m aidetect report tests\fixtures\mixed_hc3.txt --out-dir $tmp\out
```
退出码 **3**；stderr「缺少模型：clf, ppl。先运行 python -m aidetect fetch clf ppl」；无 Traceback、无「内部错误」；临时目录 0 产物。

## 8. doctor

`python -m aidetect doctor` → 退出码 **0**，末行「自检通过：可以运行检测。」；标定 v1.0（strict 0.0618 / balanced 0.0087 / lenient 0.0042，测试集 Acc 0.9435 / AUC 0.993 / FPR 0.1309 balanced）。

## 9. 单测

```
python -m pytest -q -p no:cacheprovider        # 115 passed, 1 skipped, exit 0
python -m pytest -q -p no:cacheprovider tests\test_e2e.py   # 25 passed, 1 skipped
```
（1 skipped = 真实 report 子进程 opt-in，需 AIDETECT_E2E=1。）

## 10. 产物路径

- `out\e2e_result.json`（39 条断言 + 逐段 p_ai + 耗时 + 人类单篇 record-only 数据）
- `out\mixed_hc3_report.html` / `out\mixed_hc3_report.json`
- `out\human\balanced\essay_3000_report.*`、`out\human\strict\essay_3000_report.*`
- `out\mode_strict\mixed_hc3_report.*`、`out\mode_lenient\mixed_hc3_report.*`
- `out\timing\timing.json`（AIDETECT_THREADS 对照原始数据）
