# ai-detect 独立复核报告（reviewer）

> 复核人：reviewer（独立复核员，不改产品代码）
> 复核对象：`C:\Users\mikasa\Documents\DeepSeek Harness\ai-detect`，版本 1.0.0
> 当前状态：**终局复核完成（2026-10-26 13:36）**。task-6 清单 1–8 全部逐项通过；工程验收结论=**可交付（blocked: none）**；唯一「未达标」属**方法学局限**（域外 FPR 6.55%/13.09%/19.80% vs 目标 1%/5%/10%），已在 `eval/report.md` 与 `README.md` 如实披露。详见 **§六 终局复核**；**§6.8（21:07，R-11/R-16/R-27）与 §6.9（21:17，R-28/R-30）增量复核均通过，裁定不变**。
> 复核约束：只写本文件；未 `pip install`、未联网下载、未执行 `python -m aidetect fetch` 下载；真实模型复核按 lead 放行执行（§五、§六）；复核新增文件仅 `eval/check_t2.py`（lead 授权保留的可复跑资产），一次性探针与日志写 `%TEMP%\aidetect_review\`；§六 重跑 T4 原命令会按 e2e 自身逻辑重新生成 `out\mixed_hc3_report.*` 与 `out\e2e_result.json`（T4 产物重生成，非污染）。

## 0. 方法与限制

- **静态复核**：通读 `report.py` / `cli.py` / `fusion.py` / `aggregate.py` / `segment.py` / `textio.py` / `fetch.py` / `doctor.py` / `engines\*.py`，用 AST 脚本扫描未使用 import 与死代码。
- **动态复核（不加载模型）**：用假引擎（`tests\conftest.py` 的 `FakeEngine`）与手写边界文件驱动 `textio` / `segment` / `aggregate` / `report` / CLI 分支；需要模型的分支用 `monkeypatch` 打桩验证调用顺序与返回值。
- **未覆盖**：真实模型推理质量、标定数字、e2e 实测耗时（属 T2/T5 产出，放行后复测）；docx/pdf 真实文件解析（无 fixture）；标准输入 `report -` 的完整链路（仅验证产物路径生成，见 R-15）。
- 证据脚本与样例文件：`%TEMP%\aidetect_review\{probe_pre.py, probe_cli.py, scan_static.py, files\*}`。

## 一、已复核（预复核，只读，不依赖模型/标定）

### 1.1 基线快照

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 单测（早期基线） | `python -m pytest -p no:cacheprovider` | `73 passed in 1.71s`，exit 0 |
| 单测（当前快照，T2/T5 并行进行中） | 同上 | `1 failed, 88 passed, 8 skipped, 1 xfailed in 2.17s`，唯一失败为 `tests/test_calibrate.py::test_run_meta_section_reports_scale_and_bench`（断言 `'601'` 出现在 `_run_meta_section({})` 输出，实际为 `602`）—— **属 T2 进行中的预期波动，不作为本复核结论** |
| doctor（标定未就绪） | `python -m aidetect doctor` | 依赖/模型/语料 ✓；标定 ✗（`aidetect\resources\calibration.json` 不存在）；末行「自检发现缺失项」，exit 1（按 README 属正常流程） |
| 资源就绪 | `python -c "from aidetect import fetch; print({n: fetch.is_ready(n) for n in ('clf','ppl','hc3zh')})"` | `{'clf': True, 'ppl': True, 'hc3zh': True}` |
| 产物现状 | `Test-Path aidetect\resources\calibration.json` / `eval\report.md` | 均不存在（待 T2） |
| README | `README.md` 522 行（T3 产出） | 已通读，承诺清单见 §1.4 |

### 1.2 发现清单

> 「是否阻塞」= 是否建议阻塞本次发布（或是否影响 task-6 验收条目）；「严重度」为独立判断。
> **说明**：以下为预复核（lead 修复前）的原始证据，保留作历史记录；修复后的复测结果见 §二。

| 编号 | 问题 | 证据 | 建议 | 是否阻塞 |
| --- | --- | --- | --- | --- |
| **R-01** | **参考文献剔除在「标题与首条文献同块」时整体失效**：`REF_HEADING.match(b) and len(b) <= 24`，当「参考文献」标题与 `[1]…` 条目之间没有空行、被拼成同一段落块时长度 >24，匹配失败 → 整块参考文献当正文计分。 | `%TEMP%\aidetect_review\files\refs_noblank.txt`（`参考文献` 后紧跟条目）：`build_doc` → `body_chars=38, excluded=0, units=1`，且该 unit `kind=body`、参与了 `make_units`；对照组（标题后有空行）：`body_chars=0, excluded=38, units=0`。源码：`aidetect\segment.py:222`。 | 判更鲁棒：对块首行做 `REF_HEADING.match(b.splitlines()[0])` 或 `REF_HEADING.match(b[:24])`；并把命中的整块计入 `excluded_chars`。改后补一条单测（无空行 case）。 | 否（但 README §4.2 的「参考文献剔除」承诺在常见排版下不成立，建议修后复测） |
| **R-02** | **纯零宽字符文件被当作「内部错误」退出码 4，而 README 承诺输入类错误为 2**：`textio.load` 对零宽字符文件能成功读出，`segment.build_doc` 随后抛 `ValueError`，CLI 只在 `textio.ReadError` 上转 2，`ValueError` 落到 `except Exception` → 4。 | `python -m aidetect report %TEMP%\aidetect_review\files\zerowidth.txt` → exit 4，`内部错误：ValueError: 切分后没有任何段落，文本可能只有空白字符`。源码：`aidetect\cli.py:77-84`（只捕 ReadError）、`aidetect\segment.py:237`。 | `cmd_report`/`cmd_json` 增加 `except ValueError` → `EXIT_INPUT`（或让 `textio.load` 在 `build_doc` 前预检可见字符数）。 | 否（边界输入；与 README 退出码表冲突，建议修） |
| **R-03** | **`json` 子命令缺少 `body_chars == 0` 校验**：`cmd_report` 有（`cli.py:82-84`），`cmd_json` 没有 → 正文全部被判为 meta/参考文献的文档，`json` 仍加载模型、跑 `analyze`、输出 `ai_rate=0` 的 JSON 且退出码 0。 | 打桩 `cli._models_ready→(True,[])`、`cli.aggregate.analyze→记录调用`：`report refs_only => rc=2 且 analyze 未被调用`；`json refs_only => rc=0 且 analyze 被调用`，输出 `document.body_chars=0`。源码：`aidetect\cli.py:110-129`。 | 在 `cmd_json` 的 `_load_doc` 后复制 `cmd_report` 的空正文判断，返回 `EXIT_INPUT`。 | 否（一致性缺陷，建议修） |
| **R-04** | **HTML 报告首屏图例阈值与实际分级不一致**：图例沿用 `calibration.recipe.thresholds`（未标定为 0.80/0.60/0.40），而段落四档实际用 `fusion.grade_cutoffs_for(threshold)` 由模式阈值 θ 推导。 | `report.render` 生成的 HTML 中出现「极高（≥0.80）偏高（≥0.60）中等（≥0.40）」，而同一数据下 `grade_cutoffs_for(0.6)={'high':0.85,'mid_high':0.62,'mid':0.45}`，`grade_of(0.80)='mid_high'` —— 图例与判级矛盾。源码：`aidetect\report.py:331-333`；README §4.4 承诺按 θ 推导（与 `fusion.grade_cutoffs_for` 一致）。 | 图例改用 `fusion.grade_cutoffs_for(analysis.threshold)` 的值；或注明「图示为标定固定档」。 | 否（真实逻辑 bug、用户可见误导，建议修） |
| **R-05** | **合并判定单元的句级概率错位**：`_suspicious_sentences` 先对合并单元的全部句子求 `probs`，再对每个段落 `zip(para.sentences, probs)` —— `zip` 每次从 `probs[0]` 重新对齐。 | 2 个 11 字短段合并为 1 个 unit；`score_sentences` 返回 `[0.99, 0.01]` 时，`_suspicious_sentences` 输出两段都是 `0.99`（第 2 段应 0.01）。源码：`aidetect\aggregate.py:253-254`。 | 按段累加偏移切片：`offset=0; for para: k=len(para.sentences); probs[offset:offset+k]; offset+=k`。 | 否（影响逐段证据可信度，建议修） |
| **R-06** | **`python -m aidetect.fetch --help` / `-h` 不可用且输出误导**：`fetch.py` 自研参数解析没有 argparse，未知名字一律当资源名 → `[FAIL] --help: '未知资源: --help'`，且无条件打印「分片已保留，直接重跑同一命令即可续传」。 | `python -m aidetect.fetch --help` → exit 1；`python -m aidetect.fetch -h` → exit 1；对比 `python -m aidetect fetch --help` → exit 0 正常帮助。源码：`aidetect\fetch.py:362-391`（无 help 分支）、`fetch.py:384`（无条件续传提示）。 | 识别 `-h/--help` 打印 usage 并 `return 0`；「分片已保留」只在实际发生失败/中断时打印。 | **是（按 task-6 第 8 条字面不通过，需 lead 裁定口径：视为 fetch.py 模块级 bug 还是仅要求 CLI 子命令）** |
| **R-07** | **缺模型时的提示指向不存在子命令 `fetch-model`**：模块内 `available()` 的错误文本要求「先跑：python -m aidetect fetch-model」，而 CLI 只有 `fetch clf|ppl|hc3zh|all`。 | grep：`aidetect\engines\classifier.py:60`、`aidetect\engines\perplexity.py:39`；`python -m aidetect fetch-model` → argparse `invalid choice` exit 2。 | 改为 `python -m aidetect fetch clf` / `fetch ppl`（或统一成 `python -m aidetect fetch {name}`）。 | 否（用户可见误导，建议修） |
| **R-08** | **Markdown 标题不被识别为 heading**：README §4.2 声明支持 `.md` 且「章节标题…标记为 heading 不独立判定」，但 `#`/`##` 前缀使 `HEADING_RE` 与 `_is_heading` 失配，标题行按 body 进入判定。 | `build_doc("# 第一章 绪论\n\n## 1.1 研究背景")` → 两段 `kind='body'`、`body_chars=15`、`units=1`（标题文本参与打分）。源码：`aidetect\segment.py:132-136`、`segment.py:181-190`。 | `_classify`/`_is_heading` 先去掉 `^\s*#{1,6}\s+` 再匹配；补 md 标题单测。 | 否（影响 md 输入的口径与段落计数，建议修） |
| **R-09** | **`generated_at` 未转义**：`report.render` 中唯一未经过 `_esc` 的插值；`report.write()` 不传该参数，当前 CLI 无法注入。 | `report.render(a, generated_at='<b>INJECT</b><script>alert(\'gen\')</script>')` → 字符串原样出现在 HTML；其余注入字段（name/正文/warnings/recipe.note/engine_info）raw payload 出现次数均为 0。源码：`aidetect\report.py:288`。 | `{_esc(generated_at)}`。 | 否（低危，防御性修复） |
| **R-10** | **死代码 / 未使用 import**：`segment.py:42 SENT_FINAL` 定义后 0 引用；`segment.py:233-234` `excluded += 0` 空操作；`aggregate.py:20` `Para` 未用；`textio.py:7` `import io` 未用；`cli.py:241` `import os` 放在 `main` 定义之后（运行正常，异味）。`calibrate.py:35` `sigmoid` 未用（该文件 T2 正在改，需复看）。 | `python %TEMP%\aidetect_review\scan_static.py` 输出（AST 扫描，已排除 `__future__` 与 `engines\__init__.py` 的有意 re-export）。 | 清理；`SENT_FINAL` 若计划与 `SENT_END` 合并则补注释。 | 否 |
| **R-11** | **纯英文/不含中文的文本没有任何护栏**：中文判别器与中文 GPT2 对非中文文本的打分没有语义，但流程照常出分，`warnings` 不提示。 | `build_doc` + `aggregate.analyze` 对 1650 字纯英文文档正常输出分数与 HTML；`aggregate._warnings` 五项条件（`aggregate.py:60-110`）无语言/CJK 检查；README §一 已声明「不适合以非中文为主」。 | `_warnings` 增加：正文 CJK 字符占比 < 5%（或 <20 字）→「非中文文本，结果不具解释力」；或 CLI 直接拒绝。 | 否（README 已声明不适用；但加护栏更稳） |
| **R-12** | **超长英文/无标点段未按承诺切开**：`_split_long_para` 只在 `SENT_END`（中英文标点集不含英文句点）边界切；无句读的 1650 字英文段、1440 字中文无标点段都 > `MAX_PARA_CHARS=800` 却未切分，与 README §4.2「单个单元不超过 800 字」不符。 | `build_doc(纯英文长段)` → 唯一 unit `chars=1650`，`over800=True`。源码：`aidetect\segment.py:163-178`（`SENT_END` 见 `segment.py:38` 不含 `.`）。注：`window_texts` 对超长句有硬切兜底，故推理窗口不会溢出，但仍与切分承诺不一致。 | `_split_long_para` 无句读时按 `MAX_PARA_CHARS` 字符硬切（可复用 `window_texts` 逻辑）。 | 否（承诺与实现不一致，建议修） |
| **R-13** | **`doctor` 声称「只读不联网」，但会写探针文件**：`run()` 在 `OUT_DIR` 下 `mkdir` 并写 `.doctor_write_test` 再删除（代码确实删除，但 docstring 与 README §二/§七称「只读」）。 | `python -m aidetect doctor` 后可见 `out\.doctor_write_test` 的创建/删除；源码：`aidetect\doctor.py:135-139`、模块 docstring 与 `README.md:90`。 | 去掉写探针（改用 `os.access` 判断可写），或 README 改为「除一次可写性探测外只读」。 | 否（文档与实现表述冲突，建议二选一） |
| **R-14** | **未标定 `mode_thresholds` 出现浮点尾数**：`fusion._fallback_mode_thresholds` 由 `mid_high - 0.15` 推出 `lenient=0.44999999999999996`，会进 JSON/HTML 展示。 | `fusion` 未标定时：`{'strict':0.75,'balanced':0.6,'lenient':0.44999999999999996}`。 | 生成阈值时 `round(x, 4)`，或展示层统一 `:.2f`。 | 否（观感问题） |
| **R-15** | **`report -`（标准输入）在 Windows 上必然失败**：`textio.load('-')` 返回展示名 `<stdin>`，`_resolve_out` 直接拼文件名 → `out\<stdin>_report.html`，`<` `>` 是 Windows 非法文件名字符 → `OSError` → 退出码 4。README 参数表把 `-` 列为 `report`/`json` 均可用的输入。 | 假引擎 + `cli._resolve_out(str(tmp), '<stdin>', 'report.html')` → 路径 `...\<stdin>_report.html`；`report.write(...)` → `OSError: [Errno 22] Invalid argument`。源码：`aidetect\cli.py:64-67`、`aidetect\textio.py:99-103`。 | `_resolve_out` 对 `doc_name == '<stdin>'` 使用 `stdin` 作为 stem（或 `textio` 返回 `-`）；补一条 CLI stdin 单测。 | 否（可低成本修；若 lead 认为 stdin 属承诺功能，建议修后再发布） |
| **R-16** | **二进制/不可解码文件静默变成 mojibake 正文**：`_decode` 各编码失败后 `errors="replace"` 兜底，`binary.bin` 读出 `'Ā＂þ'` 被当正文（3 字）。 | `textio.load(binary.bin)` 不抛错；对比空文件/纯空白会正确 `ReadError`。源码：`aidetect\textio.py` `_decode`。 | 若替换字符（U+FFFD）占比高或含 `\x00` → `ReadError("疑似二进制文件，无法按文本读取")`。 | 否 |
| **R-17** | **README 对 `--out-dir` 默认值的描述不准确**：参数表写「默认项目下的 `out/`（相对当前工作目录解析）」，实际默认是 `config.OUT_DIR = HOME/"out"`（`AIDETECT_HOME` 或项目根），只有显式传相对路径时才相对 cwd。 | `aidetect\config.py:10-15`、`aidetect\cli.py:64-67`。 | 改为「默认 `$AIDETECT_HOME/out`（未设置时为项目根下 `out/`）」；显式相对路径才相对 cwd。 | 否（文档；归属 engineer-docs / lead 派单） |
| **R-18** | **独立 `[1]` 式参考文献条目：被判 meta 但不计入「已剔除字数」**：`excluded_chars` 只在 `ref_cut` 分支累计；README 表述为剔除计入字数。 | `build_doc("[1] 张三…\n\n[2] 李四…")` → `paras` 均 `kind='meta'`、`excluded_chars=0`、`body_chars=0`（报告会显示「已剔除 0 字」）。源码：`aidetect\segment.py:224-227`。 | 在 `_classify` 返回 meta 时也累加 `excluded`，或 README 说明「仅参考文献标题后的内容计入剔除字数」。 | 否（口径展示问题，低） |

### 1.3 已验证正常的行为（正面证据）

| 项 | 证据 |
| --- | --- |
| XSS 转义覆盖 | 向 `name`/正文/`warnings`/`recipe.note`/`engine_info(label+note)` 注入 `<script>alert('xss1')</script>`、`onerror=alert('xss2')`、`<b>` 等 payload 后，HTML 中 raw payload 出现 0 次，`&lt;script&gt;` 出现 6 次（`report.render` 直接调用，假引擎） |
| 报告自包含 | 300 段 / 14290 字文档：`report.render` 0.002s、HTML 197KB、`href`/`src` 外部资源 0 个、`http(s)://` 不出现（单测 `test_report.py` 同口径亦通过） |
| 0 判定单元不崩 | 仅参考文献的文档 `report.render` 正常（11.5KB HTML），置信区间显示「样本不足」，warnings 含「已剔除…100%」 |
| 全角空格混排 | `build_doc` 将 `\u3000` 归一化为半角空格，正文 90 字、单元切分正常 |
| 编码支持 | UTF-8（含 BOM）、GB18030、UTF-16 解码正确；空文件/纯空白/BOM-only 抛中文 `ReadError`；不存在路径、目录的报错均为中文 |
| CLI 退出码（已验证部分） | no-args=2、`--help`=0、`--version`=0、`bogus`=2、`report` 缺参数=2、不存在文件=2、空文件=2、纯空白=2、仅参考文献（有空行）=2、`--mode weird`=2、`bench` 无标定=3、`AIDETECT_HOME` 空目录时 `report`=3（缺模型提示含正确命令）、`doctor`=1；均与 README 退出码表一致 |
| `fetch` 子命令默认值 | `cmd_fetch`: `args.names or ["clf","ppl"]`（`cli.py:139`），与 README「只做检测可只跑 `python -m aidetect fetch`（默认 clf+ppl）」一致 |
| 零网络（代码面） | 全仓 grep：仅 `fetch.py` 使用 `urllib.request`；`engines\base.py` `load_pretrained` 固定 `local_files_only=True`；`report`/`aggregate`/`engines`/`fusion` 无 `requests`/`urllib`/`socket`/`http` |
| 报告性能 | 300 个判定单元 14290 字：`render` 2ms，HTML 197KB，无外链 |

### 1.4 与 README 承诺的一致性核对

| README 承诺 | 实现核对 | 结论 |
| --- | --- | --- |
| §4.2 参考文献标题后整块剔除 | `segment.py:222` 仅当整块 ≤24 字才识别 | **不一致（R-01）** |
| §4.2 单个单元 ≤800 字（超长段按句切） | 无句读长段不切 | **不一致（R-12）** |
| §4.2 支持 `.md`、章节标题标 heading | `#` 标题被当 body | **不一致（R-08）** |
| §4.4 段落四档由 θ 推导（极高=min(.95,θ+.25) 等） | `fusion.grade_cutoffs_for` 正确；`report` 图例却显示固定 thresholds | **报告展示不一致（R-04）** |
| §3.6 可疑句阈值 max(0.70, θ)、只对判 AI 单元 | `aggregate._suspicious_sentences` 与调用方一致；合并单元存在概率错位 | **基本一致（对齐缺陷 R-05）** |
| §3.7 五项护栏 | `aggregate._warnings` 五项齐全 | 一致 |
| §二 doctor「只读不联网」 | 会写可写性探针文件 | 表述与实现冲突（R-13） |
| §二/退出码 2 = 输入不可用 | 零宽字符文件 → 4；`report -` → 4（stdin 产物命名） | **不一致（R-02/R-15）** |
| §二 fetch 唯一联网命令 | 代码面成立 | 一致 |
| §3.2 图例与判级 | 见 R-04 | **不一致** |

## 二、正式复核（第一批：不依赖标定/模型加载的修复项复测，2026-10-26 12:52 起）

> 依据 lead m00189 的放行裁定：先复测已修复项与 task-6 清单 4/5/6/7/8，再在 T2/T4 产出后补清单 1/2/3。复测期间 T2 标定进程（PID 2272，12:48 启动）仍在运行，除 §2.4 注明的一次误触外未加载真实模型。

### 2.1 环境与命令

| 项 | 值 |
| --- | --- |
| 代码快照 | cli.py 12:49:23 / aggregate.py 12:48:49 / segment.py 12:48:16 / report.py 12:49:42 / fetch.py 12:50:03 / classifier.py+perplexity.py 12:50:23 / calibrate.py 12:45:28 / README.md 12:43:10 |
| 单测 | `python -m pytest -p no:cacheprovider -o addopts="" -q` → rc=0，`96 passed, 8 skipped in 3.63s`（注意：pyproject `addopts=-q` 与命令行 `-q` 叠加会变 `-qq` 吞掉汇总行） |
| 复测脚本 | `%TEMP%\aidetect_review\probe_fix.py`（假引擎 + 子类化 `ClassifierEngine`，不加载模型）+ 旧 `probe_cli.py` 回归 |

### 2.2 逐项复测结果

| 编号 | 修复内容 | 复测证据 | 结论 |
| --- | --- | --- | --- |
| R-01 | 参考文献同块剔除、不误切正文 | `参考文献\n[1]…`（标题+条目同块）→ `body=0 excluded=42 units=0 kinds=['meta']`；`正文+参考文献`（无空行）→ `body=15 excluded=23`；负例 `参考文献的格式要求如下：…`（52 字正文）→ `body=52 excluded=0` kind=body（不误切）；24 字短负例 → `body=24` | **已复测通过** |
| R-02 | `build_doc` ValueError → 退出码 2 | `report zerowidth` / `json zerowidth` → rc=2 `无法解析文档：切分后没有任何段落…`；空文件仍 rc=2 `读取失败：文件没有可读文本` | **已复测通过** |
| R-03 | `json` 补空正文校验 | `report refs_only` rc=2、`json refs_only` rc=2；`probe_cli.py` 打桩确认 `analyze_called=[]`（不再调用 analyze） | **已复测通过** |
| R-04 | 图例改用 `Analysis.cutoffs`、JSON 增 `grade_cutoffs` | strict：θ=0.75 → cutoffs 0.95/0.77/0.60，legend==cutoffs；balanced：0.85/0.62/0.45；lenient：0.70/0.47/0.30；HTML legend 数值逐一相等；JSON `grade_cutoffs` 同值（round4）、`threshold`=0.75/0.6/0.45 | **已复测通过** |
| R-05 | 合并单元句级概率按偏移切片 | 双短段合并 unit（chars=22, para_indexes=[0,1]），`score_sentences=[0.99,0.01]` → 仅 `SentenceRow(para0,'…',0.99)`；para1 的 0.01 正确落于阈值下被过滤（旧版会错配成 0.99） | **已复测通过** |
| R-06 | fetch 模块 CLI（--help/未知资源/缺参） | `python -m aidetect.fetch --help`／`-h`／`help` → rc=0 + usage；`bogus` → rc=2，stderr `未知资源: bogus` + usage，无「分片已保留」；`--workers`（缺参）→ rc=2；`python -m aidetect fetch --help` 仍 rc=0 | **已复测通过（task-6 第 8 条字面达标）** |
| R-07 | 引擎提示命令名 | 静态扫描 `fetch-model` 0 处；`classifier.py:60` 现为 `python -m aidetect fetch clf` | **已复测通过** |
| R-08 | markdown 标题识别 | `# 第一章 绪论\n\n## 1.1 研究背景\n\n正文` → kinds=['heading','heading','body']、body=26、units=1 | **已复测通过**（新增误判见 R-21） |
| R-09 | `generated_at` 转义 | 注入 `<b>INJ</b><script>alert('gen')</script>` → raw_script=0、raw_bold=0、`&lt;b&gt;INJ&lt;/b&gt;`=1 | **已复测通过** |
| R-10 | 死代码清理 | 静态扫描 SENT_FINAL 0 处、`excluded += 0` 0 处；残留 `textio.py:7 import io`（未用、无害），`cli.py:262 import os` 为 `# noqa: E402` 有意放置 | 基本通过 |
| R-12 | 超长段硬切 ≤800 | `long_no_punct`（1440 字）→ unit sizes `[800, 640]`，max=800；`pure_en`（1920 字）→ `[800, 800, 319]`；所有推理窗口 `max_win=300`（nwin=6/7） | **已复测通过** |
| R-15 | stdin 产物命名（`_safe_stem`） | `<stdin>`→`stdin`；`a:b*c?.txt`→`b_c`；`'...'`→`document`；`_resolve_out(...,'<stdin>','report.html')` → `...\out_stdin\stdin_report.html`（无非法字符）；`json -` 默认打印 stdout 不受影响 | **已复测通过（单元级；端到端待 T2 后）** |
| 新增 | `bench --limit` | `python -m aidetect bench --help` → rc=0，含 `--limit LIMIT`；`python -m aidetect bench --limit 5`（无标定）→ rc=3 `还没有 calibration.json…`；`tests/test_cli.py::test_bench_parser_accepts_limit` 通过 | **已复测通过** |
| 回归 | XSS + 0 单元 render | name/engine label/warnings/generated_at 注入 payload → raw 计数全 0；0 单元文档 render 正常（html=11255、无 `<svg>`、warnings 3 条） | **通过** |
| 回归 | 零网络（假引擎运行期） | 将 `socket.socket`/`socket.create_connection` 替换为抛错后，`aggregate.analyze + report.render + report.write` 正常完成并落盘（`net_probe.html`） | **通过（真实模型路径待补）** |

### 2.3 新发现残余问题（低危/文档，均非阻塞）

| 编号 | 问题 | 证据 | 建议 | 阻塞 |
| --- | --- | --- | --- | --- |
| **R-19** | `eval/report.md` 的「复现命令」段与实际 CLI 不一致：`calibrate.py:552-556` 生成 `python -m aidetect.calibrate bench --limit N`，并硬编码「CLI 的 `bench` 子命令不接受 `--limit`」；而 CLI 现已支持 `python -m aidetect bench --limit N`（cli.py:240-243、tests/test_cli.py:90-94）。 | 源码 `calibrate.py:550-556`；`bench --help` 含 --limit | 改为 `python -m aidetect bench --limit N`，删除过时说明；T2 重跑 report.md 后复测 | 否 |
| **R-20** | `doctor.py:1-4` 模块 docstring 仍称「只读不写」，实际 `_check_runtime` 在 OUT_DIR 写 `.doctor_write_test` 后 unlink（doctor.py:135-138）；README §七 已改口径，模块 docstring 未同步。 | 源码对照；README §七 已改「写临时探针后立即删除」 | 同步 docstring（或改为 `os.access` 判断可写） | 否 |
| **R-21** | markdown 标题识别过宽：`_is_heading` 对 `#` 开头不要求空格 → `build_doc("#1号样本的观测结果如下，这是一行正文。")` 整段被判 heading（不计分）。 | probe_fix `R-08:hash_no_cjk_heading` → kinds=['heading']；源码 `segment.py:136-142` | 仅当 `^\s*#{1,6}\s+` 才按 md 标题处理 | 否 |
| **R-22** | README 未记录新增的 `bench --limit`：命令表只写 `python -m aidetect bench`（README.md:93/495），而 CLI 已支持 `--limit N`（cli.py:240-243），且 T2 的 `eval/report.md` 复现命令依赖它复现同一 test 子集。 | README 这两行无 `--limit`；`python -m aidetect bench --help` 输出含 `--limit LIMIT` | engineer-docs 在命令表/§十 补 `bench [--limit N]` 一句并说明用途 | 否 |
| **R-14 残余** | 未标定 `mode_thresholds.lenient=0.44999999999999996` 的浮点尾数仍在 `Recipe` 内部；展示层（HTML `:.2f`/JSON round4）已正常，标定路径 `calibrate.py:315/343-345` 已 round4。 | probe_fix `R-04:lenient`：内部 float 尾数存在、JSON/图例为 0.45 | T2 标定文件生成后确认三键均为 4 位小数即可关闭 | 否 |

### 2.4 说明与限制

- 【补记】本节列举的「尚未覆盖」项（真实模型 stdin、真实产物 XSS/边界、doctor=0、e2e、calibration 一致性、退出码 4）已在 §五 与 §六 全部补齐并复测通过。- 复测期间 T2 标定进程（PID 2272）在跑；为满足「不同时起多个重型模型进程」，本批只用假引擎/无需推理路径。**一处误触**：`report binary.bin` 边界用例实际加载了真实模型（16.57s，rc=0），随后已停止任何模型加载类复测；该结果同时证明真实模型成功路径 rc=0（但 binary.bin 属 mojibake 负例，R-16 仍成立）。
- 尚未覆盖：真实模型 `report -`（stdin）端到端、真实模型 html/json 产物 XSS/边界、doctor=0、e2e 耗时、calibration 数字一致性、退出码 4 的合法触发用例。
- 本轮未改任何产品代码/测试；仅新增本文件内容与 `%TEMP%` 下探针脚本。

## 三、待模型/标定就绪后复核（task-6 清单 1–8）

> 【历史快照，已被 §六 取代】以下为 12:52 时的「待办视角」状态；清单 1–8 的终局证据与结论以 **§六** 为准。
> 放行条件：`aidetect\resources\calibration.json` 与 `eval\report.md` 生成、pytest 全绿、doctor=0。以下状态为「预复核已完成的证据」，12:57 复测更新见 §二。

| # | 验收项 | 现状 | 放行后的复核方式 |
| --- | --- | --- | --- |
| 1 | `python -m pytest -q -p no:cacheprovider` 全绿 | 当前 1 failed（`test_calibrate.py::test_run_meta_section_reports_scale_and_bench`，T2 进行中）；早期基线 73 passed | 重跑并记录通过数与耗时；若仍失败，定位是否 T2 产物或测试预期问题，通知 engineer-calibrate |
| 2 | `python -m aidetect doctor` 退出码 0 | 当前 exit 1（缺 calibration.json，符合预期） | 重跑；核对标定/模型/语料全 ✓、退出码 0、输出无异常 warning 噪声（fitz deprecation 已观察到，属可选依赖提示） |
| 3 | 重跑 `tools\e2e.py` 核对实测耗时与产物 | 未跑（T5 产出，`tools\e2e.py` 已存在、`tests\fixtures\essay_3000.txt` 已存在） | 在假模型/真实模型允许的情况下重跑；记录 3000 字报告耗时、产物路径、退出码；与 README §九 FAQ「3000 字耗时」回填值一致 |
| 4 | `calibration.json` 与 `eval\report.md` 数字一致、`mode_thresholds` 三键且 strict>balanced>lenient | 文件不存在（T2 未完成） | 用脚本比对 JSON 的 `evaluation.dev/test`、`thresholds`、`mode_thresholds` 与 `eval/report.md` 表格逐项一致；校验 `set(keys)=={strict,balanced,lenient}` 与 `strict > balanced > lenient` |
| 5 | 报告 HTML XSS 与边界（空文件/纯英文/单句/只有参考文献/全角空格混排），临时目录不污染 out/ | **XSS 与四个边界已提前验证**（§1.3；R-09 除外）；真实模型下的报告待复测 | 用真实模型在 `%TEMP%` 下跑 `report`（`--out-dir` 指向临时目录），验证转义、0 单元、纯英文护栏提示、单句、全角空格；核对 `out/` 无新增污染 |
| 6 | CLI 退出码 0/2/3/4 与 cli.py/README 一致 | 已验大部分（§1.3）；**R-02（零宽→4）、R-15（stdin→4）为已知偏差** | 重跑矩阵并补测 `report`/`json` 成功路径（真实模型，临时目录）；偏差项按 R-02/R-15 建议修复后复测 |
| 7 | 运行期零网络（fetch.py 除外） | 代码面已 grep 验证（§1.3） | 运行期证明：在 `report`/`json` 路径（假引擎或真实模型）中 patch `socket.socket.connect`/`urllib.request.urlopen` 为抛错，确认不触发；`fetch.py` 的联网点单独确认只在其路径内 |
| 8 | `python -m aidetect.fetch --help` 与 `is_ready` 正常 | `is_ready` ✓（clf/ppl/hc3zh 均 True）；**`python -m aidetect.fetch --help` exit 1（R-06）** | 待 lead 裁定 R-06 口径；修复后复测 `--help`/`-h` 退出码 0 且输出 usage |

**12:57 复测更新**：item 1 → `96 passed, 8 skipped`、rc=0；item 8 → R-06 已修复，`--help`/`-h`/`help` rc=0、未知资源 rc=2、`--workers` 缺参 rc=2（详见 §2.2）；item 5/6/7 的假引擎/无需推理部分已通过（详见 §2.2），真实模型部分待 T2 后补；item 2/3/4 仍待 T2/T4 产出。

## 四、结论与建议处理顺序（预复核 + 正式复核第一批）

> 【历史快照】本节为第一批复核（12:52）时的排序与阻塞判定；最终裁定见 **§六.5**（可交付，blocked: none）。

- **预复核未发现数据安全类高危问题**：XSS 转义覆盖完整（唯一例外 R-09 不可经 CLI 注入）；报告自包含；不联网；边界输入大多有中文错误提示。
- **建议修复（按性价比排序，均不改动核心算法）**：
  1. R-06 `fetch.py` `--help` 支持 + 续传提示条件化（task-6 第 8 条字面验收项）。
  2. R-15 `report -` stdin 产物命名（Windows 上文档化功能不可用）。
  3. R-02 零宽字符/空切分 `ValueError` → 退出码 2。
  4. R-03 `cmd_json` 补空正文校验。
  5. R-04 报告图例改用 `grade_cutoffs_for(θ)`。
  6. R-05 合并单元句级概率偏移切片。
  7. R-01 参考文献剔除的块首行匹配 + R-08 md 标题识别 + R-12 无句读硬切（同属 `segment.py`，可一次性改）。
  8. R-07 引擎提示命令名；R-13 doctor 只读承诺；R-10 死代码；R-17 README `--out-dir` 表述；R-14 浮点尾数；R-16/R-18 低危。
- **阻塞判定**：R-06 按 task-6 第 8 条字面构成阻塞候选（需 lead 裁定）；其余 17 项均为非阻塞缺陷，建议修后复测或记入 known issues。
- 放行后我将按 §二 1–8 逐项复测，并把结果补进本文件（届时更新「已复核/待复核」状态与最终结论）。

### 4.1 正式复核第一批（2026-10-26 12:52 起）结论

- **lead 已修复并复测通过的 12 项**：R-01 / R-02 / R-03 / R-04 / R-05 / R-06 / R-07 / R-08 / R-09 / R-12 / R-15 + 新增 `bench --limit`（证据见 §2.2，均为命令+实测输出，非主观判断）。
- **R-10 基本通过**：SENT_FINAL、`excluded += 0`、`fetch-model` 提示均已清零；仅剩 `textio.py:7 import io` 未用（无害）与 `cli.py:262 import os` 的 `# noqa: E402` 有意放置。
- **当前无阻塞项**：R-06 已修复并通过 task-6 第 8 条字面验收（`python -m aidetect.fetch --help` rc=0、未知资源 rc=2、`--workers` 缺参 rc=2、误导性「分片已保留」仅失败分支打印）。
- **新残留 3 项 + 1 项残余，均低危非阻塞**：R-19（`eval/report.md` 复现命令过时，归 engineer-calibrate）；R-20（`doctor.py` docstring 与「写探针后删除」实现不一致）；R-21（markdown `#` 免空格即判标题，正文以 `#` 开头会被误判 heading）；R-14 残余（未标定内部阈值浮点尾数，展示/JSON 已 round4，标定文件生成后确认即可关闭）。
- **待 T2/T4 产出后补测**：task-6 清单 2（doctor=0）、3（e2e 耗时与产物）、4（calibration.json 与 eval/report.md 数字一致、mode_thresholds 三键且 strict>balanced>lenient）、真实模型 `report -`/`json -` 端到端、真实模型产物 XSS/边界、真实模型运行期零网络、退出码 4 的合法触发用例。

## 五、正式复核（第二批：T2 标定产物 + 真实模型路径，2026-10-26 13:10 起）

### 5.1 T2 产物核对：检查器口径审计 + 独立第二路复核

**背景**：engineer-calibrate 在我建的 `eval/check_t2.py` 上打了两处 patch（表格列偏移自动识别、R-19 改为同时接受 CLI 与模块入口两条复现命令），按 lead 要求先审计 patch 是否放松检查。

审计结论（读 diff + 负向对照实测）：
- 列偏移 patch 本身**不是放松指标核对**：新模板在「数据」列后插入「阈值」「目标FPR」两列，原 `data_start` 只识别 1 个前置列会导致整行错位；跳过这两列后 n/Acc/AUC/FPR/FNR/Precision/Brier 仍是逐项字符串相等（容差 0）。若列数判断错误，指标错位会**报 FAIL** 而不是静默通过，已用负向对照确认。
- R-19 改为「同时要求 CLI 与模块入口两条命令」符合 lead m00327 裁定，不算放松。
- **但发现两处静默漏检（已修复）**：
  1. 新模板下 `data_start` 返回 3，而原阈值列核对条件是 `if start == 2:`，导致 **6 行评测表 + 4 行 per_domain 的「阈值」列核对被静默跳过**（共 20 项检查消失，通过数从 137 降到 113 即由此而来）；「目标FPR」列同样无核对。已改为 `start >= 2` 核阈值、`start >= 3` 核目标FPR，并增加「6 行/4 行均实际核对了阈值与目标FPR 列」的计数断言，防止列偏移再次静默跳过。
  2. §七.1 门槛表（6.55% / 13.09% / 19.80% 等）**只检查了「未全部达标」等文案，未与 `calibration.json` 的 `evaluation.*.test` 交叉核对**。负向对照把 6.55% 改成 6.56% 时检查器仍 rc=0——已新增第 10 节：门槛表逐行核对目标FPR/实测FPR(2位)/Acc/AUC/Brier(4位)/判定列，并要求 3 行全部核对。

**增强后的结果**（命令 `python -B eval/check_t2.py`）：
- `== T2 产出核对：156 项通过, 0 项失败 ==`，rc=0。
- 覆盖：schema/features、thresholds/mode_thresholds/temperature 均 ≤4 位小数（R-14）、三键齐全且 strict>balanced>lenient、thresholds high>mid_high>mid、evaluation 6 行 + per_domain 4 行 + coef 10 行的阈值/目标FPR/指标逐项与 report.md 一致（容差 0）、§七.1 门槛表逐项、dataset n_dev/n_test/limit/wall_clock 与 run_meta 一致、R-19 两条复现命令、R-24 文案与可操作建议。

**负向对照**（`%TEMP%\aidetect_review\negctrl_check_t2.py`，验证检查器「会红」而不是永远绿）：基线副本 rc=0；逐一篡改副本后 11/11 均按预期 rc=1：
| # | 篡改 | 结果 |
|---|---|---|
| 1 | 评测表 strict·dev 阈值 0.0618→0.0619 | FAIL（阈值列） |
| 2 | strict·dev Acc 0.997→0.998 | FAIL（acc） |
| 3 | balanced·dev 目标FPR 5%→4% | FAIL（目标FPR） |
| 4 | per_domain open_qa 阈值 0.0087→0.0088 | FAIL（open_qa.threshold） |
| 5 | 门槛表 strict 实测 FPR 6.55%→6.56% | FAIL（gate 列） |
| 6 | 删除「未全部达标」 | FAIL（R-24） |
| 7 | 删除 CLI 复现命令 | FAIL（R-19） |
| 8 | temperature 改 0.684812（6 位小数） | FAIL（R-14 ×2） |
| 9 | mode_thresholds 倒序 | FAIL（单调性） |
| 10 | dataset n_dev 1135→1134 | FAIL（与 run_meta 不一致） |

**独立第二路复核**（不调用 check_t2 的任何解析逻辑，`%TEMP%\aidetect_review\verify_t2_independent.py`）：`== 独立交叉核对：127 项通过，0 项失败 ==`，rc=0。覆盖 evaluation 6 行/per_domain 4 行/门槛表 3 行/§一与 README 的三模式域外 FPR 披露（6.6%/13.1%/19.8%）/bench 段/run_meta 数字/「未全部达标」「无一次达标」。两路实现独立，结论一致。

**T2 产物现状（13:17:36 版）**：
- `aidetect\resources\calibration.json`（5052B，13:15:32）：`thresholds {high 0.0348, mid_high 0.0087, mid 0.0043}` 可由 `fusion.grade_cutoffs_for(0.0087)` 复现且 round4；`mode_thresholds {strict 0.0618, balanced 0.0087, lenient 0.0042}`；`temperature 0.6848`；三键齐全、strict>balanced>lenient。
- `eval\report.md`（13046B）：§三 6 行与 §各域表现 4 行逐项等于 JSON；§六 bench 差异根因（阈值边界 0.0042 取整 2.1e-5 → 2 个样本翻转 → lenient FPR 差 0.0033，strict/balanced 0.0000）；§七 7 节诊断齐全（门槛判定/特征消融/C 网格/域分离/阈值迁移成因/四档倒挂修复/bench 交叉验证）；§一 明写「未全部达标」+ strict 6.6% / balanced 13.1% / lenient 19.8% + 三条可操作建议；全文无「一次达标」；§五 同时给出 `aidetect bench --limit 150`（CLI）与 `aidetect.calibrate bench --limit 150`（模块入口）两条复现路径。
- README 同步披露域外 FPR（6.6%/13.1%/19.8%）、阈值迁移说明与 bench ≤0.0032 差异口径。

**结论：task-6 清单第 4 条（calibration.json 与 eval\report.md 数字一致、mode_thresholds 三键且 strict>balanced>lenient）已复测通过，T2 产出可签字。** 剩余为「方法学局限」而非工程缺陷：域外 balanced FPR 13.1% 超出 [3%,8%] 目标区间（strict 6.6%、lenient 19.8% 同未达标），按协议未用 test 调阈值，报告与 README 已如实披露——写入终局结论，不作为阻塞项。

### 5.2 真实模型路径复核（batch 2，无需 e2e 产物）

环境：真实模型（clf 410MB + ppl 421MB，CPU），单次 report 约 13–23s；`AIDETECT_THREADS` 默认 8。

| 检查 | 证据 | 结论 |
|---|---|---|
| stdin `report -`（真实子进程） | `python -m aidetect report - --out-dir <tmp> < cn_doc.txt` → rc=0，生成 `stdin_report.html/json`，body_chars 480、units 1、ai_rate 0.9987 | 通过（R-15 已修） |
| stdin `json -`（真实子进程） | `python -m aidetect json - --out <tmp>\stdin_json_sub.json` → rc=0 且文件落盘 | 通过 |
| 空文件 / 零宽字符 / 只有参考文献 | rc=2，0.00–0.01s（不加载模型），提示「无法解析文档：切分后没有任何段落」「文档正文为空…」 | 通过（R-01/R-02 已修） |
| 纯英文 1650 字 | rc=0，units [682,679,199] 均 ≤800，ai_rate 1.0；warnings 仅 HC3 域外误报提示 | **无「非中文/纯英文」护栏**（R-11 仍为文档/产品限制，非阻塞） |
| 单句 9 字 | rc=0，ai_rate 0.9992，含「正文仅 9 字，低于 300 字可靠性下限」警告 | 通过 |
| 全角空格混排 | rc=0，units [29,480]，ai_rate 0.9418 | 通过 |
| 超长无标点段落 | 1440 字 → units [800,640]；纯英文 1920 字 → [800,800,319] | 通过（R-12 已修） |
| 真实产物 XSS | 注入 `<script>alert('xss')</script>`、`<img src=x onerror=alert(1)>`、`"><svg/onload=alert(2)>`、`<iframe src="http://evil.example">` 的文档跑真实 report：payload 全部转义为 `&lt;script&gt;…` / `&#x27;` 纯文本；raw `<img`/`<iframe` 计数 0；`onerror=`/`http://evil.example` 只出现在转义文本内；HTML 中 raw `<script>`/`<svg>` 各 1 个是报告自身内联脚本与图表（非 payload） | 通过（无标签注入） |
| 运行期零网络（真实模型） | 先把 `socket.socket`/`socket.create_connection`/`socket.getaddrinfo` 打成抛 `OSError`，再跑真实 report+json → rc=0、json 落盘 | 通过 |
| 内部错误 | patch `aggregate.analyze` 抛 RuntimeError → rc=4，提示「内部错误：RuntimeError: …（设 AIDETECT_DEBUG=1 看堆栈）」；AIDETECT_DEBUG=1 时重抛 | 通过 |
| Ctrl-C | patch analyze 抛 KeyboardInterrupt → `cli.main` 返回 130 | 通过 |
| 缺模型 | `AIDETECT_HOME` 指向空目录：report rc=3「缺少模型：clf, ppl。先运行 python -m aidetect fetch clf ppl」、doctor rc=1 | 通过 |
| doctor 全就绪 | rc=0，6.3s，末行「自检通过：可以运行检测。」；fitz deprecation 噪声已消失（R-13 转好） | 通过 |
| fetch 帮助/就绪（清单第 8 条） | `python -m aidetect.fetch --help` rc=0（首行「用法: python -m aidetect.fetch [资源 ...] [--workers N] [--chunk-mb N] [--retries N]」）、`-h` rc=0、`python -m aidetect fetch --help` rc=0、`bogus` rc=2 stderr「未知资源: bogus」；`is_ready={"clf":true,"ppl":true,"hc3zh":true}` | **通过（R-06 已修，授权书第 8 条字面验收满足）** |

说明：tools/e2e.py 的「自包含」断言是**子串级**（不得含 `http://`、`<link`、`src=` 等）。含 URL 文本的普通文档正文会被渲染为转义纯文本（非外部资源），但仍会命中该子串断言——这是测试口径限制而非产品缺陷，记为 R-23（低危、不阻塞；建议 e2e 改为检查 `href=`/`src=`/`<link` 等真实引用，或 fixture 避免 URL 文本）。

### 5.3 收尾结果（已由 §六 终局复核覆盖）

- e2e 于 13:34 完成重做与收口（真实 HC3 混排 fixture，39 条必需断言），我随后按 lead 要求重跑 pytest、`tools\e2e.py` 与 T2 一致性检查器，**全部通过**；最终数字、证据与裁定见 **§六**。
- 此前 2 failed/103 passed 的中间态已消失（根因：`tests/test_e2e.py` 与 `out\essay_3000_report.json` 旧版并存，非产品缺陷）。

## 六、终局复核（T4 收口后，2026-10-26 13:30 起）

> 触发：lead m00658（T4/task-5 completed）+ engineer-e2e m00652 通知 e2e 重做完成。以下是 task-6 清单 1–8 的最终签字证据与裁定。
> 环境：Windows / 18 逻辑核 / CPU-only torch 2.14.0+cpu / transformers 5.20.2 / Python 3.12.10；命令均在 `ai-detect` 下、`$env:PYTHONIOENCODING='utf-8'`。

### 6.1 清单 1–8 终局逐项结论

| # | 清单项 | 结论 | 关键证据（命令 + 输出） | 是否阻塞 |
| --- | --- | --- | --- | --- |
| 1 | pytest 全绿 | **PASS** | 默认：`python -m pytest -p no:cacheprovider -o addopts="" -q` → `114 passed, 1 skipped in 2.35s`，rc=0；skip 为 `tests\test_e2e.py:430`「真实 report 会加载模型…设 AIDETECT_E2E=1 显式开启」。显式开启：`$env:AIDETECT_E2E='1'` 后同命令 → `115 passed in 17.41s`，rc=0。全量无失败。 | 否 |
| 2 | doctor 退出码 0 | **PASS** | `python -m aidetect doctor` → rc=**0**，末行「自检通过：可以运行检测。」（e2e 内实测 8.3s；我复测 6.3s）。`out\` 无 `.doctor_write_test` 残留 → 写探针后立即删除，与 `doctor.py:1-5` 新 docstring 一致（R-20 关闭）。 | 否 |
| 3 | 重跑 `tools\e2e.py` 核对耗时与产物 | **PASS** | 命令 `python tools\e2e.py --repeat 2 --compare-human --json-out out\e2e_result.json` → **EXIT=0**；`out\e2e_result.json` 中 `required=true` 断言 **39/39 PASS、0 FAIL、6 INFO**（总计 45 条）；耗时 `[18.19, 21.26]s`（<60s PASS，机器负载波动，engineer 实测 16.32/15.77s）；`--repeat 2` 两轮 `ai_rate_chars/ai_rate_paras` 完全一致 `(0.5424, 0.5)`；ai_rate_ci `[0.1564, 0.8897]`；`document={total_chars:2319, body_chars:2319, excluded_chars:0, paragraphs:8, units:8, ref_cut:false}`；降级路径 rc=3、无栈、0 产物。产物：`out\mixed_hc3_report.html` 30,529B、`out\mixed_hc3_report.json` 17,015B。我的独立产品校验脚本（`%TEMP%\aidetect_review\check_final_products.py`）**40/40 PASS**（断言计数、逐段标签/p_ai、HTML 自包含、JSON 数值一致）。 | 否 |
| 4 | calibration.json ↔ eval/report.md 一致；三键且 strict>balanced>lenient | **PASS** | `python -B eval\check_t2.py` → **156 项通过 / 0 失败**，rc=0（逐项容差 0）；独立第二路 `verify_t2_independent.py` → **127/0**；负向对照 `negctrl_check_t2.py` → **11/11 全部按预期变红**。实测值：`thresholds={high 0.0348, mid_high 0.0087, mid 0.0043}`、`mode_thresholds={strict 0.0618, balanced 0.0087, lenient 0.0042}`、`temperature 0.6848`（均 ≤4 位小数）；`n_dev=1135 ≥600`、`n_test=1417 ≥1200`；标定 wall-clock 874s≈14.6min ≤25min；bench 566.1s（strict/balanced max_abs_diff=0.0；lenient 0.0032 已有取整边界根因，见 report.md:161-164）。文件：`aidetect\resources\calibration.json`（13:15:32）、`eval\report.md`（13:17:36）、`eval\run_meta.json`（13:17:36）。 | 否 |
| 5 | 报告 HTML 的 XSS 与边界 | **PASS** | XSS：把 `<script>alert('xss')</script>` / `<img src=x onerror=alert(1)>` / `"><svg/onload=alert(2)>` / `<iframe src="http://evil.example">` 注入文档正文后跑**真实模型** report → rc=0；产物中 payload 全部转为 `&lt;script&gt;…` 等转义文本，raw `<img`/`<iframe`=0、`onerror=`/`http://evil.example` 仅出现在转义文本内；raw `<script`/`<svg` 各 1 个是报告自身内联 JS/图表（非 payload）。边界（临时目录文件，未污染 out/）：空文件/纯空白/BOM-only/零宽字符/只有参考文献 → 均 rc=**2**（零宽字符为「无法解析文档：切分后没有任何段落」，不再 rc=4）；纯英文 1650 字 → rc=0，units `[682,679,199]`（均 ≤800），ai_rate=**1.0**（仅域外 FPR 警告，无非中文护栏=R-11，见 §6.4）；单句 9 字 → rc=0，ai_rate 0.9992 +「正文仅 9 字，低于 300 字可靠性下限」；全角空格混排 → rc=0，units `[29,480]`，ai_rate 0.9418；超长无标点 1440 字 → `[800,640]`、英文 1920 字 → `[800,800,319]`。终局产物 HTML（mixed）30,529B：`http://`/`https://`/`<link`/`src=`/`@import`/`url(`/`<img`/`<iframe`/`<script src=` 全 0，含 1 个内联 `<script>`，`onerror` 0。 | 否 |
| 6 | CLI 退出码 0/2/3/4 与实现/README 一致 | **PASS** | 0：真实 report 成功（e2e 两轮、mixed/essay 均 0）、`--help`/`--version` rc=0；2：空文件「读取失败：文件没有可读文本」、仅参考文献「文档正文为空…」、不存在文件、未知资源 `python -m aidetect.fetch bogus`（stderr 未知资源+usage）、参数错误；3：`AIDETECT_HOME` 空目录 → report「缺少模型：clf, ppl。先运行 python -m aidetect fetch clf ppl」、doctor rc=1；4：打桩让 `aggregate.analyze` 抛 RuntimeError →「内部错误：RuntimeError: …（设 AIDETECT_DEBUG=1 看堆栈）」，`AIDETECT_DEBUG=1` 时重抛；130：打桩抛 KeyboardInterrupt → rc=130。README 退出码表 0/1/2/3/4/130 与实现一致。`bench --help` rc=0 且含 `--limit`；真实成功路径退出码 0 已验证。 | 否 |
| 7 | 运行期零网络（fetch.py 除外） | **PASS** | 静态：`aidetect\*.py`+`aidetect\engines\*.py` 扫描 `urllib|requests|http.client|socket|httpx`，除 `fetch.py` 外 **0 命中**；`engines\base.py:78/83` 一律 `from_pretrained(..., local_files_only=True)`。动态：把 `socket.socket`/`socket.create_connection`/`socket.getaddrinfo` 替换为抛 OSError 后跑**真实模型** `report` → rc=0 且 JSON 正常落盘。未执行任何 `fetch` 下载。 | 否 |
| 8 | fetch.py：`python -m aidetect.fetch --help` 与 is_ready | **PASS** | `python -m aidetect.fetch --help` → rc=**0**，首行 `用法: python -m aidetect.fetch [资源 ...] [--workers N] [--chunk-mb N] [--retries N]`、资源/默认/续传说明；`-h` rc=0；bogus rc=2（stderr「未知资源: bogus」+usage，无误导的「分片已保留」）；`from aidetect.fetch import is_ready` → `{'clf': True, 'ppl': True, 'hc3zh': True}`。R-06 字面验收满足。 | 否 |

**组内说明**：lead/engineer 汇报的「115 passed / 1 skipped」是把两个口径混在一起了——默认跑=114 passed/1 skipped（真实模型用例 skip），`AIDETECT_E2E=1` 跑=115 passed/0 skipped；两者 rc 均 0（记为 R-26，汇报口径问题，非缺陷）。

### 6.2 端到端 fixture 审计（lead 要求「git-less 检查」）

- 命令：`python -B %TEMP%\aidetect_review\check_fixture.py` → **32/32 PASS**：`dataset_sha256` 重算与 `tests\fixtures\mixed_hc3.meta.json` 一致（`425e7071…`，对应本地 `data\hc3zh\all.jsonl`）；按 meta 的 8 组 `line_no/field/answer_index` 从 `all.jsonl` 重建 8 段，与 `mixed_hc3.txt` 逐段逐字一致；`order=ai|human` 交替、8 段标签与 `field` 对应；fixture 全文无 `http(s)://`、`www.`、`<script`、`<img`、`<iframe`、`onerror=`、`javascript:`、`@import`、可执行后缀；meta 自带 `dataset/fixture/total_chars/selection/sha256/行号` 可审计。
- 可重建性：`tools/e2e.py --build-mixed-fixture` 生成内容与仓库 fixture **字节级一致**（我的 `check_fixture_build.py` 在临时目录重建后比对通过；`tests\test_e2e.py::test_mixed_fixture_is_reproducible` 亦覆盖）。
- 口径小瑕（R-25，非阻塞）：`meta.size_bytes=6603` 是 LF 归一的正文 UTF-8 字节数；Windows 上 `write_text` 落盘为 CRLF，实际文件 6618B（差 15B=换行）。`total_chars=2319`（去空白）与实际一致。复现性不受影响，建议字段改名或注明。

### 6.3 工程验收结论（逐项 PASS/FAIL + 证据）

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 单测全绿 | **PASS** | 清单 1：114 passed/1 skipped（默认）、115 passed/0 skipped（AIDETECT_E2E=1），rc=0 |
| doctor rc=0 | **PASS** | 清单 2 |
| e2e 耗时与产物 | **PASS** | 清单 3：39/39 必需断言，18.19/21.26s（<60s），repeat 一致，降级 rc3 |
| T2 数字一致/三阈值单调 | **PASS** | 清单 4：156/0 + 负向 11/11 + 独立 127/0 |
| HTML XSS/边界 | **PASS** | 清单 5：真实产物转义通过；边界矩阵全符合预期 |
| CLI 退出码一致 | **PASS** | 清单 6：0/2/3/4 + 130 全实测 |
| 运行期零网络 | **PASS** | 清单 7：静态 0 命中 + 断 socket 真实 report rc=0 |
| fetch --help/is_ready | **PASS** | 清单 8：rc=0 + 三资源 ready |
| 交付门槛（不可越过） | **PASS（工程）** | 代码、测试、报告、CLI、离线承诺均满足；无阻塞缺陷 |

**结论：工程验收全部通过，blocked: none。**

### 6.4 方法学局限（交付必须随附披露，非工程缺陷）

1. **域外 FPR 未达目标**（`eval/report.md:17-18,118-120,123`；`README.md:366,382,398,483`）：目标 1%/5%/10%（strict/balanced/lenient），域外 test 实测 **6.55%/13.09%/19.80%**，三档均超「目标 ±(0.02,0.03)」区间。协议为「dev（财经/法律/医学）按目标 FPR 取阈 → 域外 test 只测不改」，**未用 test 调阈值/选参**。其它指标正常：AUC 0.9930、Acc 0.9435（balanced）、Brier 0.0168、各域 AUC 最低 0.979（nlpcc_dbqa，该域 FPR 37.5%）。
2. **阈值迁移**是本版主要误差源：dev 人类文本与域外人类文本分布不同，AUC 高说明区分度正常，但阈值位置迁移（report.md:107-110、§七 诊断；消融/C 网格/域分离见 report.md:112-164）。交付物已写明「重要场合用 strict / 用自有语料重标定 / 不可作处分唯一证据」。
3. **人类长文单篇仍会被标记**：`tests\fixtures\essay_3000.txt`（手写人类长文）在 balanced/strict 下 ai_rate_chars 均 **0.1247**（1/8 段 high；段落 p_ai 两极：0.9956 vs ≤0.0011，阈值 [0.0042,0.0618] 落在空档，故两模式同值）。e2e 已改为 record-only，报告注明成因；这提示 essay 类边界文本的误报风险，交付说明应保留。
4. **混合真实 HC3 文本可分**：`mixed_hc3`（4 真实 chatgpt 回答 + 4 真实 human 回答，2319 字）AI 段 p_ai 中位数 **0.9999**、人类段 **0.0013**（人类段最大值 0.0052），被判 AI 的 4 段恰为 AI 段——与域外 FPR 结论不矛盾：HC3 式文本接近训练分布，真实自写文本更远。
5. **无「非中文/纯英文」护栏**（R-11，产品限制）：纯英文 1650 字实跑 rc=0、ai_rate=1.0，仅给域外 FPR 警告。**非中文输入结果不可用**，README 已声明适用中文但未硬拦截；建议后续加「拉丁字符占比过高 → 结果不可靠」护栏。
6. e2e 自包含断言为子串级（R-23），本版 mixed fixture 不含 URL，未影响结论；普通文档正文含 URL 文本会被误判为外链（实测转义纯文本，非注入）。

### 6.5 终局裁定

- **可交付：是（blocked: none）**。task-6 清单 1–8 全部 PASS，工程验收无阻塞项；R-14/R-19/R-24/R-20/R-21 已关闭并复测。
- **交付条件（必须）**：随交付说明披露 §6.4 全部 6 条，尤其是「域外 FPR 6.55%/13.09%/19.80% vs 1%/5%/10%」与「不可作处分唯一证据」。产品文档目前已经写入（report.md §一/§七、README §5.5/§八/§十），无需返工。
- **不计入阻塞的低危残余**（详见 §6.6）：R-11 无非中文护栏、R-16 二进制 mojibake、R-18 独立 `[1]` 条目不进 excluded、R-23 e2e 子串断言、R-25 meta.size_bytes 口径、R-26 pytest 计数口径、R-27 README pytest 命令 -qq 吞汇总。均不影响验收结论。

### 6.6 残余问题清单（非阻塞，建议后续处理）

| 编号 | 问题 | 证据 | 建议 |
| --- | --- | --- | --- |
| R-11 | 无非中文/纯英文护栏 | 纯英文 1650 字真实 report：ai_rate=1.0，仅域外 FPR 警告 | 后续加字符集占比护栏；README 已声明适用中文，可接受 |
| R-16 | 二进制文件按文本解码成 mojibake 后计分 | `binary.bin` → 正文 'Ā＂þ'（3 字），未报「不是文本」 | 按 NUL/控制字符比例拒绝，报输入错误 2 |
| R-18 | 独立 `[1] 张三…` 行标 meta 但 excluded_chars=0 | 预复核实测；与 README §4.2「计入已剔除字数」口径有偏差 | 文档注明或累计 excluded |
| R-23 | e2e 自包含断言为子串级 | 正文含 `http://` 文本会误判为外链（实为转义纯文本） | 改检查 `href=`/`src=`/`<link` 等真实引用 |
| R-25 | meta.size_bytes 为 LF 归一字节，磁盘 CRLF 差 15B | 6603 vs 6618B；重建字节级一致 | 字段改名/注明，或写入磁盘字节数 |
| R-26 | pytest 计数口径混用 | 默认 114+1skip；AIDETECT_E2E=1 时 115+0skip；汇报写成「115/1」 | 统一口径（建议汇报默认口径） |
| R-27 | README `python -m pytest -q` 与 pyproject `addopts=-q` 叠加成 `-qq` 吞汇总行 | 实际跑无「N passed」行，退出码仍正确 | README 命令改 `-o addopts="" -q` |

### 6.7 复核资产（可复跑）

- **`eval/check_t2.py`（156 项，0 失败）**：核 `calibration.json` schema（version/features/mean/std/coef/intercept/temperature/thresholds/mode_thresholds/evaluation/per_domain/dataset）；`thresholds`/`mode_thresholds`/`temperature` ≤4 位小数；三键齐全且 `strict>balanced>lenient`、`thresholds.high>mid_high>mid`；`eval/report.md` 的评测表 6 行 + 各域 4 行逐项与 JSON 相等（阈值 4dp、目标FPR、n/Acc/AUC/FPR/FNR/Precision/Brier 3dp，容差 0）；权重表 10 行；`dataset.n_dev/n_test`、`--limit 150`、wall-clock 与 `run_meta.json` 一致；复现命令 CLI 与脚本双路径（R-19）；第 9 节 R-24 门槛判定（FPR 区间 `[tgt-0.02,tgt+0.03]`、未全部达标文案、无「一次达标」、含 strict/重标定/唯一证据建议）；第 10 节 §七.1 门槛表 3 行逐项核对；另有「行数计数断言」防止列偏移静默跳过。
- **负向对照 11/11**（`%TEMP%\aidetect_review\negctrl_check_t2.py`）：baseline rc0；篡改阈值尾数/Acc/目标FPR/per_domain 阈值/gate 实测 FPR/「未全部达标」文案/CLI 命令/temperature 6 位小数/mode 倒序/n_dev 各 1 例 → 全部 rc=1 并定位正确；**证明检查器会红**。
- **独立第二路 127/0**（`%TEMP%\aidetect_review\verify_t2_independent.py`）：自写 markdown 解析，不复用 check_t2 逻辑。
- **产品/产物校验 40/40**（`check_final_products.py`）、**fixture 审计 32/32**（`check_fixture.py`）、**真实模型 CLI/XSS/零网络**（`probe_real_cli.py`、`probe_real_stdin.py`）、**第一批复测**（`probe_fix.py`）、**CLI/退出码**（`probe_final_bits.py`）。

### 6.8 增量复核（Lead 修复 R-11/R-16/R-27 后，2026-10-26 21:07）

> 背景：lead 在 13:36 终局签字后于 13:41 修改 3 处（R-11 非中文护栏 `aggregate.py`、R-16 二进制嗅探 `textio.py`、R-27 README 测试命令）并新增 5 条测试与 `LICENSE`/`.gitignore`。按 lead 要求做**增量复核**：只验这三处及关联文档，不重跑全量性能基准（e2e 全量由 lead 复跑 EXIT=0/39）。**R-11 / R-16 / R-27 关闭**；本节结论对 §6.1–§6.7 有效，终局裁定不变。

#### 6.8.1 实现核实（读码）

| 项 | 实现 | 结论 |
| --- | --- | --- |
| R-11 非中文护栏 | `aggregate.py:270-281` `_script_ratio_chinese(doc)`：`cjk`=汉字（U+4E00–U+9FFF）计数、`latin`=ASCII 字母计数、`total=cjk+latin`（≤0 → `(0.0, 0)`）；`aggregate.py:284-292` `_warnings`：`if script_total >= 50 and cjk_ratio < 0.5` 追加「正文中汉字只占 X%（非中文内容为主）…分数不可用，请勿引用。」 | PASS（与 lead 描述一致） |
| R-16 二进制嗅探 | `textio.py:15-46`：采样前 4096B；UTF-16 BOM 白名单；含 NUL→二进制；控制字符 `b<0x09 or 0x0E<=b<0x20` 占比 >0.05→二进制；`read_text_file` 文本/无后缀/未知后缀分支解码前嗅探→`ReadError(BINARY_HINT)`；docx/pdf 专用分支不嗅探 | PASS |
| 新增测试 | `tests/test_textio.py` +3（二进制拒绝/未知后缀二进制/UTF-16 不误判）、`tests/test_aggregate.py` +2（非中文护栏/中文不误报） | PASS（119−114=+5 对得上） |

#### 6.8.2 ① 非中文护栏：触发口径与中文零误伤

命令：`python -B %TEMP%\aidetect_review\probe_incr.py`（假引擎 `conftest.fake_engines`，不加载模型）。逐例（样例 / 汉字占比 / 汉字+拉丁字符数 / 是否触发）：zh_100pct 1.00/135 否；en_0pct 0.00/198 **是**（warning 原文「正文中汉字只占 0%（非中文内容为主）…」）；ratio_60_zh 0.22/603 是；ratio_50_exact 0.50/100 否；ratio_49_zh 0.49/100 是；short_en（19 letters）0/19 否；digits_punct 0/0 否；russian 0/0 否；japanese_kanji_kana 1.00/87 否；korean 0/0 否；arabic 0/0 否；emoji 0/0 否。

- 判定条件落地正确（<50% 且 汉字+拉丁≥50；边界 50% 不触发）；**中文零误伤**：纯中文、`out/mixed_hc3_report.json`、`out/essay_3000_report.json` 均无该护栏（后两者 warnings 只有「域外阈值提醒」+「段间标准差 0.50/0.33」）。
- 真实模型（`probe_real_incr.py`，6.6–6.8s）：纯英文 1650 字 rc=0、`ai_rate_chars=0.9926`、JSON warnings 与 HTML 均出现护栏（`guard_in_html=True`）；纯中文 rc=0、无护栏。
- 残余 **R-28（非阻塞）**：护栏只统计汉字/拉丁，俄/日/韩/阿拉伯/emoji 等 `script_total=0` 或占比 1.0 → 不触发；README §6.2「汉字占比低于 50%…」与 §6.3 护栏清单表述过宽。建议按 Unicode script 统计或收敛 README 口径。

#### 6.8.3 ② 二进制嗅探：误杀检查

文本层（`probe_incr.py`）：GB18030 `.txt`/`.md`、UTF-16 LE/BE（带 BOM）、UTF-8 BOM、md 表格+公式、tab/换行、控制字符 4%、emoji、未知后缀文本、docx（专用分支）全部正常读取；PNG-like、含 NUL、控制字符 6% → `ReadError(BINARY_HINT)`。

CLI 真实模型（`probe_legal.py`，6 个临时文件顺序跑）：

| 文件 | rc | body_chars/units | 结论 |
| --- | --- | --- | --- |
| gb18030.txt | 0 | 399/1 | 未被误杀 |
| utf16le_bom.txt | 0 | 401/1 | 未被误杀 |
| utf16be_bom.txt | 0 | 403/1 | 未被误杀 |
| md_table_formula.md | 0 | 479/3 | 表格+公式保留 |
| mixed_zh_en.txt | 0 | 540/4 | 无护栏误伤 |
| doc.docx | 0 | 435/2 | 正常 |

二进制：`png_like.bin` rc=2、stderr「读取失败：无法解析文档：这看起来是二进制文件，不是文本（支持 .txt / .md / .docx / .pdf）（后缀 .bin）」、未加载模型（0.0x s）；`one_nul.txt` rc=2 同文案。文字类边界：ANSI 彩色 `.log`（`\x1b[31mERROR\x1b[0m`×20）控制字符占比 5.13% → 被判二进制（**R-30**，`.log` 是宣称支持后缀，低危；建议 ESC 序列白名单或提示去色）；BOM-less UTF-16 → 拒绝为二进制（**R-32**，低危行为变化，建议加 BOM 或嗅探 UTF-16 交替 NUL）。**无「合法中文文档被误杀」证据。**

#### 6.8.4 ③ pytest / e2e 复跑

- 默认：`python -m pytest -p no:cacheprovider -o addopts="" -q` → **119 passed, 1 skipped, rc=0**（skip 为真实模型 e2e 用例，默认不加载）。
- `$env:AIDETECT_E2E='1'; python -m pytest ...` → **120 passed, 0 skipped, rc=0**（17.7s / 33.1s 两次，机器负载波动）。
- e2e：13:45 全量 `python tools\e2e.py --repeat 2 --compare-human --json-out out\e2e_result.json` → EXIT=0、必需断言 **39/39 PASS**、耗时 [12.99,13.43]s、repeat 两轮 AI 率 (0.5424,0.5) 一致、doctor 5.7s、降级 rc3、essay 0.1247、HTML 29.6KB；21:05 `--repeat 1 --json-out %TEMP%\aidetect_review\e2e_incr8.json` → EXIT=0、**37/37 PASS**（repeat 1 少 2 条）、mixed 0.5424、essay 0.1247（该轮 44.1s 因与合法文本探针并发争 CPU，不作性能证据；性能结论仍以 §6.1 的 [18.19,21.26]s 与 lead 复跑为准）。
- 产物：`out/mixed_hc3_report.{html,json}` 与 `out/e2e_result.json` 由 e2e 重生成（不污染 `out/` 之外）；当前产物 warnings 均无「非中文」。

#### 6.8.5 ④ README / LICENSE / .gitignore 一致性（无夸大）

| 文档 | 核实内容 | 结论 |
| --- | --- | --- |
| README §6.2（400-412 行） | 二进制「读取阶段报错（退出码 2）」与实测一致；非中文段「只提示、不硬性拒绝」与实现一致；触发条件描述不完整见 R-28 | PASS（附 R-28） |
| README §6.3（414-420 行） | 护栏清单含「非中文为主」 | PASS（覆盖范围同 R-28） |
| README §十（507-526 行） | `python -m pytest -o addopts="" -q` 口径正确（不加会被 `addopts` 叠加成 `-qq` 吞汇总，实测过）；`AIDETECT_E2E=1` 门控真实存在（`tests\test_e2e.py:430`）且 120 passed；「约 20 秒」实测 17.7s 量级一致 | PASS |
| README §十一（532-539 行） | LICENSE 链接存在；MIT 全文正确；`pyproject.toml` `license = { text = "MIT" }` 同步；「模型/语料不随仓库分发」与 `.gitignore /models/ /data/` 一致；第三方声明与 `models/*/.provenance.json` 的 repo 名一致 | PASS（附 R-31） |
| LICENSE（1614B） | MIT 全文 + 第三方许可声明；clf「Apache-2.0（以模型卡片为准）」——离线无法核验 HF 卡片 SPDX（R-31，低危，保留对冲即可） | PASS |
| .gitignore（217B） | `/models/ /data/ /out/ *.log __pycache__ *.py[cod] *.egg-info/ build/ dist/ .pytest_cache/` 正确；README 未提及 .gitignore（**R-29**，信息级） | PASS |

#### 6.8.6 ⑤ 增量终局裁定

| 项 | 结论 | 阻塞 |
| --- | --- | --- |
| ① 非中文护栏口径 + 中文零误伤 | PASS（附 R-28 覆盖缺口/文案） | 否 |
| ② 二进制误杀检查 | PASS（附 R-30 / R-32 低危边界） | 否 |
| ③ pytest / e2e | PASS（119+1skip / 120；e2e EXIT=0 37–39/39） | 否 |
| ④ README/LICENSE/.gitignore | PASS（附 R-29 / R-31 信息级） | 否 |
| ⑤ 终局裁定 | **仍可交付，blocked: none**；§6.4 方法学局限不变（FPR 6.55/13.09/19.80 vs 1/5/10；essay 类人类长文 0.1247；无非拉丁文字护栏） | 否 |

新增残余（均非阻塞）：R-28 非拉丁文字不触发护栏（README 表述过宽）；R-29 README 未提 .gitignore（信息级）；R-30 ANSI 彩色 .log 可能被误判二进制（低）；R-31 第三方许可 SPDX 无法离线核验（低，保留对冲）；R-32 BOM-less UTF-16 被拒绝（低，行为变化）。**R-11 / R-16 / R-27 关闭。**

### 6.9 微增量复核（R-30 ANSI 修复 + R-28 文案收敛，2026-10-26 21:17）

> 背景：§6.8 之后 lead 针对 R-30/R-28 做了最后一处小改（`textio.py` ANSI 清洗 + `README` §6.2/§6.3 口径收敛），并修复「全 ANSI 序列文件被判为非二进制→走通并产出无意义报告」的缺陷。按要求只验 4 点，不重跑全量基准。**§6.8.6 残余中的 R-28（README 文案）与 R-30（ANSI 彩色日志）关闭**；终局裁定不变。

#### 6.9.1 实现核实（读码）

- `aidetect\textio.py:19-20` 新增 `_ANSI_ESCAPE = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]")`；`_looks_binary`（29-42 行）在 NUL/UTF-16 BOM 检查之后、控制字符统计之前执行 `sample = _ANSI_ESCAPE.sub(b"", sample)`，替换后为空 → `False`（纯 ANSI 不判二进制）；原无用 `import io` 已删（现 `import re`/`import sys` 均被使用）。
- `textio.py:54-70`：`_ANSI_TEXT`/`_CTRL_TEXT`/`_clean_text`；`_read_plain_text(path, hint)` = `_looks_binary` → `_decode` → `_clean_text`，清洗后 `not text.strip()` 抛 `ReadError("文件没有可读文本（只含控制字符或 ANSI 转义序列）")`；`read_text_file` 的 txt/无后缀/未知后缀分支统一走它（未知后缀保留「（后缀 .bin）」提示）。
- `textio.py:133-140`：`read_stdin()` 同样 `_clean_text` + 空则报「stdin 没有可读文本（只含控制字符或 ANSI 转义序列）」。
- `tests\test_textio.py`：新增 `test_ansi_colored_log_is_still_text`（92-97 行）、`test_pure_ansi_sequences_are_rejected_as_no_text`（100-104 行）。
- README §6.2（410-411 行）：精确口径「汉字字符数 ÷（汉字 + 拉丁字母数），低于 50% 且计数 ≥ 50」+「口径局限」段（纯俄文/日文假名/韩文/阿拉伯文/emoji 不触发、分数同样不可用）；§6.3（421 行）「非中文为主——判定口径为汉字/拉丁字母占比」。

#### 6.9.2 ① 纯 ANSI / 彩色日志 / 编码边界（真实 CLI）

命令：`python -B %TEMP%\aidetect_review\probe_incr9.py`（同一脚本内对每个文件独立 out 目录跑真实 `python -m aidetect report`）。

| 用例 | 读取层 | CLI rc | 正文/单元 | 备注 |
| --- | --- | --- | --- | --- |
| 彩色 .log（`\x1b[31mERROR\x1b[0m …`×60） | ok；ESC 已清洗（原 2760 字符 → 2220，`has_esc=False`） | **0** | body_chars 1920 / units 4 | ai_rate 1.0；英文日志触发非中文护栏 + 域外提醒（2 warnings），属预期 |
| 中文彩色日志（`\x1b[32m信息\x1b[0m …`×20） | ok；ESC 已清洗 | 0 | 320 / 1 | 正常出报告 |
| 纯 ANSI（`\x1b[31m`×200，.txt） | `ReadError`「文件没有可读文本（只含控制字符或 ANSI 转义序列）」 | **2** | — | stderr「读取失败：文件没有可读文本（只含控制字符或 ANSI 转义序列）」；**无垃圾报告、不崩栈、不误计分**（此前缺陷已闭环） |
| 纯控制字符（`\x01\x02\x03\x04`×100） | `ReadError`「二进制」 | — | — | 仍按二进制拒绝（合理） |
| GB18030 .txt | ok，43 字符 | 0 | 39 / 2 | 未误杀 |
| UTF-16 LE 带 BOM .txt | ok，18 字符 | 0 | 18 / 1 | 未误杀 |
| docx（专用分支） | ok，31 字符 | 0 | 27 / 1 | 未误杀 |
| PNG-like / 含 NUL .txt | `ReadError`「二进制」 | 2 | — | 二进制仍拒绝 |
| **stdin** 纯 ANSI | — | **2** | — | stderr「stdin 没有可读文本（只含控制字符或 ANSI 转义序列）」 |
| **stdin** 彩色日志 | — | 0 | 1920 | 正常（R-15 修复无回归） |

补充（诚实记录）：我在 21:12 的第一版探针（lead 修复前）确实复现了该缺陷——纯 ANSI 文件未被判二进制、`report` 以 rc=0 走通并产出报告；lead 于 21:14:52 修复后，本次 21:16 复测为 rc=2，缺陷已闭环。

#### 6.9.3 ② 清洗未破坏真实报告口径

- `python -m aidetect report tests\fixtures\mixed_hc3.txt --out-dir <临时目录>` → rc=0、body_chars **2319**、units 8、**ai_rate_chars 0.5424**（与 §6.1/§6.6 完全一致）。
- ANSI 日志清洗后按可见字符计 1920（60 行 × 32 字符），说明转义序列不再进入正文计分；修复前旧实现会把 ESC 计入（2460）——本次修复同时消除了「脏正文」。

#### 6.9.4 ③ pytest（含真实模型）

- 默认 `python -m pytest -p no:cacheprovider -o addopts="" -q` → **121 passed, 1 skipped, rc=0**。
- `$env:AIDETECT_E2E='1'` 同命令 → **122 passed, 0 skipped, rc=0**（31.2s，真实模型）。
- 回归测试：`test_ansi_colored_log_is_still_text`、`test_pure_ansi_sequences_are_rejected_as_no_text` 均通过；无既有用例回归。

#### 6.9.5 ④ README §6.2/§6.3 与实现一致性

- §6.2（410 行）公式「汉字字符数 ÷（汉字 + 拉丁字母数），低于 50% 且计数 ≥ 50」与 `_script_ratio_chinese`/`_warnings` 逐字对应；411 行「口径局限」列出的俄/日假名/韩/阿/emoji 与 §6.8.2 探测矩阵逐项吻合 → **R-28 文案过宽问题关闭**。
- §6.3（421 行）「非中文为主——判定口径为汉字/拉丁字母占比」准确。
- 二进制段（413 行）「读取阶段直接报错（退出码 2）」与本次实测一致；README 未对 ANSI/彩色日志做额外承诺 → 无新夸大、无遗漏承诺。
- 唯一可选的补充说明：README 未提及「纯 ANSI/控制字符文件会报『没有可读文本』」这一细节——属实现改进的补充说明，非承诺偏差，无需改。

#### 6.9.6 ⑤ 微增量裁定

| 项 | 结论 | 阻塞 |
| --- | --- | --- |
| ① 纯 ANSI→rc2、彩色日志→rc0、GB18030/UTF-16 带 BOM→rc0 | PASS | 否 |
| ② 清洗未破坏真实报告（mixed：2319 字 / 8 单元 / 0.5424） | PASS | 否 |
| ③ pytest 121+1skip / 122+0skip | PASS | 否 |
| ④ README §6.2/§6.3 与实现一致、无新夸大 | PASS | 否 |
| ⑤ 终局裁定 | **仍可交付，blocked: none**；R-28（文案）与 R-30（ANSI）关闭 | 否 |

§6.8.6 残余更新：**R-28 关闭**（README 已收敛）、**R-30 关闭**（ANSI 清洗 + 纯 ANSI 空文本报错）；仍余 R-29（README 未提 .gitignore，信息级）、R-31（第三方 SPDX 无法离线核验，低）、R-32（BOM-less UTF-16 被拒，低）——均非阻塞。

## 附录 A：复现命令

```powershell
cd ai-detect
$env:PYTHONIOENCODING='utf-8'
# 单测（默认：真实模型 report 用例 skip；注意 pyproject addopts=-q 与命令行 -q 叠加成 -qq 会吞汇总行）
python -m pytest -p no:cacheprovider -o addopts="" -q            # 期望 121 passed, 1 skipped, rc=0
$env:AIDETECT_E2E='1'
python -m pytest -p no:cacheprovider -o addopts="" -q            # 期望 122 passed, 0 skipped, rc=0（真实模型）
Remove-Item Env:AIDETECT_E2E
# 验收：doctor / e2e
python -m aidetect doctor                                        # 期望 rc=0，末行「自检通过：可以运行检测。」
python tools\e2e.py --repeat 2 --compare-human --json-out out\e2e_result.json   # 期望 EXIT=0，39 必需断言 PASS
# T2 数字一致性检查器（复核资产，156 项）
python -B eval\check_t2.py                                       # 期望 156 项通过 / 0 失败
# 终局一次性探针（均在 %TEMP%\aidetect_review\，不污染仓库）
python -B %TEMP%\aidetect_review\check_fixture.py                # mixed fixture 可审计性 32/32
python -B %TEMP%\aidetect_review\check_final_products.py         # e2e 结果/HTML 自包含/JSON 数值 40/40
python -B %TEMP%\aidetect_review\negctrl_check_t2.py             # 负向对照 11/11（期望 rc=1，证明检查器会红）
python -B %TEMP%\aidetect_review\verify_t2_independent.py        # 独立第二路 127/0
python -B %TEMP%\aidetect_review\probe_fix.py                    # 第一批复测（R-01…R-21/XSS/零网络/CLI）
python -B %TEMP%\aidetect_review\probe_real_cli.py               # 真实模型 CLI/XSS/零网络/内部错误
python -B %TEMP%\aidetect_review\check_fixture_build.py          # 重建 mixed fixture 字节级比对
python -B %TEMP%\aidetect_review\probe_incr.py                   # §6.8 护栏矩阵 + 二进制嗅探（假引擎，不加载模型）
python -B %TEMP%\aidetect_review\probe_real_incr.py              # §6.8 真实模型：纯英文护栏 / 中文零误伤
python -B %TEMP%\aidetect_review\probe_legal.py                  # §6.8 合法文本不被误判二进制（GB18030/UTF-16/md/docx）
python -B %TEMP%\aidetect_review\probe_incr9.py                  # §6.9 ANSI 清洗修复 + 纯 ANSI→rc2 / 彩色日志→rc0 / 编码/docx/mixed 回归（真实 CLI）
python tools\e2e.py --repeat 1 --json-out "$env:TEMP\aidetect_review\e2e_incr8.json"  # §6.8 增量复跑，期望 EXIT=0、37 必需断言 PASS
# 边界文件（临时目录）
python -m aidetect report "$env:TEMP\aidetect_review\files\zerowidth.txt" --out-dir "$env:TEMP\aidetect_review\out_rc"  # 期望 rc=2
python -m aidetect.fetch --help                                  # 期望 rc=0 且打印 usage
python -m aidetect.fetch bogus                                   # 期望 rc=2，stderr「未知资源: bogus」+ usage
```

## 附录 B：按 owner 的建议清单（供 lead 派单）

| 归属 | 处理项（终局） |
| --- | --- |
| 产品代码（lead 已修，复核确认） | R-01~R-10、R-12、R-14、R-15、R-19、R-20、R-21、R-24 均已关闭并复测（§二、§五、§六）；R-11/R-16/R-27 由 lead 于 13:41 修复，§6.8 复审关闭；无待办阻塞 |
| engineer-smoke（engines/tools） | R-07 已修并复测；`textio.py:7 import io` 已由 lead 一并删除；无待办 |
| engineer-calibrate（T2） | R-14/R-19/R-24 已修并复测；`calibration.json`/`eval/report.md` 数字一致、可复现；无需再动（若重跑标定，注意 check_t2 会强制 ≤4 位小数与逐项一致） |
| engineer-e2e（T4） | T4 已收口（39 断言 PASS）；后续可选：自包含断言从子串改属性级（R-23）、meta.size_bytes 注明 LF 归一口径（R-25） |
| lead / 后续版本（§6.9 后） | R-29 README 未提 .gitignore（信息级）；R-31 第三方 SPDX 无法离线核验（低）；R-32 BOM-less UTF-16（低）——R-28/R-30 已于 §6.9 关闭 | 均非阻塞 |
| engineer-docs（README） | README 数字与 `calibration.json`/`eval/report.md` 一致（check_t2 覆盖）；R-27（pytest 命令）已修；建议统一「121+1skip / 122+0skip」口径（R-26）、补 .gitignore 说明（R-29） |
| **阻塞项** | **无**（工程验收全部 PASS；仅方法学局限 §6.4 需随交付披露，产品文档已写入） |
