"""命令行入口：``python -m aidetect <子命令>``。

子命令：
    report    检测一篇文档，产出 HTML + JSON 报告
    json      只要 JSON（便于脚本化/批量）
    doctor    自检环境、模型、标定状态
    fetch     下载模型/评测语料（唯一会联网的命令）
    calibrate 用 HC3-Chinese 拟合融合参数并评测
    bench     用现成标定重新评测
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import __version__, aggregate, config, fetch, report, textio
from .segment import build_doc

EXIT_OK, EXIT_INPUT, EXIT_RESOURCE, EXIT_INTERNAL = 0, 2, 3, 4


# --------------------------------------------------------------------------- 小工具
def _banner(text: str) -> None:
    print(f"\n== {text}", flush=True)


def _models_ready() -> tuple[bool, list[str]]:
    missing = [name for name in ("clf", "ppl") if not fetch.is_ready(name)]
    return (not missing), missing


def _load_doc(source: str):
    text, name = textio.load(source)
    return build_doc(text, name)


def _print_summary(analysis, html_path: Path | None, json_path: Path | None) -> None:
    st = analysis.stats
    rate = analysis.ai_rate_chars
    ci = analysis.ai_rate_ci
    ci_txt = f"{ci[0]*100:.1f}%–{ci[1]*100:.1f}%" if ci else "样本不足"
    print(f"\n文档：{st.get('name')}  正文 {st.get('body_chars')} 字 / {st.get('units')} 段 "
          f"（剔除 {st.get('excluded_chars', 0)} 字）")
    for e in analysis.engine_info:
        state = "✓" if e["available"] else "✗"
        print(f"引擎 {state} {e['label']}" + (f"  {e['note']}" if e["note"] else ""))
    print(f"\nAI 生成占比（字数口径）：{rate*100:.1f}%   95% CI {ci_txt}")
    judged = analysis.grade_counts.get("high", 0) + analysis.grade_counts.get("mid_high", 0)
    print(f"AI 段落占比：{analysis.ai_rate_paras*100:.0f}%（{judged}/{st.get('units')} 段判为 AI）"
          f"   模式 {analysis.mode} / 阈值 {analysis.threshold:.2f}")
    if analysis.warnings:
        print("\n护栏提示：")
        for w in analysis.warnings:
            print(f"  ! {w}")
    if html_path:
        print(f"\nHTML 报告：{html_path}")
    if json_path:
        print(f"JSON 结果：{json_path}")


_BAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_stem(doc_name: str) -> str:
    """把展示名变成安全的文件名主干（stdin 的 '<stdin>'、路径、非法字符都要处理）。"""
    name = Path(doc_name).stem or "document"
    name = _BAD_NAME_CHARS.sub("_", name).strip(" ._")
    return name or "document"


def _resolve_out(out_dir: str | None, doc_name: str, suffix: str) -> Path:
    base = Path(out_dir) if out_dir else config.OUT_DIR
    return base / f"{_safe_stem(doc_name)}_{suffix}"


# --------------------------------------------------------------------------- 子命令
def cmd_report(args: argparse.Namespace) -> int:
    ready, missing = _models_ready()
    if not ready:
        print(f"缺少模型：{', '.join(missing)}。先运行 python -m aidetect fetch {' '.join(missing)}",
              file=sys.stderr)
        return EXIT_RESOURCE
    try:
        doc = _load_doc(args.file)
    except textio.ReadError as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return EXIT_INPUT
    except ValueError as exc:      # 例：去空白/零宽字符后没有可判定段落
        print(f"无法解析文档：{exc}", file=sys.stderr)
        return EXIT_INPUT
    if doc.body_chars == 0:
        print("文档正文为空（或全部被判为参考文献/元信息），无法检测。", file=sys.stderr)
        return EXIT_INPUT

    _banner(f"检测 {doc.name}（正文 {doc.body_chars} 字）")
    t0 = time.time()
    analysis = aggregate.analyze(doc, mode=args.mode, max_sentences=args.max_sentences)
    html_path = None
    if not args.no_html:
        html_path = _resolve_out(args.out_dir, doc.name, "report.html")
        report.write(analysis, html_path)
    json_path = None
    if not args.no_json:
        json_path = _resolve_out(args.out_dir, doc.name, "report.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2),
                             encoding="utf-8")
    _print_summary(analysis, html_path, json_path)
    print(f"\n总耗时 {time.time()-t0:.1f}s")
    if args.open and html_path is not None:
        try:
            import os
            os.startfile(str(html_path))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    return EXIT_OK


def cmd_json(args: argparse.Namespace) -> int:
    ready, missing = _models_ready()
    if not ready:
        print(f"缺少模型：{', '.join(missing)}", file=sys.stderr)
        return EXIT_RESOURCE
    try:
        doc = _load_doc(args.file)
    except textio.ReadError as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return EXIT_INPUT
    except ValueError as exc:
        print(f"无法解析文档：{exc}", file=sys.stderr)
        return EXIT_INPUT
    if doc.body_chars == 0:
        print("文档正文为空（或全部被判为参考文献/元信息），无法检测。", file=sys.stderr)
        return EXIT_INPUT
    analysis = aggregate.analyze(doc, mode=args.mode, max_sentences=args.max_sentences)
    payload = json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"JSON 结果：{out}")
    else:
        print(payload)
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import doctor

    return doctor.run()


def cmd_fetch(args: argparse.Namespace) -> int:
    names = args.names or ["clf", "ppl"]
    if names == ["all"]:
        names = ["clf", "ppl", "hc3zh"]
    return fetch.main([*names, "--workers", str(args.workers),
                       "--chunk-mb", str(args.chunk_mb), "--retries", str(args.retries)])


def cmd_calibrate(args: argparse.Namespace) -> int:
    from . import calibrate

    if not all(fetch.is_ready(n) for n in ("clf", "ppl", "hc3zh")):
        print("标定需要 clf/ppl/hc3zh 三类资源。先运行 python -m aidetect fetch all",
              file=sys.stderr)
        return EXIT_RESOURCE
    calib = calibrate.run(limit=args.limit)
    test = (calib.get("evaluation", {}).get(config.DEFAULT_MODE) or {}).get("test") or {}
    print(f"\n域外测试集（{config.DEFAULT_MODE}）：Acc {test.get('acc')} / AUC {test.get('auc')} / "
          f"FPR {test.get('fpr')} / FNR {test.get('fnr')}")
    return EXIT_OK


def cmd_bench(args: argparse.Namespace) -> int:
    from . import calibrate

    if not (config.RESOURCES_DIR / "calibration.json").exists():
        print("还没有 calibration.json，先运行 python -m aidetect calibrate", file=sys.stderr)
        return EXIT_RESOURCE
    calib = calibrate.bench(limit=getattr(args, "limit", None))
    print(json.dumps(calib.get("evaluation"), ensure_ascii=False, indent=2))
    return EXIT_OK


def cmd_launch(args: argparse.Namespace) -> int:
    from . import launch

    return launch.main(list(getattr(args, "files", []) or []))


# --------------------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aidetect",
        description="离线中文「AI 生成率」检测：判别器 + 困惑度 + 文体统计三引擎融合，"
                    "输出段落级证据与单文件 HTML 报告。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python -m aidetect doctor
  python -m aidetect fetch all
  python -m aidetect report 论文.docx --open
  python -m aidetect report 论文.docx --mode strict
  type 草稿.txt | python -m aidetect json -
""",
    )
    p.add_argument("--version", action="version", version=f"ai-detect {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="检测文档并生成报告")
    r.add_argument("file", help="txt/md/docx/pdf 路径，或 - 读 stdin")
    r.add_argument("--mode", default=config.DEFAULT_MODE,
                   choices=list(config.MODE_TARGET_FPR), help="严格程度（默认 balanced）")
    r.add_argument("--out-dir", default=None, help="输出目录（默认 out/）")
    r.add_argument("--max-sentences", type=int, default=20, help="最多列出的可疑句数")
    r.add_argument("--no-html", action="store_true", help="不生成 HTML")
    r.add_argument("--no-json", action="store_true", help="不生成 JSON")
    r.add_argument("--open", action="store_true", help="生成后用默认浏览器打开报告")
    r.set_defaults(func=cmd_report)

    j = sub.add_parser("json", help="只输出 JSON")
    j.add_argument("file", help="文件路径，或 - 读 stdin")
    j.add_argument("--mode", default=config.DEFAULT_MODE, choices=list(config.MODE_TARGET_FPR))
    j.add_argument("--max-sentences", type=int, default=20)
    j.add_argument("--out", default=None, help="写入文件而不是打印")
    j.set_defaults(func=cmd_json)

    d = sub.add_parser("doctor", help="自检环境/模型/标定")
    d.set_defaults(func=cmd_doctor)

    f = sub.add_parser("fetch", help="下载模型与评测语料（唯一联网命令）")
    f.add_argument("names", nargs="*", help="clf / ppl / hc3zh / all")
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--chunk-mb", type=int, default=16)
    f.add_argument("--retries", type=int, default=200)
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("calibrate", help="标定融合参数并评测")
    c.add_argument("--limit", type=int, default=None, help="每类每域最多取多少条（快速调试）")
    c.set_defaults(func=cmd_calibrate)

    b = sub.add_parser("bench", help="用现成标定重新评测")
    b.add_argument("--limit", type=int, default=None,
                   help="与 calibrate 相同的抽样上限（复现同一测试子集；默认全量，很慢）")
    b.set_defaults(func=cmd_bench)

    la = sub.add_parser("launch", help="交互式启动器：拖文件进来即可检测（双击 .bat 用的就是它）")
    la.add_argument("files", nargs="*", help="要检测的文件或文件夹（留空进入交互窗口）")
    la.set_defaults(func=cmd_launch)
    return p


def _tolerant_console() -> None:
    """输出编码装不下某个字符（Windows 传统 GBK 控制台 / 重定向到文件）时用 ? 代替，而不是抛异常。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 个别流不支持
            continue


def main(argv: list[str] | None = None) -> int:
    _tolerant_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("AIDETECT_DEBUG"):
            raise
        print(f"内部错误：{type(exc).__name__}: {exc}\n（设 AIDETECT_DEBUG=1 看堆栈）",
              file=sys.stderr)
        return EXIT_INTERNAL


import os  # noqa: E402  (main 里用到，放这里避免顶部 import 顺序噪音)

if __name__ == "__main__":
    raise SystemExit(main())
