"""双击 / 拖拽启动器：不记命令也能用。

两种进入方式：

    python -m aidetect.launch                  # 交互窗口：把文件拖进来、回车
    python -m aidetect.launch 论文.docx 摘要.pdf  # 直接检测（也接受文件夹）

模型在一次会话里只加载一次，所以连续检测多篇不会重复等加载。
生成完报告后会自动用默认浏览器打开（``AIDETECT_NO_OPEN=1`` 可关掉）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from . import aggregate, config, fetch, report, textio
from .cli import (
    EXIT_INPUT,
    EXIT_OK,
    EXIT_RESOURCE,
    _banner,
    _print_summary,
    _resolve_out,
    _tolerant_console,
)
from .segment import build_doc

DOC_SUFFIXES = {".txt", ".md", ".markdown", ".text", ".docx", ".pdf", ".log", ".csv"}
MODE_ALIASES = {
    "1": "balanced", "balanced": "balanced", "默认": "balanced", "均衡": "balanced",
    "2": "strict", "strict": "strict", "严格": "strict",
    "3": "lenient", "lenient": "lenient", "宽松": "lenient",
}
MODE_MENU = ("1=balanced（默认，均衡） 2=strict（严格，少误判人类） "
             "3=lenient（宽松，更容易判为 AI）")
QUIT_WORDS = {"q", "quit", "exit", ":q", "退出"}
HELP_WORDS = {"h", "help", "?", "菜单"}


class Command:
    """把输入的一行解析成 quit / help / mode / files。"""

    def __init__(self, kind: str, value: str | None = None) -> None:
        self.kind = kind
        self.value = value


def parse_input(line: str | None) -> Command:
    """解析交互输入：空行/exit 退出，1~3 切模式，其余当作路径（容忍拖拽带来的引号）。"""
    text = (line or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return Command("quit")
    low = text.lower()
    if low in QUIT_WORDS:
        return Command("quit")
    if low in HELP_WORDS:
        return Command("help")
    if low in MODE_ALIASES:
        return Command("mode", MODE_ALIASES[low])
    return Command("files", text)


def expand(paths: list[str]) -> list[Path]:
    """文件原样保留；文件夹展开成其中的文档；不存在的路径与重复项丢掉。"""
    found: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        try:
            candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        except OSError:
            continue
        for cand in candidates:
            if not cand.is_file() or cand.suffix.lower() not in DOC_SUFFIXES:
                continue
            key = str(cand.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(cand)
    return found


def detect(path: Path, engines: list, mode: str, open_report: bool = True) -> bool:
    """检测一篇，写 HTML + JSON；失败只打印原因、不抛异常（交互窗口要能继续）。"""
    try:
        text, name = textio.load(str(path))
        doc = build_doc(text, name)
    except textio.ReadError as exc:
        print(f"  [x] 读不了 {path.name}：{exc}")
        return False
    except Exception as exc:  # noqa: BLE001 - 交互模式下不因单篇炸掉整个会话
        print(f"  [x] {path.name}：{type(exc).__name__}: {exc}")
        return False
    if doc.body_chars == 0:
        print(f"  [x] {path.name}：正文为空（或整篇被当作参考文献），跳过")
        return False

    _banner(f"检测 {doc.name}（正文 {doc.body_chars} 字）")
    started = time.time()
    analysis = aggregate.analyze(doc, mode=mode, engines=engines)
    elapsed = time.time() - started

    html_path = _resolve_out(None, doc.name, "report.html") if os.environ.get("AIDETECT_NO_HTML") is None else None
    json_path = _resolve_out(None, doc.name, "report.json")
    if html_path is not None:
        report.write(analysis, html_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(analysis, html_path, json_path)
    print(f"本次检测耗时 {elapsed:.1f}s（不含模型加载）")
    if open_report and html_path is not None:
        _open_report(html_path)
    return True


def _open_report(path: Path) -> None:
    if os.environ.get("AIDETECT_NO_OPEN"):
        return
    opener = getattr(os, "startfile", None)
    if opener is None:  # pragma: no cover - 非 Windows
        print(f"报告：{path}")
        return
    try:
        opener(str(path))
    except OSError:  # pragma: no cover - 没有默认浏览器
        print(f"报告：{path}")


def _welcome(mode: str) -> None:
    print("=" * 64)
    print(" CQUT-AI-Detect · 中文「AI 生成率」自查工具")
    print("=" * 64)
    print(f" 当前模式：{mode}")
    print(f" 切换模式：{MODE_MENU}")
    print(" 用法：把 .docx / .pdf / .txt / .md 文件拖进本窗口，按回车；")
    print("       也可以直接输入文件或文件夹路径（文件夹会批量检测）。")
    print(" h 看帮助 · q 或空行退出")
    print(f" 报告输出目录：{config.OUT_DIR}")
    print("-" * 64)


def _help() -> None:
    print(" 文件：把文件拖进来（带引号的路径也认），或输入文件夹路径批量检测")
    print(f" 模式：{MODE_MENU}")
    print(" 命令：h / ? 帮助，q / exit / 空行 退出")
    print(" 说明：这是自查工具，结果不能当作学术不端证据（详见 README）")


def _interactive(mode: str) -> int:
    engines: list | None = None
    _welcome(mode)
    while True:
        try:
            line = input("\n文件/命令> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return EXIT_OK
        cmd = parse_input(line)
        if cmd.kind == "quit":
            print("已退出。")
            return EXIT_OK
        if cmd.kind == "help":
            _help()
            continue
        if cmd.kind == "mode":
            mode = cmd.value or mode
            print(f"模式已切换为 {mode}")
            continue
        files = expand([cmd.value or ""])
        if not files:
            print(f"没有找到可检测的文件：{cmd.value}")
            print("（支持 .docx / .pdf / .txt / .md；输入文件夹会展开其中所有文档）")
            continue
        if engines is None:
            print("首次使用，正在加载模型（约 10 秒）…", flush=True)
            engines = aggregate.build_engines()
        for index, item in enumerate(files, 1):
            if len(files) > 1:
                print(f"\n--- [{index}/{len(files)}] ---")
            detect(item, engines, mode, open_report=index == len(files))
        print(f"\n共 {len(files)} 篇完成，报告在：{config.OUT_DIR}")


def main(argv: list[str] | None = None) -> int:
    _tolerant_console()
    args = list(sys.argv[1:] if argv is None else argv)
    missing = [name for name in ("clf", "ppl") if not fetch.is_ready(name)]
    if missing:
        print(f"缺少模型：{', '.join(missing)}")
        print("先在项目目录运行：python -m aidetect fetch all  （约 830MB，只需一次）")
        return EXIT_RESOURCE

    mode = config.DEFAULT_MODE
    paths = [arg for arg in args if arg.strip()]
    if not paths:
        return _interactive(mode)

    files = expand(paths)
    if not files:
        print(f"没有找到可检测的文件：{' '.join(paths)}")
        print("（支持 .docx / .pdf / .txt / .md；文件或文件夹都行）")
        return EXIT_INPUT
    print(f"准备检测 {len(files)} 篇，正在加载模型…", flush=True)
    engines = aggregate.build_engines()
    ok = 0
    for index, item in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n--- [{index}/{len(files)}] ---")
        if detect(item, engines, mode, open_report=index == len(files)):
            ok += 1
    print(f"\n完成 {ok}/{len(files)} 篇，报告在：{config.OUT_DIR}")
    return EXIT_OK if ok else EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
