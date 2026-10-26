"""自检：一条命令看清「这台机器上能不能跑、跑的是哪套参数」。

原则：不联网，不下载；除在输出目录写一个临时探针文件（写完立刻删除，
用于确认目录可写）外不改动任何文件。任何一项失败都给可执行的下一步命令。
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys

from . import __version__, config, fusion


def symbols_for(encoding: str | None) -> tuple[str, str, str]:
    """按输出编码挑符号：老式 GBK 控制台装不下 ✓/✗，退化为 ASCII，避免自检崩在打印上。"""
    enc = encoding or "utf-8"
    try:
        "✓✗".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return ("[OK]", "[!]", "[X]")
    return ("✓", "!", "✗")


OK, WARN, BAD = symbols_for(getattr(sys.stdout, "encoding", None))


def _row(flag: str, title: str, detail: str = "") -> str:
    return f" {flag} {title}" + (f"\n     {detail}" if detail else "")


def _check_python() -> tuple[bool, list[str]]:
    ver = sys.version_info
    ok = ver >= (3, 10)
    detail = f"Python {platform.python_version()} @ {sys.executable}"
    if not ok:
        detail += "（需要 3.10+）"
    return ok, [_row(OK if ok else BAD, "解释器", detail)]


def _check_deps() -> tuple[bool, list[str]]:
    need = {
        "torch": "神经网络推理",
        "transformers": "模型加载（HF 官方 API）",
        "numpy": "数值计算",
        "scipy": "温度缩放拟合",
        "sklearn": "逻辑回归标定",
    }
    optional = {
        "docx": "读取 .docx",
        "pypdf": "读取 .pdf（首选）",
        "fitz": "读取 .pdf（兜底，PyMuPDF）",
    }
    lines, ok = [], True
    for mod, why in need.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            lines.append(_row(OK, f"{mod} {ver}".strip(), why))
        except Exception as exc:  # noqa: BLE001
            ok = False
            lines.append(_row(BAD, mod, f"缺少（{why}）：pip install {mod}"))
    for mod, why in optional.items():
        try:
            importlib.import_module(mod)
            lines.append(_row(OK, mod, why))
        except Exception:  # noqa: BLE001
            lines.append(_row(WARN, mod, f"未安装（{why} 将不可用）"))
    return ok, lines


def _check_models() -> tuple[bool, list[str]]:
    lines, ok = [], True
    for name, spec in config.MODEL_SPECS.items():
        d = config.MODELS_DIR / name
        missing = [f for f in spec["files"] if not (d / f).exists()]
        total = sum((d / f).stat().st_size for f in spec["files"] if (d / f).exists())
        if missing:
            ok = False
            lines.append(_row(BAD, f"模型 {name}（{spec['title']}）",
                              f"缺 {len(missing)} 个文件：{', '.join(missing[:3])}… "
                              f"→ python -m aidetect fetch {name}"))
        else:
            prov = d / ".provenance.json"
            src = ""
            if prov.exists():
                try:
                    meta = json.loads(prov.read_text(encoding="utf-8"))
                    main = meta.get("pytorch_model.bin") or {}
                    if main.get("fetched_at"):
                        src = f"，下载于 {main['fetched_at']}"
                except (OSError, json.JSONDecodeError):
                    pass
            lines.append(_row(OK, f"模型 {name}（{spec['title']}）",
                              f"{total/1e6:.0f} MB，{spec['repo']}{src}"))
    return ok, lines


def _check_calibration() -> tuple[bool, list[str]]:
    path = fusion.CALIBRATION_PATH
    if not path.exists():
        return False, [_row(WARN, "标定参数", f"未找到 {path} → python -m aidetect calibrate"
                                           "（未标定时只能用启发式加权，结论不可正式使用）")]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [_row(BAD, "标定参数", f"{path} 解析失败：{exc}")]
    rc = fusion.load_recipe()
    lines = [_row(OK if rc.calibrated else WARN, f"标定参数 v{rc.version}",
                  f"{path.name}，生成于 {raw.get('generated_at', '未知')}，"
                  f"标定集 {raw.get('dataset', {}).get('n_dev', '?')} 单元 / "
                  f"测试集 {raw.get('dataset', {}).get('n_test', '?')} 单元")]
    ev = rc.evaluation or {}
    for mode, item in ev.items():
        test = item.get("test") if isinstance(item, dict) else None
        if isinstance(test, dict):
            lines.append(_row("·", f"{mode} 阈值 {item.get('threshold', rc.mode_thresholds.get(mode))}",
                              f"域外测试：Acc {test.get('acc')} / AUC {test.get('auc')} / "
                              f"FPR {test.get('fpr')} / FNR {test.get('fnr')}"))
    return rc.calibrated, lines


def _check_data() -> tuple[bool, list[str]]:
    d = config.DATA_DIR / "hc3zh" / "all.jsonl"
    if d.exists():
        return True, [_row(OK, "评测语料 HC3-Chinese", f"{d.stat().st_size/1e6:.1f} MB")]
    return False, [_row(WARN, "评测语料 HC3-Chinese",
                        "未下载（只影响标定/评测，不影响检测）→ python -m aidetect fetch hc3zh")]


def _check_runtime() -> list[str]:
    lines = [_row(OK, "平台", f"{platform.system()} {platform.machine()} / "
                            f"{os.cpu_count()} 逻辑核 / PID {os.getpid()}")]
    try:
        import torch

        lines.append(_row(OK, "torch 推理",
                          f"{torch.__version__}，CUDA {'可用' if torch.cuda.is_available() else '不可用（CPU 模式）'}，"
                          f"线程 {torch.get_num_threads()}"))
    except Exception:  # noqa: BLE001
        lines.append(_row(WARN, "torch 推理", "未安装，判别器/困惑度引擎不可用"))
    env = os.environ.get("HF_ENDPOINT")
    lines.append(_row("·", "下载端点", env or "https://huggingface.co（+ hf-mirror 回退）"))
    try:
        config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = config.OUT_DIR / ".doctor_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        lines.append(_row(OK, "输出目录", str(config.OUT_DIR)))
    except OSError as exc:
        lines.append(_row(BAD, "输出目录", f"{config.OUT_DIR} 不可写：{exc}"))
    return lines


def run() -> int:
    print(f"ai-detect {__version__} 自检\n" + "=" * 62)
    hard_ok = True
    for label, fn in (("运行环境", _check_python), ("依赖", _check_deps),
                      ("模型", _check_models), ("标定", _check_calibration),
                      ("评测语料", _check_data)):
        print(f"\n[{label}]")
        ok, lines = fn()
        hard_ok = hard_ok and ok
        for line in lines:
            print(line)
    print("\n[环境细节]")
    for line in _check_runtime():
        print(line)

    print("\n" + "=" * 62)
    if hard_ok:
        print("自检通过：可以运行检测。")
        return 0
    print("自检发现缺失项。按上面的提示补齐后重试；缺模型/标定时检测会降级或拒绝给出正式结论。")
    return 1
