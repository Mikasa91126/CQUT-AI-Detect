"""模型/数据集下载：多连接 Range 分段 + 分片级断点续传 + 端点轮换。

本机实测：HuggingFace 单连接约 0.5 MB/s、4 连接约 1.5 MB/s、8 连接约 2.9 MB/s，故默认 8 连接。
小分片（16 MB）比大分片稳：CDN 对超长 Range 响应容易提前断流，分片越小重试成本越低，
且断点粒度就是分片文件，中断后重跑自动跳过已完成部分。

用法：
    python -m aidetect.fetch clf ppl hc3zh
    python -m aidetect.fetch all --workers 8
    python -m aidetect.fetch clf --chunk-mb 8 --retries 200
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

UA = {"User-Agent": "aidetect/1.0 (+local; no-telemetry)"}
CHUNK_BYTES = 16 * 1024 * 1024
SMALL_FILE = 1_000_000
READ_BLOCK = 1 << 20
NO_PROGRESS_LIMIT = 5
PART_TMPL = ".p{:03d}"


# --------------------------------------------------------------------------- 网络
def _use_proxy() -> bool:
    """是否走系统/环境代理。

    默认跟随系统设置；设置了 AIDETECT_NO_PROXY=1 时强制直连 —— 常见场景是
    本机开着 Clash/V2Ray 但端口已关闭，系统代理会让所有请求报 WinError 10061。
    """
    return os.environ.get("AIDETECT_NO_PROXY", "").strip() not in {"1", "true", "yes"}


_OPENER: urllib.request.OpenerDirector | None = None


def _opener() -> urllib.request.OpenerDirector:
    """返回 urllib opener：默认跟随系统代理，AIDETECT_NO_PROXY=1 时禁用代理。"""
    global _OPENER
    if _OPENER is None:
        if _use_proxy():
            _OPENER = urllib.request.build_opener()
        else:
            _OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _OPENER


def _open(req: urllib.request.Request, timeout: float):
    return _opener().open(req, timeout=timeout)


def _endpoints(mirror: str | None = None) -> list[str]:
    """优先 spec 指定的镜像；否则 HF_ENDPOINT（或官方），失败再退 hf-mirror。"""
    if mirror == "modelscope":
        return ["modelscope"]
    primary = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    out = [primary]
    hf_mirror = "https://hf-mirror.com"
    if primary != hf_mirror:
        out.append(hf_mirror)
    return out


def _url(repo: str, kind: str, filename: str, base: str) -> str:
    # ModelScope：https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={file}
    if base == "modelscope":
        quoted = urllib.parse.quote(filename)
        return (f"https://modelscope.cn/api/v1/models/{repo}/repo"
                f"?Revision=master&FilePath={quoted}")
    # 模型：https://huggingface.co/{repo}/resolve/main/{file}
    # 数据集：https://huggingface.co/datasets/{repo}/resolve/main/{file}
    seg = "datasets/" if kind == "dataset" else ""
    return f"{base}/{seg}{repo}/resolve/main/{filename}"


def _head_size(urls: list[str], attempts: int = 6) -> int:
    """HEAD 取文件大小；CDN 偶发 SSL 断连，故多轮重试 + Range 探测兜底。"""
    last: Exception | None = None
    for i in range(attempts):
        for url in urls:
            try:
                req = urllib.request.Request(url, headers=UA, method="HEAD")
                with _open(req, timeout=30) as r:
                    n = r.headers.get("Content-Length") or r.headers.get("content-length")
                    if n:
                        return int(n)
            except Exception as exc:  # noqa: BLE001
                last = exc
            try:  # 某些 CDN 不支持 HEAD：退回 Range 探测
                req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
                with _open(req, timeout=30) as r:
                    cr = r.headers.get("Content-Range") or ""
                    if "/" in cr:
                        return int(cr.rsplit("/", 1)[1])
            except Exception as exc:  # noqa: BLE001
                last = exc
        time.sleep(min(0.5 * (i + 1), 3.0))
    raise RuntimeError(f"无法获取文件大小: {urls[0]} ({last})")


class _Progress:
    """文件级进度：只统计真正新写入的字节。"""

    def __init__(self, filename: str, size: int, already: int = 0) -> None:
        self.filename = filename
        self.size = size
        self.done = already
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n

    def snapshot(self) -> int:
        with self._lock:
            return self.done

    def __enter__(self) -> "_Progress":
        def loop() -> None:
            t0, b0 = time.time(), self.snapshot()
            while not self._stop.wait(2.0):
                cur = self.snapshot()
                dt = max(time.time() - t0, 1e-6)
                pct = min(100.0, 100.0 * cur / max(self.size, 1))
                print(f"\r  [..] {self.filename:<26s} {cur/1e6:7.1f}/{self.size/1e6:6.1f} MB"
                      f" {pct:5.1f}% {(cur-b0)/dt/1e6:4.2f} MB/s   ", end="", flush=True)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        cur = self.snapshot()
        ok = cur >= self.size
        print(f"\r  [{'ok' if ok else '!!'}] {self.filename:<26s} {cur/1e6:7.1f}/{self.size/1e6:6.1f} MB"
              f" {100.0*cur/max(self.size,1):5.1f}%                ", flush=True)


def _stream_range(url: str, path: Path, start: int, end: int, progress: _Progress) -> None:
    """把 [start, end] 追加写入 path（调用方保证 start == 现有大小）。"""
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
    with _open(req, timeout=120) as r:
        status = getattr(r, "status", 200)
        mode = "ab"
        if status != 206:
            # 部分镜像（如 ModelScope 小文件）忽略 Range 直接 200 返回整个文件；
            # 只有「一次性下完整个文件」时才允许，避免多个分片重复下载全量。
            whole = start == 0 and end >= progress.size - 1
            if status != 200 or not whole:
                raise RuntimeError(f"服务器未按 Range 返回（HTTP {status}）")
            mode = "wb"
        with open(path, mode) as f:
            while True:
                buf = r.read(READ_BLOCK)
                if not buf:
                    break
                f.write(buf)
                f.flush()
                progress.add(len(buf))


def _fetch_chunk(urls: list[str], path: Path, start: int, end: int,
                 progress: _Progress, retries: int = 200) -> None:
    """下载一个分片；断流就地续传，无进展时轮换端点。"""
    want = end - start + 1
    attempt = 0
    stall = 0
    while True:
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            if have > want:  # 极端情况：多余字节截断
                with open(path, "r+b") as f:
                    f.truncate(want)
            return
        url = urls[attempt % len(urls)]
        err: Exception | None = None
        try:
            _stream_range(url, path, start + have, end, progress)
        except Exception as exc:  # noqa: BLE001
            err = exc
        after = path.stat().st_size if path.exists() else 0
        if after >= want:
            continue
        if after > have:
            stall = 0
        else:
            stall += 1
        attempt += 1
        if stall >= NO_PROGRESS_LIMIT:
            raise RuntimeError(f"分片无进展 {path.name} ({after}/{want}): {err}")
        if attempt > retries:
            raise RuntimeError(f"分片重试超限 {path.name} ({after}/{want}): {err}")
        time.sleep(min(0.5 * stall, 3.0))


# --------------------------------------------------------------------------- 分片布局
def _layout_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".layout")


def _part_paths(dest: Path, n: int) -> list[Path]:
    if n <= 1:
        return [dest.with_name(dest.name + ".part")]
    return [dest.with_name(dest.name + PART_TMPL.format(i)) for i in range(n)]


def _delete_parts(dest: Path) -> None:
    """删除该文件的所有临时分片（不改动最终文件）。"""
    for p in dest.parent.iterdir():
        if not p.is_file() or p.name == dest.name:
            continue
        if p.name in (dest.name + ".part",) or p.name.startswith(dest.name + ".p") \
                or p.name.startswith(dest.name + ".chunk") or p.name == dest.name + ".layout":
            try:
                p.unlink()
            except OSError:
                pass


def _plan_parts(dest: Path, size: int, workers: int,
                chunk_bytes: int) -> tuple[list[Path], list[tuple[int, int]]]:
    """决定分片布局；布局与上次不同则丢弃旧分片，避免错位拼接。"""
    if size < SMALL_FILE or workers <= 1:
        per, n = size, 1
    else:
        per = max(chunk_bytes, math.ceil(size / max(workers * 4, 1)))
        n = max(1, math.ceil(size / per))
    spans = [(i * per, min(size - 1, (i + 1) * per - 1)) for i in range(n)]
    layout = {"size": size, "per": per, "n": n}
    lp = _layout_path(dest)
    old = None
    if lp.exists():
        try:
            old = json.loads(lp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = None
    if old != layout:
        _delete_parts(dest)
        lp.write_text(json.dumps(layout), encoding="utf-8")
    return _part_paths(dest, n), spans


def _write_provenance(dest_dir: Path, filename: str, info: dict) -> None:
    path = dest_dir / ".provenance.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data[filename] = info
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def download_file(repo: str, kind: str, filename: str, dest: Path,
                  workers: int = 8, expect_size: int | None = None,
                  chunk_bytes: int = CHUNK_BYTES, retries: int = 200,
                  mirror: str | None = None) -> Path:
    """把仓库中的单个文件下到 dest（原子替换，支持分片续传）。"""
    if dest.exists() and (expect_size is None or dest.stat().st_size == expect_size):
        print(f"  [skip] {filename} 已存在", flush=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    urls = [_url(repo, kind, filename, base) for base in _endpoints(mirror)]
    size = expect_size or _head_size(urls)
    parts, spans = _plan_parts(dest, size, workers, chunk_bytes)

    done_bytes = 0
    pending: list[tuple[Path, int, int]] = []
    for path, (start, end) in zip(parts, spans):
        want = end - start + 1
        have = path.stat().st_size if path.exists() else 0
        if have == want:
            done_bytes += have
        else:
            if have > want:  # 防御：异常大的分片截断
                with open(path, "r+b") as f:
                    f.truncate(want)
                have = want
                done_bytes += have
            else:
                done_bytes += have
            pending.append((path, start, end))
    print(f"  [..] {filename}: {size/1e6:.1f} MB / {len(spans)} 分片 / "
          f"{len(pending)} 待下 / 已有 {done_bytes/1e6:.1f} MB", flush=True)

    errs: list[Exception] = []
    err_lock = threading.Lock()
    if pending:
        q: "queue.Queue[tuple[Path, int, int]]" = queue.Queue()
        for item in pending:
            q.put(item)

        def worker() -> None:
            while True:
                try:
                    path, start, end = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    _fetch_chunk(urls, path, start, end, prog, retries=retries)
                except Exception as exc:  # noqa: BLE001
                    with err_lock:
                        errs.append(exc)

        with _Progress(filename, size, already=done_bytes) as prog:
            threads = [threading.Thread(target=worker, daemon=True)
                       for _ in range(max(1, min(workers, len(pending))))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    else:
        print(f"  [skip] {filename}: 分片已齐，直接合并", flush=True)

    if not errs:
        got = sum((p.stat().st_size if p.exists() else 0) for p in parts)
        if got != size:
            errs.append(RuntimeError(f"分片汇总大小不符 {filename}: {got}/{size}"))
    if errs:
        raise RuntimeError(f"下载未完成 {repo}/{filename}: {errs[0]}（分片已保留，重跑可续传）")

    if len(parts) == 1:
        # 单分片：分片就是最终内容，直接原子改名（若再 copy 会把源文件截断）
        part = parts[0]
        got = part.stat().st_size if part.exists() else 0
        if got != size:
            raise RuntimeError(f"大小校验失败 {filename}: {got}/{size}")
        os.replace(part, dest)
    else:
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 20)
        got = tmp.stat().st_size
        if got != size:
            raise RuntimeError(f"大小校验失败 {filename}: {got}/{size}")
        os.replace(tmp, dest)
    _delete_parts(dest)
    _write_provenance(dest.parent, filename, {
        "repo": repo, "kind": kind, "bytes": size, "urls": urls,
        "mirror": mirror or "hf",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"  [ok] {filename} ({size/1e6:.1f} MB)", flush=True)
    return dest


# --------------------------------------------------------------------------- 资源
def _spec(name: str) -> tuple[dict, Path]:
    if name in config.MODEL_SPECS:
        return config.MODEL_SPECS[name], config.MODELS_DIR / name
    if name in config.DATASET_SPECS:
        return config.DATASET_SPECS[name], config.DATA_DIR / name
    raise KeyError(f"未知资源: {name}")


def ensure(name: str, workers: int = 8, chunk_bytes: int = CHUNK_BYTES,
           retries: int = 200) -> list[Path]:
    """确保某个逻辑资源（模型或数据集）的全部白名单文件就位。"""
    spec, dest_dir = _spec(name)
    print(f"== {name}: {spec['repo']} -> {dest_dir}", flush=True)
    out: list[Path] = []
    errors: list[Exception] = []
    for fn in spec["files"]:
        try:
            out.append(download_file(spec["repo"], spec["kind"], fn, dest_dir / fn,
                                     workers=workers, chunk_bytes=chunk_bytes,
                                     retries=retries, mirror=spec.get("mirror")))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            print(f"  [ERR] {fn}: {exc}", flush=True)
    if errors:
        raise RuntimeError(f"{name} 有 {len(errors)} 个文件未完成: {errors[0]}")
    return out


def is_ready(name: str) -> bool:
    try:
        spec, d = _spec(name)
    except KeyError:
        return False
    return all((d / fn).exists() for fn in spec["files"])


def _usage() -> str:
    return (
        "用法: python -m aidetect.fetch [资源 ...] [--workers N] [--chunk-mb N] [--retries N] [--no-proxy]\n"
        "资源: clf | cmj | zhv3 | ppl | hc3zh | qwen | all"
        "（默认 clf cmj zhv3 ppl；all = 默认 + hc3zh；qwen 为可选实验组件）\n"
        "zhv3 是站点对照轨用的判别器（约 390MB），不下也能跑，只是看不到站点对照数字。\n"
        "这是唯一联网的命令；失败后再跑同一条命令会自动续传。\n"
        "--no-proxy: 忽略系统代理直连（本机代理端口已关闭时用这个，等价于 AIDETECT_NO_PROXY=1）。\n"
    )


def main(argv: list[str] | None = None) -> int:
    global _OPENER
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help", "help") for a in argv):
        print(_usage(), end="", flush=True)
        return 0
    if "--no-proxy" in argv:
        argv.remove("--no-proxy")
        os.environ["AIDETECT_NO_PROXY"] = "1"
        _OPENER = None
    opts = {"--workers": 8, "--chunk-mb": CHUNK_BYTES // (1024 * 1024), "--retries": 200}
    for key in list(opts):
        if key in argv:
            i = argv.index(key)
            if i + 1 >= len(argv) or not argv[i + 1].lstrip("-").isdigit():
                print(f"{key} 需要一个整数参数\n{_usage()}", file=sys.stderr, flush=True)
                return 2
            opts[key] = int(argv[i + 1])
            del argv[i:i + 2]
    names = argv or ["clf", "cmj", "zhv3", "ppl"]
    if names == ["all"]:
        names = ["clf", "cmj", "zhv3", "ppl", "hc3zh"]
    known = set(config.MODEL_SPECS) | set(config.DATASET_SPECS)
    unknown = [n for n in names if n not in known]
    if unknown:
        print(f"未知资源: {', '.join(unknown)}\n{_usage()}", file=sys.stderr, flush=True)
        return 2
    names = [n for n in dict.fromkeys(names) if n in known]
    failures: list[str] = []
    for n in names:
        try:
            ensure(n, workers=opts["--workers"],
                   chunk_bytes=opts["--chunk-mb"] * 1024 * 1024,
                   retries=opts["--retries"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{n}: {exc}")
            print(f"[FAIL] {n}: {exc}", flush=True)
    if failures:
        print("\n未完成：\n  " + "\n  ".join(failures), flush=True)
        print("（分片已保留，直接重跑同一命令即可续传）", flush=True)
        return 1
    print("\n全部就绪。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
