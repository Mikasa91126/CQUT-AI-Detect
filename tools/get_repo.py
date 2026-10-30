"""从 hf-mirror 下载一个仓库的白名单文件到本地目录（绕过系统代理）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aidetect import fetch  # noqa: E402

DEFAULT_FILES = [
    "config.json",
    "model.safetensors",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法: python tools/get_repo.py <repo> <dest> [files...]")
        return 2
    repo, dest = argv[0], Path(argv[1])
    files = argv[2:] or DEFAULT_FILES
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    for fn in files:
        try:
            got = fetch.download_file(repo, "model", fn, dest / fn, workers=4)
            print(f"ok   {fn} -> {got} bytes", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"fail {fn}: {type(exc).__name__}: {exc}", flush=True)
            errors.append(fn)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
