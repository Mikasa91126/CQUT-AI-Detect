"""引擎协议与公共工具。

每个引擎约定：
- ``available()`` 返回 ``(是否可用, 说明)``，缺模型时给用户可执行的提示；
- ``score(units)`` 返回 ``EngineResult``，其中 ``scores`` 是 0–1 的「像 AI」启发式分；
- ``features`` 是给融合层用的原始特征（键名稳定，标定文件按这些键取参数）。

引擎不负责阈值判定，也不负责任制报告 —— 这两件事在 fusion / report 层。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from ..segment import Unit


@dataclass
class EngineResult:
    name: str
    label: str
    available: bool
    scores: list[float] = field(default_factory=list)
    features: dict[str, list[float]] = field(default_factory=dict)
    note: str = ""
    elapsed: float = 0.0

    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


class Engine(Protocol):
    name: str
    label: str

    def available(self) -> tuple[bool, str]: ...

    def score(self, units: list[Unit]) -> EngineResult: ...


# --- 文本哈希缓存（窗口级去重，重复句不再重复推理） -------------------------
class TextCache:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text: str):
        return self._store.get(self.key(text))

    def put(self, text: str, value) -> None:
        self._store[self.key(text)] = value

    def __len__(self) -> int:
        return len(self._store)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def sigmoid(x: float) -> float:
    from math import exp

    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def load_pretrained(factory, path: str, **kwargs):
    """兼容 transformers 4.x / 5.x：老式 .bin 权重在 5.x 下需显式关掉 safetensors。"""
    try:
        return factory.from_pretrained(path, local_files_only=True, **kwargs)
    except (TypeError, OSError, ValueError) as exc:  # noqa: PERF203
        msg = str(exc).lower()
        if "safetensors" in msg or "use_safetensors" in msg:
            return factory.from_pretrained(
                path, local_files_only=True, use_safetensors=False, **kwargs
            )
        raise
