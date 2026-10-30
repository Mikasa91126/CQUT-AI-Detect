"""分类器引擎：HC3-Chinese 判别模型（BertForSequenceClassification）。

- AI 概率取 ``softmax(logits)[1]``，标签顺序从 ``config.id2label`` 自动识别（已核实本模型为
  ``{0: Human, 1: ChatGPT}``），识别不出时退回 HF 二分类惯例（1 = AI）。
- 512 token 上限：长段落按句子边界切窗口（≤300 字），段落分 = 窗口分按字数加权；
  同时保留 max（最可疑窗口）与离散度，供融合层使用。
- 窗口级文本缓存：重复句/重复段不重复推理。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .. import config
from ..segment import Unit, char_count, window_texts
from .base import EngineResult, TextCache, clamp01, load_pretrained

_TORCH_READY = False


def _init_torch() -> None:
    global _TORCH_READY
    if _TORCH_READY:
        return
    import torch

    torch.set_grad_enabled(False)
    threads = int(os.environ.get("AIDETECT_THREADS", "0") or 0)
    if threads <= 0:
        physical = max(1, (os.cpu_count() or 4) // 2)
        threads = min(8, physical * 2)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _TORCH_READY = True


class ClassifierEngine:
    name = "clf"
    label = "中文判别模型"
    spec_key = "clf"
    dir_name = "clf"
    FEAT_PREFIX = "clf"
    MIN_SENT_CHARS = 20

    def __init__(self, spec: dict | None = None, model_dir: Path | None = None) -> None:
        self.spec = spec or config.MODEL_SPECS[self.spec_key]
        self.model_dir = model_dir or (config.MODELS_DIR / self.dir_name)
        self._cache = TextCache()
        self._tok = None
        self._model = None
        self._ai_index = 1
        self._load_error: str | None = None

    # --- 生命周期 ------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        weights = [f for f in ("pytorch_model.bin", "model.safetensors")
                   if (self.model_dir / f).exists()]
        tokenizer = [f for f in ("tokenizer.json", "vocab.txt")
                     if (self.model_dir / f).exists()]
        missing: list[str] = []
        if not (self.model_dir / "config.json").exists():
            missing.append("config.json")
        if not weights:
            missing.append("pytorch_model.bin")
        if not tokenizer:
            missing.append("vocab.txt")
        if missing:
            return False, f"缺少模型文件 {missing}，先跑：python -m aidetect fetch {self.spec_key}"
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "未安装 transformers（pip install transformers）"
        return True, "就绪"

    def _load(self) -> None:
        if self._model is not None:
            return
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        _init_torch()
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = load_pretrained(AutoTokenizer, str(self.model_dir))
        self._model = load_pretrained(AutoModelForSequenceClassification, str(self.model_dir))
        self._model.eval()
        self._ai_index = self._resolve_ai_index()

    def _resolve_ai_index(self) -> int:
        explicit = self.spec.get("ai_label")
        if explicit is not None:
            return int(explicit)
        cfg = getattr(self._model, "config", None)
        labels = getattr(cfg, "id2label", None) or {}
        for k, v in labels.items():
            low = str(v).lower()
            if any(x in low for x in ("ai", "chatgpt", "gpt", "machine", "fake")):
                return int(k)
        return 1

    @property
    def ai_label_name(self) -> str:
        cfg = getattr(self._model, "config", None)
        labels = getattr(cfg, "id2label", None) or {}
        return str(labels.get(self._ai_index, "AI"))

    # --- 推理 ----------------------------------------------------------
    def _forward(self, texts: list[str]) -> list[float]:
        """批量算 AI 概率；texts 会先去重（同文本只跑一次推理）。"""
        if not texts:
            return []
        self._load()
        import torch

        pending = [t for t in dict.fromkeys(texts) if self._cache.get(t) is None]
        bs = int(self.spec.get("batch_size", 8))
        max_len = int(self.spec.get("max_tokens", 512))
        for i in range(0, len(pending), bs):
            batch = pending[i:i + bs]
            enc = self._tok(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
            )
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self._ai_index]
            for text, p in zip(batch, probs.tolist()):
                self._cache.put(text, float(p))
        return [float(self._cache.get(t)) for t in texts]  # type: ignore[arg-type]

    def score(self, units: list[Unit]) -> EngineResult:
        t0 = time.time()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, label=self.label, available=False, note=why)
        scores: list[float] = []
        p = self.FEAT_PREFIX
        feats: dict[str, list[float]] = {k: [] for k in
                                        (f"{p}_mean", f"{p}_max", f"{p}_min",
                                         f"{p}_std", f"{p}_nwin")}
        for unit in units:
            windows = window_texts(unit.text, int(self.spec.get("window_chars", 300)))
            probs = self._forward(windows)
            weights = [max(1, char_count(w)) for w in windows]
            mean = sum(p_ * w for p_, w in zip(probs, weights)) / sum(weights)
            mn = min(probs)
            mx = max(probs)
            scores.append(clamp01(mean))
            feats[f"{p}_mean"].append(mean)
            feats[f"{p}_max"].append(mx)
            feats[f"{p}_min"].append(mn)
            feats[f"{p}_std"].append((mx - mn))
            feats[f"{p}_nwin"].append(float(len(windows)))
        return EngineResult(
            name=self.name, label=self.label, available=True,
            scores=scores, features=feats,
            note=f"AI 类 = {self.ai_label_name}；窗口 {len(self._cache)} 次推理",
            elapsed=time.time() - t0,
        )

    # --- 句级 ----------------------------------------------------------
    def score_sentences(self, sentences: list[str]) -> list[float]:
        """句级 AI 概率；过短句子（<20 字）拼接邻句后判定，避免分布外误判。"""
        ok, _ = self.available()
        if not ok or not sentences:
            return []
        windows: list[str] = []
        for i, s in enumerate(sentences):
            if char_count(s) >= self.MIN_SENT_CHARS:
                windows.append(s)
                continue
            left = i - 1
            right = i + 1
            parts = [s]
            total = char_count(s)
            while total < self.MIN_SENT_CHARS and (left >= 0 or right < len(sentences)):
                if right < len(sentences):
                    parts.append(sentences[right])
                    total += char_count(sentences[right])
                    right += 1
                elif left >= 0:
                    parts.insert(0, sentences[left])
                    total += char_count(sentences[left])
                    left -= 1
            windows.append("".join(parts))
        return self._forward(windows)


class CmjClassifierEngine(ClassifierEngine):
    """第二判别器：zhv3 微调版（1:1 平衡、跨多模型家族的 AIGC 语料）。

    与 ``clf``（HC3 v1，2022 年 ChatGPT 语料）互补：本模型见过的生成器更多，
    对「现代大模型 + 人工润色」的文本更敏感，代价是误报略高。
    """

    name = "cmj"
    label = "中文判别模型 v2"
    spec_key = "cmj"
    dir_name = "cmj"
    FEAT_PREFIX = "cmj"


class Zhv3ClassifierEngine(ClassifierEngine):
    """第三判别器：zhv3 原版（站点对照轨用）。

    它没有做 1:1 平衡微调，原始分整体偏高（对人写正式文本也常给 0.2-0.5），
    正适合对齐商业网站的激进口径；**不要**让它参与保守口径的融合判定。
    """

    name = "zhv3"
    label = "中文判别模型 v3"
    spec_key = "zhv3"
    dir_name = "zhv3"
    FEAT_PREFIX = "zhv3"

