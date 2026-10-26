"""困惑度引擎：用中文 GPT2（CLUECorpusSmall）算 token 级对数概率与 GLTR 分桶。

零样本法的核心假设是「AI 文本是 LM 的高概率路径、且波动小」：
- 低困惑度（平均 logprob 高）、top-10 桶占比高 → 更像模型自己写出来的；
- burstiness（token/句子间波动）低 → 更像 AI。

注意：绝对困惑度强依赖语料域（学术文本天然偏高），所以除绝对特征外还产出
「文档内相对 z 值」，供融合层在国内比较。本引擎自己的启发式分只是兜底展示，
不参与正式判定（正式判定走 fusion 的标定权重）。
"""
from __future__ import annotations

import time
from pathlib import Path

from .. import config
from ..segment import Unit, char_count, split_sentences, window_texts
from .base import EngineResult, TextCache, clamp01, load_pretrained, sigmoid

# 回退分词器的词表规模下限：低于它说明 vocab.txt 根本没被读进去（所有汉字都会变成
# [UNK]，困惑度成为无意义数字），必须直接报错，不能静默继续。
_MIN_FALLBACK_VOCAB = 1000


def _build_fallback_tokenizer(model_dir: Path):
    """``AutoTokenizer`` 失败时，从 ``vocab.txt`` 手工构建 BERT 分词器。

    transformers 4.x 的构造参数是 ``vocab_file=``，5.x 改成 ``vocab=``（可直接传路径）；
    在 5.x 里旧的 ``vocab_file=`` 会被 ``**kwargs`` 静默吞掉，只得到 5 个 special token
    （实测 5.20.2：vocab_size=5，分词结果全 [UNK]），再喂给 GPT2 就是垃圾困惑度。
    所以这里两个参数名都试，并在词表明显过小时抛错——宁可让用户看到明确报错，
    也不要给出看起来正常、实际错误的检测结果。
    """
    from transformers import BertTokenizerFast

    vocab_path = model_dir / "vocab.txt"
    attempts: list[str] = []
    for kwargs in ({"vocab": str(vocab_path)}, {"vocab_file": str(vocab_path)}):
        param = next(iter(kwargs))
        try:
            tok = BertTokenizerFast(**kwargs)
        except TypeError as exc:  # 参数名对不上（另一大版本）
            attempts.append(f"{param}= -> TypeError: {exc}")
            continue
        size = int(getattr(tok, "vocab_size", 0))
        if size >= _MIN_FALLBACK_VOCAB:
            return tok
        attempts.append(f"{param}= -> vocab_size={size}（过小，疑似参数被忽略）")
    import transformers

    raise RuntimeError(
        f"无法从 {vocab_path} 构建可用的回退分词器（transformers={transformers.__version__}）；"
        f"尝试记录：{attempts}"
    )


class PerplexityEngine:
    name = "ppl"
    label = "困惑度/流畅度"
    MAX_WINDOW = 240          # 字符数：256 token 内留余量
    MAX_TOKENS = 256

    def __init__(self, spec: dict | None = None, model_dir: Path | None = None) -> None:
        self.spec = spec or config.MODEL_SPECS["ppl"]
        self.model_dir = model_dir or (config.MODELS_DIR / "ppl")
        self._cache = TextCache()
        self._tok = None
        self._model = None

    # --- 生命周期 ------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        need = ["config.json", "pytorch_model.bin", "vocab.txt"]
        missing = [f for f in need if not (self.model_dir / f).exists()]
        if missing:
            return False, f"缺少模型文件 {missing}，先跑：python -m aidetect fetch ppl"
        try:
            import transformers  # noqa: F401
        except ImportError:
            return False, "未安装 transformers（pip install transformers）"
        return True, "就绪"

    def _load(self) -> None:
        if self._model is not None:
            return
        from ..engines.classifier import _init_torch

        _init_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self._tok = load_pretrained(AutoTokenizer, str(self.model_dir))
        except Exception:  # noqa: BLE001 - GPT2-Chinese 仓库的 tokenizer_class 有时缺失
            self._tok = _build_fallback_tokenizer(self.model_dir)
        self._model = load_pretrained(AutoModelForCausalLM, str(self.model_dir))
        self._model.eval()

    # --- 单窗口特征 ----------------------------------------------------
    def _window_features(self, windows: list[str]) -> list[dict]:
        if not windows:
            return []
        self._load()
        import torch

        uniq = [w for w in dict.fromkeys(windows) if self._cache.get(w) is None]
        bs = int(self.spec.get("batch_size", 2))
        topk = int(self.spec.get("topk", 100))
        for i in range(0, len(uniq), bs):
            batch = uniq[i:i + bs]
            enc = self._tok(
                batch, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=self.MAX_TOKENS, padding=True,
            )
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            if ids.shape[1] < 2:
                for w in batch:
                    self._cache.put(w, self._one_token_stats([], None))
                continue
            with torch.no_grad():
                logits = self._model(input_ids=ids, attention_mask=mask).logits
                logits = logits[:, :-1, :]
                target = ids[:, 1:].unsqueeze(-1)
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                token_lp = logprobs.gather(-1, target).squeeze(-1)
                valid = mask[:, 1:].bool()
                probs = logprobs.exp()
                entropy = -(probs * logprobs).sum(-1)
                k = min(topk, logits.shape[-1])
                topk_idx = logits.topk(k, dim=-1).indices
                rank_of = []
                for b in range(ids.shape[0]):
                    tgt = target[b, :, 0]
                    hit = (topk_idx[b] == tgt.unsqueeze(-1)).float()
                    has = hit.sum(-1) > 0
                    pos = (hit * torch.arange(1, k + 1, device=hit.device)).sum(-1)
                    rank_of.append(torch.where(has, pos, torch.full_like(pos, float(k + 1))))
                rank = torch.stack(rank_of, dim=0)
                maxprob = torch.softmax(logits, dim=-1).max(-1).values
            for b, w in enumerate(batch):
                keep = valid[b]
                lps = token_lp[b][keep]
                rk = rank[b][keep]
                ent = entropy[b][keep]
                mp = maxprob[b][keep]
                self._cache.put(w, self._one_token_stats(
                    lps.tolist(), None, rk.tolist(), ent.tolist(), mp.tolist()
                ))
                if not keep.any():
                    self._cache.put(w, self._one_token_stats([]))
        return [self._cache.get(w) for w in windows]  # type: ignore[list-item]

    @staticmethod
    def _one_token_stats(lps: list[float], _unused, ranks: list[float] | None = None,
                         ents: list[float] | None = None, mps: list[float] | None = None) -> dict:
        import statistics as st

        ranks = ranks or []
        ents = ents or []
        mps = mps or []
        if not lps:
            return {"ppl": 1000.0, "meanlprob": -6.9, "top10": 0.0, "top100": 0.0,
                    "tail": 1.0, "maxprob": 0.0, "entropy": 8.0, "lp_std": 0.0,
                    "lp_p10": -9.0, "n_tokens": 0.0}
        mean_lp = sum(lps) / len(lps)
        srt = sorted(lps)
        p10 = srt[max(0, len(srt) // 10)]
        n = len(lps)
        return {
            "ppl": float(min(1e4, pow(2.718281828, -mean_lp))),
            "meanlprob": mean_lp,
            "top10": sum(1 for r in ranks if r <= 10) / n if ranks else 0.0,
            "top100": sum(1 for r in ranks if r <= 100) / n if ranks else 0.0,
            "tail": sum(1 for r in ranks if r > 100) / n if ranks else 0.0,
            "maxprob": sum(mps) / len(mps) if mps else 0.0,
            "entropy": sum(ents) / len(ents) if ents else 0.0,
            "lp_std": st.pstdev(lps) if n > 1 else 0.0,
            "lp_p10": p10,
            "n_tokens": float(n),
        }

    # --- 句级辅助：burstiness -----------------------------------------
    def _burstiness(self, text: str) -> float:
        sents = [s.text for s in split_sentences(text) if char_count(s.text) >= 6]
        if len(sents) < 3:
            return float("nan")
        feats = self._window_features(sents[:24])
        means = [f["meanlprob"] for f in feats if f.get("n_tokens", 0) > 0]
        if len(means) < 3:
            return float("nan")
        import statistics as st

        return float(st.pstdev(means))

    # --- 对外接口 ------------------------------------------------------
    def score(self, units: list[Unit]) -> EngineResult:
        t0 = time.time()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, label=self.label, available=False, note=why)
        keys = ("ppl", "meanlprob", "top10", "top100", "tail", "maxprob", "entropy",
                "lp_std", "lp_p10", "burst")
        feats: dict[str, list[float]] = {k: [] for k in keys}
        scores: list[float] = []
        mid = float(config.PPL_HEURISTIC["mid"])
        scale = float(config.PPL_HEURISTIC["scale"])
        for unit in units:
            windows = window_texts(unit.text, self.MAX_WINDOW)
            wf = self._window_features(windows)
            # 窗口按 token 数加权平均
            wts = [max(1.0, f["n_tokens"]) for f in wf]
            tot = sum(wts) or 1.0

            def wavg(key: str) -> float:
                return sum(f.get(key, 0.0) * w for f, w in zip(wf, wts)) / tot

            ppl = wavg("ppl")
            burst = self._burstiness(unit.text)
            for k, v in (
                ("ppl", ppl), ("meanlprob", wavg("meanlprob")), ("top10", wavg("top10")),
                ("top100", wavg("top100")), ("tail", wavg("tail")),
                ("maxprob", wavg("maxprob")), ("entropy", wavg("entropy")),
                ("lp_std", wavg("lp_std")), ("lp_p10", wavg("lp_p10")),
                ("burst", 0.0 if burst != burst else burst),
            ):
                feats[k].append(float(v))
            scores.append(clamp01(sigmoid((mid - ppl) / scale)))
        return EngineResult(
            name=self.name, label=self.label, available=True,
            scores=scores, features=feats,
            note=f"GPT2-Chinese；{len(self._cache)} 次窗口推理", elapsed=time.time() - t0,
        )
