"""zhv3 判别器探针：比较目标论文、HC3 人类、新闻人类、现代 AI 文本的窗口级分布。"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402
from transformers import BertForSequenceClassification, BertTokenizerFast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/zhv3"
CSV_PATH = ROOT / "data/anx/test.csv"
WIN = 240
LINES = [0.30, 0.50, 0.60, 0.70, 0.80, 0.90]


class Win:
    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


class Zhv3:
    def __init__(self) -> None:
        self.tok = BertTokenizerFast(vocab=str(MODEL_DIR / "vocab.txt"))
        self.model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR), use_safetensors=False)
        self.model.eval()

    @torch.no_grad()
    def score(self, texts: list[str], batch: int = 8) -> list[float]:
        out: list[float] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            enc = self.tok(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
            probs = torch.softmax(self.model(**enc).logits, -1)[:, 1]
            out.extend(float(x) for x in probs)
        return out


def windows(text: str, size: int = WIN) -> list[str]:
    out: list[str] = []
    cur = ""
    for sent_obj in segment.split_sentences(segment.normalize(text)):
        sent = sent_obj.text
        if not sent:
            continue
        if len(sent) > size:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(sent), size):
                out.append(sent[i : i + size])
            continue
        if len(cur) + len(sent) > size and cur:
            out.append(cur)
            cur = sent
        else:
            cur += sent
    if cur:
        out.append(cur)
    return out


def load_group(source: str, label: str, per: int) -> list[str]:
    texts: list[str] = []
    with CSV_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("source") or "") != source or (row.get("label") or "") != label:
                continue
            t = (row.get("text") or "").strip()
            if len(t) >= 200:
                texts.append(t[:1200])
    random.Random(20241025).shuffle(texts)
    return texts[:per]


def body_of(path: Path) -> str:
    text, _ = textio.load(str(path))
    doc = segment.build_doc(text, path.stem)
    return "\n".join(p.text for p in doc.paras if p.kind == "body")


def summarize(name: str, docs: list[str], eng: Zhv3) -> dict:
    flat: list[str] = []
    for t in docs:
        flat.extend(windows(t))
    scores = eng.score(flat)
    total = sum(len(w) for w in flat) or 1
    shares = {f"{ln:.2f}": round(sum(len(w) for w, s in zip(flat, scores) if s >= ln) / total, 4) for ln in LINES}
    entry = {
        "docs": len(docs),
        "n": len(flat),
        "mean": round(sum(scores) / len(scores), 4),
        "shares": shares,
    }
    print(
        f"{name:20s} docs={len(docs):3d} win={len(flat):4d} mean={entry['mean']:.3f} "
        + " ".join(f"@{k}={v:.2f}" for k, v in shares.items()),
        flush=True,
    )
    return entry


def main() -> int:
    eng = Zhv3()
    out: dict[str, dict] = {}
    docs = [
        ("刘颐刚", [body_of(Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx"))]),
        ("夏崇僖", [body_of(Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx"))]),
        ("张昊坤", [body_of(Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx"))]),
        ("mixed_hc3", [(ROOT / "tests/fixtures/mixed_hc3.txt").read_text(encoding="utf-8")]),
        ("essay_3000", [(ROOT / "tests/fixtures/essay_3000.txt").read_text(encoding="utf-8")]),
    ]
    for name, texts in docs:
        out[name] = summarize(name, texts, eng)

    for name, source, label, per in (
        ("human_hc3", "hc3_human", "0", 40),
        ("human_thucnews", "thucnews", "0", 40),
        ("ai_deepseek", "parallel_deepseek-v3.2", "1", 30),
        ("ai_gpt41mini", "parallel_gpt-4.1-mini", "1", 30),
        ("ai_kimi_k2", "parallel_Kimi-K2", "1", 30),
    ):
        out[name] = summarize(name, load_group(source, label, per), eng)

    (ROOT / "out/_probe_zhv3.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote out/_probe_zhv3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
