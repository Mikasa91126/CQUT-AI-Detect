"""Qwen2.5-0.5B 窗口级惊奇度探针：看它能不能把「AI 味」拉出来。"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aidetect.segment as segment  # noqa: E402
import aidetect.textio as textio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models/qwen"
WIN = 240
MAX_TOKENS = 256


class Win:
    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


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


def build() -> tuple[object, object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    model.eval()
    return torch, model, tok


def window_stats(torch, model, tok, texts: list[str]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(texts), 4):
        batch = texts[i : i + 4]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"]
        attn = enc["attention_mask"]
        logits = model(input_ids=ids, attention_mask=attn).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for j in range(ids.shape[0]):
            seq = int(attn[j].sum())
            if seq < 6:
                out.append({"n": seq, "mean_lp": -9.0, "ppl": 8000.0, "top10": 0.0})
                continue
            tgt = ids[j, 1:seq]
            lp = logprobs[j, 0 : seq - 1].gather(1, tgt.unsqueeze(1)).squeeze(1)
            top10 = (logprobs[j, 0 : seq - 1].topk(10, dim=-1).indices == tgt.unsqueeze(1)).any(dim=1).float().mean()
            mean_lp = float(lp.mean())
            out.append({
                "n": seq - 1,
                "mean_lp": round(mean_lp, 4),
                "ppl": round(math.exp(-mean_lp), 2),
                "top10": round(float(top10), 4),
            })
    return out


def load_hc3(per: int = 40) -> tuple[list[str], list[str]]:
    rng = random.Random(20241025)
    path = ROOT / "data/hc3zh/all.jsonl"
    hum: list[str] = []
    ai: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            for a in row.get("human_answers", [])[:1]:
                if a and len(str(a)) >= 200 and len(hum) < per:
                    hum.append(str(a)[:1200])
            for a in row.get("chatgpt_answers", [])[:1]:
                if a and len(str(a)) >= 200 and len(ai) < per:
                    ai.append(str(a)[:1200])
            if len(hum) >= per and len(ai) >= per:
                break
    rng.shuffle(hum)
    rng.shuffle(ai)
    return hum[:per], ai[:per]


DOCS = [
    ("刘颐刚", Path(r"B:\论文\生物医学工程专业导论论文_刘颐刚.docx")),
    ("夏崇僖", Path(r"B:\论文\生物医学工程专业导论论文夏崇僖.docx")),
    ("张昊坤", Path(r"B:\论文\测控技术与仪器专业导论_考核小论文_张昊坤.docx")),
    ("mixed_hc3", ROOT / "tests/fixtures/mixed_hc3.txt"),
    ("essay_3000", ROOT / "tests/fixtures/essay_3000.txt"),
]


def body_of(path: Path) -> str:
    text, _ = textio.load(str(path))
    doc = segment.build_doc(text, path.stem)
    return "\n".join(p.text for p in doc.paras if p.kind == "body")


def main() -> int:
    torch, model, tok = build()
    print("qwen 加载完成", flush=True)

    report: dict[str, dict] = {}
    for name, path in DOCS:
        if not path.exists():
            continue
        wins = windows(body_of(path))
        stats = window_stats(torch, model, tok, wins)
        lps = [s["mean_lp"] for s in stats]
        report[name] = {
            "n": len(wins),
            "class": "doc",
            "mean_lp": round(sum(lps) / len(lps), 4),
            "ppl": round(math.exp(-sum(lps) / len(lps)), 2),
            "top10": round(sum(s["top10"] for s in stats) / len(stats), 4),
            "max_lp": round(max(lps), 4),
            "p90_lp": round(sorted(lps)[int(len(lps) * 0.9)], 4),
            "lps": [round(x, 3) for x in lps],
        }
        print(f"  {name:10s} n={len(wins):3d} ppl={report[name]['ppl']:7.2f} mean_lp={report[name]['mean_lp']:7.3f} top10={report[name]['top10']:.3f}", flush=True)

    hum, ai = load_hc3()
    for label, texts in (("HC3人类", hum), ("HC3AI", ai)):
        wins: list[str] = []
        for t in texts:
            wins.extend(windows(t))
        stats = window_stats(torch, model, tok, wins)
        lps = [s["mean_lp"] for s in stats]
        report[label] = {
            "n": len(wins),
            "class": "corpus",
            "mean_lp": round(sum(lps) / len(lps), 4),
            "ppl": round(math.exp(-sum(lps) / len(lps)), 2),
            "top10": round(sum(s["top10"] for s in stats) / len(stats), 4),
            "lps": [round(x, 3) for x in lps],
        }
        print(f"  {label:10s} n={len(wins):3d} ppl={report[label]['ppl']:7.2f} mean_lp={report[label]['mean_lp']:7.3f} top10={report[label]['top10']:.3f}", flush=True)

    # AUC：用 mean_lp 区分 AI（高）与人类（低）
    def auc(neg: list[float], pos: list[float]) -> float:
        pairs = [(x, 0) for x in neg] + [(y, 1) for y in pos]
        pairs.sort()
        rank = 0.0
        i = 0
        n = len(pairs)
        while i < n:
            j = i
            while j < n and pairs[j][0] == pairs[i][0]:
                j += 1
            avg = (i + j + 1) / 2
            for k in range(i, j):
                if pairs[k][1] == 1:
                    rank += avg
            i = j
        npos = sum(1 for _, l in pairs if l == 1)
        nneg = n - npos
        return (rank - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else 0.5

    h = [s for s in report["HC3人类"]["lps"]]
    a = [s for s in report["HC3AI"]["lps"]]
    print(f"窗口级 AUC（mean_lp 高=AI）: {auc(h, a):.4f}")

    out = ROOT / "out/_probe_qwen.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
