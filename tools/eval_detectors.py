"""评测候选检测器：把文档/样本打成「文档级 AI 概率」，并输出对照表。

用法：
    python tools/eval_detectors.py --report-json out/xxx_report.json --text doc.txt ...
    python tools/eval_detectors.py --builtin      # 内置评测集（用户文档 + 人类样本 + HC3）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EVAL_DOCS = ROOT / "out" / "_eval_docs"

DEFAULT_MODELS = {
    "clf_hc3": MODELS / "clf",
    "cmj_zhv3": MODELS / "cmj",
}

USER_REPORTS = {
    "刘颐刚_导论论文": ROOT / "out" / "生物医学工程专业导论论文_刘颐刚_report.json",
    "夏崇僖_导论论文": ROOT / "out" / "生物医学工程专业导论论文夏崇僖_report.json",
    "张昊坤_测控小论文": ROOT / "out" / "测控技术与仪器专业导论_考核小论文_张昊坤_report.json",
}


# ---------------------------------------------------------------- 文档重建
def doc_from_report(path: Path) -> tuple[str, list[tuple[int, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paras = sorted(data.get("paragraphs") or [], key=lambda r: r.get("index", 0))
    rows: list[tuple[int, str, str]] = []
    for i, r in enumerate(paras):
        text = (r.get("text") or "").strip()
        if text:
            rows.append((r.get("index", i), r.get("kind") or "body", text))
    full = "\n\n".join(t for _, _, t in rows)
    body = [(i, t) for i, kind, t in rows if kind == "body"]
    return full, body


def save_doc(name: str, text: str) -> Path:
    EVAL_DOCS.mkdir(parents=True, exist_ok=True)
    path = EVAL_DOCS / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- 模型打分
class Detector:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.tok = None
        self.model = None
        self.ai_index = 1
        self.max_len = 480

    def load(self) -> None:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

        cfg = AutoConfig.from_pretrained(str(self.path))
        id2label = {int(k): str(v) for k, v in (getattr(cfg, "id2label", {}) or {}).items()}
        ai_keys = ("ai", "chatgpt", "gpt", "machine", "fake", "aigc", "generated")
        for idx, label in id2label.items():
            if any(k in label.lower() for k in ai_keys):
                self.ai_index = idx
                break
        try:
            self.tok = AutoTokenizer.from_pretrained(str(self.path), use_fast=True)
        except Exception:  # noqa: BLE001
            from transformers import BertTokenizerFast
            self.tok = BertTokenizerFast(vocab_file=str(self.path / "vocab.txt"))
        if not self.tok.is_fast:
            pass
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path), config=cfg)
        self.model.eval()
        self.model.to("cpu")
        max_pos = int(getattr(cfg, "max_position_embeddings", 512) or 512)
        self.max_len = min(max_pos, 512) - 2
        self.id2label = id2label

    def _windows(self, text: str) -> list[str]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_len:
            return [text]
        return [self.tok.decode(ids[i:i + self.max_len]) for i in range(0, len(ids), self.max_len)]

    @torch.inference_mode()
    def score(self, texts: list[str], batch: int = 8) -> list[float]:
        windows: list[str] = []
        owners: list[int] = []
        for i, t in enumerate(texts):
            for w in self._windows(t) or [t]:
                windows.append(w)
                owners.append(i)
        scores = [0.0] * len(texts)
        counts = [0] * len(texts)
        for start in range(0, len(windows), batch):
            chunk = windows[start:start + batch]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_len + 2, return_tensors="pt")
            logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, self.ai_index].tolist()
            for j, p in enumerate(probs):
                owner = owners[start + j]
                scores[owner] += p
                counts[owner] += 1
        return [scores[i] / max(counts[i], 1) for i in range(len(texts))]


# ---------------------------------------------------------------- 评测
def human_samples() -> dict[str, str]:
    out: dict[str, str] = {}
    essay = ROOT / "tests" / "fixtures" / "essay_3000.txt"
    if essay.exists():
        out["人类样本_essay3000"] = essay.read_text(encoding="utf-8")
    try:
        import docx
        for label, p in {
            "人类样本_班会总结": Path(r"C:\Users\mikasa\Desktop\药学与生物工程学院生物医学工程402班第九周班会总结.docx"),
            "人类样本_入党申请书": Path(r"C:\Users\mikasa\Desktop\入党申请书\入党申请书-刘颐刚-可编辑版.docx"),
        }.items():
            if p.exists():
                d = docx.Document(str(p))
                out[label] = "\n\n".join(x.text.strip() for x in d.paragraphs if x.text.strip())
    except Exception as exc:  # noqa: BLE001
        print("docx 读数失败:", exc)
    return out


def hc3_samples(per_domain: int = 12) -> dict[str, str]:
    path = ROOT / "data" / "hc3zh" / "all.jsonl"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    buckets: dict[str, list[tuple[str, str]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            src = rec.get("source")
            b = buckets.setdefault(src, [])
            if len(b) >= per_domain:
                continue
            if rec.get("chatgpt_answers"):
                b.append(("AI", rec["chatgpt_answers"][0]))
            if len(b) < per_domain and rec.get("human_answers"):
                b.append(("HUMAN", rec["human_answers"][0]))
            if len(b) >= per_domain:
                continue
    for src, items in buckets.items():
        for i, (lab, text) in enumerate(items[:per_domain]):
            out[f"HC3_{src}_{lab}_{i:02d}"] = text
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builtin", action="store_true", help="跑内置评测集")
    ap.add_argument("--models", nargs="*", default=None, help="name=path 形式")
    ap.add_argument("--text", nargs="*", default=None, help="额外文本文件")
    ap.add_argument("--out", default=str(ROOT / "out" / "_eval_detectors.json"))
    args = ap.parse_args(argv)

    models_spec = {}
    if args.models:
        for item in args.models:
            name, _, path = item.partition("=")
            models_spec[name] = Path(path)
    else:
        models_spec = {k: v for k, v in DEFAULT_MODELS.items() if (v / "config.json").exists()}

    docs: dict[str, str] = {}
    if args.builtin:
        for label, rp in USER_REPORTS.items():
            if rp.exists():
                full, _ = doc_from_report(rp)
                docs[label] = full
        docs.update(human_samples())
        docs.update(hc3_samples())
    for p in args.text or []:
        path = Path(p)
        docs[path.stem] = path.read_text(encoding="utf-8")

    print(f"文档/样本数: {len(docs)}；模型: {list(models_spec)}", flush=True)
    for name, d in docs.items():
        print(f"  {name}: {len(d)} 字", flush=True)

    result: dict[str, dict[str, float]] = {}
    for mname, mpath in models_spec.items():
        if not mpath.exists():
            print(f"[skip] {mname} 缺 {mpath}")
            continue
        t0 = time.time()
        det = Detector(mname, mpath)
        det.load()
        print(f"[{mname}] loaded in {time.time()-t0:.1f}s id2label={det.id2label} ai_index={det.ai_index}", flush=True)
        t1 = time.time()
        keys = list(docs)
        scores = det.score([docs[k] for k in keys])
        print(f"[{mname}] 打分完成 {time.time()-t1:.1f}s", flush=True)
        result[mname] = dict(zip(keys, scores))

    order = list(docs)
    head = "样本".ljust(34) + "".join(m.ljust(12) for m in result)
    print("\n" + head)
    for k in order:
        print(k.ljust(34) + "".join(f"{result[m].get(k, float('nan')):.4f}".ljust(12) for m in result))

    Path(args.out).write_text(json.dumps({"docs": {k: len(v) for k, v in docs.items()}, "scores": result},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n写出", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
