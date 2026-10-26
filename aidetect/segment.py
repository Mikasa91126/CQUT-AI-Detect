"""文档切分：段落 / 句子 / 判定单元，以及参考文献区剔除。

目标不是完美的排版还原，而是给检测器提供「语义完整、长度可控」的单元：
- 段落太短（<24 字）不独立判定，与相邻段合并成一个判定单元；
- 段落太长（>800 字）按句子边界切成多段，避免单次 512 token 截断失真。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

# --- 规范化 ---------------------------------------------------------------
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\xa0]")
_SPACES = re.compile(r"[ \t\u3000]+")
_BLANKS = re.compile(r"\n{3,}")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# --- 识别规则 -------------------------------------------------------------
REF_HEADING = re.compile(
    r"^\s*(参\s*考\s*文\s*献|参\s*考\s*资\s*料|引\s*用\s*文\s*献|references?|bibliography|works\s+cited)\s*[:：]?\s*$",
    re.I,
)
# 标题与首条文献挤在同一块（无空行）时只匹配开头，真正的切分条件见 build_doc
REF_HEADING_START = re.compile(
    r"^\s*(参\s*考\s*文\s*献|参\s*考\s*资\s*料|引\s*用\s*文\s*献|references?|bibliography|works\s+cited)\s*[:：]?",
    re.I,
)
HEADING_RE = re.compile(
    r"^\s*(?:"
    r"摘\s*要|内\s*容\s*摘\s*要|关\s*键\s*词|关键字|abstract|key\s*words?|"
    r"引\s*言|前\s*言|绪\s*论|导\s*论|结\s*论|结\s*语|讨\s*论|致\s*谢|附\s*录|"
    r"研\s*究\s*背\s*景|研\s*究\s*方\s*法|实\s*验\s*结\s*果|相\s*关\s*工\s*作|"
    r"[一二三四五六七八九十]+\s*[、.．]|"
    r"第\s*[一二三四五六七八九十0-9]+\s*[章节部分篇]|"
    r"\d+(?:\.\d+)*\s+[^\d\s]"
    r")",
    re.I,
)
BIB_ENTRY = re.compile(r"^\s*\[\s*\d+\s*\]")
URL_ONLY = re.compile(r"^(?:https?://|www\.|doi:|DOI:)", re.I)
SENT_END = "。！？!?；;…"
CLOSERS = "”’」』）)】》〉\"'"
MIN_PARA_JOIN = 600          # 无空行时强制断段的字符阈值
MAX_PARA_CHARS = 800         # 单段超过它就在句子边界切开


@dataclass
class Sentence:
    text: str
    index: int


@dataclass
class Para:
    index: int
    text: str
    kind: str = "body"          # body | heading | meta
    sentences: list[Sentence] = field(default_factory=list)
    merged_into: int | None = None

    @property
    def chars(self) -> int:
        return char_count(self.text)

    @property
    def is_scorable(self) -> bool:
        return self.kind == "body"


@dataclass
class Unit:
    """一次建模/打分的粒度：单段，或若干个相邻短段的合并。"""
    index: int
    paras: list[Para]
    text: str

    @property
    def chars(self) -> int:
        return char_count(self.text)

    @property
    def para_indexes(self) -> list[int]:
        return [p.index for p in self.paras]


@dataclass
class Doc:
    name: str
    paras: list[Para]
    raw_text: str
    ref_cut: bool = False
    excluded_chars: int = 0

    @property
    def body_paras(self) -> list[Para]:
        return [p for p in self.paras if p.is_scorable]

    @property
    def body_chars(self) -> int:
        return sum(p.chars for p in self.body_paras)

    @property
    def total_chars(self) -> int:
        return char_count(self.raw_text)


def char_count(text: str) -> int:
    """按「可见字符」计字数：去掉所有空白。中文场景比 len() 更接近检测系统的口径。"""
    return len(re.sub(r"\s", "", text))


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def _join_lines(lines: list[str]) -> str:
    """把被硬换行折断的行拼回去：中英之间加空格，中文之间不加。"""
    out = ""
    for line in lines:
        if not out:
            out = line
            continue
        if CJK_RE.match(out[-1:]) and CJK_RE.match(line[:1]):
            out += line
        else:
            out += " " + line
    return out


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 48:
        return False
    if s.startswith("#"):          # markdown 标题：# / ## 后必须跟空格（或独占一行）
        if not re.match(r"^#{1,6}(?:\s|$)", s):
            return False           # 「#1号样本…」这类正文不当作标题
        return bool(CJK_RE.search(s) or re.search(r"[A-Za-z]", s))
    return bool(HEADING_RE.match(s))


def split_sentences(text: str) -> list[Sentence]:
    out: list[Sentence] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in SENT_END:
            j = i + 1
            while j < n and (text[j] in SENT_END or text[j] in CLOSERS):
                j += 1
            seg = text[start:j].strip()
            if seg:
                out.append(Sentence(seg, len(out)))
            start = j
            i = j
        else:
            i += 1
    tail = text[start:].strip()
    if tail:
        out.append(Sentence(tail, len(out)))
    return out


def _split_long_para(text: str) -> list[str]:
    """在句子边界把超长段落切成 ≤MAX_PARA_CHARS 的块。

    单个句子本身超过上限时（长引用、无标点长串）按字符硬切，
    保证返回的每一块 len(piece) <= MAX_PARA_CHARS。
    """
    if len(text) <= MAX_PARA_CHARS:
        return [text]
    joiner = "" if CJK_RE.search(text) else " "
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            pieces.append(joiner.join(buf))
            buf, buf_len = [], 0

    for sent in split_sentences(text):
        s = sent.text
        while len(s) > MAX_PARA_CHARS:
            flush()
            pieces.append(s[:MAX_PARA_CHARS])
            s = s[MAX_PARA_CHARS:]
        if buf and buf_len + len(s) > MAX_PARA_CHARS:
            flush()
        buf.append(s)
        buf_len += len(s)
    flush()
    return pieces


def _classify(text: str) -> str:
    if BIB_ENTRY.match(text):
        return "meta"
    if URL_ONLY.match(text) and len(text) < 200:
        return "meta"
    if char_count(text) < 3:
        return "meta"
    if _is_heading(text):
        return "heading"
    return "body"


def build_doc(text: str, name: str = "<doc>") -> Doc:
    """把原始文本切成 Doc。"""
    norm = normalize(text)
    lines = norm.split("\n")

    blocks: list[str] = []
    cur: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            if cur:
                blocks.append(_join_lines(cur))
                cur = []
            continue
        if cur and (_is_heading(s) or char_count(_join_lines(cur)) >= MIN_PARA_JOIN):
            blocks.append(_join_lines(cur))
            cur = []
        cur.append(s)
    if cur:
        blocks.append(_join_lines(cur))

    paras: list[Para] = []
    ref_cut = False
    excluded = 0
    idx = 0
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        if not ref_cut and REF_HEADING.match(b) and len(b) <= 24:
            ref_cut = True
        if not ref_cut:
            m = REF_HEADING_START.match(b)
            if m:
                rest = b[m.end():].strip()
                # 「参考文献」与首条文献同块（中间没空行）时，只有紧跟 [1] 这类条目才切，
                # 避免把正文里「参考文献的格式要求…」误判为参考文献区。
                if BIB_ENTRY.match(rest) or (not rest and len(b) <= 24):
                    ref_cut = True
        kind = "meta" if ref_cut else _classify(b)
        if ref_cut:
            kind = "meta"
            excluded += char_count(b)
        for piece in _split_long_para(b):
            p = Para(index=idx, text=piece, kind=kind)
            p.sentences = split_sentences(piece) if kind == "body" else []
            paras.append(p)
            idx += 1

    if not paras:
        raise ValueError("切分后没有任何段落，文本可能只有空白字符")

    return Doc(name=name, paras=paras, raw_text=norm, ref_cut=ref_cut, excluded_chars=excluded)


def make_units(doc: Doc, min_chars: int | None = None) -> list[Unit]:
    """把可判定段落聚成判定单元：短段并入相邻段，保证单元 ≥ min_chars。"""
    min_chars = config.MIN_UNIT_CHARS if min_chars is None else min_chars
    units: list[Unit] = []
    buf: list[Para] = []
    buf_chars = 0

    def flush() -> None:
        nonlocal buf, buf_chars
        if not buf:
            return
        merged = []
        for p in buf:
            merged.append(p)
            p.merged_into = units.__len__()
        units.append(Unit(index=len(units), paras=list(merged), text=_join_paras(merged)))
        buf, buf_chars = [], 0

    for p in doc.body_paras:
        if buf and buf_chars + p.chars > MAX_PARA_CHARS:
            flush()
        buf.append(p)
        buf_chars += p.chars
        if buf_chars >= min_chars:
            flush()
    flush()
    return units


def _join_paras(paras: list[Para]) -> str:
    out = ""
    for p in paras:
        if not out:
            out = p.text
        elif CJK_RE.match(out[-1:]) and CJK_RE.match(p.text[:1]):
            out += p.text
        else:
            out += " " + p.text
    return out


def window_texts(text: str, max_chars: int | None = None) -> list[str]:
    """把一段文本按句子边界切成 ≤max_chars 的窗口（用于分类器 512 token 限制）。"""
    max_chars = config.MAX_CHUNK_CHARS if max_chars is None else max_chars
    if char_count(text) <= max_chars:
        return [text]
    windows: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            windows.append("".join(buf))
            buf, buf_len = [], 0

    for sent in split_sentences(text):
        s = sent.text
        # 单句本身超长（长公式、无标点串）时按字符硬切，保证每个窗口 <= max_chars
        while len(s) > max_chars:
            flush()
            windows.append(s[:max_chars])
            s = s[max_chars:]
        if buf and buf_len + len(s) > max_chars:
            flush()
        buf.append(s)
        buf_len += len(s)
    flush()
    return windows


# --- 供困惑度引擎使用：去掉引用标记与网址，避免 token 分布被污染 -------------
_CLEAN = [
    (re.compile(r"https?://\S+"), ""),
    (re.compile(r"\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]"), ""),   # [1] [1-3] [1, 2]
    (re.compile(r"（[^）]{0,24}(?:等|et\s+al|20\d\d)[^）]{0,24}）"), ""),
    (re.compile(r"\(\s*[A-Za-z][^)]{0,40}(?:et\s+al\.?|19\d\d|20\d\d)[^)]{0,40}\)"), ""),
    (re.compile(r"\$[^$]{0,200}\$"), ""),
]


def clean_for_ppl(text: str) -> str:
    for pat, repl in _CLEAN:
        text = pat.sub(repl, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def stats(doc: Doc) -> dict:
    body = doc.body_paras
    return {
        "name": doc.name,
        "total_chars": doc.total_chars,
        "body_chars": doc.body_chars,
        "excluded_chars": doc.excluded_chars,
        "paragraphs": len(doc.paras),
        "body_paragraphs": len(body),
        "ref_cut": doc.ref_cut,
        "avg_para_chars": round(doc.body_chars / len(body), 1) if body else 0.0,
    }
