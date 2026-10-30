"""站点对照轨道：把句级判别分映射到 0-100，按 PaperPass 档位统计「疑似文字占比」。

**这是给商业站点对照用的激进口径，不是证据口径。**
- 档位线固定为 70 / 60 / 50（PaperPass 的「≥70 高疑似 / 60–70 中疑似 / 50–60 低疑似 /
  其余无 AI 疑似」），不是按本机 FPR 标定出来的阈值，天生比保守口径宽松；
- 判别器优先用 zhv3（原始分整体偏高，与商业网站的激进判定更接近），不可用时回退 cmj；
- 结论只能用于「和网站数字大致对得上」的参考，报告里必须与保守口径并列展示，
  不能单独当作学术不端证据。

与 ``aggregate`` 的关系：``aggregate.analyze`` 在最后调用 :func:`evaluate` 生成
``Analysis.site``；该轨道失败（引擎缺失/分数长度不符等）时静默降级为 ``None``，
不影响既有 risk / grade / ai_rate 逻辑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import config
from .engines.base import clamp01

# 档位从高到低；比例字典固定包含这几个键（另加 suspected = high+mid_high+mid）。
BAND_ORDER = ("high", "mid_high", "mid", "none")
# 分数低于此长度的句子不参与判定（去空白后），太短的句子判别器基本是瞎猜。
MIN_SENT_CHARS = 6

_LINEAR_ANCHORS = [(0.0, 0.0), (1.0, 100.0)]
_LINEAR_META = {"source": "linear-fallback"}


class _Win:
    """站点轨的窗口对象：判别器只要 ``.text`` 与 ``.chars`` 两个属性。"""

    __slots__ = ("text", "chars")

    def __init__(self, text: str) -> None:
        self.text = text
        self.chars = len(text)


@dataclass
class SiteSentence:
    index: int          # 全文句序号（0 开始，按 body 段落顺序）
    para_index: int     # 所属段落 index（与 Doc.paras 的 index 字段一致）
    text: str
    p: float            # 原始模型分 0-1
    score: float        # 映射后的 0-100 分
    band: str           # "high"/"mid_high"/"mid"/"none"


@dataclass
class SiteReport:
    engine: str
    sentences: list[SiteSentence]
    ratios: dict[str, float]     # 字数占比：high/mid_high/mid/none/suspected
    suspected_ratio: float       # = ratios["suspected"]，0-1
    headline: float              # 0-100 综合分 = 已判定句的字数加权平均 score
    meta: dict                   # 映射文件里的 meta（外加 requested_engine）

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "headline": round(self.headline, 2),
            "suspected_ratio": round(self.suspected_ratio, 4),
            "ratios": {k: round(float(v), 4) for k, v in self.ratios.items()},
            "bands": dict(config.SITE_TRACK["bands"]),
            "meta": dict(self.meta),
            "sentences": [
                {
                    "index": s.index,
                    "para_index": s.para_index,
                    "text": s.text,
                    "p": round(s.p, 4),
                    "score": round(s.score, 2),
                    "band": s.band,
                }
                for s in self.sentences
            ],
        }

# --- 映射文件 -------------------------------------------------------------
def _parse_anchors(raw) -> list[tuple[float, float]] | None:
    """把 JSON 里的 anchors 解析成按 p 升序的浮点对；格式不对返回 None。"""
    if not isinstance(raw, list) or not raw:
        return None
    pts: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
        if x != x or y != y:        # NaN
            return None
        pts.append((x, y))
    pts.sort(key=lambda t: t[0])
    return pts


def load_map() -> tuple[list[tuple[float, float]], dict]:
    """读取 ``resources/site_map.json`` 的 (anchors, meta)。

    文件不存在或损坏时回退线性映射 ``[(0,0),(1,100)]``，meta 标注 linear-fallback。
    """
    path = config.RESOURCES_DIR / str(config.SITE_TRACK["map_file"])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return list(_LINEAR_ANCHORS), dict(_LINEAR_META)
    if not isinstance(data, dict):
        return list(_LINEAR_ANCHORS), dict(_LINEAR_META)
    anchors = _parse_anchors(data.get("anchors"))
    if anchors is None:
        return list(_LINEAR_ANCHORS), dict(_LINEAR_META)
    meta = data.get("meta")
    return anchors, dict(meta) if isinstance(meta, dict) else {}


def _clamp100(x: float) -> float:
    return 0.0 if x < 0.0 else 100.0 if x > 100.0 else x


def score_of(p: float, anchors: list[tuple[float, float]] | None = None) -> float:
    """把 0-1 模型分按 anchors 分段线性插值成 0-100 分。

    超出首/末锚点时取端点分数（不外推），结果 clamp 到 0-100。
    ``anchors`` 为 None 时读映射文件（缺失则线性）。
    """
    pts = _parse_anchors(anchors) if anchors is not None else load_map()[0]
    if not pts:
        pts = list(_LINEAR_ANCHORS)
    try:
        x = float(p)
    except (TypeError, ValueError):
        x = 0.0
    if x <= pts[0][0]:
        out = pts[0][1]
    elif x >= pts[-1][0]:
        out = pts[-1][1]
    else:
        out = pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                out = y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                break
    return _clamp100(out)


def band_of(score: float) -> str:
    """0-100 分 → PaperPass 四档（阈值取 config.SITE_TRACK["bands"]）。"""
    bands = config.SITE_TRACK["bands"]
    if score >= bands["high"]:
        return "high"
    if score >= bands["mid_high"]:
        return "mid_high"
    if score >= bands["mid"]:
        return "mid"
    return "none"


# --- 引擎解析（优先 zhv3，回退 cmj） --------------------------------------
def _engine_classes() -> dict[str, type]:
    from .engines.classifier import (ClassifierEngine, CmjClassifierEngine,
                                     Zhv3ClassifierEngine)

    return {"clf": ClassifierEngine, "cmj": CmjClassifierEngine,
            "zhv3": Zhv3ClassifierEngine}


def _available(engine) -> bool:
    try:
        ok, _why = engine.available()
    except Exception:  # noqa: BLE001 - 第三方/假引擎实现不一致时不拖垮主流程
        return False
    return bool(ok)


def _resolve_engine(engines: list | None):
    """按 config.SITE_TRACK["engines"] 的优先级挑一个可用引擎；都不可用返回 None。

    调用方给了 ``engines`` 列表时只在列表里找（生产路径；避免测试里用假引擎时
    意外加载真实模型）；``engines is None`` 时才惰性构造真实引擎。
    """
    cands = [str(name) for name in (config.SITE_TRACK.get("engines") or [])]
    if not cands:
        return None
    if engines:
        for cname in cands:
            for eng in engines:
                if getattr(eng, "name", "") == cname and _available(eng):
                    return eng
    if engines is not None:
        return None
    classes = _engine_classes()
    for cname in cands:
        cls = classes.get(cname)
        if cls is None:
            continue
        try:
            eng = cls()
        except Exception:  # noqa: BLE001 - 缺 spec/依赖时当作不可用，继续下一个候选
            continue
        if _available(eng):
            return eng
    return None


# --- 句级/窗口 ------------------------------------------------------------
def _body_sentences(doc) -> list[tuple[int, str]]:
    """按 body 段落顺序收集句子 [(para_index, text)]；标题/参考文献段一律跳过。"""
    out: list[tuple[int, str]] = []
    for para in getattr(doc, "paras", None) or []:
        if getattr(para, "kind", "body") != "body":
            continue
        for sent in getattr(para, "sentences", None) or []:
            text = getattr(sent, "text", None)
            if text is None:
                text = str(sent)
            text = text.strip()
            if len(text) < MIN_SENT_CHARS:
                continue
            out.append((int(getattr(para, "index", 0)), text))
    return out


def _make_windows(items: list[tuple[int, str]], window_chars: int):
    """把连续句子拼成 ≤window_chars 的窗口；单句超长时整句单独成窗。

    返回 (wins, win_ids)：wins 给引擎打分，win_ids 记录每个窗口包含的句子下标。
    """
    wins: list[_Win] = []
    win_ids: list[list[int]] = []
    cur: list[str] = []
    cur_len = 0
    cur_ids: list[int] = []

    def flush() -> None:
        nonlocal cur, cur_len, cur_ids
        if cur:
            wins.append(_Win("".join(cur)))
            win_ids.append(cur_ids)
            cur, cur_len, cur_ids = [], 0, []

    for i, (_para_index, text) in enumerate(items):
        n = len(text)
        if n > window_chars:            # 单句超长：整句作一窗（不硬切，保持语义）
            flush()
            wins.append(_Win(text))
            win_ids.append([i])
            continue
        if cur and cur_len + n > window_chars:
            flush()
        cur.append(text)
        cur_len += n
        cur_ids.append(i)
    flush()
    return wins, win_ids


# --- 主入口 ---------------------------------------------------------------
def evaluate(doc, engines: list | None = None) -> SiteReport | None:
    """对一篇文档跑站点对照轨；没有可用引擎/没有可判定句子时返回 None。

    流程：只取 body 段落句子（短句跳过）→ 按 window_chars 拼窗口 → 判别器给每窗
    一个 0-1 分 → 窗口分回填给窗口内每个句子 → 映射到 0-100 并分档 → 按字数算占比。
    """
    engine = _resolve_engine(engines)
    if engine is None:
        return None
    items = _body_sentences(doc)
    if not items:
        return None
    window_chars = int(config.SITE_TRACK.get("window_chars", 240))
    wins, win_ids = _make_windows(items, window_chars)
    if not wins:
        return None
    res = engine.score(wins)
    if res is None or not getattr(res, "available", True):
        return None
    scores = list(getattr(res, "scores", None) or [])
    if len(scores) != len(wins):
        return None

    sent_p: list[float | None] = [None] * len(items)
    for raw_p, ids in zip(scores, win_ids):
        p = clamp01(float(raw_p))
        for i in ids:
            sent_p[i] = p

    anchors, meta = load_map()
    sentences: list[SiteSentence] = []
    band_chars = {band: 0 for band in BAND_ORDER}
    total_chars = 0
    weighted = 0.0
    for i, (para_index, text) in enumerate(items):
        p = sent_p[i]
        if p is None:
            continue
        score = score_of(p, anchors)
        band = band_of(score)
        chars = len(text)
        total_chars += chars
        weighted += score * chars
        band_chars[band] += chars
        sentences.append(SiteSentence(index=i, para_index=para_index, text=text,
                                      p=p, score=score, band=band))
    if total_chars <= 0:
        return None

    ratios = {band: band_chars[band] / total_chars for band in BAND_ORDER}
    ratios["suspected"] = ratios["high"] + ratios["mid_high"] + ratios["mid"]
    meta = dict(meta)
    meta["requested_engine"] = list(config.SITE_TRACK.get("engines") or [])
    return SiteReport(
        engine=str(getattr(engine, "name", "")),
        sentences=sentences,
        ratios=ratios,
        suspected_ratio=ratios["suspected"],
        headline=weighted / total_chars,
        meta=meta,
    )
