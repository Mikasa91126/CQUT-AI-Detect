"""aidetect 全局配置：路径、模型清单、默认判定阈值。

只放常量与浅层环境变量解析，不做任何 IO。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("AIDETECT_HOME") or ROOT).resolve()

MODELS_DIR = HOME / "models"
DATA_DIR = HOME / "data"
OUT_DIR = HOME / "out"
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"

# --- 模型清单 -------------------------------------------------------------
# kind: "model" / "dataset"；files: 只下白名单（HF 仓库里 flax/tf 等副本一律不取）。
MODEL_SPECS: dict[str, dict] = {
    "clf": {
        "kind": "model",
        "repo": "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
        "title": "中文判别模型（HC3-Chinese RoBERTa）",
        "license": "Apache-2.0",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        # id2label 已核实为 {0: Human, 1: ChatGPT}；ai_label=None 表示按 id2label 自动识别。
        "ai_label": None,
        "window_chars": 300,
        "max_tokens": 512,
        "batch_size": 8,
    },
    "ppl": {
        "kind": "model",
        "repo": "uer/gpt2-chinese-cluecorpussmall",
        "title": "中文概率语言模型（GPT2-Chinese，用于困惑度/GLTR）",
        "license": "Apache-2.0",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "vocab.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "max_tokens": 512,
        "batch_size": 4,
        "topk": 100,
    },
}

DATASET_SPECS: dict[str, dict] = {
    "hc3zh": {
        "kind": "dataset",
        "repo": "Hello-SimpleAI/HC3-Chinese",
        "title": "HC3-Chinese 评测/标定集",
        "files": ["all.jsonl"],
    },
}

# --- 判定阈值（calibration.json 存在时以标定结果为准） ---------------------
DEFAULT_THRESHOLDS = {"high": 0.80, "mid_high": 0.60, "mid": 0.40}
MODE_TARGET_FPR = {"strict": 0.01, "balanced": 0.05, "lenient": 0.10}
DEFAULT_MODE = "balanced"

# 困惑度引擎的兜底启发式映射（未标定时用于展示，正式判定以标定权重为准）：
#   score = sigmoid((mid - ppl) / scale)
PPL_HEURISTIC = {"mid": 25.0, "scale": 8.0}

# --- 文本处理 -------------------------------------------------------------
MIN_RELIABLE_CHARS = 300      # 低于此字数，报告标注「不可靠」
MIN_UNIT_CHARS = 24           # 短于此的段落不独立判定，并入相邻段
MAX_CHUNK_CHARS = 300         # 分类器单窗口最大字符数（512 token 内安全）
BOOTSTRAP_ROUNDS = 1000

GRADE_LABELS = {
    "high": "高（疑似 AI）",
    "mid_high": "中高",
    "mid": "中",
    "low": "低",
}
