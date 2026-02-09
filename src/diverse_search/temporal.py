"""
时间敏感性查询识别 + 新鲜度感知重排

思路：
1. 用规则检测查询中的时间敏感信号（关键词、年份、角色查询等）
2. 时间敏感查询 → 给新鲜内容加分（β > 0）
3. 非时间敏感查询 → 不加分（β = 0），完全不影响原有逻辑

示例：
- "latest iPhone price" → 时间敏感 → β=0.12, 新鲜内容加分
- "how does photosynthesis work" → 永恒话题 → β=0, 无变化

与意图分类器配合：
- 强时间敏感查询会小幅增加λ（偏重相关+新鲜，而非多样+过时）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TemporalConfig:
    """时间敏感性配置"""
    beta_min: float = 0.0    # 非时间敏感查询的β
    beta_max: float = 0.15   # 强时间敏感查询的β
    # 各来源的新鲜度分数（0=旧, 1=新）
    freshness_by_source: Dict[str, float] = field(default_factory=lambda: {
        "wiki": 0.7,     # Wikipedia 2025年版本，较新
        "msmarco": 0.3,  # MS MARCO 较旧的 Bing 搜索结果
    })


# 模块级缓存，同 INTENT_DICTIONARY 模式
TEMPORAL_CACHE: Dict[str, Tuple[float, str]] = {}


# --- 关键词信号表 ---

# 强时间信号 (score ~0.85)
_STRONG_KEYWORDS = {
    "now", "current", "currently", "latest", "today", "tonight",
    "2025", "2026", "breaking", "updated", "just",
}

# 中等时间信号 (score ~0.5)
_MODERATE_KEYWORDS = {
    "new", "newest", "trending", "price", "cost", "worth",
    "recently", "upcoming", "modern",
}

# 正则模式
_YEAR_PATTERN = re.compile(r"\b20[2-3]\d\b")
_ROLE_PATTERN = re.compile(
    r"\bwho\s+is\b.*?\b(?:president|ceo|king|queen|prime\s+minister|mayor|governor|chairman)\b",
    re.IGNORECASE,
)
_TIME_PHRASE_PATTERN = re.compile(
    r"\b(?:this\s+(?:year|month|week)|right\s+now|at\s+the\s+moment|these\s+days)\b",
    re.IGNORECASE,
)


def classify_temporal(query: str) -> Tuple[float, str]:
    """
    检测查询的时间敏感性

    Returns:
        (temporal_score, explanation): 0=永恒话题, 1=强时间敏感
    """
    query_lower = query.lower().strip()

    # 先检查缓存
    if query_lower in TEMPORAL_CACHE:
        return TEMPORAL_CACHE[query_lower]

    signals = []
    words = set(query_lower.split())

    # 强关键词匹配
    strong_hits = words & _STRONG_KEYWORDS
    if strong_hits:
        signals.append((0.85, f"strong: {', '.join(sorted(strong_hits))}"))

    # 中等关键词匹配
    moderate_hits = words & _MODERATE_KEYWORDS
    if moderate_hits:
        signals.append((0.5, f"moderate: {', '.join(sorted(moderate_hits))}"))

    # 年份引用
    year_match = _YEAR_PATTERN.search(query_lower)
    if year_match:
        signals.append((0.85, f"year: {year_match.group()}"))

    # 角色查询 ("who is the president of ...")
    if _ROLE_PATTERN.search(query_lower):
        signals.append((0.75, "role query"))

    # 时间短语 ("this year", "right now")
    time_match = _TIME_PHRASE_PATTERN.search(query_lower)
    if time_match:
        signals.append((0.8, f"time phrase: {time_match.group()}"))

    if not signals:
        result = (0.0, "no temporal signals")
        TEMPORAL_CACHE[query_lower] = result
        return result

    # Max-pool across signals, slight boost for multiple signals
    max_score = max(s[0] for s in signals)
    multi_boost = min(0.1, 0.03 * (len(signals) - 1))
    score = min(1.0, max_score + multi_boost)

    explanation = "; ".join(s[1] for s in signals)
    result = (score, explanation)
    TEMPORAL_CACHE[query_lower] = result
    return result


def temporal_beta(
    query: str,
    config: Optional[TemporalConfig] = None,
) -> Tuple[float, float, str]:
    """
    计算时间敏感性β值

    Returns:
        (beta, temporal_score, explanation)
    """
    if config is None:
        config = TemporalConfig()

    temporal_score, explanation = classify_temporal(query)

    # 线性插值
    beta = config.beta_min + temporal_score * (config.beta_max - config.beta_min)

    return beta, temporal_score, explanation


# --- 事实型查询模式 ---
# 这类查询意图非常明确，不需要多样性，ML 模型容易低估

_FACTUAL_PATTERNS: list[Tuple[re.Pattern, float, str]] = [
    # "who is/was ..." → 找特定人物
    (re.compile(r"^who\s+(?:is|was|are|were)\b", re.I), 0.85, "who-is query"),
    # "what is/are ..." → 找定义/事实
    (re.compile(r"^what\s+(?:is|are|was|were)\b", re.I), 0.80, "what-is query"),
    # "when did/was ..." → 找具体时间
    (re.compile(r"^when\s+(?:did|was|is|will)\b", re.I), 0.85, "when query"),
    # "where is/was ..." → 找具体地点
    (re.compile(r"^where\s+(?:is|are|was|were)\b", re.I), 0.80, "where query"),
    # "how many/much/old/tall ..." → 找具体数值
    (re.compile(r"^how\s+(?:many|much|old|tall|long|far|fast)\b", re.I), 0.85, "how-many query"),
    # "名词 + of + 名词" 式的具体实体查询
    (re.compile(r"(?:president|capital|population|ceo|founder|author)\s+of\b", re.I), 0.80, "entity-of query"),
]


def classify_factual(query: str) -> Tuple[float, str]:
    """
    检测查询是否为事实型（有明确唯一答案）

    Returns:
        (specificity_score, explanation): 0=探索性, 0.8+=事实型
    """
    for pattern, score, label in _FACTUAL_PATTERNS:
        if pattern.search(query):
            return score, label
    return 0.0, ""


def query_aware_lambda_and_beta(
    query: str,
    intent_classifier,
    temporal_config: Optional[TemporalConfig] = None,
    lambda_min: float = 0.3,
    lambda_max: float = 0.9,
) -> Dict:
    """
    综合意图分析 + 事实型识别 + 时间敏感性，返回λ和β

    三层信号叠加:
    1. ML 意图分类器 → 基础 intent_score
    2. 事实型查询规则 → 兜底提升 intent_score (ML 容易低估问句)
    3. 时间敏感性 → β (新鲜度加分) + λ 提升 (偏重相关性)

    Args:
        query: 查询文本
        intent_classifier: IntentClassifier 实例 (需要 predict_lambda 方法)
        temporal_config: 时间敏感性配置
        lambda_min: 最小λ
        lambda_max: 最大λ

    Returns:
        dict with: lambda, beta, intent_score, temporal_score, temporal_explanation
    """
    if temporal_config is None:
        temporal_config = TemporalConfig()

    # 1) ML 意图分析 → 基础 intent_score
    lam, intent_score = intent_classifier.predict_lambda(query, lambda_min, lambda_max)

    # 2) 事实型查询规则兜底: 如果是明确事实问句, 保证 intent_score 不低于阈值
    factual_score, factual_label = classify_factual(query)
    if factual_score > intent_score:
        intent_score = factual_score
        # 重新计算 λ
        lam = lambda_min + intent_score * (lambda_max - lambda_min)

    # 3) 时间敏感性 → β
    beta, temporal_score, temporal_explanation = temporal_beta(query, temporal_config)

    # 时间敏感 + 事实型查询: 更激进地拉高 λ (要相关+新鲜, 不要多样+过时)
    if temporal_score > 0.5:
        boost = 0.1 * temporal_score  # 最高 +0.1 (之前是 0.05)
        lam = min(lambda_max, lam + boost)

    # 补充事实型标签到 explanation
    if factual_label:
        temporal_explanation = (
            f"{factual_label}; {temporal_explanation}" if temporal_explanation
            else factual_label
        )

    return {
        "lambda": float(lam),
        "beta": float(beta),
        "intent_score": float(intent_score),
        "temporal_score": float(temporal_score),
        "temporal_explanation": temporal_explanation,
    }


def generate_freshness_scores(
    sources: list[str],
    config: Optional[TemporalConfig] = None,
) -> np.ndarray:
    """
    根据来源标签生成新鲜度分数数组

    Args:
        sources: 每条passage的来源标签 (e.g. ["wiki", "msmarco", ...])
        config: 配置 (包含 freshness_by_source 映射)

    Returns:
        (n,) float32 数组, 值在 [0, 1]
    """
    if config is None:
        config = TemporalConfig()

    default_freshness = 0.5
    freshness = np.array(
        [config.freshness_by_source.get(s, default_freshness) for s in sources],
        dtype=np.float32,
    )
    return freshness
