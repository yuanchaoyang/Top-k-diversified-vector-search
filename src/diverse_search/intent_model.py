"""
训练好的意图分类模型 - 用于推理

使用词向量(embedding)作为特征，能更好地理解语义！

使用方法:
    from diverse_search.intent_model import IntentClassifier

    classifier = IntentClassifier.load("models/intent_v3")
    score = classifier.predict("apple")  # 返回 ~0.23 (模糊)
    score = classifier.predict("machine learning")  # 返回 ~0.85 (明确)
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class IntentClassifier:
    """意图分类器 - 预测查询词的意图明确度"""

    def __init__(
        self,
        model=None,
        scaler=None,
        label_cache: Dict[str, float] = None,
        use_embeddings: bool = False,
        embedding_model=None,
    ):
        self.model = model
        self.scaler = scaler
        self.label_cache = label_cache or {}
        self.use_embeddings = use_embeddings
        self._embedding_model = embedding_model

    @property
    def embedding_model(self):
        """懒加载embedding模型"""
        if self._embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not installed")
                self.use_embeddings = False
        return self._embedding_model

    @classmethod
    def load(cls, model_dir: str) -> "IntentClassifier":
        """从目录加载模型"""
        model_dir = Path(model_dir)

        model = None
        scaler = None
        use_embeddings = False

        # 加载模型
        model_path = model_dir / "intent_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                model = data.get("model")
                scaler = data.get("scaler")
                use_embeddings = data.get("use_embeddings", False)

        # 加载标签缓存
        labels_path = model_dir / "intent_labels.json"
        label_cache = {}
        if labels_path.exists():
            with open(labels_path, encoding='utf-8') as f:
                label_cache = json.load(f)

        return cls(
            model=model,
            scaler=scaler,
            label_cache=label_cache,
            use_embeddings=use_embeddings,
        )

    def predict(self, word: str) -> float:
        """预测意图分数 (0=模糊, 1=明确)"""
        word_lower = word.lower().strip()

        # 1. 先查缓存
        if word_lower in self.label_cache:
            return self.label_cache[word_lower]

        # 2. 用模型预测
        if self.model is not None and self.scaler is not None:
            if self.use_embeddings and self.embedding_model is not None:
                embedding = self.embedding_model.encode([word_lower])
                features_scaled = self.scaler.transform(embedding)
                score = self.model.predict(features_scaled)[0]
                return max(0.0, min(1.0, score))

        # 3. 默认返回中等分数
        return 0.5

    def predict_lambda(
        self,
        word: str,
        lambda_min: float = 0.4,
        lambda_max: float = 0.85
    ) -> Tuple[float, float]:
        """
        预测意图分数并转换为λ值

        Returns:
            (lambda_value, intent_score)
        """
        score = self.predict(word)
        lam = lambda_min + score * (lambda_max - lambda_min)
        return lam, score


# ============================================================================
# 便捷函数
# ============================================================================

_default_classifier: Optional[IntentClassifier] = None


def get_classifier(model_dir: str = "models/intent_v3") -> IntentClassifier:
    """获取或创建分类器实例"""
    global _default_classifier
    if _default_classifier is None:
        model_path = Path(model_dir)
        if (model_path / "intent_model.pkl").exists():
            _default_classifier = IntentClassifier.load(model_dir)
        else:
            _default_classifier = IntentClassifier()
    return _default_classifier


def predict_intent(word: str) -> float:
    """快速预测意图分数"""
    return get_classifier().predict(word)


def predict_lambda(word: str, lambda_min: float = 0.4, lambda_max: float = 0.85) -> float:
    """快速预测λ值"""
    lam, _ = get_classifier().predict_lambda(word, lambda_min, lambda_max)
    return lam
