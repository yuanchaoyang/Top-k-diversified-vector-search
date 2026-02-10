#!/usr/bin/env python3
"""
调试Intent分类器 - 检查clarity值的来源
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diverse_search.intent_model import IntentClassifier


def debug_intent():
    """调试intent模型"""

    print("=" * 60)
    print("Intent分类器调试")
    print("=" * 60)

    # 加载模型
    model_path = Path("models/intent_v3")
    if not model_path.exists():
        print(f"❌ 模型目录不存在: {model_path}")
        return

    classifier = IntentClassifier.load(str(model_path))

    print(f"\n模型信息:")
    print(f"  有模型: {classifier.model is not None}")
    print(f"  有scaler: {classifier.scaler is not None}")
    print(f"  使用embeddings: {classifier.use_embeddings}")
    print(f"  缓存词数: {len(classifier.label_cache)}")

    # 测试关键词
    test_words = [
        "apple",
        "Apple",
        "APPLE",
        "bank",
        "python",
        "Python",
        "mercury",
        "Einstein",
        "president",
        "what is machine learning",
        "who is the president",
    ]

    print("\n" + "=" * 60)
    print("测试查询的clarity值")
    print("=" * 60)
    print(f"{'查询':<30} | {'Clarity':<8} | {'λ (0.5-0.9)':<12} | 来源")
    print("-" * 60)

    for word in test_words:
        # 预测
        intent_score = classifier.predict(word)
        lam, _ = classifier.predict_lambda(word, lambda_min=0.5, lambda_max=0.9)

        # 检查来源
        word_lower = word.lower().strip()
        if word_lower in classifier.label_cache:
            source = f"缓存 ({classifier.label_cache[word_lower]:.2f})"
        elif classifier.model is not None:
            source = "模型预测"
        else:
            source = "默认值"

        clarity_pct = intent_score * 100
        print(f"{word:<30} | {clarity_pct:>6.1f}% | {lam:>6.2f}      | {source}")

    # 显示缓存中的示例
    print("\n" + "=" * 60)
    print("缓存中的样例（前20个）")
    print("=" * 60)

    cache_items = list(classifier.label_cache.items())[:20]
    for word, score in cache_items:
        clarity_pct = score * 100
        print(f"  {word:<20} → {clarity_pct:>6.1f}%")

    # 检查特定词
    print("\n" + "=" * 60)
    print("歧义词检查")
    print("=" * 60)

    ambiguous = ["apple", "bank", "python", "mercury", "light", "run", "set"]

    for word in ambiguous:
        word_lower = word.lower()
        if word_lower in classifier.label_cache:
            score = classifier.label_cache[word_lower]
            clarity_pct = score * 100

            # 评估是否合理
            if clarity_pct < 40:
                assessment = "✅ 合理（歧义→低clarity）"
            elif clarity_pct > 60:
                assessment = "❌ 不合理（歧义词应该低clarity）"
            else:
                assessment = "⚠️ 中等（可能偏高）"

            print(f"  {word:<15} {clarity_pct:>6.1f}%  {assessment}")
        else:
            print(f"  {word:<15} {'缓存中无':>6}  ⚠️ 未缓存")


if __name__ == "__main__":
    debug_intent()
