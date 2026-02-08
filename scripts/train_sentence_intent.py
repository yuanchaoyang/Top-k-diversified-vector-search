#!/usr/bin/env python3
"""
第二步: 合并单词+句子标注数据, 重新训练 intent classifier.

流程:
  1. 加载原有单词标注 (data/chatgpt_labels.txt)
  2. 加载句子标注 (data/chatgpt_sentence_labels.txt)
  3. 合并训练
  4. 保存到 models/intent_v2/

用法:
  python scripts/train_sentence_intent.py

前置:
  python scripts/label_sentences_for_intent.py  # 生成 prompt
  # 用 ChatGPT 标注 → 保存到 data/chatgpt_sentence_labels.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def load_labels_file(filepath: str) -> Dict[str, float]:
    """加载标注文件 (支持多个 JSON 对象)."""
    with open(filepath) as f:
        content = f.read()

    labels = {}
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)

    for json_str in json_objects:
        try:
            data = json.loads(json_str)
            for batch_name, batch_data in data.items():
                if isinstance(batch_data, list):
                    for item in batch_data:
                        text = item["word"].lower().strip()
                        score = float(item["score"])
                        labels[text] = score
        except (json.JSONDecodeError, KeyError):
            continue

    return labels


def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    print("  加载 sentence-transformers 模型...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def train_and_evaluate(
    texts: List[str],
    scores: np.ndarray,
    embedding_model,
) -> Tuple:
    """训练模型."""
    print(f"\n  编码 {len(texts)} 条文本...")
    X = embedding_model.encode(texts, show_progress_bar=True)

    print(f"\n训练数据统计:")
    print(f"  总数: {len(texts)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  标签: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}, std={scores.std():.2f}")
    print(f"\n分数分布:")
    print(f"  模糊 (0-0.3):   {sum(1 for s in scores if s < 0.3)}")
    print(f"  中等 (0.3-0.7): {sum(1 for s in scores if 0.3 <= s < 0.7)}")
    print(f"  明确 (0.7-1.0): {sum(1 for s in scores if s >= 0.7)}")

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, scores, test_size=0.2, random_state=42,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "MLP": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1500, random_state=42, early_stopping=True,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, random_state=42,
        ),
    }

    print(f"\n模型训练:")
    print("-" * 55)

    best_model = None
    best_r2 = -999
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred_train = model.predict(X_train_s)
        y_pred_test = model.predict(X_test_s)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        print(f"  {name:20s} Train-R²={train_r2:.4f}  Test-R²={test_r2:.4f}  MSE={test_mse:.4f}")

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model = model
            best_name = name

    print("-" * 55)
    print(f"  最佳: {best_name} (R²={best_r2:.4f})")

    return best_model, scaler


def test_model(model, scaler, embedding_model, all_labels: Dict[str, float]):
    """测试: 单词 + 句子都要测."""
    test_cases = [
        # 单词 - 模糊
        "apple", "python", "bank", "spring",
        # 单词 - 明确
        "photosynthesis", "iphone 15", "machine learning",
        # 句子 - 模糊/宽泛
        "how to get better at life",
        "things to know",
        "what is the meaning of love",
        "best ways to improve",
        # 句子 - 中等
        "how does photosynthesis work",
        "what causes climate change",
        "python tutorial for beginners",
        # 句子 - 明确
        "what year was the eiffel tower built",
        "boiling point of water in celsius",
        "who wrote romeo and juliet",
        "what is the population of tokyo",
        "how many calories in a banana",
    ]

    embeddings = embedding_model.encode(test_cases)
    embeddings_s = scaler.transform(embeddings)
    preds = model.predict(embeddings_s)

    print("\n测试结果:")
    print("-" * 80)
    print(f"  {'文本':<45} {'预测':>6}  {'类别':<6}  {'标注':>6}")
    print("-" * 80)

    for text, pred in zip(test_cases, preds):
        pred = float(np.clip(pred, 0, 1))
        cat = "模糊" if pred < 0.3 else ("中等" if pred < 0.7 else "明确")
        actual = all_labels.get(text.lower(), None)
        actual_str = f"{actual:.2f}" if actual is not None else "  --"
        print(f"  {text:<45} {pred:>5.2f}   {cat:<6}  {actual_str}")


def main():
    word_labels_file = Path("data/chatgpt_labels.txt")
    sentence_labels_file = Path("data/chatgpt_sentence_labels.txt")
    output_dir = Path("models/intent_v2")

    print("=" * 70)
    print("训练句子级 Intent Classifier (v2)")
    print("=" * 70)

    # 加载单词标注
    word_labels = {}
    if word_labels_file.exists():
        print(f"\n[1] 加载单词标注: {word_labels_file}")
        word_labels = load_labels_file(str(word_labels_file))
        print(f"  单词标注: {len(word_labels)} 条")
    else:
        print(f"\n[1] 跳过 (文件不存在: {word_labels_file})")

    # 加载句子标注
    sentence_labels = {}
    if sentence_labels_file.exists():
        print(f"\n[2] 加载句子标注: {sentence_labels_file}")
        sentence_labels = load_labels_file(str(sentence_labels_file))
        print(f"  句子标注: {len(sentence_labels)} 条")
    else:
        print(f"\n[2] 错误: {sentence_labels_file} 不存在!")
        print("   请先运行:")
        print("     python scripts/label_sentences_for_intent.py")
        print("   然后用 ChatGPT 标注, 保存回复到 data/chatgpt_sentence_labels.txt")
        return

    # 合并
    all_labels = {**word_labels, **sentence_labels}
    print(f"\n[3] 合并: {len(word_labels)} 单词 + {len(sentence_labels)} 句子 = {len(all_labels)} 条")

    # 训练
    print(f"\n[4] 加载 embedding 模型...")
    embedding_model = load_embedding_model()

    print(f"\n[5] 训练模型...")
    texts = list(all_labels.keys())
    scores = np.array([all_labels[t] for t in texts])
    model, scaler = train_and_evaluate(texts, scores, embedding_model)

    # 保存
    print(f"\n[6] 保存模型到 {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "intent_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "use_embeddings": True,
        }, f)

    with open(output_dir / "intent_labels.json", "w") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    print(f"  保存完成")

    # 测试
    print(f"\n[7] 测试模型...")
    test_model(model, scaler, embedding_model, all_labels)

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")
    print(f"""
模型已保存到: {output_dir}/

下一步 - 用新模型重新评估:

  PYTHONPATH=src python scripts/evaluate_msmarco.py \\
    --data-dir data/msmarco \\
    --model-dir {output_dir} \\
    --k 10 --topN 100 \\
    --out-dir outputs/msmarco_eval_v2
""")


if __name__ == "__main__":
    main()
