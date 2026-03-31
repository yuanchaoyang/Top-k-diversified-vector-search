#!/usr/bin/env python3
"""
合并所有标注数据 (单词 + 句子 + 多样化查询), 重新训练 intent classifier.

数据来源:
  1. data/chatgpt_labels.txt          — 单词标注 (~3500 条)
  2. data/chatgpt_sentence_labels.txt  — 句子标注 (~200 条)
  3. data/chatgpt_diverse_labels.txt   — 多样化查询标注 (~300 条, 新增)

用法:
  python scripts/train_diverse_intent.py

前置:
  python scripts/generate_diverse_labels.py  # 生成 prompt
  # 用 ChatGPT 标注 → 保存到 data/chatgpt_diverse_labels.txt
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
    print(f"  标签: min={scores.min():.2f}, max={scores.max():.2f}, "
          f"mean={scores.mean():.2f}, std={scores.std():.2f}")
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
    """测试: 重点测试之前表现差的查询类型."""
    test_cases = [
        # 单词 - 模糊
        "apple", "python", "bank", "spring",
        # 单词 - 明确
        "photosynthesis", "kubernetes", "tensorflow",
        # 产品 + 型号 + 属性 (之前的弱项, 应该 0.85+)
        "macbook pro M4 price",
        "iPhone 16 Pro Max battery life",
        "RTX 4090 power consumption watts",
        "Tesla Model Y 2024 range miles",
        "PS5 Pro price USD",
        # 产品对比 (应该 0.70-0.85)
        "macbook pro vs dell xps 15",
        "iPhone 16 vs Samsung Galaxy S24",
        # 具体操作 (应该 0.75-0.90)
        "how to factory reset macbook pro",
        "how to install Python on Windows 11",
        # 宽泛查询 (应该 0.10-0.35)
        "how to be more productive",
        "tips for better life",
        "interesting stuff",
        # 具体事实 (应该 0.85-0.98)
        "who painted the Mona Lisa",
        "height of Mount Everest in meters",
        "boiling point of water in celsius",
        # 带限定词的多义词 (应该 0.65-0.85)
        "apple fruit nutrition facts",
        "python programming beginner tutorial",
        "jaguar car price 2024",
    ]

    embeddings = embedding_model.encode(test_cases)
    embeddings_s = scaler.transform(embeddings)
    preds = model.predict(embeddings_s)

    print("\n测试结果:")
    print("-" * 85)
    print(f"  {'文本':<45} {'预测':>6}  {'类别':<6}  {'标注':>6}")
    print("-" * 85)

    for text, pred in zip(test_cases, preds):
        pred = float(np.clip(pred, 0, 1))
        cat = "模糊" if pred < 0.3 else ("中等" if pred < 0.7 else "明确")
        actual = all_labels.get(text.lower(), None)
        actual_str = f"{actual:.2f}" if actual is not None else "  --"
        print(f"  {text:<45} {pred:>5.2f}   {cat:<6}  {actual_str}")


def main():
    label_files = [
        ("data/chatgpt_labels.txt", "单词标注"),
        ("data/chatgpt_sentence_labels.txt", "句子标注"),
        ("data/chatgpt_diverse_labels.txt", "多样化查询标注"),
        ("data/chatgpt_targeted_labels.txt", "针对性补充标注 (under-specified commercial + temporal/factual)"),
    ]
    output_dir = Path("models/intent_v3")

    print("=" * 70)
    print("训练 Intent Classifier v3 (单词 + 句子 + 多样化查询)")
    print("=" * 70)

    # 加载所有标注数据
    all_labels = {}
    for filepath, desc in label_files:
        p = Path(filepath)
        if p.exists():
            labels = load_labels_file(str(p))
            print(f"\n  [{desc}] {filepath}: {len(labels)} 条")
            all_labels.update(labels)
        else:
            print(f"\n  [{desc}] {filepath}: 不存在, 跳过")

    if not all_labels:
        print("\n错误: 没有加载到任何标注数据!")
        return

    print(f"\n合并后总计: {len(all_labels)} 条")

    # 训练
    print(f"\n加载 embedding 模型...")
    embedding_model = load_embedding_model()

    print(f"\n训练模型...")
    texts = list(all_labels.keys())
    scores = np.array([all_labels[t] for t in texts])
    model, scaler = train_and_evaluate(texts, scores, embedding_model)

    # 保存
    print(f"\n保存模型到 {output_dir}...")
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
    print(f"\n测试模型...")
    test_model(model, scaler, embedding_model, all_labels)

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")
    print(f"""
模型已保存到: {output_dir}/

下一步 - 更新 demo 使用新模型:
  修改 demo/search_api.py 中的模型加载顺序, 添加 "models/intent_v3":

    for model_dir in ["models/intent_v3", "models/intent_v2", "models/intent_chatgpt"]:

  然后重启 demo:
    python demo/search_api.py
""")


if __name__ == "__main__":
    main()
