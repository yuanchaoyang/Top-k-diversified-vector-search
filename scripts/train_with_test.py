#!/usr/bin/env python3
"""
训练意图分类模型 - 使用独立的训练集和测试集

训练集: chatgpt_labels.txt 
测试集: chatgpt_test (104词)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pickle
import re
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_chatgpt_labels(filepath: str) -> Dict[str, float]:
    """加载ChatGPT标注的数据 (支持多个JSON对象和嵌套结构)"""
    with open(filepath) as f:
        content = f.read()

    labels = {}

    def extract_items(obj):
        """递归提取所有带word和score的项"""
        if isinstance(obj, dict):
            if "word" in obj and "score" in obj:
                word = obj["word"].lower().strip()
                score = float(obj["score"])
                labels[word] = score
            else:
                for value in obj.values():
                    extract_items(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_items(item)

    # 尝试解析整个文件为JSON
    try:
        data = json.loads(content)
        extract_items(data)
    except json.JSONDecodeError:
        # 如果失败，尝试找多个JSON对象
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        for json_str in json_objects:
            try:
                data = json.loads(json_str)
                extract_items(data)
            except json.JSONDecodeError:
                continue

    return labels


def load_test_set(filepath: str) -> Dict[str, float]:
    """加载测试集"""
    with open(filepath) as f:
        content = f.read()

    labels = {}

    def extract_items(obj):
        if isinstance(obj, dict):
            if "word" in obj and "score" in obj:
                word = obj["word"].lower().strip()
                score = float(obj["score"])
                labels[word] = score
            else:
                for value in obj.values():
                    extract_items(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_items(item)

    data = json.loads(content)
    extract_items(data)

    return labels


def get_category(score):
    """分数转类别"""
    if score < 0.3:
        return "ambiguous"
    elif score < 0.7:
        return "moderate"
    else:
        return "specific"


def count_by_category(scores):
    """统计各类别数量"""
    amb = sum(1 for s in scores if s < 0.3)
    mod = sum(1 for s in scores if 0.3 <= s < 0.7)
    spe = sum(1 for s in scores if s >= 0.7)
    return amb, mod, spe


def main():
    print("=" * 70)
    print("Intent Model Training with Train/Test Split")
    print("=" * 70)

    # 1. 加载训练集
    print("\n[Step 1] Loading training data...")

    train_labels = load_chatgpt_labels("data/chatgpt_labels.txt")
    print(f"  chatgpt_labels.txt: {len(train_labels)} words")

    # 2. 加载测试集
    print("\n[Step 2] Loading test data...")
    test_labels = load_test_set("data/chatgpt_test.txt")
    print(f"  Test set: {len(test_labels)} words")

    # 检查是否有重复
    overlap = set(train_labels.keys()) & set(test_labels.keys())
    if overlap:
        print(f"  WARNING: {len(overlap)} words overlap between train and test!")
        print(f"    Overlapping words: {list(overlap)[:10]}...")
        # 从测试集中移除重复的词
        for w in overlap:
            del test_labels[w]
        print(f"  Removed overlaps. Test set now: {len(test_labels)} words")

    # 3. 统计分布
    train_scores = list(train_labels.values())
    test_scores = list(test_labels.values())

    train_amb, train_mod, train_spe = count_by_category(train_scores)
    test_amb, test_mod, test_spe = count_by_category(test_scores)

    print(f"\n  Training set distribution:")
    print(f"    Ambiguous (0-0.3):   {train_amb}")
    print(f"    Moderate (0.3-0.7):  {train_mod}")
    print(f"    Specific (0.7-1.0):  {train_spe}")

    print(f"\n  Test set distribution:")
    print(f"    Ambiguous (0-0.3):   {test_amb}")
    print(f"    Moderate (0.3-0.7):  {test_mod}")
    print(f"    Specific (0.7-1.0):  {test_spe}")

    # 4. 生成词向量
    print("\n[Step 3] Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    words_train = list(train_labels.keys())
    words_test = list(test_labels.keys())
    y_train = np.array([train_labels[w] for w in words_train])
    y_test = np.array([test_labels[w] for w in words_test])

    print("\n[Step 4] Generating embeddings...")
    X_train = embedding_model.encode(words_train, show_progress_bar=True)
    X_test = embedding_model.encode(words_test, show_progress_bar=True)

    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")

    # 5. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. 训练模型
    print("\n[Step 5] Training models...")
    print("-" * 70)

    # 计算样本权重 (平衡类别)
    print("\n  Computing sample weights for class balance...")
    sample_weights = np.ones(len(y_train))

    # 统计各类别数量
    n_amb = sum(1 for y in y_train if y < 0.3)
    n_mod = sum(1 for y in y_train if 0.3 <= y < 0.7)
    n_spe = sum(1 for y in y_train if y >= 0.7)
    n_total = len(y_train)

    # 计算权重 (少数类获得更高权重)
    w_amb = n_total / (3 * n_amb) if n_amb > 0 else 1
    w_mod = n_total / (3 * n_mod) if n_mod > 0 else 1
    w_spe = n_total / (3 * n_spe) if n_spe > 0 else 1

    for i, y in enumerate(y_train):
        if y < 0.3:
            sample_weights[i] = w_amb
        elif y < 0.7:
            sample_weights[i] = w_mod
        else:
            sample_weights[i] = w_spe

    print(f"    Ambiguous weight: {w_amb:.2f}")
    print(f"    Moderate weight:  {w_mod:.2f}")
    print(f"    Specific weight:  {w_spe:.2f}")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            random_state=42
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), max_iter=2000,
            random_state=42, early_stopping=True, validation_fraction=0.1
        ),
    }

    # 标记支持sample_weight的模型
    supports_weights = {"RandomForest", "GradientBoosting"}

    best_model = None
    best_acc = 0
    best_name = ""
    results = []

    for name, model in models.items():
        if name in supports_weights:
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_scaled, y_train)

        # 训练集评估
        y_pred_train = np.clip(model.predict(X_train_scaled), 0, 1)
        train_r2 = r2_score(y_train, y_pred_train)
        train_cat = [get_category(y) for y in y_train]
        pred_cat_train = [get_category(y) for y in y_pred_train]
        train_acc = sum(1 for t, p in zip(train_cat, pred_cat_train) if t == p) / len(train_cat)

        # 测试集评估
        y_pred_test = np.clip(model.predict(X_test_scaled), 0, 1)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        test_cat = [get_category(y) for y in y_test]
        pred_cat_test = [get_category(y) for y in y_pred_test]
        test_acc = sum(1 for t, p in zip(test_cat, pred_cat_test) if t == p) / len(test_cat)

        print(f"\n  {name}:")
        print(f"    Train R²: {train_r2:.4f}  Train Accuracy: {train_acc*100:.1f}%")
        print(f"    Test R²:  {test_r2:.4f}  Test Accuracy:  {test_acc*100:.1f}%")
        print(f"    Test MSE: {test_mse:.4f}  Test MAE: {test_mae:.4f}")

        results.append({
            "name": name,
            "train_r2": train_r2,
            "train_acc": train_acc,
            "test_r2": test_r2,
            "test_acc": test_acc,
            "test_mse": test_mse,
            "predictions": list(zip(words_test, y_test, y_pred_test))
        })

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model
            best_name = name
            best_result = results[-1]

    print("\n" + "-" * 70)
    print(f"  Best Model: {best_name}")
    print(f"    Test Accuracy: {best_acc*100:.1f}%")
    print(f"    Test R²: {best_result['test_r2']:.4f}")

    # 7. 保存模型
    print("\n[Step 6] Saving model...")
    output_dir = Path("models/intent_chatgpt")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "intent_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": scaler,
            "use_embeddings": True,
        }, f)

    with open(output_dir / "intent_labels.json", "w") as f:
        json.dump(train_labels, f, indent=2, ensure_ascii=False)

    # 保存测试集结果
    with open("data/test_results.json", "w") as f:
        json.dump({
            "model": best_name,
            "test_accuracy": best_acc,
            "test_r2": best_result["test_r2"],
            "predictions": [
                {"word": w, "true": float(t), "pred": float(p),
                 "true_cat": get_category(t), "pred_cat": get_category(p)}
                for w, t, p in best_result["predictions"]
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"  Saved model to: {output_dir}")
    print(f"  Saved test results to: data/test_results.json")

    # 8. 详细测试结果
    print("\n" + "=" * 70)
    print("TEST SET DETAILED RESULTS")
    print("=" * 70)

    predictions = best_result["predictions"]

    # 按类别统计准确率
    by_category = {"ambiguous": [], "moderate": [], "specific": []}
    for word, true, pred in predictions:
        cat = get_category(true)
        by_category[cat].append((word, true, pred, get_category(pred)))

    for cat in ["ambiguous", "moderate", "specific"]:
        items = by_category[cat]
        correct = sum(1 for _, _, _, pred_cat in items if pred_cat == cat)
        print(f"\n  {cat.upper()} ({len(items)} words): {correct}/{len(items)} = {correct/len(items)*100:.1f}%")

        # 显示错误案例
        errors = [(w, t, p, pc) for w, t, p, pc in items if pc != cat]
        if errors:
            print(f"    Errors:")
            for w, t, p, pc in errors[:5]:
                print(f"      {w}: true={t:.2f}({cat}) pred={p:.2f}({pc})")

    # 9. 总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Dataset:
    Training set:     {len(train_labels)} words
    Test set:         {len(test_labels)} words

  Best Model: {best_name}
    Test Accuracy:    {best_acc*100:.1f}%
    Test R²:          {best_result['test_r2']:.4f}
    Test MSE:         {best_result['test_mse']:.4f}

  Files saved:
    models/intent_chatgpt/intent_model.pkl
    models/intent_chatgpt/intent_labels.json
    data/test_results.json
""")


if __name__ == "__main__":
    main()
