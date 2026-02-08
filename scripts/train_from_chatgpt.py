#!/usr/bin/env python3
"""
用ChatGPT标注的数据 + 词向量特征 训练意图分类模型

改进: 使用词向量(embedding)作为特征，而不是手工特征
这样模型能学到语义信息，准确率更高！
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pickle
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_chatgpt_labels(filepath: str) -> Dict[str, float]:
    """加载ChatGPT标注的数据 (支持多个JSON对象)"""
    import re

    with open(filepath) as f:
        content = f.read()

    labels = {}

    # 找到所有JSON对象
    json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)

    for json_str in json_objects:
        try:
            data = json.loads(json_str)
            for batch_name, batch_data in data.items():
                if isinstance(batch_data, list):
                    for item in batch_data:
                        word = item["word"].lower().strip()
                        score = float(item["score"])
                        labels[word] = score
        except json.JSONDecodeError:
            continue

    return labels


def get_embedding_model():
    """加载词向量模型"""
    try:
        from sentence_transformers import SentenceTransformer
        print("  加载 sentence-transformers 模型...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 384维，快速且效果好
        return model
    except ImportError:
        print("  安装 sentence-transformers...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model


def get_word_embeddings(words: List[str], model) -> np.ndarray:
    """获取词的embedding向量"""
    print(f"  生成 {len(words)} 个词的embedding...")
    embeddings = model.encode(words, show_progress_bar=True)
    return embeddings


def train_model_with_embeddings(
    labels: Dict[str, float],
    embedding_model,
) -> Tuple:
    """用词向量特征训练模型"""

    # 准备数据
    words = list(labels.keys())
    y = np.array([labels[w] for w in words])

    # 获取词向量
    X = get_word_embeddings(words, embedding_model)

    print(f"\n训练数据统计:")
    print(f"  总词数: {len(words)}")
    print(f"  特征维度: {X.shape[1]} (词向量)")
    print(f"  标签分布: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    # 按分数段统计
    print(f"\n分数分布:")
    print(f"  模糊 (0-0.3):   {sum(1 for s in y if s < 0.3)} 个")
    print(f"  中等 (0.3-0.7): {sum(1 for s in y if 0.3 <= s < 0.7)} 个")
    print(f"  明确 (0.7-1.0): {sum(1 for s in y if s >= 0.7)} 个")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练多个模型比较
    models = {
        "MLP-Small": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=1000, random_state=42, early_stopping=True
        ),
        "MLP-Large": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1500, random_state=42, early_stopping=True
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, random_state=42
        ),
    }

    print(f"\n模型训练与评估:")
    print("-" * 50)

    best_model = None
    best_score = -999
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        # 训练集评估
        y_pred_train = model.predict(X_train_scaled)
        train_r2 = r2_score(y_train, y_pred_train)

        # 测试集评估
        y_pred_test = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"  {name:20s} Train-R²={train_r2:.4f}  Test-R²={test_r2:.4f}  MSE={test_mse:.4f}")

        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_name = name

    print("-" * 50)
    print(f"  最佳模型: {best_name} (R²={best_score:.4f})")

    return best_model, scaler, embedding_model, labels, words


def save_model(model, scaler, labels: Dict[str, float], output_dir: Path):
    """保存模型"""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "intent_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "use_embeddings": True,  # 标记使用词向量
        }, f)

    with open(output_dir / "intent_labels.json", "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\n模型已保存到: {output_dir}")


def test_model(model, scaler, embedding_model, labels):
    """测试模型"""
    print("\n" + "=" * 60)
    print("模型测试")
    print("=" * 60)

    test_words = [
        # 模糊词 (应该预测低分)
        "apple", "python", "java", "bank", "cell", "go", "spring", "rust",
        # 中等 (应该预测中等分)
        "music", "movie", "computer", "phone", "game", "travel",
        # 明确 (应该预测高分)
        "iphone 15", "machine learning", "eiffel tower", "albert einstein",
        "neural network", "playstation 5", "tesla model 3",
        # 新词 (测试泛化能力)
        "tensorflow", "kubernetes", "docker", "chatgpt", "openai",
        "banana", "orange", "microsoft", "google", "facebook",
    ]

    # 获取embedding
    embeddings = embedding_model.encode(test_words)
    embeddings_scaled = scaler.transform(embeddings)
    predictions = model.predict(embeddings_scaled)

    print("\n预测结果:")
    print("-" * 70)
    print(f"{'词':<25} {'预测':<8} {'类别':<8} {'实际':<8}")
    print("-" * 70)

    for word, pred in zip(test_words, predictions):
        pred = float(np.clip(pred, 0, 1))
        actual = labels.get(word.lower(), None)

        # 判断类别
        if pred < 0.3:
            cat = "模糊"
        elif pred < 0.7:
            cat = "中等"
        else:
            cat = "明确"

        actual_str = f"{actual:.2f}" if actual is not None else "新词"
        print(f"  {word:<23} {pred:.2f}     {cat:<6}   {actual_str}")


def main():
    print("=" * 60)
    print("用词向量特征训练意图分类模型")
    print("=" * 60)

    # 加载数据
    data_file = Path("data/chatgpt_labels.txt")
    if not data_file.exists():
        print(f"错误: {data_file} 不存在")
        return

    print(f"\n[Step 1] 加载ChatGPT标注数据...")
    labels = load_chatgpt_labels(data_file)
    print(f"  成功加载 {len(labels)} 个词的标签")

    # 加载词向量模型
    print(f"\n[Step 2] 加载词向量模型...")
    embedding_model = get_embedding_model()

    # 训练
    print(f"\n[Step 3] 训练模型...")
    model, scaler, embedding_model, labels, words = train_model_with_embeddings(
        labels, embedding_model
    )

    # 保存
    print(f"\n[Step 4] 保存模型...")
    output_dir = Path("models/intent_chatgpt")
    save_model(model, scaler, labels, output_dir)

    # 测试
    test_model(model, scaler, embedding_model, labels)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"""
模型已保存到 models/intent_chatgpt/

改进说明:
- 使用词向量(384维)作为特征，而不是手工特征(17维)
- 词向量能捕捉语义信息，相似的词有相似的向量
- 模型能更好地泛化到未见过的词
""")


if __name__ == "__main__":
    main()
