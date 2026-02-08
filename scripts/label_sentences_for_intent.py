#!/usr/bin/env python3
"""
第一步: 从 MS MARCO 查询中采样, 生成 ChatGPT 标注 prompt.

用法:
  python scripts/label_sentences_for_intent.py

输出:
  1. 打印 prompt (复制到 ChatGPT 对话框)
  2. 你把 ChatGPT 的回复粘贴保存到 data/chatgpt_sentence_labels.txt

然后运行第二步:
  python scripts/train_sentence_intent.py
"""

from __future__ import annotations

import random
from pathlib import Path


def sample_diverse_queries(queries_file: str, n: int = 200) -> list[str]:
    """从查询集中采样多样化的查询."""
    with open(queries_file) as f:
        all_queries = [line.strip() for line in f if line.strip()]

    # 按长度分桶，确保短查询和长查询都有
    short = [q for q in all_queries if len(q.split()) <= 3]
    medium = [q for q in all_queries if 4 <= len(q.split()) <= 8]
    long = [q for q in all_queries if len(q.split()) > 8]

    random.seed(42)
    sampled = []
    for bucket, count in [(short, n // 4), (medium, n // 2), (long, n // 4)]:
        sampled.extend(random.sample(bucket, min(count, len(bucket))))

    # 补足
    remaining = [q for q in all_queries if q not in set(sampled)]
    if len(sampled) < n:
        sampled.extend(random.sample(remaining, min(n - len(sampled), len(remaining))))

    return sampled[:n]


def generate_prompt(queries: list[str], batch_size: int = 50) -> list[str]:
    """生成 ChatGPT 标注用的 prompt."""
    prompts = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_num = i // batch_size + 1

        queries_list = "\n".join(f"{j+1}. {q}" for j, q in enumerate(batch))

        prompt = f"""I have {len(batch)} search queries. Please rate each query's "intent clarity" from 0 to 1:

- 0.0-0.2: Very ambiguous / broad (the user's need is unclear, many possible interpretations)
  Examples: "rock" (music? stone?), "how to get better", "things to do"
- 0.3-0.4: Somewhat ambiguous (multiple reasonable interpretations)
  Examples: "apple products", "best treatment", "python tutorial"
- 0.5-0.6: Moderately specific (general topic is clear but details vary)
  Examples: "how does photosynthesis work", "cost of living in new york"
- 0.7-0.8: Specific (clear intent, limited interpretations)
  Examples: "what year was the eiffel tower built", "symptoms of type 2 diabetes"
- 0.9-1.0: Very specific (single clear answer expected)
  Examples: "boiling point of water in celsius", "who wrote hamlet"

Queries:
{queries_list}

Return ONLY a JSON object in this exact format:
{{"batch{batch_num}": [
  {{"word": "the full query text", "score": 0.X, "reason": "brief explanation"}},
  ...
]}}"""
        prompts.append(prompt)

    return prompts


def main():
    queries_file = Path("data/msmarco/msmarco_queries.txt")

    if not queries_file.exists():
        print(f"错误: {queries_file} 不存在")
        print("请先运行: python scripts/prepare_msmarco.py")
        return

    print("=" * 70)
    print("生成句子级 Intent 标注 Prompt")
    print("=" * 70)

    # 采样查询
    queries = sample_diverse_queries(str(queries_file), n=200)
    print(f"\n采样了 {len(queries)} 条查询")
    print(f"  短查询 (<=3词): {sum(1 for q in queries if len(q.split()) <= 3)}")
    print(f"  中查询 (4-8词): {sum(1 for q in queries if 4 <= len(q.split()) <= 8)}")
    print(f"  长查询 (>8词):  {sum(1 for q in queries if len(q.split()) > 8)}")

    # 保存采样的查询 (方便检查)
    sampled_file = Path("data/msmarco/sampled_queries_for_labeling.txt")
    with open(sampled_file, "w") as f:
        for q in queries:
            f.write(q + "\n")
    print(f"\n采样查询已保存到: {sampled_file}")

    # 生成 prompt
    prompts = generate_prompt(queries, batch_size=50)

    print(f"\n生成了 {len(prompts)} 个 prompt (每个 50 条查询)")
    print("\n" + "=" * 70)
    print("请把以下每个 PROMPT 复制到 ChatGPT, 收集回复")
    print("然后把所有回复合并保存到: data/chatgpt_sentence_labels.txt")
    print("=" * 70)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"PROMPT {i+1}/{len(prompts)}")
        print("=" * 70)
        print(prompt)

    print(f"\n{'='*70}")
    print("完成后运行:")
    print("  python scripts/train_sentence_intent.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
