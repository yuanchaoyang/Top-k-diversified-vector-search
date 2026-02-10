#!/usr/bin/env python3
"""
构建改进的混合语料库

混合策略（50-60k段落）：
- A. 定义性百科片段 (~15k): 维基百科首段
- B. 实体卡片 (~26k): Wikidata描述
- C. 词典/词义 (~8-12k): WordNet定义
- D. 消歧义路由器 (~2-5k): 维基消歧义页面
- E. 网页/问答 (~3-8k): LoTTE/BEIR数据集

使用方法:
    python scripts/build_improved_corpus.py --output data/improved
"""

import re
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

# 确保能导入datasets
try:
    from datasets import load_dataset
except ImportError:
    print("正在安装 datasets 库...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "-q"])
    from datasets import load_dataset

# 确保能导入sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("正在安装 sentence-transformers 库...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer


def normalize_title(s: str) -> str:
    """标准化标题用于匹配"""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()


def clip_chars(text: str, lo=80, hi=600) -> Optional[str]:
    """裁剪文本到指定长度范围"""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < lo:
        return None
    if len(text) > hi:
        # 在单词边界处截断
        text = text[:hi].rsplit(" ", 1)[0] + "…"
    return text


def lead_paragraph(wiki_text: str) -> str:
    """提取首段（粗略但有效）"""
    # 按双换行或多个换行分段
    parts = [p.strip() for p in re.split(r'\n\n+', wiki_text) if p.strip()]
    if not parts:
        # 退化到单换行
        parts = [p.strip() for p in wiki_text.split("\n") if p.strip()]
    return parts[0] if parts else ""


class CorpusBuilder:
    """混合语料库构建器"""

    def __init__(self, output_dir: str, max_passages: int = 60000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_passages = max_passages
        self.passages = []

    def build_seed_lists(self) -> tuple:
        """构建种子列表：高频词 + 热门实体"""
        print("\n=== 步骤1: 构建种子列表 ===")

        # A) 高频词列表
        print("加载高频词列表...")
        try:
            freq_ds = load_dataset("Maximax67/English-Valid-Words", split="train", trust_remote_code=True)
            # 检查列名
            print(f"  列名: {freq_ds.column_names}")

            # 适配不同的列名
            if "word" in freq_ds.column_names:
                word_col = "word"
            elif "token" in freq_ds.column_names:
                word_col = "token"
            else:
                word_col = freq_ds.column_names[0]

            # 提取前10k词
            common_words = [row[word_col] for row in list(freq_ds)[:10000]]
            print(f"  获取了 {len(common_words)} 个高频词")
        except Exception as e:
            print(f"  加载高频词失败: {e}")
            print("  使用备用词表...")
            # 备用：常见英语词汇
            common_words = self._get_fallback_common_words()

        # B) 热门实体标题
        print("加载Wikidata实体列表...")
        try:
            wd_desc = load_dataset("masaki-sakata/wikidata_descriptions", split="en", trust_remote_code=True)
            print(f"  列名: {wd_desc.column_names}")

            popular_titles = [row["wiki_title"] for row in wd_desc if "wiki_title" in row]
            print(f"  获取了 {len(popular_titles)} 个实体标题")
        except Exception as e:
            print(f"  加载Wikidata失败: {e}")
            popular_titles = []

        return common_words, popular_titles

    def _get_fallback_common_words(self) -> List[str]:
        """备用常见词列表"""
        return [
            "time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand",
            "part", "child", "eye", "woman", "place", "work", "week", "case", "point", "government",
            "company", "number", "group", "problem", "fact", "water", "food", "money", "music",
            "light", "sound", "color", "word", "sentence", "book", "paper", "letter", "name",
            "idea", "question", "answer", "reason", "story", "example", "system", "program",
            "apple", "bank", "president", "country", "city", "state", "church", "school",
            "student", "teacher", "doctor", "lawyer", "engineer", "artist", "writer", "singer"
        ] * 150  # 重复以达到足够数量

    def build_entity_cards(self, popular_titles: List[str]) -> List[Dict]:
        """Bucket B: 实体卡片 (~26k)"""
        print("\n=== 步骤2: 构建实体卡片 ===")
        entity_passages = []

        try:
            wd_desc = load_dataset("masaki-sakata/wikidata_descriptions", split="en", trust_remote_code=True)

            for row in tqdm(wd_desc, desc="处理Wikidata实体"):
                title = row.get("wiki_title", "").strip()
                desc = row.get("description", "").strip()

                if not title or not desc:
                    continue

                # 格式: "标题 — 描述."
                txt = clip_chars(f"{title} — {desc}.", lo=40, hi=300)
                if txt:
                    entity_passages.append({
                        "text": txt,
                        "source": "wikidata_desc",
                        "title": title
                    })

            print(f"  生成了 {len(entity_passages)} 个实体卡片")
        except Exception as e:
            print(f"  构建实体卡片失败: {e}")

        return entity_passages

    def build_wordnet_definitions(self, common_words: List[str]) -> List[Dict]:
        """Bucket C: 词典定义 (~8-12k)"""
        print("\n=== 步骤3: 构建WordNet定义 ===")
        def_passages = []

        try:
            wn = load_dataset("marksverdhei/wordnet-definitions-en-2021", split="train", trust_remote_code=True)
            print(f"  列名: {wn.column_names}")

            # 构建 lemma -> [(definition, example)] 映射
            lemma2defs = defaultdict(list)
            for row in wn:
                # 适配不同的列名
                lemma = row.get("word") or row.get("lemma") or row.get("term")
                gloss = row.get("definition") or row.get("gloss")
                ex = row.get("example") or row.get("examples", "")

                if lemma and gloss:
                    lemma2defs[normalize_title(lemma)].append((gloss, ex))

            print(f"  加载了 {len(lemma2defs)} 个词条")

            # 为高频词生成定义段落
            for w in tqdm(common_words[:5000], desc="生成定义段落"):
                key = normalize_title(w)
                senses = lemma2defs.get(key, [])

                # 每个词最多包含2个词义
                for gloss, ex in senses[:2]:
                    base = f"{w}: {gloss}"
                    if ex:
                        base += f" Example: {ex}"

                    txt = clip_chars(base, lo=60, hi=400)
                    if txt:
                        def_passages.append({
                            "text": txt,
                            "source": "wordnet",
                            "title": w
                        })

            print(f"  生成了 {len(def_passages)} 个定义段落")
        except Exception as e:
            print(f"  构建WordNet定义失败: {e}")
            print("  使用备用简单定义...")
            def_passages = self._get_fallback_definitions(common_words[:100])

        return def_passages

    def _get_fallback_definitions(self, words: List[str]) -> List[Dict]:
        """备用简单定义"""
        simple_defs = {
            "apple": "A round fruit with red or green skin and white flesh.",
            "bank": "A financial institution that accepts deposits and provides loans.",
            "word": "A single unit of language that has meaning and can be spoken or written.",
            "music": "Vocal or instrumental sounds combined to produce beauty of form and expression.",
            "light": "The natural agent that stimulates sight and makes things visible.",
            "president": "The elected head of a republican state.",
            "time": "The indefinite continued progress of existence and events.",
            "water": "A colorless, transparent, odorless liquid that forms rivers, lakes, and seas.",
        }

        passages = []
        for w in words:
            if w in simple_defs:
                passages.append({
                    "text": f"{w}: {simple_defs[w]}",
                    "source": "fallback_def",
                    "title": w
                })
        return passages

    def build_wikipedia_leads(self, popular_titles: List[str], target_count: int = 15000) -> List[Dict]:
        """Bucket A: 维基百科首段 (~15k)"""
        print("\n=== 步骤4: 构建维基百科首段 ===")
        wiki_passages = []

        try:
            print("  加载FineWiki数据集（流式）...")
            finewiki = load_dataset(
                "HuggingFaceFW/finewiki",
                "CC-MAIN-2024-10",  # 使用一个具体的配置
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            # 构建需要的标题集合
            need = set(normalize_title(t) for t in popular_titles[:target_count * 2])
            print(f"  目标标题数: {len(need)}")

            count = 0
            for row in tqdm(finewiki, desc="处理维基百科", total=target_count):
                if count >= target_count:
                    break

                # 提取标题和文本
                title = row.get("title", "")
                text = row.get("text", "")

                if not title or not text:
                    continue

                # 检查是否在目标标题列表中（或者如果标题列表不足，接受所有）
                if need and normalize_title(title) not in need:
                    continue

                # 提取首段
                lead = lead_paragraph(text)
                if not lead:
                    continue

                # 格式: "{标题}. {首段}"
                snippet = clip_chars(f"{title}. {lead}", lo=80, hi=600)
                if snippet:
                    wiki_passages.append({
                        "text": snippet,
                        "source": "finewiki_lead",
                        "title": title
                    })
                    count += 1

            print(f"  生成了 {len(wiki_passages)} 个维基百科首段")
        except Exception as e:
            print(f"  构建维基百科首段失败: {e}")
            print("  使用备用简单维基内容...")
            wiki_passages = self._get_fallback_wiki_content()

        return wiki_passages

    def _get_fallback_wiki_content(self) -> List[Dict]:
        """备用简单维基内容"""
        fallback = [
            {
                "text": "Apple Inc. Apple Inc. is an American multinational technology company headquartered in Cupertino, California.",
                "source": "fallback_wiki",
                "title": "Apple Inc."
            },
            {
                "text": "Bank. A bank is a financial institution that accepts deposits from the public and creates a demand deposit.",
                "source": "fallback_wiki",
                "title": "Bank"
            },
            {
                "text": "Word. A word is a basic unit of language that carries meaning and consists of one or more morphemes.",
                "source": "fallback_wiki",
                "title": "Word"
            }
        ]
        return fallback

    def build_disambiguation_pages(self, target_count: int = 3000) -> List[Dict]:
        """Bucket D: 消歧义页面 (~2-5k)"""
        print("\n=== 步骤5: 构建消歧义页面 ===")
        disamb = []

        try:
            print("  加载完整维基百科数据集（用于提取消歧义页）...")
            # 注意：完整维基百科数据集很大，这里使用流式加载
            wiki_full = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",  # 使用一个可用的日期配置
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            count = 0
            for row in tqdm(wiki_full, desc="查找消歧义页", total=target_count):
                if count >= target_count:
                    break

                title = row.get("title", "")
                text = row.get("text", "")

                if not title or not text:
                    continue

                # 检查是否是消歧义页面
                is_disamb = "(disambiguation)" in title.lower()

                if not is_disamb:
                    # 检查首段是否包含"may refer to"等标志性短语
                    lead = lead_paragraph(text).lower()
                    is_disamb = any(phrase in lead for phrase in [
                        "may refer to",
                        "can refer to",
                        "may also refer to",
                        "can also refer to"
                    ])

                if is_disamb:
                    snippet = clip_chars(f"{title}. {lead_paragraph(text)}", lo=80, hi=600)
                    if snippet:
                        disamb.append({
                            "text": snippet,
                            "source": "wiki_disamb",
                            "title": title
                        })
                        count += 1

            print(f"  生成了 {len(disamb)} 个消歧义页面")
        except Exception as e:
            print(f"  构建消歧义页面失败: {e}")
            print("  使用备用消歧义内容...")
            disamb = self._get_fallback_disambiguation()

        return disamb

    def _get_fallback_disambiguation(self) -> List[Dict]:
        """备用消歧义内容"""
        fallback = [
            {
                "text": "Apple (disambiguation). Apple may refer to: Apple Inc., technology company; Apple (fruit), the fruit; Apple Records, record label.",
                "source": "fallback_disamb",
                "title": "Apple (disambiguation)"
            },
            {
                "text": "Bank (disambiguation). Bank may refer to: Bank (finance), financial institution; River bank, the land alongside a body of water; Memory bank, in computing.",
                "source": "fallback_disamb",
                "title": "Bank (disambiguation)"
            },
            {
                "text": "Mercury (disambiguation). Mercury may refer to: Mercury (element), chemical element; Mercury (planet), planet in the solar system; Mercury (mythology), Roman god.",
                "source": "fallback_disamb",
                "title": "Mercury (disambiguation)"
            }
        ]
        return fallback

    def build_web_qa_passages(self, target_count: int = 5000) -> List[Dict]:
        """Bucket E: 网页/问答段落 (~3-8k)"""
        print("\n=== 步骤6: 构建网页/问答段落 ===")
        webish = []

        try:
            print("  加载LoTTE数据集...")
            # 尝试加载LoTTE的一个子集
            lotte = load_dataset(
                "colbertv2/lotte_passages",
                "lifestyle",  # 或 "writing", "recreation", "science", "technology"
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            for row in tqdm(lotte, desc="处理LoTTE", total=target_count):
                if len(webish) >= target_count:
                    break

                # 适配不同的列名
                body = row.get("text") or row.get("passage") or row.get("body", "")
                doc_id = row.get("doc_id") or row.get("id", "")

                snippet = clip_chars(body, lo=80, hi=600)
                if snippet:
                    webish.append({
                        "text": snippet,
                        "source": "lotte",
                        "title": str(doc_id)
                    })

            print(f"  生成了 {len(webish)} 个网页/问答段落")
        except Exception as e:
            print(f"  构建网页/问答段落失败: {e}")
            print("  跳过此部分...")

        return webish

    def assemble_corpus(self, parts: Dict[str, List[Dict]]) -> List[Dict]:
        """组装最终语料库"""
        print("\n=== 步骤7: 组装语料库 ===")

        corpus = []
        for name, passages in parts.items():
            print(f"  {name}: {len(passages)} 条")
            corpus.extend(passages)

        print(f"  总计: {len(corpus)} 条")

        # 打乱顺序
        random.shuffle(corpus)

        # 限制到目标数量
        if len(corpus) > self.max_passages:
            corpus = corpus[:self.max_passages]
            print(f"  裁剪到: {len(corpus)} 条")

        return corpus

    def generate_embeddings(self, corpus: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        """生成嵌入向量"""
        print("\n=== 步骤8: 生成嵌入向量 ===")
        print(f"  加载模型: {model_name}")

        model = SentenceTransformer(model_name)

        # 提取文本
        texts = [p["text"] for p in corpus]

        # 批量编码
        print(f"  编码 {len(texts)} 条文本...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        # L2标准化
        print("  L2标准化...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def save_corpus(self, corpus: List[Dict], embeddings: np.ndarray):
        """保存语料库和嵌入"""
        print("\n=== 步骤9: 保存文件 ===")

        # 保存段落文本
        passages_file = self.output_dir / "passages.txt"
        with open(passages_file, "w", encoding="utf-8") as f:
            for p in corpus:
                f.write(p["text"] + "\n")
        print(f"  已保存: {passages_file}")

        # 保存来源标签
        sources_file = self.output_dir / "passage_sources.txt"
        with open(sources_file, "w", encoding="utf-8") as f:
            for p in corpus:
                f.write(p["source"] + "\n")
        print(f"  已保存: {sources_file}")

        # 保存标题
        titles_file = self.output_dir / "passage_titles.txt"
        with open(titles_file, "w", encoding="utf-8") as f:
            for p in corpus:
                f.write(p.get("title", "") + "\n")
        print(f"  已保存: {titles_file}")

        # 保存嵌入
        embeddings_file = self.output_dir / "passage_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"  已保存: {embeddings_file}")

        # 保存元数据
        metadata_file = self.output_dir / "metadata.json"
        metadata = {
            "total_passages": len(corpus),
            "embedding_dim": embeddings.shape[1],
            "source_distribution": self._count_sources(corpus),
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  已保存: {metadata_file}")

        # 打印统计
        print("\n=== 语料库统计 ===")
        print(f"  总段落数: {len(corpus)}")
        print(f"  嵌入维度: {embeddings.shape[1]}")
        print("\n来源分布:")
        for source, count in sorted(metadata["source_distribution"].items()):
            print(f"    {source}: {count}")

    def _count_sources(self, corpus: List[Dict]) -> Dict[str, int]:
        """统计来源分布"""
        counts = defaultdict(int)
        for p in corpus:
            counts[p["source"]] += 1
        return dict(counts)

    def build(self, skip_web: bool = False):
        """执行完整的构建流程"""
        print("=" * 60)
        print("构建改进的混合语料库")
        print("=" * 60)

        # 1. 构建种子列表
        common_words, popular_titles = self.build_seed_lists()

        # 2. 构建各个部分
        parts = {}

        # B. 实体卡片
        parts["entity_cards"] = self.build_entity_cards(popular_titles)

        # C. 词典定义
        parts["wordnet_defs"] = self.build_wordnet_definitions(common_words)

        # A. 维基百科首段
        parts["wiki_leads"] = self.build_wikipedia_leads(popular_titles, target_count=15000)

        # D. 消歧义页面
        parts["disambiguation"] = self.build_disambiguation_pages(target_count=3000)

        # E. 网页/问答（可选）
        if not skip_web:
            parts["web_qa"] = self.build_web_qa_passages(target_count=5000)

        # 3. 组装语料库
        corpus = self.assemble_corpus(parts)

        # 4. 生成嵌入
        embeddings = self.generate_embeddings(corpus)

        # 5. 保存
        self.save_corpus(corpus, embeddings)

        print("\n" + "=" * 60)
        print("构建完成！")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="构建改进的混合语料库")
    parser.add_argument(
        "--output",
        type=str,
        default="data/improved",
        help="输出目录"
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=60000,
        help="最大段落数"
    )
    parser.add_argument(
        "--skip-web",
        action="store_true",
        help="跳过网页/问答部分（如果加载失败）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="句子编码模型"
    )

    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 构建语料库
    builder = CorpusBuilder(args.output, args.max_passages)
    builder.build(skip_web=args.skip_web)


if __name__ == "__main__":
    main()
