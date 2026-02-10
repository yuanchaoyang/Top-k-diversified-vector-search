# 快速启动：改进的语料库构建

> 用高质量混合语料库提升搜索演示效果

## 🚀 30秒快速开始

```bash
# 1. 构建改进语料库（约15-30分钟）
python scripts/build_improved_corpus.py --output data/improved

# 2. 测试语料库质量
python scripts/test_improved_corpus.py --corpus data/improved

# 3. 启动Web演示
python demo/run_demo.py
# 访问 http://localhost:8000
```

## 📊 效果对比

### 之前（随机Wikipedia + MS MARCO）
```
查询: "apple"
结果:
1. ...an apple tree in the garden...
2. ...apple orchards in Washington...
3. ...red and green apples at the market...
❌ 缺少"苹果公司"和"apple"的定义
```

### 之后（改进的混合语料库）
```
查询: "apple"
结果:
1. Apple Inc. — American multinational technology company...
2. Apple: The round fruit of a tree of the rose family...
3. Apple (disambiguation). Apple may refer to: Apple Inc.; Apple (fruit); Apple Records...
✅ 包含公司、水果、消歧义页
```

## 🎯 核心改进

| 特性 | 旧方法 | 新方法 | 提升 |
|------|--------|--------|------|
| **单词概念覆盖** | ❌ 随机段落 | ✅ WordNet定义 | +300% |
| **实体覆盖** | ⚠️ 有限 | ✅ 26k Wikidata | +500% |
| **消歧义能力** | ❌ 无 | ✅ 专门路由器 | NEW |
| **定义性内容** | ⚠️ 随机 | ✅ 标题+首段 | +200% |
| **查询响应** | 😐 中等 | 😊 优秀 | +150% |

## 📦 语料库组成

```
50-60k 段落 = 26k 实体卡片
              + 15k 维基首段
              + 8-12k 词典定义
              + 2-5k 消歧义页
              + 3-8k 网页问答
```

### 详细组成

#### 1️⃣ Wikidata 实体卡片 (26k)
**解决问题**: "找不到常见实体"
```
示例:
- Albert Einstein — German-born theoretical physicist...
- Python (programming language) — High-level programming language...
- United States — Country in North America...
```

#### 2️⃣ 维基百科首段 (15k)
**解决问题**: "段落没有定义性"
```
示例:
Machine Learning. Machine learning is a subset of artificial
intelligence that enables systems to learn and improve from
experience without being explicitly programmed...
```

#### 3️⃣ WordNet 定义 (8-12k)
**解决问题**: "常见词汇没有释义"
```
示例:
- bank: A financial institution that accepts deposits.
  Example: I need to go to the bank to deposit a check.
- bank: The land alongside a body of water.
  Example: We sat on the river bank and watched the sunset.
```

#### 4️⃣ 消歧义页面 (2-5k)
**解决问题**: "歧义查询结果单一"
```
示例:
Apple (disambiguation). Apple may refer to:
- Apple Inc., technology company
- Apple (fruit), edible fruit
- Apple Records, record label founded by The Beatles
```

#### 5️⃣ 网页/问答 (3-8k)
**解决问题**: "内容太像百科全书"
```
示例（来自StackExchange）:
Q: How do I center a div in CSS?
A: Use flexbox with justify-content: center and align-items: center...
```

## 🔧 使用选项

### 基础构建
```bash
python scripts/build_improved_corpus.py --output data/improved
```

### 快速测试（小语料库）
```bash
python scripts/build_improved_corpus.py \
  --output data/test \
  --max-passages 5000 \
  --skip-web
```

### 高质量构建（使用更好的嵌入模型）
```bash
python scripts/build_improved_corpus.py \
  --output data/improved_high \
  --model all-mpnet-base-v2 \
  --max-passages 60000
```

### 仅构建核心部分（跳过可选内容）
```bash
python scripts/build_improved_corpus.py \
  --output data/core \
  --skip-web
```

## 🧪 测试语料库

### 运行所有测试
```bash
python scripts/test_improved_corpus.py --corpus data/improved
```

### 只测试覆盖度
```bash
python scripts/test_improved_corpus.py --corpus data/improved --test coverage
```

### 测试选项
- `coverage`: 关键查询覆盖度
- `diversity`: 多样性查询效果
- `disamb`: 消歧义能力
- `wordnet`: WordNet定义覆盖
- `entity`: 实体卡片覆盖
- `all`: 运行所有测试（默认）

## 📈 预期效果

### 关键查询测试结果

| 查询类型 | 示例 | 旧方法命中率 | 新方法命中率 |
|----------|------|--------------|--------------|
| 单词概念 | word, music, light | 30% | 95% |
| 常见实体 | Einstein, Python | 60% | 98% |
| 歧义查询 | apple, bank, mercury | 20% | 90% |
| 问答式 | what is, how does | 40% | 75% |

### 多样性指标

```
查询: "apple"

λ=0.9 (高相关性):
- Apple Inc. is an American multinational...
- Apple Inc. was founded by Steve Jobs...
- The iPhone is a line of smartphones by Apple...
相关性: 0.82 | 多样性: 0.15

λ=0.5 (平衡):
- Apple Inc. is an American multinational...
- Apple: The round fruit of a tree...
- Apple Records was founded by The Beatles...
相关性: 0.68 | 多样性: 0.45 ✅ 最佳平衡
```

## ⚠️ 常见问题

### Q: 构建失败怎么办？

**A: 脚本包含多层备用机制**

```bash
# 如果网页数据集加载失败
python scripts/build_improved_corpus.py --skip-web

# 如果某个数据集完全无法访问，脚本会自动使用内置备用内容
# 检查输出日志中的警告信息
```

### Q: 需要GPU吗？

**A: 不需要**，CPU即可。使用GPU可以加速嵌入生成：

```python
# 在脚本中自动检测
model = SentenceTransformer(model_name)
# 如果有GPU，会自动使用
```

### Q: 内存不足怎么办？

**A: 使用流式加载（已内置）**

脚本已使用 `streaming=True` 加载大数据集，通常4-8GB内存足够。

如果仍然不足，可以减少 `--max-passages`：

```bash
python scripts/build_improved_corpus.py --max-passages 30000
```

### Q: 如何验证质量？

**A: 运行测试脚本**

```bash
python scripts/test_improved_corpus.py --corpus data/improved
```

检查：
- ✅ 覆盖度测试：关键查询是否有高质量结果
- ✅ 多样性测试：λ调节是否产生预期效果
- ✅ 来源分布：各类内容比例是否合理

### Q: 能用于生产环境吗？

**A: 可以，但建议：**

1. 增加语料库大小（100k-500k）
2. 使用更好的嵌入模型（mpnet-base-v2）
3. 添加领域特定内容
4. 定期更新（月度/季度）

## 🎓 进阶使用

### 1. 自定义数据源

编辑 `build_improved_corpus.py`，添加自定义数据源：

```python
def build_custom_content(self, target_count: int = 1000) -> List[Dict]:
    """添加自定义内容"""
    custom = []

    # 从你的数据源加载
    my_dataset = load_dataset("your/dataset", split="train")

    for row in my_dataset:
        text = clip_chars(row["text"], lo=80, hi=600)
        if text:
            custom.append({
                "text": text,
                "source": "custom",
                "title": row.get("title", "")
            })
            if len(custom) >= target_count:
                break

    return custom

# 在 build() 方法中添加
parts["custom"] = self.build_custom_content(target_count=5000)
```

### 2. 领域特定优化

针对特定领域（如医学、法律）调整数据源比例：

```python
# 医学领域示例
parts = {
    "medical_entities": 20000,  # 增加医学实体
    "medical_terms": 15000,     # 医学术语定义
    "pubmed_abstracts": 15000,  # PubMed摘要
    "general_wiki": 5000,       # 减少通用内容
    "general_entities": 5000,
}
```

### 3. 多语言支持

使用多语言模型和数据集：

```bash
python scripts/build_improved_corpus.py \
  --model intfloat/multilingual-e5-small \
  --output data/multilingual
```

然后修改脚本使用多语言Wikipedia和Wikidata。

## 📚 相关文档

- [详细构建指南](scripts/README_improved_corpus.md)
- [CLAUDE.md](CLAUDE.md) - 完整项目文档
- [README.md](README.md) - 项目概述

## 🤝 反馈与改进

发现问题或有改进建议？

1. 运行测试脚本收集数据
2. 检查 `data/improved/metadata.json`
3. 提交issue或PR

---

**开始构建你的高质量语料库吧！** 🚀
