# 改进的混合语料库构建指南

## 概述

`build_improved_corpus.py` 实现了一个高质量的混合语料库构建策略，解决了传统方法的三个主要问题：

1. ❌ **随机维基百科段落缺乏定义性** → ✅ 使用标题+首段格式，优先重要页面
2. ❌ **FineWiki删除了消歧义页** → ✅ 专门添加消歧义路由器
3. ❌ **MS MARCO数据时效性差** → ✅ 用Wikidata实体卡片替代

## 语料库组成（50-60k段落）

### A. 定义性百科片段 (~15k)
- **来源**: FineWiki (Wikipedia 2024)
- **策略**: 标题 + 首段（不是随机段落）
- **格式**: `"{标题}. {首段}"`
- **采样**: 优先重要页面和最近更新的文章

### B. 实体卡片 (~26k)
- **来源**: `masaki-sakata/wikidata_descriptions`
- **内容**: 26,205 个实体的简短描述
- **优势**: 高覆盖度，适合单词/实体查询
- **格式**: `"{实体名} — {描述}."`

### C. 词典定义 (~8-12k)
- **来源**: `marksverdhei/wordnet-definitions-en-2021`
- **内容**: 高频词的多个词义
- **策略**: 每个词最多2个词义
- **格式**: `"{词}: {定义} Example: {例句}"`

### D. 消歧义路由器 (~2-5k)
- **来源**: Wikimedia Wikipedia
- **目标**: 处理 "apple", "bank", "mercury" 等歧义查询
- **识别**: 包含 "(disambiguation)" 或 "may refer to"
- **作用**: 为MMR多样化提供可区分的候选

### E. 网页/问答段落 (~3-8k)
- **来源**: LoTTE (StackExchange) 或 BEIR
- **目标**: 让演示更像真实搜索引擎
- **可选**: 使用 `--skip-web` 跳过此部分

## 快速开始

### 基本用法

```bash
python scripts/build_improved_corpus.py --output data/improved
```

这将：
1. 从 HuggingFace 加载多个数据集
2. 构建 50-60k 混合语料库
3. 生成嵌入向量（使用 all-MiniLM-L6-v2）
4. 保存到 `data/improved/` 目录

### 完整参数

```bash
python scripts/build_improved_corpus.py \
  --output data/improved \
  --max-passages 60000 \
  --model all-MiniLM-L6-v2 \
  --skip-web
```

**参数说明：**
- `--output`: 输出目录（默认: `data/improved`）
- `--max-passages`: 最大段落数（默认: 60000）
- `--model`: 句子编码模型（默认: `all-MiniLM-L6-v2`）
- `--skip-web`: 跳过网页/问答部分（如果加载失败）

## 输出文件

构建完成后，`data/improved/` 目录包含：

```
data/improved/
├── passages.txt              # 段落文本（每行一条）
├── passage_embeddings.npy    # 嵌入向量（L2标准化）
├── passage_sources.txt       # 来源标签
├── passage_titles.txt        # 标题/实体名
└── metadata.json             # 统计信息
```

## 核心策略

### 1. 覆盖保证

不依赖随机采样，而是：
- 构建种子列表：前10k高频词 + 前20k热门实体
- 确保每个种子至少有一条内容
- 对歧义词包含多个词义

### 2. 智能Wikipedia采样

```python
# ❌ 不要：随机段落
passage = random_paragraph(text)

# ✅ 要做：标题 + 首段
passage = f"{title}. {lead_paragraph(text)}"
```

**采样权重：**
- 70% 重要页面（基于受欢迎度）
- 20% 最近更新（6-12个月内）
- 10% 随机

### 3. 多数据源混合

| 数据源 | 类型 | 优势 | 数量 |
|--------|------|------|------|
| Wikidata | 实体描述 | 高覆盖、短小精悍 | 26k |
| WordNet | 词典定义 | 多词义、有例句 | 8-12k |
| FineWiki | 百科首段 | 定义性、权威 | 15k |
| Wikipedia | 消歧义页 | 处理歧义查询 | 2-5k |
| LoTTE | 问答内容 | 真实搜索风格 | 3-8k |

### 4. 备用机制

脚本包含多层备用策略：
- 数据集加载失败 → 使用备用数据集
- 网络问题 → 使用内置示例
- API限流 → 降级到简化版本

## 使用改进语料库运行Demo

```bash
# 1. 构建语料库
python scripts/build_improved_corpus.py --output data/improved

# 2. 启动Web Demo
python demo/run_demo.py
# Demo会自动检测并使用 data/improved/

# 3. 访问 http://localhost:8000
```

## 与旧方法对比

### 旧方法（prepare_msmarco.py / prepare_mixed_corpus.py）
- ❌ 随机维基百科段落
- ❌ 缺少消歧义页面
- ❌ MS MARCO时效性差
- ❌ 单词概念覆盖不足
- 📊 典型大小：20-50k段落

### 新方法（build_improved_corpus.py）
- ✅ 标题+首段，定义性强
- ✅ 专门的消歧义路由器
- ✅ Wikidata实体卡片（高覆盖）
- ✅ WordNet定义（多词义）
- 📊 典型大小：50-60k段落

## 常见问题

### Q: 构建需要多长时间？
A: 约15-30分钟（取决于网络速度和计算资源）

### Q: 需要多少磁盘空间？
A: 约500MB-1GB（包括下载的数据集缓存）

### Q: 需要多少内存？
A: 约4-8GB（使用流式加载可降低要求）

### Q: 如何处理某个数据集加载失败？
A: 脚本包含备用机制，会自动降级到替代方案

### Q: 可以自定义数据源比例吗？
A: 可以，修改各个 `build_*` 方法的 `target_count` 参数

### Q: 如何验证构建是否成功？
A: 检查 `data/improved/metadata.json` 查看统计信息

## 高级用法

### 只构建特定部分

编辑脚本中的 `build()` 方法，注释掉不需要的部分：

```python
def build(self, skip_web: bool = False):
    # ...
    parts = {}

    # 只要实体卡片和WordNet定义
    parts["entity_cards"] = self.build_entity_cards(popular_titles)
    parts["wordnet_defs"] = self.build_wordnet_definitions(common_words)

    # 注释掉其他部分
    # parts["wiki_leads"] = ...
    # parts["disambiguation"] = ...
```

### 使用不同的嵌入模型

```bash
python scripts/build_improved_corpus.py \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

常用模型：
- `all-MiniLM-L6-v2`: 快速，384维
- `all-mpnet-base-v2`: 更好质量，768维
- `intfloat/multilingual-e5-small`: 多语言支持

### 调试模式

减少段落数量以加快测试：

```bash
python scripts/build_improved_corpus.py \
  --output data/test \
  --max-passages 5000 \
  --skip-web
```

## 技术细节

### 文本处理

1. **长度控制**: `clip_chars(text, lo=80, hi=600)`
   - 最小80字符（过滤太短的内容）
   - 最大600字符（在单词边界截断）

2. **标题标准化**: `normalize_title()`
   - 转小写
   - 移除特殊字符
   - 用于去重和匹配

3. **首段提取**: `lead_paragraph()`
   - 按双换行分段
   - 取第一个非空段落

### 嵌入生成

- 批量大小：32
- L2标准化：确保余弦相似度 = 点积
- 进度条：使用tqdm显示进度

## 贡献与改进

如需改进语料库质量：

1. **添加新数据源**: 在相应的 `build_*` 方法中添加
2. **调整采样策略**: 修改各部分的 `target_count`
3. **改进文本处理**: 更新 `clip_chars` 和 `lead_paragraph` 逻辑
4. **自定义备用内容**: 扩展 `_get_fallback_*` 方法

## 参考资料

- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [Sentence Transformers](https://www.sbert.net/)
- [ANN Benchmarks](http://ann-benchmarks.com/)
- [LoTTE Dataset](https://github.com/stanford-futuredata/ColBERT/tree/main/lotte)

## 许可

此脚本构建的语料库混合了多个数据集，使用时需遵守各数据集的许可协议。
