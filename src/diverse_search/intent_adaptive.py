"""
基于LLM意图分析的Adaptive Lambda

思路：
1. 用LLM分析查询词的"意图明确度"
2. 明确查询 → 高λ（偏重相关性）
3. 模糊查询 → 低λ（偏重多样性）

示例：
- "apple" → 模糊（水果？公司？）→ 低λ
- "iPhone 15 Pro" → 明确 → 高λ

支持的LLM后端：
- OpenAI API (需要API key)
- Ollama (本地免费LLM)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import requests
import json


# LLM后端类型
LLMBackend = Literal["openai", "ollama", "rule"]


@dataclass
class IntentConfig:
    """意图分析配置"""
    lambda_clear: float = 0.8      # 明确查询的λ
    lambda_ambiguous: float = 0.4  # 模糊查询的λ
    lambda_default: float = 0.6    # 默认λ
    llm_backend: LLMBackend = "ollama"  # 默认使用Ollama
    ollama_model: str = "qwen2.5:0.5b"  # Ollama模型
    ollama_url: str = "http://localhost:11434"  # Ollama API地址


# 预定义的意图词典（可由LLM生成）
# 格式: word -> (intent_score, reason)
# intent_score: 0=模糊, 1=明确
INTENT_DICTIONARY: Dict[str, Tuple[float, str]] = {}


def analyze_query_intent_rule_based(query_word: str) -> Tuple[float, str]:
    """
    基于规则的意图分析（简单版本）

    Returns:
        (intent_score, reason): 0-1分数，0=模糊，1=明确
    """
    word = query_word.lower().strip()

    # 规则1：长词通常更明确
    length_score = min(len(word) / 15, 1.0)

    # 规则2：包含数字通常更明确
    has_digit = any(c.isdigit() for c in word)
    digit_score = 0.3 if has_digit else 0.0

    # 规则3：复合词通常更明确
    has_compound = '-' in word or '_' in word or word.count(' ') > 0
    compound_score = 0.2 if has_compound else 0.0

    # 规则4：常见多义词标记为模糊
    ambiguous_words = {
        'apple', 'amazon', 'python', 'java', 'spring', 'ruby',
        'bank', 'cell', 'rock', 'star', 'light', 'book',
        'paper', 'mouse', 'windows', 'office', 'word'
    }
    ambiguous_penalty = -0.4 if word in ambiguous_words else 0.0

    # 综合得分
    score = 0.5 + length_score * 0.3 + digit_score + compound_score + ambiguous_penalty
    score = max(0.0, min(1.0, score))

    reason = f"length={len(word)}, has_digit={has_digit}, compound={has_compound}"

    return score, reason


def analyze_query_intent_ollama(
    query_word: str,
    model: str = "qwen2.5:0.5b",
    base_url: str = "http://localhost:11434",
    use_cache: bool = True,
) -> Tuple[float, str]:
    """
    使用Ollama本地LLM分析查询意图

    Args:
        query_word: 查询词
        model: Ollama模型名称
        base_url: Ollama API地址
        use_cache: 是否使用缓存
    """
    # 先检查缓存
    if use_cache and query_word.lower() in INTENT_DICTIONARY:
        return INTENT_DICTIONARY[query_word.lower()]

    try:
        prompt = f"""Analyze the search query "{query_word}".

Rate its "intent clarity" from 0 to 1:
- 0.0-0.3: Very ambiguous (e.g., "apple" = fruit or company, "python" = snake or language)
- 0.4-0.6: Moderately specific
- 0.7-1.0: Very specific (single clear meaning)

Return ONLY valid JSON: {{"score": 0.X, "reason": "brief explanation"}}"""

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=30,
        )
        response.raise_for_status()

        result = response.json()
        result_text = result.get("response", "").strip()

        # 尝试解析JSON
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        # 尝试提取JSON部分
        import re
        json_match = re.search(r'\{[^}]+\}', result_text)
        if json_match:
            result_text = json_match.group()

        data = json.loads(result_text)
        score = float(data.get("score", 0.5))
        reason = data.get("reason", "")

        # 缓存结果
        INTENT_DICTIONARY[query_word.lower()] = (score, reason)

        return score, reason

    except Exception as e:
        print(f"  Ollama error for '{query_word}': {e}")
        return analyze_query_intent_rule_based(query_word)


def analyze_query_intent_llm(
    query_word: str,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    backend: LLMBackend = "ollama",
    config: "IntentConfig" = None,
) -> Tuple[float, str]:
    """
    使用LLM分析查询意图

    Args:
        query_word: 查询词
        api_key: OpenAI API key (仅OpenAI后端需要)
        use_cache: 是否使用缓存
        backend: LLM后端 ("openai", "ollama", "rule")
        config: 配置对象
    """
    # 先检查缓存
    if use_cache and query_word.lower() in INTENT_DICTIONARY:
        return INTENT_DICTIONARY[query_word.lower()]

    # 根据后端选择
    if backend == "rule":
        return analyze_query_intent_rule_based(query_word)
    elif backend == "ollama":
        model = config.ollama_model if config else "qwen2.5:0.5b"
        url = config.ollama_url if config else "http://localhost:11434"
        return analyze_query_intent_ollama(query_word, model=model, base_url=url, use_cache=use_cache)
    elif backend == "openai":
        if api_key is None:
            return analyze_query_intent_rule_based(query_word)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You analyze search query intent. Return JSON only."},
                    {"role": "user", "content": f"""Analyze the search query "{query_word}".

Rate its "intent clarity" from 0 to 1:
- 0.0-0.3: Very ambiguous (e.g., "apple" = fruit or company, "python" = snake or language)
- 0.4-0.6: Moderately specific
- 0.7-1.0: Very specific (single clear meaning)

Return ONLY valid JSON: {{"score": 0.X, "reason": "brief explanation"}}"""}
                ],
                temperature=0.1,
                max_tokens=100,
            )

            result_text = response.choices[0].message.content.strip()
            # 尝试解析JSON
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            data = json.loads(result_text)
            score = float(data.get("score", 0.5))
            reason = data.get("reason", "")

            # 缓存结果
            INTENT_DICTIONARY[query_word.lower()] = (score, reason)

            return score, reason

        except Exception as e:
            print(f"  OpenAI error for '{query_word}': {e}")
            return analyze_query_intent_rule_based(query_word)
    else:
        return analyze_query_intent_rule_based(query_word)


def intent_based_lambda(
    query_word: str,
    config: IntentConfig = None,
    api_key: Optional[str] = None,
) -> Tuple[float, str]:
    """
    根据查询意图选择λ

    Returns:
        (lambda_value, reason)
    """
    if config is None:
        config = IntentConfig()

    intent_score, reason = analyze_query_intent_llm(
        query_word,
        api_key=api_key,
        backend=config.llm_backend,
        config=config
    )

    # 线性插值
    lam = config.lambda_ambiguous + intent_score * (config.lambda_clear - config.lambda_ambiguous)

    return lam, f"intent={intent_score:.2f} ({reason})"


def batch_intent_lambdas(
    query_words: List[str],
    config: IntentConfig = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """批量计算意图λ"""
    if config is None:
        config = IntentConfig()

    lambdas = []
    for i, word in enumerate(query_words):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Analyzing {i+1}/{len(query_words)}...")
        lam, _ = intent_based_lambda(word, config, api_key)
        lambdas.append(lam)
    return np.array(lambdas, dtype=np.float32)


def batch_analyze_intent_ollama(
    words: List[str],
    model: str = "qwen2.5:0.5b",
    base_url: str = "http://localhost:11434",
    batch_size: int = 10,
) -> Dict[str, Tuple[float, str]]:
    """
    批量用Ollama分析意图

    Args:
        words: 查询词列表
        model: Ollama模型名称
        base_url: Ollama API地址
        batch_size: 每批处理的词数
    """
    results = {}

    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}: analyzing {len(batch)} words...")

        words_list = "\n".join([f"- {w}" for w in batch])

        prompt = f"""Analyze these search queries for intent clarity.

Rate each from 0 to 1:
- 0.0-0.3: Very ambiguous (multiple meanings, e.g., "apple", "python", "java")
- 0.4-0.6: Moderately specific
- 0.7-1.0: Very specific (single clear meaning)

Words:
{words_list}

Return ONLY a JSON array: [{{"word": "...", "score": 0.X, "reason": "..."}}]"""

        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            result_text = result.get("response", "").strip()

            # 尝试解析JSON
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            # 尝试提取JSON数组
            import re
            json_match = re.search(r'\[[\s\S]*\]', result_text)
            if json_match:
                result_text = json_match.group()

            try:
                data = json.loads(result_text)
                for item in data:
                    word = item.get("word", "").lower()
                    score = float(item.get("score", 0.5))
                    reason = item.get("reason", "")
                    results[word] = (score, reason)
                    INTENT_DICTIONARY[word] = (score, reason)
            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}")
                # Fallback to individual analysis
                for w in batch:
                    if w.lower() not in results:
                        score, reason = analyze_query_intent_ollama(w, model, base_url)
                        results[w.lower()] = (score, reason)

        except Exception as e:
            print(f"    Batch error: {e}")
            # Fallback to individual analysis
            for w in batch:
                if w.lower() not in results:
                    score, reason = analyze_query_intent_rule_based(w)
                    results[w.lower()] = (score, reason)

    return results


def test_ollama_connection(
    base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:0.5b",
) -> bool:
    """测试Ollama连接是否正常"""
    try:
        # 测试API是否可用
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        if model in model_names or any(model in n for n in model_names):
            print(f"  ✓ Ollama connected, model '{model}' available")
            return True
        else:
            print(f"  ✗ Model '{model}' not found. Available: {model_names}")
            return False

    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        return False


def batch_analyze_intent_llm(
    words: List[str],
    api_key: str,
    batch_size: int = 20,
) -> Dict[str, Tuple[float, str]]:
    """
    批量用LLM分析意图（更高效）

    一次请求分析多个词
    """
    try:
        from openai import OpenAI
        import json

        client = OpenAI(api_key=api_key)
        results = {}

        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}: analyzing {len(batch)} words...")

            words_list = "\n".join([f"- {w}" for w in batch])

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You analyze search query intent. Return JSON array only."},
                    {"role": "user", "content": f"""Analyze these search queries for intent clarity.

Rate each from 0 to 1:
- 0.0-0.3: Very ambiguous (multiple meanings, e.g., "apple", "python", "java")
- 0.4-0.6: Moderately specific
- 0.7-1.0: Very specific (single clear meaning)

Words:
{words_list}

Return ONLY a JSON array: [{{"word": "...", "score": 0.X, "reason": "..."}}]"""}
                ],
                temperature=0.1,
                max_tokens=1500,
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            try:
                data = json.loads(result_text)
                for item in data:
                    word = item.get("word", "").lower()
                    score = float(item.get("score", 0.5))
                    reason = item.get("reason", "")
                    results[word] = (score, reason)
                    INTENT_DICTIONARY[word] = (score, reason)
            except json.JSONDecodeError as e:
                print(f"    JSON parse error: {e}")
                # Fallback to individual analysis
                for w in batch:
                    if w.lower() not in results:
                        score, reason = analyze_query_intent_rule_based(w)
                        results[w.lower()] = (score, reason)

        return results

    except Exception as e:
        print(f"  Batch LLM error: {e}")
        return {w.lower(): analyze_query_intent_rule_based(w) for w in words}


# ============================================================================
# LLM批量生成意图词典
# ============================================================================

def generate_intent_dictionary_prompt(words: List[str]) -> str:
    """生成让LLM批量分析的prompt"""

    prompt = """Analyze these search query words and rate their "intent clarity" from 0 to 1.

- Score 0.0-0.3: Very ambiguous (multiple meanings, e.g., "apple", "python")
- Score 0.4-0.6: Moderately specific (some ambiguity)
- Score 0.7-1.0: Very specific (clear single meaning)

Words to analyze:
"""
    for word in words:
        prompt += f"- {word}\n"

    prompt += """
Return as JSON array:
[
  {"word": "...", "score": 0.X, "reason": "..."},
  ...
]
"""
    return prompt


def parse_llm_intent_response(response: str) -> Dict[str, Tuple[float, str]]:
    """解析LLM的响应"""
    import json
    try:
        data = json.loads(response)
        result = {}
        for item in data:
            word = item["word"].lower()
            score = float(item["score"])
            reason = item.get("reason", "")
            result[word] = (score, reason)
        return result
    except:
        return {}


def update_intent_dictionary(new_entries: Dict[str, Tuple[float, str]]):
    """更新意图词典"""
    global INTENT_DICTIONARY
    INTENT_DICTIONARY.update(new_entries)


# ============================================================================
# 预生成的示例词典（可以用LLM扩展）
# ============================================================================

# 这些是用规则+人工标注生成的示例
SAMPLE_INTENT_DICTIONARY = {
    # 明确查询 (score > 0.7)
    "iphone": (0.8, "specific product brand"),
    "shakespeare": (0.9, "specific person"),
    "photosynthesis": (0.95, "specific scientific term"),
    "eiffel tower": (0.95, "specific landmark"),

    # 中等 (score 0.4-0.7)
    "music": (0.5, "broad category"),
    "travel": (0.5, "broad concept"),
    "science": (0.45, "broad field"),

    # 模糊查询 (score < 0.4)
    "apple": (0.2, "fruit or company"),
    "python": (0.25, "snake or programming language"),
    "java": (0.2, "island or programming language"),
    "amazon": (0.2, "river or company"),
    "bank": (0.3, "financial or river bank"),
    "cell": (0.3, "biology or phone or prison"),
}

# 初始化词典
INTENT_DICTIONARY.update(SAMPLE_INTENT_DICTIONARY)
