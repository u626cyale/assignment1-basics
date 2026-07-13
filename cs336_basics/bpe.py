from collections import defaultdict
import regex
import heapq
from typing import List, Tuple, Dict

# 严格对齐作业文档给出的GPT2风格预分词正则，分支顺序不可修改
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def build_vocab(input_path: str, special_tokens: List[str]) -> Dict[bytes, int]:
    """
    统计预分词词频，严格遵循作业要求：
    1. 特殊token作为硬分割边界，跨边界不合并
    2. 特殊token本身不参与合并统计
    """
    counter = defaultdict(int)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        # 转义特殊token中的特殊字符，用捕获组分割，保留特殊token用于判断跳过
        special_pattern = "|".join(regex.escape(tok) for tok in special_tokens)
        parts = regex.split(f"({special_pattern})", text)
    else:
        parts = [text]

    for part in parts:
        if not part:
            continue
        # 跳过特殊token本身，不参与统计
        if part in special_tokens:
            continue
        # 用finditer遍历，避免一次性生成所有pre-token列表
        for match in regex.finditer(PAT, part):
            word_bytes = match.group().encode("utf-8")
            counter[word_bytes] += 1
    return counter


def merge_pair(symbols: List[bytes], pair: Tuple[bytes, bytes]) -> List[bytes]:
    """左到右无重叠合并，保证合并结果确定性，对齐BPE标准逻辑"""
    a, b = pair
    result = []
    i = 0
    n = len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
            result.append(a + b)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result


def get_best_pair(heap: list, pair_freq: Dict[Tuple[bytes, bytes], int]) -> Tuple[bytes, bytes] | None:
    """
    懒删除最小堆实现，严格对齐作业优先级规则：
    1. 频次越高优先级越高
    2. 频次相同时，字节对字典序越大优先级越高
    """
    while heap:
        neg_freq, pair = heapq.heappop(heap)
        stored_freq = -neg_freq
        real_freq = pair_freq.get(pair, 0)
        if real_freq != stored_freq:
            continue
        # 堆中频次可能因懒删除而偏低，需扫描 pair_freq 确认全局最高频
        max_freq = max(pair_freq.values())
        return max(
            (p for p, f in pair_freq.items() if f == max_freq),
            key=lambda p: p,
        )
    return None


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    高性能字节级BPE训练实现，完全符合CS336作业要求
    核心优化：
    1. 懒删除堆O(logN)取最优pair，替代朴素O(N)全量遍历
    2. pair_words索引：每次合并仅处理受影响的预分词，而非全量语料
    """
    # 1. 初始化256个基础字节词汇（ID 0-255对应单字节）
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # 2. 添加特殊token到词汇表，记录字节序列用于合并校验
    special_bytes: List[bytes] = []
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[next_id] = token_bytes
        special_bytes.append(token_bytes)
        next_id += 1

    # 参数合法性校验（对齐作业定义：vocab_size包含所有类型token）
    initial_vocab_size = len(vocab)
    if vocab_size < initial_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {initial_vocab_size} "
            f"(256 base bytes + {len(special_tokens)} special tokens)"
        )

    # 3. 统计预分词词频
    word_freq = build_vocab(input_path, special_tokens)
    if not word_freq:
        raise ValueError("Training corpus is empty")

    # 4. 初始化BPE状态
    word_symbols: Dict[bytes, List[bytes]] = {}  # 每个预分词对应的当前字节符号序列
    pair_freq: Dict[Tuple[bytes, bytes], int] = defaultdict(int)  # 字节对全局频次
    pair_words: Dict[Tuple[bytes, bytes], set[bytes]] = defaultdict(set)  # 字节对所属的预分词集合

    for word, count in word_freq.items():
        # 预分词拆分为单字节符号序列
        symbols = [bytes([b]) for b in word]
        word_symbols[word] = symbols
        # 统计初始相邻字节对
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freq[pair] += count
            pair_words[pair].add(word)

    # 5. 初始化最小堆
    merges: List[Tuple[bytes, bytes]] = []
    target_merges = vocab_size - initial_vocab_size
    heap = []
    for pair, freq in pair_freq.items():
        heapq.heappush(heap, (-freq, pair))

    # 6. 迭代合并
    for _ in range(target_merges):
        # 循环获取合法最优pair（跳过合并后等于特殊token的pair）
        best_pair = None
        while True:
            candidate = get_best_pair(heap, pair_freq)
            if candidate is None:
                break
            merged_token = candidate[0] + candidate[1]
            if merged_token not in special_bytes:
                best_pair = candidate
                break
            # 禁止合并出特殊 token，从索引中移除以免反复选中
            pair_words.pop(candidate, None)
            pair_freq.pop(candidate, None)
        if best_pair is None:
            break  # 无更多可合并pair，提前终止

        # 记录合并规则
        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[next_id] = merged_token
        next_id += 1

        # 获取所有包含当前pair的预分词（核心优化：仅处理受影响的词）
        affected_words = list(pair_words.pop(best_pair))
        for word in affected_words:
            old_symbols = word_symbols[word]
            count = word_freq[word]

            # 移除旧符号序列所有字节对的频次
            for i in range(len(old_symbols) - 1):
                p = (old_symbols[i], old_symbols[i + 1])
                pair_freq[p] -= count
                if pair_freq[p] == 0:
                    del pair_freq[p]
                    pair_words.pop(p, None)
                else:
                    pair_words[p].discard(word)
                    heapq.heappush(heap, (-pair_freq[p], p))

            # 生成合并后的新符号序列
            new_symbols = merge_pair(old_symbols, best_pair)
            word_symbols[word] = new_symbols

            # 添加新符号序列所有字节对的频次，并入堆
            for i in range(len(new_symbols) - 1):
                p = (new_symbols[i], new_symbols[i + 1])
                pair_freq[p] += count
                pair_words[p].add(word)
                heapq.heappush(heap, (-pair_freq[p], p))

    return vocab, merges
