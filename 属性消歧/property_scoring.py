from __future__ import annotations
import re
from typing import List

def split_tokens(text: str) -> List[str]:
    """将属性值切分为细粒度的 token，用于相似度计算"""
    # 提取英文和数字
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", text)
    # 提取中文词组（按 2-3 个字滑动切分，或直接切单字，这里用 2-gram 模拟词组）
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_tokens = []
    if len(chinese_chars) > 1:
        for i in range(len(chinese_chars) - 1):
            chinese_tokens.append(chinese_chars[i] + chinese_chars[i+1])
    else:
        chinese_tokens = chinese_chars
        
    return ascii_tokens + chinese_tokens

def calculate_similarity(val1: str, val2: str) -> float:
    """计算两个属性值的 Jaccard 相似度"""
    tokens1 = set(split_tokens(val1))
    tokens2 = set(split_tokens(val2))
    
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0

def should_merge_values(val1: str, val2: str, threshold: float = 0.4) -> bool:
    """判断两个属性值是否语义高度重合，应该被合并"""
    # 1. 包含关系直接合并
    if val1 in val2 or val2 in val1:
        return True
        
    # 2. 相似度超过阈值合并
    sim = calculate_similarity(val1, val2)
    return sim >= threshold
