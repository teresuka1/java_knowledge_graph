from __future__ import annotations

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple

from property_patterns import PROPERTY_PATTERNS, PropertyPattern

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_TEXT_DIR = BASE_DIR / "原始文本"
ENTITY_DIR = BASE_DIR / "实体抽取结果"
OUT_DIR = BASE_DIR / "属性抽取结果"

SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;]\s*")

def read_text(path: Path) -> str:
    """健壮的文本读取"""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def split_sentences(text: str) -> List[str]:
    """文本分句"""
    pieces: List[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        parts = SENTENCE_SPLIT_RE.split(block)
        for part in parts:
            sentence = part.strip()
            if sentence:
                pieces.append(sentence)
    return pieces

def load_valid_entities(entity_dir: Path) -> Set[str]:
    """从实体抽取结果中加载所有合法的实体，确保属性抽取有的放矢"""
    valid_entities = set()
    if not entity_dir.exists():
        print(f"警告：找不到实体目录 {entity_dir}")
        return valid_entities
    
    for csv_file in entity_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            # 兼容各种列名 (entity 或 main_entity)
            if "main_entity" in df.columns:
                valid_entities.update(df["main_entity"].dropna().astype(str).tolist())
            elif "entity" in df.columns:
                valid_entities.update(df["entity"].dropna().astype(str).tolist())
            
            # 如果有 aliases 列，也加进去
            if "aliases" in df.columns:
                for aliases_str in df["aliases"].dropna():
                    valid_entities.update([a.strip() for a in str(aliases_str).split("|") if a.strip()])
        except Exception as e:
            print(f"读取实体文件失败 {csv_file}: {e}")
            
    return {e for e in valid_entities if len(e) >= 2}

def clean_property_value(value: str) -> str:
    """清洗属性值"""
    value = re.sub(r"^[：:,\s]+", "", value)
    value = re.sub(r"[\s]+$", "", value)
    return value.strip()

def extract_properties_from_sentence(sentence: str, valid_entities: Set[str]) -> List[Dict]:
    """基于模式和合法实体从单句中抽取属性"""
    results = []
    for pattern in PROPERTY_PATTERNS:
        for cue in pattern.cues:
            # 构建正则，捕获实体和属性值
            if pattern.direction == "lr":
                # 实体在左，cue在中，值在右
                regex = f"([A-Za-z0-9_\\-\u4e00-\u9fff]{{2,30}}?)(?:{cue})(.*)"
            else:
                # 值为左，cue在中，实体在右
                regex = f"(.*?)(?:{cue})([A-Za-z0-9_\\-\u4e00-\u9fff]{{2,30}}?)$"
                
            matches = re.finditer(regex, sentence)
            for match in matches:
                if pattern.direction == "lr":
                    entity = match.group(1).strip()
                    value = match.group(2).strip()
                else:
                    value = match.group(1).strip()
                    entity = match.group(2).strip()
                    
                value = clean_property_value(value)
                
                # 严格校验：实体必须是之前实体抽取阶段识别出来的合法实体
                if entity in valid_entities and 2 <= len(value) <= pattern.max_value_len:
                    results.append({
                        "entity": entity,
                        "property_name": pattern.name,
                        "property_value": value,
                        "evidence": sentence,
                        "pattern_name": cue,
                        "confidence": pattern.base_score,
                    })
    return results

def run_extraction():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_entities = load_valid_entities(ENTITY_DIR)
    print(f"已加载 {len(valid_entities)} 个合法实体作为抽取基准。")

    all_properties = []
    
    for file_path in RAW_TEXT_DIR.glob("*.txt"):
        text = read_text(file_path)
        domain = file_path.stem
        sentences = split_sentences(text)
        
        domain_properties = []
        for sentence in sentences:
            props = extract_properties_from_sentence(sentence, valid_entities)
            for p in props:
                p["domain"] = domain
                p["source_file"] = file_path.name
                domain_properties.append(p)
                all_properties.append(p)
                
        # 按领域保存，即使为空也生成文件以保持结构一致
        columns = ["entity", "property_name", "property_value", "evidence", "pattern_name", "confidence", "domain", "source_file"]
        df_domain = pd.DataFrame(domain_properties, columns=columns)
        df_domain.to_csv(OUT_DIR / f"{domain}.csv", index=False, encoding="utf-8-sig")

    # 保存全量
    if all_properties:
        df_all = pd.DataFrame(all_properties)
        out_file = OUT_DIR / "all_properties.csv"
        df_all.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"属性抽取完成，共抽取 {len(df_all)} 条属性，保存至 {out_file}")
    else:
        print("未抽取到任何属性。")

if __name__ == "__main__":
    run_extraction()
