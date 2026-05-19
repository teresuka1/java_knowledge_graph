from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

from property_scoring import should_merge_values

BASE_DIR = Path(__file__).resolve().parent.parent
IN_DIR = BASE_DIR / "属性抽取结果"
OUT_DIR = BASE_DIR / "属性消歧结果"

def deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def fuse_property_values(values: List[str]) -> str:
    """融合相似的属性值，保留最长/最完整的表述"""
    if not values:
        return ""
        
    fused: List[str] = []
    for val in sorted(values, key=len, reverse=True): # 从长到短处理
        # 如果当前 val 和已有的 fused 中的任何一个相似，则跳过（因为已经保留了较长的）
        is_redundant = False
        for existing in fused:
            if should_merge_values(existing, val):
                is_redundant = True
                break
        if not is_redundant:
            fused.append(val)
            
    return " | ".join(fused)

def run_disambiguation():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not IN_DIR.exists():
        print(f"输入目录不存在: {IN_DIR}")
        return
        
    all_dfs = []
    for csv_file in IN_DIR.glob("*.csv"):
        if csv_file.name == "all_properties.csv":
            continue
        all_dfs.append(pd.read_csv(csv_file, encoding="utf-8-sig"))
        
    if not all_dfs:
        print("未找到属性抽取结果。")
        return
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # 按 Entity + PropertyName 分组进行融合
    grouped_data = []
    
    grouped = df.groupby(["entity", "property_name", "domain"])
    for (entity, prop_name, domain), group in grouped:
        raw_values = group["property_value"].dropna().astype(str).tolist()
        evidences = group["evidence"].dropna().astype(str).tolist()
        source_files = group["source_file"].dropna().astype(str).tolist()
        
        merged_value = fuse_property_values(raw_values)
        method = "基于 Jaccard Token Overlap 的贪心融合" if len(raw_values) > 1 else "直接保留单值"
        
        grouped_data.append({
            "main_entity": entity,
            "property_name": prop_name,
            "merged_values": merged_value,
            "value_count": len(raw_values),
            "evidence": "；".join(deduplicate_preserve_order(evidences)[:3]), # 最多保留3条证据
            "source_file": source_files[0] if source_files else "",
            "domain": domain,
            "confidence": round(group["confidence"].mean(), 4),
            "disambiguation_method": method
        })
        
    final_df = pd.DataFrame(grouped_data, columns=[
        "main_entity", "property_name", "merged_values", "value_count", 
        "evidence", "source_file", "domain", "confidence", "disambiguation_method"
    ])
    if not final_df.empty:
        final_df.sort_values(by=["value_count", "main_entity"], ascending=[False, True], inplace=True)
    
    # 按领域分别保存
    domains = df["domain"].unique() if not df.empty else []
    for domain in domains:
        domain_df = final_df[final_df["domain"] == domain]
        domain_df.to_csv(OUT_DIR / f"{domain}.csv", index=False, encoding="utf-8-sig")
        
    # 保存全量汇总文件
    out_file = OUT_DIR / "all_properties.csv"
    final_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"属性消歧完成，共生成 {len(final_df)} 条高质量规范化属性。")
    print(f"已按照领域分发生成: {', '.join(str(d)+'.csv' for d in domains)}")

if __name__ == "__main__":
    run_disambiguation()
