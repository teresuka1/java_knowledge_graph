from __future__ import annotations  # 允许把类型注解写成字符串形式，减少前向引用问题

import argparse  # 用来解析命令行参数
from pathlib import Path  # 用面向对象的方式处理文件路径
from typing import List, Tuple  # 给列表和元组添加类型注解

import pandas as pd  # 用来把关系结果写成 csv

from relation_extractor import extract_document_relations, load_entities, read_text  # 导入关系抽取主流程、实体加载和文本读取函数


def build_default_paths(script_path: Path) -> Tuple[Path, Path, Path]:  # 根据当前脚本位置推导默认输入输出目录
    project_root = script_path.parent.parent  # `关系抽取/main.py` 的上两级就是项目根目录
    raw_dir = project_root / "原始文本"  # 默认原始文本目录
    entity_dir = project_root / "实体消歧" / "实体消歧结果"  # 优先使用实体消歧后的实体表目录
    if not entity_dir.exists():  # 如果实体消歧结果目录不存在
        entity_dir = project_root / "实体抽取结果"  # 就退回使用实体抽取结果目录
    output_dir = project_root / "关系抽取结果"  # 关系抽取结果默认输出到根目录下这个文件夹
    return raw_dir, entity_dir, output_dir  # 返回默认的文本目录、实体目录和输出目录


def parse_args() -> argparse.Namespace:  # 解析命令行参数，支持外部指定输入输出路径
    script_path = Path(__file__).resolve()  # 获取当前脚本的绝对路径
    default_raw, default_entity, default_output = build_default_paths(script_path)  # 基于脚本位置计算默认目录
    parser = argparse.ArgumentParser(description="通用关系抽取：输入文本与实体表，输出知识图谱三元组。")  # 创建命令行参数解析器
    parser.add_argument("--raw-dir", type=Path, default=default_raw, help="原始文本目录（*.txt）")  # 注册原始文本目录参数
    parser.add_argument("--entity-dir", type=Path, default=default_entity, help="实体表目录（*.csv）")  # 注册实体表目录参数
    parser.add_argument("--output-dir", type=Path, default=default_output, help="关系结果输出目录")  # 注册关系结果输出目录参数
    return parser.parse_args()  # 真正解析并返回参数对象


def discover_pairs(raw_dir: Path, entity_dir: Path) -> List[Tuple[Path, Path]]:  # 自动发现 txt 与 csv 的同名文件对
    pairs: List[Tuple[Path, Path]] = []  # 用来保存所有成功匹配上的文件对
    for raw_path in sorted(raw_dir.glob("*.txt")):  # 遍历原始文本目录下全部 txt 文件
        csv_path = entity_dir / f"{raw_path.stem}.csv"  # 按同名规则拼出对应的实体 csv 路径
        if csv_path.exists():  # 只有对应 csv 真实存在时才算匹配成功
            pairs.append((raw_path, csv_path))  # 把这一对文件加入待处理列表
    return pairs  # 返回全部可处理的 txt/csv 文件对


def save_rows(rows: List[dict], output_path: Path) -> None:  # 把关系抽取结果保存为 csv 文件
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 如果输出目录不存在就自动创建
    df = pd.DataFrame(rows)  # 把关系字典列表转成 DataFrame
    df.to_csv(output_path, index=False, encoding="utf-8-sig")  # 以 utf-8-sig 编码写出 csv，便于 Excel 打开中文


def main() -> None:  # 整个关系抽取脚本的总入口
    args = parse_args()  # 先读取用户传入或默认的目录参数
    pairs = discover_pairs(args.raw_dir, args.entity_dir)  # 自动发现可以处理的文本/实体文件对
    if not pairs:  # 如果一对可处理文件都没找到
        print("未找到可配对的 txt/csv 文件。请检查 --raw-dir 与 --entity-dir。")  # 打印提示信息
        return  # 直接结束程序

    all_rows: List[dict] = []  # 用来汇总所有文件的关系结果，最终生成总表
    total_entities = 0  # 统计所有文件中参与关系抽取的实体总数
    for raw_path, csv_path in pairs:  # 逐组处理每个 txt/csv 文件对
        text = read_text(raw_path)  # 读取当前原始文本内容
        entities = load_entities(csv_path)  # 读取并规范化当前文件对应的实体表
        total_entities += len(entities)  # 累加当前文件的实体数量

        rows = extract_document_relations(text=text, entities=entities, source_file=raw_path.name)  # 对当前整篇文档执行关系抽取
        all_rows.extend(rows)  # 把当前文件抽到的关系追加到总表缓存里

        output_path = args.output_dir / f"{raw_path.stem}.csv"  # 为当前文件生成单独结果 csv 路径
        save_rows(rows, output_path)  # 把当前文件的关系结果写出到 csv
        print(  # 打印当前文件的处理统计信息
            f"[done] {raw_path.stem}: entities={len(entities)}, relations={len(rows)}, output={output_path}"  # 展示文件名、实体数、关系数和输出路径
        )

    merged_output = args.output_dir / "all_relations.csv"  # 生成总表输出路径
    save_rows(all_rows, merged_output)  # 把所有文件的关系结果汇总写出
    print(  # 打印整批数据的最终统计信息
        f"处理完成：files={len(pairs)}, entities={total_entities}, "  # 第一段展示处理文件数和实体总数
        f"relations={len(all_rows)}, output={merged_output}"  # 第二段展示关系总数和总表路径
    )


if __name__ == "__main__":  # 只有直接运行这个脚本时才会进入这里
    main()  # 调用主函数启动整个关系抽取流程
