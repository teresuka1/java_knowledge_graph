# 关系消歧说明

本目录实现了一个面向知识图谱三元组 CSV 的通用关系消歧工具。它不是针对当前 Java 后端知识图谱写死规则，而是把可变的关系体系放在 `default_relation_taxonomy.json` 中；换到其他知识图谱时，优先修改这个 JSON，而不是改 Python 代码。

## 文件结构

- `main.py`：命令行入口，默认读取 `../关系抽取结果/*.csv`，输出到 `关系消歧结果/`。
- `default_relation_taxonomy.json`：关系词表、同义关系、互逆关系、对称关系、证据触发词和反触发词配置。
- `taxonomy.py`：加载关系词表，把原始关系标签映射到规范关系。
- `disambiguator.py`：核心消歧逻辑，完成方向统一、冲突评分、三元组合并和审计记录生成。
- `io_utils.py`：CSV 读写、字段别名适配和报告生成。
- `models.py` / `text_utils.py`：数据结构与文本规范化工具。
- `关系消歧结果/`：运行后生成的结果目录。

## 使用方法

在本目录执行：

```bash
python main.py
```

默认输入：

```text
../关系抽取结果/*.csv
```

默认输出：

```text
关系消歧结果/
```

也可以处理单个文件：

```bash
python main.py --input-file ../关系抽取结果/all_relations.csv --output-dir 关系消歧结果
```

如果换成其他知识图谱，可以准备同样结构的 CSV。默认识别字段包括：

- `head` / `subject` / `source`：头实体
- `relation` / `predicate` / `edge`：关系
- `tail` / `object` / `target`：尾实体
- `confidence` / `score`：抽取置信度
- `evidence` / `context` / `sentence`：证据文本

`head_type`、`tail_type`、`pattern_name`、`section_title`、`source_file`、`method` 都是可选字段。

## 消歧方法

本工具使用的是“关系词表规范化 + 方向规范化 + 证据加权投票”的规则/统计混合方法。

### 1. 关系标签规范化

不同抽取器或不同语料可能会给同一语义贴不同标签，例如：

- `same_as`、`synonym_of`、`别名` 统一为 `alias_of`
- `contains`、`includes`、`包含` 统一为 `has_part`
- `requires`、`based_on`、`依赖` 统一为 `depends_on`

这些映射都配置在 `default_relation_taxonomy.json` 的 `aliases` 中。

### 2. 互逆关系方向统一

有些关系表达的是同一个语义，但方向相反。例如：

```text
A part_of B
```

会被规范成：

```text
B has_part A
```

这类规则由 JSON 中的 `"swap": true` 控制。当前默认把 `part_of`、`component_of`、`member_of` 统一成反向的 `has_part`；把 `implements` 统一成反向的 `implemented_by`。

### 3. 对称关系端点排序

`alias_of` 这类关系没有方向，`A alias_of B` 和 `B alias_of A` 应视为同一条边。工具会按实体规范化 key 排序端点，避免重复边。

### 4. 同一实体对多关系冲突消歧

如果同一组规范化实体对上出现多个候选关系，例如：

```text
传输层 has_part TCP
传输层 used_for TCP
```

工具会对候选关系分别打分，然后只保留分数最高的关系。分数由四部分组成：

- 抽取置信度：原始 `confidence` 的最大值和平均值。
- 证据触发词：证据文本、规则名、方法名、章节标题中是否出现该关系的触发词。
- 关系先验：关系类型本身的优先级，配置在 JSON 的 `priority`。
- 支持度：同一规范三元组被多少行支持。

公式在 `disambiguator.py` 中：

```text
score = 0.62 * confidence + 0.23 * cue_score + 0.10 * relation_priority + 0.05 * support_score
```

同时支持反触发词。例如 `is_a` 如果证据里出现“用于、基于、包含、导致”等词，会降低 `is_a` 的证据分，因为这些词更可能指向用途、依赖、组成或因果关系。

### 5. 重复规范三元组合并

如果多行经过标签规范、方向统一后变成同一个三元组，会合并为一行输出，并保留：

- `record_count`：合并了多少条原始记录
- `source_row_ids`：原始行号
- `source_relations`：原始关系标签
- `evidence`：去重后的证据摘要
- `disambiguation_basis`：评分依据

## 输出文件

每个输入 CSV 会生成两个文件：

```text
<原文件名>_disambiguated.csv
<原文件名>_decisions.csv
```

`*_disambiguated.csv` 是最终规范关系结果，核心字段包括：

- `head, relation, tail`：消歧后的三元组
- `confidence`：保留关系的平均抽取置信度
- `conflicting_relations`：被消掉的竞争关系
- `disambiguation_score`：最终消歧分数
- `disambiguation_method`：使用的方法
- `disambiguation_basis`：候选关系分数与证据依据

`*_decisions.csv` 是审计文件，每条原始关系都会有一条决策记录：

- `decision=kept`：原记录被保留
- `decision=canonicalized`：原记录做了标签、方向或对称端点规范化
- `decision=relation_conflict_resolved`：原记录所在实体对有多关系冲突，该记录对应关系未被保留

还会生成 `消歧报告.md`，汇总输入行数、输出行数、冲突实体对数量、方向统一数量和示例。

## 本次消了什么歧

在当前知识图谱关系抽取结果中，主要处理了四类歧义：

1. `part_of` 与 `has_part` 的互逆方向歧义。
2. `alias_of` 的对称端点歧义。
3. 同一实体对上的多关系标签冲突，例如 `has_part` 与 `used_for`、`is_a` 与 `depends_on`。
4. 规范化后三元组重复，合并为一条可追溯的关系。

示例：

```text
原始候选：
传输层 has_part TCP
传输层 used_for TCP

消歧后：
传输层 has_part TCP
```

原因是证据类似“传输层：TCP、UDP、SCTP”，更像层级/枚举组成关系，而不是用途关系。

另一个方向统一示例：

```text
原始：
A part_of B

规范：
B has_part A
```

这个转换不依赖当前 Java 文本，只依赖关系词表中 `part_of` 的 `"swap": true` 配置，因此可以复用到其他知识图谱。

## 适配其他知识图谱

如果你的新图谱有自己的关系集合，只需要复制并修改 `default_relation_taxonomy.json`：

```json
{
  "canonical": "located_in",
  "aliases": ["located_in", "in_location", "位于"],
  "priority": 0.78,
  "cues": ["位于", "坐落于", "located in"],
  "negative_cues": ["用于", "导致"]
}
```

如果某个关系是互逆表达，增加：

```json
{
  "canonical": "located_in",
  "aliases": ["contains_place"],
  "swap": true
}
```

如果某个关系是对称关系，增加：

```json
{
  "canonical": "related_to",
  "aliases": ["related_to", "关联"],
  "symmetric": true
}
```

这样代码仍然不需要修改。
