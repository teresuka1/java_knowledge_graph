from __future__ import annotations  # 允许更灵活地使用类型注解，减少前向引用带来的问题

import re  # 提供正则表达式能力，用于切句、标题识别和关系触发词匹配
from dataclasses import dataclass  # 用 dataclass 简化数据对象定义
from pathlib import Path  # 用面向对象方式处理文件路径
from collections import Counter  # 统计实体类型出现频次，选出更可信的主类型
from typing import Dict, Iterable, List, Sequence, Tuple  # 给核心数据结构补充类型注解

import pandas as pd  # 读取实体 csv 文件

from relation_patterns import RELATION_PATTERNS, RelationPattern  # 导入全部关系规则和规则数据结构


ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-./]+")  # 匹配英文、数字和常见符号组成的实体，例如 SQL、JVM、HashMap
SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;]\s*")  # 按中英文句末符号切句，这是句内关系抽取的第一步
HEADING_RE = re.compile(  # 识别知识笔记里的标题行，用于先分章节再抽关系
    r"^\s*(?:"  # 允许标题前面有空白，并从这里开始匹配标题编号
    r"第?[一二三四五六七八九十百千万零两\d]+(?:章|节|部分)?"  # 匹配“第一章”“第2节”这类形式
    r"|[一二三四五六七八九十百千万零两]+"  # 匹配“一、二、三”这类中文编号
    r"|[0-9]+(?:\.[0-9]+)*"  # 匹配“1”“1.1”“2.3.4”这类数字编号
    r"|[A-Za-z]"  # 匹配“A.”这类英文字母编号
    r")"  # 编号部分到此结束
    r"(?:[、.．)）]|(?:\s+))+\s*(.+?)\s*$"  # 匹配编号后的分隔符，并提取真正的标题文字
)
GENERIC_ENTITY_SUFFIXES = (  # 这些后缀常让短语更像“标题”而不是具体图谱节点
    "方式",  # 如“创建方式”通常更像标题而不是实体
    "方法",  # 如“实现方法”也偏泛化
    "特性",  # 如“核心特性”通常是概括词
    "过程",  # 如“执行过程”更偏描述过程
    "关系",  # 如“继承关系”更多是知识点名
    "定义",  # 如“事务定义”偏标题化
    "分类",  # 如“索引分类”偏章节名
    "概述",  # 如“系统概述”偏总结标题
    "简介",  # 如“JVM简介”偏介绍标题
)


@dataclass  # 把这个类定义成轻量数据对象，方便存实体信息
class EntityNode:  # 表示“实体表中的一个实体节点”
    canonical: str  # 标准实体名，也就是最终统一后的主实体名
    entity_type: str  # 这个实体所属的实体类型
    aliases: List[str]  # 这个实体可能出现的别名列表
    mention_count: int  # 这个实体在前序步骤中累计出现的次数


@dataclass  # 这个类也只是保存数据，不需要复杂行为
class Mention:  # 表示“一个实体在某个句子里的一次出现”
    canonical: str  # 它最终映射到哪个标准实体
    surface: str  # 它在句子中实际出现的文本表面形式
    start: int  # mention 在句子中的起始位置
    end: int  # mention 在句子中的结束位置
    entity_type: str  # mention 对应实体的类型


@dataclass  # 用 dataclass 表达“带章节信息的句子”更直观
class SectionSentence:  # 表示“句子 + 它所属章节”的组合对象
    section_title: str  # 这句话位于哪个章节标题下
    sentence: str  # 具体的句子文本


def read_text(path: Path) -> str:  # 以尽量稳健的方式读取文本文件
    raw = path.read_bytes()  # 先按原始字节读取，避免直接指定错误编码导致失败
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):  # 依次尝试常见中文文本编码
        try:  # 尝试用当前编码解码
            return raw.decode(enc)  # 解码成功就立刻返回文本
        except UnicodeDecodeError:  # 当前编码不适合这个文件时
            continue  # 换下一个编码继续尝试
    return raw.decode("utf-8", errors="ignore")  # 如果都失败，就尽量忽略非法字符做兜底读取


def normalize_text(text: str) -> str:  # 统一文本格式，减少匹配时的噪声
    text = (text or "").replace("\u3000", " ")  # 把全角空格替换成普通空格
    text = re.sub(r"\r\n?", "\n", text)  # 把不同系统的换行符统一成 \n
    text = re.sub(r"[ \t]+", " ", text)  # 把连续空格和制表符压缩成一个空格
    return text.strip()  # 去掉首尾空白后返回


def split_sentences(text: str) -> List[str]:  # 把整段文本切成句子列表
    pieces: List[str] = []  # 用来保存切好的句子
    for block in normalize_text(text).split("\n"):  # 先按换行切成块，适应笔记式文本结构
        block = block.strip()  # 去掉块前后的空白
        if not block:  # 如果这一块是空的
            continue  # 就直接跳过
        parts = SENTENCE_SPLIT_RE.split(block)  # 再按句末标点切句
        if len(parts) == 1:  # 如果没有切开
            parts = [block]  # 就把整个块当成一个句子
        for part in parts:  # 遍历切出来的每一个片段
            sentence = part.strip()  # 去掉片段前后的空白
            if sentence:  # 只有非空句子才保留
                pieces.append(sentence)  # 加入句子列表
    return pieces  # 返回全部句子


def normalize_name(text: str) -> str:  # 标准化实体名或短语表达
    text = normalize_text(text)  # 先做基础空白归一化
    text = text.strip("`'\"“”‘’[]{}【】<>《》")  # 去掉首尾常见包裹符号
    text = re.sub(r"\s*([()/._:+\-])\s*", r"\1", text)  # 去掉标点附近多余空格，保证写法统一
    text = re.sub(r"\s+", " ", text)  # 再次压缩连续空格
    return text.strip()  # 去掉首尾空白后返回


def build_loose_key(text: str) -> str:  # 构建更宽松的实体归一化 key
    text = normalize_name(text).lower()  # 先标准化文本并统一成小写
    return re.sub(r"[\s\-_/.:：,，;；()（）\[\]【】]+", "", text)  # 去掉空格和常见分隔符，便于聚合近似实体


def split_aliases(raw_value: object) -> List[str]:  # 把 csv 中的别名字段拆成列表
    if pd.isna(raw_value):  # 如果这一列本身是空值
        return []  # 就返回空列表
    parts = [normalize_name(part) for part in str(raw_value).split("|")]  # 按竖线拆分并逐项标准化
    return [p for p in parts if p]  # 只保留非空别名


def split_type_tokens(text: str) -> List[str]:  # 把实体类型拆成可比较的小 token
    normalized = normalize_name(text)  # 先规范类型字符串
    ascii_tokens = re.findall(r"[A-Za-z]+", normalized)  # 提取英文 token
    chinese_tokens = re.findall(r"[\u4e00-\u9fff]{1,6}", normalized)  # 提取中文 token
    return ascii_tokens + chinese_tokens  # 合并后返回


def token_overlap_score(left: str, right: str) -> float:  # 计算两个类型字符串的 token 重叠比例
    left_tokens = set(split_type_tokens(left))  # 把左侧类型拆成 token 集合
    right_tokens = set(split_type_tokens(right))  # 把右侧类型拆成 token 集合
    if not left_tokens or not right_tokens:  # 如果任意一边没有可比较 token
        return 0.0  # 直接认为重叠度为 0
    union = left_tokens | right_tokens  # 计算并集
    if not union:  # 理论上的额外保护
        return 0.0  # 避免除零
    return len(left_tokens & right_tokens) / len(union)  # 用交集除以并集得到重叠分数


def _entity_score(canonical: str, mention_count: int) -> Tuple[int, int]:  # 给候选标准名一个简单代表性分数
    return (max(mention_count, 1), len(canonical))  # 优先看出现次数，其次看实体名长度


def load_entities(csv_path: Path) -> List[EntityNode]:  # 从实体 csv 中读取并整理可用于关系抽取的实体列表
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # 读取实体表 csv
    if df.empty:  # 如果文件是空表
        return []  # 就直接返回空实体列表

    grouped: Dict[str, Dict[str, object]] = {}  # 用松散 key 聚合实体，减少格式差异带来的重复
    for _, row in df.iterrows():  # 逐行遍历实体表
        canonical = normalize_name(  # 读取当前行的主实体名并做标准化
            str(row.get("main_entity", row.get("entity", "")) or "")  # 优先取 main_entity，没有时兼容 entity 字段
        )
        if not canonical:  # 如果标准实体名为空
            continue  # 跳过这一行
        entity_type = normalize_name(str(row.get("entity_type", "") or ""))  # 读取并标准化实体类型
        mention_count = int(row.get("mention_count", 1) or 1)  # 读取出现次数，默认至少为 1
        aliases = [canonical]  # 先把标准实体名本身加入别名集合
        aliases.extend(split_aliases(row.get("aliases")))  # 合并 aliases 列中的别名
        aliases.extend(split_aliases(row.get("mentions")))  # 合并 mentions 列中的表面提及写法
        aliases = dedup_preserve_order(normalize_name(a) for a in aliases if a and len(normalize_name(a)) >= 2)  # 去重并过滤过短别名

        key = build_loose_key(canonical) or canonical  # 为当前实体生成松散归一化 key
        if key not in grouped:  # 如果这是这个 key 第一次出现
            grouped[key] = {  # 初始化这个聚合桶
                "canonical": canonical,  # 先暂存当前标准名
                "mention_count": max(mention_count, 1),  # 先暂存当前出现次数
                "aliases": list(aliases),  # 先暂存当前别名列表
                "type_counter": Counter(),  # 用计数器累计实体类型频次
            }
        bucket = grouped[key]  # 取出当前 key 对应的聚合桶
        if _entity_score(canonical, mention_count) > _entity_score(  # 如果当前标准名代表性更强
            str(bucket["canonical"]), int(bucket["mention_count"])  # 就和桶里已保存的标准名做比较
        ):
            bucket["canonical"] = canonical  # 用当前更优的实体名覆盖旧标准名
        bucket["mention_count"] = int(bucket["mention_count"]) + max(mention_count, 1)  # 累加出现次数
        bucket["aliases"] = dedup_preserve_order([*bucket["aliases"], *aliases])  # 合并并去重别名集合
        if entity_type:  # 如果当前实体有类型信息
            bucket["type_counter"][entity_type] += max(mention_count, 1)  # 按出现次数给该类型加权计数

    entities: List[EntityNode] = []  # 用来保存最终整理好的实体对象
    for bucket in grouped.values():  # 遍历每个聚合后的实体桶
        type_counter: Counter = bucket["type_counter"]  # 取出这个实体的类型计数器
        entity_type = type_counter.most_common(1)[0][0] if type_counter else ""  # 选出现次数最多的类型作为主类型
        entities.append(  # 构造最终实体对象
            EntityNode(  # 创建一个标准化后的实体节点
                canonical=str(bucket["canonical"]),  # 写入最终标准实体名
                entity_type=entity_type,  # 写入最终主类型
                aliases=dedup_preserve_order(bucket["aliases"]),  # 写入去重后的别名列表
                mention_count=int(bucket["mention_count"]),  # 写入累计出现次数
            )
        )
    entities.sort(key=lambda e: (-e.mention_count, -len(e.canonical), e.canonical))  # 按出现次数和长度排序，便于后续优先处理重要实体
    return entities  # 返回全部整理好的实体节点


def dedup_preserve_order(items: Iterable[str]) -> List[str]:  # 去重但尽量保持原有顺序不变
    seen = set()  # 记录已经出现过的元素
    result: List[str] = []  # 保存去重后的结果
    for item in items:  # 按顺序遍历输入项
        if item in seen:  # 如果这个元素已经出现过
            continue  # 就跳过，避免重复
        seen.add(item)  # 把新元素加入已见集合
        result.append(item)  # 并追加到最终结果中
    return result  # 返回按原顺序去重后的列表


def _match_ascii_alias(sentence: str, alias: str) -> List[Tuple[int, int]]:  # 在句子中匹配英文/ASCII 类型实体
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", re.IGNORECASE)  # 构造带边界控制的正则，防止误匹配更长单词内部
    return [(m.start(), m.end()) for m in pattern.finditer(sentence)]  # 返回每个命中的起止位置


def _match_non_ascii_alias(sentence: str, alias: str) -> List[Tuple[int, int]]:  # 在句子中匹配中文或其他非 ASCII 实体
    hits: List[Tuple[int, int]] = []  # 保存全部命中的区间
    start = 0  # 从句首开始查找
    while True:  # 循环找出所有出现位置
        idx = sentence.find(alias, start)  # 从当前起点开始寻找 alias
        if idx < 0:  # 如果再也找不到
            break  # 退出循环
        hits.append((idx, idx + len(alias)))  # 保存当前这次命中的区间
        start = idx + 1  # 从下一个字符继续向后查找，支持重叠式扫描
    return hits  # 返回全部命中区间


def find_mentions(sentence: str, entities: Sequence[EntityNode]) -> List[Mention]:  # 在一个句子里找到所有实体出现位置
    sentence_norm = normalize_text(sentence)  # 先规范句子格式，减少空白噪声
    mentions: List[Mention] = []  # 保存句中找到的所有实体 mention
    alias_items: List[Tuple[str, str, str, int]] = []  # 暂存 alias、canonical、type、mention_count 的组合
    for entity in entities:  # 遍历所有候选实体
        for alias in entity.aliases:  # 遍历该实体的全部别名
            alias_items.append((alias, entity.canonical, entity.entity_type, entity.mention_count))  # 展平为统一匹配项
    alias_items.sort(key=lambda x: (-len(x[0]), x[0]))  # 先按别名长度降序排序，优先匹配更长更具体的实体

    for alias, canonical, entity_type, _ in alias_items:  # 遍历排序后的全部 alias 匹配项
        if len(alias) < 2:  # 过短 alias 容易误命中
            continue  # 所以直接跳过
        if ASCII_TOKEN_RE.fullmatch(alias):  # 如果这个 alias 是英文/ASCII 实体
            spans = _match_ascii_alias(sentence_norm, alias)  # 用带边界控制的英文匹配逻辑
        else:  # 否则把它当作中文或非 ASCII 实体
            spans = _match_non_ascii_alias(sentence_norm, alias)  # 用简单子串扫描匹配
        for start, end in spans:  # 遍历当前 alias 的全部命中位置
            mentions.append(  # 为每一次命中构造 mention 对象
                Mention(  # 创建一个句内实体出现记录
                    canonical=canonical,  # 记录它映射到哪个标准实体
                    surface=sentence_norm[start:end],  # 记录它在句子里的实际文本
                    start=start,  # 记录起始位置
                    end=end,  # 记录结束位置
                    entity_type=entity_type,  # 记录实体类型
                )
            )
    mentions.sort(key=lambda m: (m.start, -(m.end - m.start)))  # 先按起点排序，同起点时优先更长 mention
    return _resolve_overlaps(mentions)  # 最后解决重叠问题后返回 mention 列表


def _resolve_overlaps(mentions: Sequence[Mention]) -> List[Mention]:  # 解决同一区间被多个实体同时命中的问题
    selected: List[Mention] = []  # 保存最终选中的 mention
    occupied: List[Tuple[int, int]] = []  # 记录已经被占用的字符区间
    for mention in mentions:  # 按排序后的顺序遍历 mention
        conflict = False  # 先假设当前 mention 不冲突
        for s, e in occupied:  # 遍历已经占用的区间
            if not (mention.end <= s or mention.start >= e):  # 如果当前 mention 与已有区间发生重叠
                conflict = True  # 标记为冲突
                break  # 不必继续检查其他区间
        if conflict:  # 如果当前 mention 与已选 mention 冲突
            continue  # 就跳过它
        selected.append(mention)  # 否则把它加入最终结果
        occupied.append((mention.start, mention.end))  # 并把它的区间标记为已占用
    return selected  # 返回去重叠后的 mention 列表


def _pair_gap(left: Mention, right: Mention) -> int:  # 计算两个实体 mention 之间的字符间隔
    return max(0, right.start - left.end)  # 如果右实体紧接着左实体，间隔最小就是 0


def _contains_cue(text: str, cues: Sequence[str]) -> str | None:  # 检查两个实体中间文本是否命中了某个关系触发词
    for cue in cues:  # 逐个遍历这条关系模式的触发词
        if re.search(cue, text):  # 如果当前触发词在文本中命中
            return cue  # 就返回命中的 cue
    return None  # 如果都没命中，则返回 None


def _is_generic_entity_name(name: str) -> bool:  # 判断一个实体名是否过于泛化、太像标题而不像实体
    s = normalize_name(name)  # 先做基础标准化
    if not s:  # 如果标准化后为空
        return True  # 认为它无效且泛化
    if len(s) <= 2:  # 太短的实体先不按泛化标题处理
        return False  # 直接返回 False
    return any(s.endswith(sfx) for sfx in GENERIC_ENTITY_SUFFIXES)  # 只要它以后缀表中的某个词结尾，就认为偏泛化


def _is_noisy_enumeration_between(text: str) -> bool:  # 判断两个实体之间是否只是纯枚举符号，没有真实关系语义
    s = normalize_text(text)  # 先规范这段中间文本
    if not s:  # 如果中间文本本身为空
        return False  # 不把它当成噪声枚举
    if re.fullmatch(r"[、,，/和及与或以及 ]+", s):  # 如果只由顿号、逗号、连词等构成
        return True  # 说明更像纯枚举连接
    if re.fullmatch(r"[：:、,， ]*(是|属于|继承|实现|包括|包含)[、,， ]*", s):  # 如果只是非常弱的模板壳子
        return True  # 也当成噪声处理，避免误连边
    return False  # 否则认为不是纯枚举噪声


def _relation_specific_guard(  # 在关系模式命中之后，再做一轮质量过滤
    pattern: RelationPattern,  # 当前命中的关系规则
    left: Mention,  # 左侧实体 mention
    right: Mention,  # 右侧实体 mention
    between: str,  # 两个实体之间的文本
    head: str,  # 根据方向推导出的头实体
    tail: str,  # 根据方向推导出的尾实体
) -> bool:  # 返回这条候选关系是否应该保留
    if _is_noisy_enumeration_between(between):  # 如果两个实体中间只是纯枚举噪声
        return False  # 就不保留这条候选关系

    if pattern.relation in {"extends", "implemented_by"} and _is_generic_entity_name(head):  # 如果头实体本身太像概括标题
        return False  # 继承/实现关系就不可靠
    if pattern.relation in {"extends", "implemented_by", "is_a"} and _is_generic_entity_name(left.surface):  # 如果左侧表面词太泛化
        return False  # 这些关系也容易误判，直接过滤
    if pattern.relation == "alias_of" and abs(len(left.surface) - len(right.surface)) > 10:  # 别名关系通常两个实体长度不会差太多
        return False  # 长度差过大时别名关系不可信
    if pattern.relation == "causes" and len(between) < 1:  # 因果关系至少需要中间有明确连接文本
        return False  # 没有中间文本就不保留
    if pattern.relation in {"implemented_by", "extends"}:  # 对实现和继承关系额外做类型相似度约束
        overlap = token_overlap_score(left.entity_type, right.entity_type)  # 计算两边实体类型的 token 重叠度
        if overlap <= 0 and left.entity_type and right.entity_type and len(normalize_text(between)) > 10:  # 类型完全无关且中间跨度还大时
            return False  # 说明大概率误抽，直接过滤
    if head == tail:  # 头尾实体如果完全相同
        return False  # 就不能保留自环关系
    return True  # 通过全部过滤条件后，保留这条候选关系


def _pattern_match(  # 对一个实体对尝试套用某一条关系模式
    pattern: RelationPattern,  # 当前尝试的关系规则
    left: Mention,  # 左侧 mention
    right: Mention,  # 右侧 mention
    sentence: str,  # 当前所在句子
) -> Tuple[str, str, float] | None:  # 如果命中则返回 head、tail、confidence，否则返回 None
    if right.start <= left.end:  # 如果右实体没有出现在左实体后面
        return None  # 当前实现里不处理这种顺序
    gap = _pair_gap(left, right)  # 计算两个实体之间的字符间隔
    if gap > pattern.max_entity_gap:  # 如果间距超过该关系规则允许的最大值
        return None  # 认为它们离得太远，不适合建立关系

    between = sentence[left.end : right.start]  # 取出两个实体之间的中间文本
    cue = _contains_cue(between, pattern.cues)  # 检查中间文本是否命中该关系的触发词
    if cue is None:  # 如果一个触发词都没命中
        return None  # 当前关系模式不成立

    if pattern.direction == "lr":  # 如果规则方向是左到右
        head, tail = left.canonical, right.canonical  # 就把左实体当 head，右实体当 tail
    elif pattern.direction == "rl":  # 如果规则方向是右到左
        head, tail = right.canonical, left.canonical  # 就把右实体当 head，左实体当 tail
    else:  # 其余情况就是弱方向关系
        if left.canonical == right.canonical:  # 如果两个实体标准名本来就相同
            return None  # 就不建立关系
        head, tail = sorted([left.canonical, right.canonical], key=len)  # 否则简单按长度排序得到稳定输出顺序

    if not _relation_specific_guard(pattern, left, right, between, head, tail):  # 如果没有通过关系级质量检查
        return None  # 直接丢弃

    confidence = score_confidence(pattern.base_score, gap, left, right)  # 根据基础分和上下文因素计算最终置信度
    return head, tail, confidence  # 返回这条候选关系


def score_confidence(base_score: float, gap: int, left: Mention, right: Mention) -> float:  # 为候选关系打启发式分数
    gap_penalty = min(0.20, gap * 0.006)  # 实体间距越远，扣分越多，但最多扣 0.20
    type_bonus = 0.03 if left.entity_type and right.entity_type else 0.0  # 两边都有类型信息时给予少量奖励
    diversity_bonus = 0.02 if left.entity_type != right.entity_type else 0.0  # 类型不同但同时存在时给予一点区分度奖励
    score = base_score - gap_penalty + type_bonus + diversity_bonus  # 合成最终启发式分数
    return max(0.0, min(0.99, score))  # 把分数截断到 0 到 0.99 之间


def _extract_enumeration_relations(  # 针对“标题/冒号 + 枚举项”结构补充抽取整体-部分关系
    sentence: str,  # 当前句子
    mentions: Sequence[Mention],  # 当前句子中的全部实体 mention
    source_file: str,  # 来源文件名
    section_title: str,  # 当前句子所属章节
) -> List[Dict[str, object]]:  # 返回通过枚举结构抽到的关系列表
    rows: List[Dict[str, object]] = []  # 用来保存抽取出的关系
    colon_pos = max(sentence.find("："), sentence.find(":"))  # 找到中英文冒号位置，知识笔记常用它引出枚举结构
    if colon_pos < 0:  # 如果句子里根本没有冒号
        return rows  # 就不适用枚举结构抽取

    left_mentions = [m for m in mentions if m.end <= colon_pos]  # 收集冒号左边的实体
    right_mentions = [m for m in mentions if m.start > colon_pos]  # 收集冒号右边的实体
    if not left_mentions or len(right_mentions) < 2:  # 左边没有头实体，或右边没有足够的枚举项时
        return rows  # 不抽取枚举关系

    head_mention = max(left_mentions, key=lambda m: m.end)  # 取最靠近冒号的左侧实体作为头实体
    if _is_generic_entity_name(head_mention.surface):  # 如果这个头实体太像标题而不像具体实体
        return rows  # 就不抽这种关系

    for tail_mention in right_mentions:  # 遍历冒号右边的每个候选部分实体
        if head_mention.canonical == tail_mention.canonical:  # 如果头尾其实是同一个标准实体
            continue  # 就跳过，避免自环
        between = sentence[head_mention.end : tail_mention.start]  # 取头实体和当前尾实体之间的文本
        if len(normalize_text(between)) > 24:  # 如果头尾之间距离过大
            continue  # 说明它可能并不属于同一局部枚举结构
        if not re.search(r"[、,，/和及与或]|包括|包含|分为", sentence[colon_pos + 1 :]):  # 如果冒号右边整体上没有枚举痕迹
            continue  # 就不把它当成 has_part 结构
        confidence = score_confidence(0.76, _pair_gap(head_mention, tail_mention), head_mention, tail_mention)  # 给这条枚举关系一个启发式分数
        rows.append(  # 把枚举结构抽到的关系写成统一字典格式
            {  # 关系结果统一按字典保存，便于后续写 csv
                "head": head_mention.canonical,  # 头实体是冒号左边的主题实体
                "relation": "has_part",  # 枚举结构默认抽成整体包含部分关系
                "tail": tail_mention.canonical,  # 尾实体是右边的某一个枚举项
                "head_type": head_mention.entity_type,  # 头实体类型
                "tail_type": tail_mention.entity_type,  # 尾实体类型
                "evidence": sentence,  # 保留完整证据句，方便人工审核
                "pattern_name": "枚举结构关系",  # 标记这是由枚举结构规则抽出的关系
                "section_title": section_title,  # 保留章节信息
                "source_file": source_file,  # 保留来源文件名
                "confidence": round(confidence, 4),  # 置信度保留四位小数
                "method": "rule_pattern_sentence_level",  # 记录当前采用的方法类型
            }
        )
    return rows  # 返回全部枚举结构关系


def extract_sentence_relations(  # 在单个句子中抽取全部候选关系
    sentence: str,  # 当前句子文本
    entities: Sequence[EntityNode],  # 整个文档对应的实体表
    source_file: str,  # 来源文件名
    section_title: str = "全文",  # 当前句子所在章节，默认是全文
) -> List[Dict[str, object]]:  # 返回当前句子中抽到的关系列表
    mentions = find_mentions(sentence, entities)  # 先识别这个句子里有哪些实体出现
    if len(mentions) < 2:  # 如果句子里不足两个实体
        return []  # 就不可能形成实体间关系

    rows: List[Dict[str, object]] = _extract_enumeration_relations(  # 先尝试抽枚举结构关系
        sentence=sentence,  # 传入当前句子
        mentions=mentions,  # 传入句中 mention 列表
        source_file=source_file,  # 传入来源文件名
        section_title=section_title,  # 传入章节标题
    )
    n = len(mentions)  # 当前句子的 mention 数量
    for i in range(n):  # 枚举左侧 mention 下标
        for j in range(i + 1, n):  # 枚举右侧 mention 下标，构成所有 mention 对
            left = mentions[i]  # 取左侧 mention
            right = mentions[j]  # 取右侧 mention
            if left.canonical == right.canonical:  # 如果两者映射到同一个标准实体
                continue  # 就不建立关系
            for pattern in RELATION_PATTERNS:  # 让这对 mention 依次尝试全部关系规则
                hit = _pattern_match(pattern, left, right, sentence)  # 用当前规则判断是否形成关系
                if hit is None:  # 如果这条规则没命中
                    continue  # 就试下一条规则
                head, tail, confidence = hit  # 拿到命中的头实体、尾实体和置信度
                if head == tail:  # 再做一次保护，防止自环
                    continue  # 自环关系不保留
                rows.append(  # 把当前命中的关系写成统一结果格式
                    {  # 每一条关系都写成字典，便于后续去重和导出
                        "head": head,  # 头实体
                        "relation": pattern.relation,  # 关系类型
                        "tail": tail,  # 尾实体
                        "head_type": left.entity_type if head == left.canonical else right.entity_type,  # 根据方向确定头实体类型
                        "tail_type": right.entity_type if tail == right.canonical else left.entity_type,  # 根据方向确定尾实体类型
                        "evidence": sentence,  # 保留证据句
                        "pattern_name": pattern.name,  # 记录命中的关系模式名
                        "section_title": section_title,  # 记录章节信息
                        "source_file": source_file,  # 记录来源文件
                        "confidence": round(confidence, 4),  # 置信度保留四位小数
                        "method": "rule_pattern_sentence_level",  # 标记采用的是句级规则抽取
                    }
                )
    return rows  # 返回当前句子的全部关系候选


def deduplicate_relations(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:  # 对整篇文档抽到的关系做去重
    best: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}  # 用 (head, relation, tail, source_file) 作为去重 key
    for row in rows:  # 遍历全部关系结果
        key = (  # 为当前关系构造唯一键
            str(row["head"]),  # 键的第一部分：头实体
            str(row["relation"]),  # 键的第二部分：关系类型
            str(row["tail"]),  # 键的第三部分：尾实体
            str(row["source_file"]),  # 键的第四部分：来源文件
        )
        prev = best.get(key)  # 查看这条关系之前是否已经出现过
        if prev is None:  # 如果第一次出现
            best[key] = dict(row)  # 就直接保存它
            continue  # 然后处理下一条
        if float(row.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):  # 如果当前这条置信度更高
            best[key] = dict(row)  # 就用当前这条覆盖之前那条
    output = list(best.values())  # 把字典里的最优关系取出来形成列表
    output.sort(key=lambda r: (-float(r.get("confidence", 0.0)), str(r.get("head", "")), str(r.get("tail", ""))))  # 按置信度降序排序，便于人工查看
    return output  # 返回去重后的关系结果


def extract_heading(line: str) -> str | None:  # 尝试从一行文本中识别章节标题
    m = HEADING_RE.match(line.strip())  # 用标题正则匹配这一行
    if not m:  # 如果匹配失败
        return None  # 说明这一行不是标题
    heading = normalize_name(m.group(1))  # 取出标题正文并做标准化
    return heading[:80] if heading else None  # 返回标题，顺便把长度限制在 80 字内


def parse_sections(text: str) -> List[Tuple[str, str]]:  # 把全文切成多个“章节标题 + 章节内容”
    sections: List[Tuple[str, str]] = []  # 保存全部章节结果
    current_title = "全文"  # 如果识别不到标题，就把默认标题设为“全文”
    buffer: List[str] = []  # 暂存当前章节的正文行

    def flush() -> None:  # 把当前缓冲区内容写入 sections
        content = "\n".join([line for line in buffer if line]).strip()  # 把当前章节行拼成完整文本
        if content:  # 只有章节正文非空时才写入
            sections.append((current_title, content))  # 保存“标题 + 正文”
        buffer.clear()  # 清空缓冲区，为下一个章节做准备

    for raw_line in normalize_text(text).split("\n"):  # 逐行遍历标准化后的全文
        line = raw_line.strip()  # 去掉每一行前后空白
        if not line:  # 如果这一行为空
            continue  # 就直接跳过
        heading = extract_heading(line)  # 尝试把这一行识别成标题
        if heading:  # 如果识别到了标题
            flush()  # 先把上一章节写出去
            current_title = heading  # 再更新当前章节标题
        else:  # 如果这一行不是标题
            buffer.append(line)  # 就把它归入当前章节正文
    flush()  # 循环结束后把最后一个章节也写出去
    if not sections:  # 如果整篇文本都没有识别出章节
        fallback = normalize_text(text)  # 就把全文做一次标准化作为兜底内容
        if fallback:  # 如果全文内容非空
            sections.append(("全文", fallback))  # 以“全文”为标题保存
    return sections  # 返回全部章节


def iter_section_sentences(text: str) -> List[SectionSentence]:  # 把全文展开成“带章节信息的句子列表”
    items: List[SectionSentence] = []  # 保存最终结果
    for section_title, section_text in parse_sections(text):  # 先遍历全文的每个章节
        for sentence in split_sentences(section_text):  # 再把当前章节正文切成句子
            items.append(SectionSentence(section_title=section_title, sentence=sentence))  # 把句子和章节标题打包保存
    return items  # 返回全部“章节 + 句子”对象


def extract_document_relations(text: str, entities: Sequence[EntityNode], source_file: str) -> List[Dict[str, object]]:  # 整篇文档关系抽取的总入口
    all_rows: List[Dict[str, object]] = []  # 保存整篇文档中抽出的所有候选关系
    for item in iter_section_sentences(text):  # 遍历全文切分后的每个带章节信息的句子
        sentence_rows = extract_sentence_relations(  # 对单个句子执行句级关系抽取
            item.sentence,  # 传入句子文本
            entities,  # 传入当前文档的实体表
            source_file=source_file,  # 传入来源文件名
            section_title=item.section_title,  # 传入当前句子所属章节
        )
        all_rows.extend(sentence_rows)  # 把当前句子的结果追加到整篇文档结果中
    return deduplicate_relations(all_rows)  # 最后对整篇文档关系去重并返回
