from __future__ import annotations  # 允许更灵活地书写类型注解

from dataclasses import dataclass  # 用 dataclass 简化纯数据配置类写法
from typing import List, Sequence  # 为列表和序列配置类型注解


@dataclass(frozen=True)  # 关系模式是配置数据，不希望在运行过程中被随意修改
class RelationPattern:  # 定义“单条关系规则”的数据结构
    relation: str  # 最终输出到三元组里的关系名，例如 is_a
    name: str  # 这条规则的人类可读名称，便于调试和答辩说明
    cues: Sequence[str]  # 关系触发词或触发模式列表，会在两个实体中间文本里匹配
    direction: str  # 关系方向，lr 表示左到右，rl 表示右到左，both 表示弱方向关系
    base_score: float  # 这条关系规则的基础置信度
    max_entity_gap: int = 48  # 两个实体之间允许的最大字符间隔，过远通常不可靠


RELATION_PATTERNS: List[RelationPattern] = [  # 维护系统支持的全部通用关系模式
    RelationPattern(  # 定义“别名关系”规则
        relation="alias_of",  # 输出关系名：别名关系
        name="别名关系",  # 便于人工理解的中文名
        cues=(r"又称", r"也称", r"也叫", r"亦称", r"简称", r"缩写", r"即", r"俗称"),  # 典型别名触发词
        direction="both",  # 别名关系方向较弱，因此用 both
        base_score=0.92,  # 别名触发词一般比较可靠，所以基础分较高
        max_entity_gap=20,  # 别名关系通常两个实体挨得比较近
    ),
    RelationPattern(  # 定义“上下位关系”规则
        relation="is_a",  # 输出关系名：is_a
        name="上下位关系",  # 中文说明
        cues=(r"是", r"属于", r"归类为", r"可视为", r"可看作"),  # 常见上下位触发词
        direction="lr",  # 默认左实体是下位项，右实体是上位项
        base_score=0.83,  # 该规则基础可靠度
        max_entity_gap=36,  # 上下位关系允许的实体间距
    ),
    RelationPattern(  # 定义“由整体列出部分”的反向上下位关系
        relation="is_a",  # 仍然输出 is_a 关系
        name="上位包含关系",  # 中文说明
        cues=(r"包括", r"包含", r"分为", r"由.+?组成"),  # 常见整体列举子项的表达方式
        direction="rl",  # 这类句子通常是右侧实体属于左侧整体，所以方向反过来
        base_score=0.78,  # 相比显式“是”略弱一些，因此分数稍低
        max_entity_gap=48,  # 这类表达往往跨度更大，所以放宽一点间距
    ),
    RelationPattern(  # 定义“部分属于整体”规则
        relation="part_of",  # 输出关系名：part_of
        name="部分整体关系",  # 中文说明
        cues=(r"是.+?一部分", r"组成", r"构成", r"隶属于"),  # 常见 part_of 触发模式
        direction="lr",  # 左实体是部分，右实体是整体
        base_score=0.80,  # 该规则的基础置信度
        max_entity_gap=36,  # 允许的最大实体间距
    ),
    RelationPattern(  # 定义“整体包含部分”规则
        relation="has_part",  # 输出关系名：has_part
        name="整体部分关系",  # 中文说明
        cues=(r"包含", r"包括", r"由.+?构成", r"由.+?组成"),  # 常见 has_part 触发表达
        direction="lr",  # 左实体是整体，右实体是部分
        base_score=0.76,  # 基础置信度略低于显式关系
        max_entity_gap=48,  # 允许的实体跨度稍大
    ),
    RelationPattern(  # 定义“用途关系”规则
        relation="used_for",  # 输出关系名：used_for
        name="用途关系",  # 中文说明
        cues=(r"用于", r"用来", r"用作", r"可用于", r"常用于", r"适用于"),  # 常见用途触发词
        direction="lr",  # 一般左实体用于右侧目标或任务
        base_score=0.81,  # 用途关系触发词相对明确
        max_entity_gap=48,  # 允许适中的实体间距
    ),
    RelationPattern(  # 定义“依赖关系”规则
        relation="depends_on",  # 输出关系名：depends_on
        name="依赖关系",  # 中文说明
        cues=(r"依赖", r"基于", r"建立在", r"取决于"),  # 常见依赖关系触发词
        direction="lr",  # 左实体依赖右实体
        base_score=0.82,  # 依赖关系表达通常较明确
        max_entity_gap=42,  # 允许的最大实体间距
    ),
    RelationPattern(  # 定义“因果关系”规则
        relation="causes",  # 输出关系名：causes
        name="因果关系",  # 中文说明
        cues=(r"导致", r"引起", r"造成", r"使得", r"会使"),  # 常见因果触发词
        direction="lr",  # 左实体导致右实体
        base_score=0.80,  # 因果关系有一定可靠度
        max_entity_gap=42,  # 允许的最大字符间距
    ),
    RelationPattern(  # 定义“实现关系”规则
        relation="implemented_by",  # 输出关系名：implemented_by
        name="实现关系",  # 中文说明
        cues=(r"实现", r"由.+?实现"),  # 常见实现关系表达
        direction="rl",  # 这类句子里往往是“接口 由 实现类 实现”，所以右到左输出更合理
        base_score=0.84,  # 实现关系表达较明确，基础分较高
        max_entity_gap=40,  # 允许的最大实体间距
    ),
    RelationPattern(  # 定义“继承关系”规则
        relation="extends",  # 输出关系名：extends
        name="继承关系",  # 中文说明
        cues=(r"继承", r"扩展自", r"派生自"),  # 常见继承关系触发词
        direction="lr",  # 左实体继承右实体
        base_score=0.85,  # 继承关系通常较明确，因此基础分较高
        max_entity_gap=36,  # 允许的最大实体间距
    ),
]  # 所有通用关系模式定义结束
