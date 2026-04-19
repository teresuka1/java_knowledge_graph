from __future__ import annotations

import re


HEADING_RE = re.compile(
    r"^\s*(?:"
    r"第?[一二三四五六七八九十百千万零两\d]+(?:章|节|部分)?"
    r"|[一二三四五六七八九十百千万零两]+"
    r"|[0-9]+(?:\.[0-9]+)*"
    r"|[A-Za-z]"
    r")"
    r"(?:[、.．)）]|(?:\s+))+\s*(.+?)\s*$"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
ASCII_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./+\-]+$")
GENERIC_SUFFIXES = {
    "模型",
    "方法",
    "算法",
    "机制",
    "协议",
    "接口",
    "系统",
    "框架",
    "过程",
    "流程",
    "结构",
    "类型",
    "原理",
    "版本",
    "命令",
    "工具",
    "工具类",
}
