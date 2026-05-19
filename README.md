# java_knowledge_graph

Java 后端面试知识图谱。

当前离线构建链路：

1. `实体抽取`
2. `实体消歧`
3. `关系抽取`
4. `关系消歧`
5. `属性抽取`
6. `属性消歧`
7. `知识存储`

后四步入口：

- `关系消歧/main.py`
- `属性抽取/main.py`
- `属性消歧/main.py`
- `知识存储/main.py`

一键执行关系抽取后的完整流水线：

```bash
python run_knowledge_graph_pipeline.py
```

最终输出位于 `知识存储/graph_store/`：

- `nodes.csv`
- `edges.csv`
- `attributes.csv`
- `graph.json`
- `knowledge_graph.db`
