from domain_ner_common import DomainConfig, run_bert_bilstm_crf_ner


def build_config() -> DomainConfig:
    return DomainConfig(
        domain_name="mysql",
        source_file="mysql.txt",
        output_csv="mysql.csv",
        default_type="领域实体",
        seed_entities=[
            # 一、领域实体
            "MySQL", "RDBMS", "关系型数据库管理系统", "SQL", "数据库", "存储引擎", "事务", "索引", "锁机制",
            "主从复制", "读写分离", "分库分表", "SQL优化", "性能调优",
            # 二、基础架构实体
            "服务层", "存储引擎层", "连接管理", "SQL解析", "优化器", "查询缓存", "执行器", "插件式架构",
            # 三、存储引擎实体
            "InnoDB", "MyISAM",
            # 四、数据库设计与规范实体
            "数据库三范式", "1NF", "2NF", "3NF", "反范式", "冗余字段", "联表查询", "主键", "唯一键", "外键", "逻辑外键", "参照完整性",
            # 五、数据类型实体
            "数值型", "int", "bigint", "double", "decimal", "字符串型", "varchar", "char", "text", "日期型", "datetime", "timestamp", "date",
            # 六、SQL分类实体
            "DDL", "DML", "DQL", "CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE", "SELECT",
            "WHERE", "ORDER BY", "LIMIT", "GROUP BY", "HAVING", "DISTINCT", "LIKE",
            # 七、查询与连接实体
            "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "UNION", "UNION ALL", "子查询", "相关子查询", "非相关子查询",
            "条件查询", "分组查询", "排序", "分页查询",
            # 八、函数实体
            "聚合函数", "COUNT()", "SUM()", "AVG()", "MAX()", "MIN()",
            "字符串函数", "CONCAT()", "SUBSTRING()", "LENGTH()",
            "日期函数", "NOW()", "DATE_FORMAT()", "DATE_ADD()",
            "数学函数", "ROUND()", "CEIL()", "FLOOR()",
            # 九、索引相关实体
            "索引", "索引下推", "B+树", "B树", "二叉树", "哈希表", "单列索引", "联合索引", "复合索引", "主键索引",
            "唯一索引", "普通索引", "全文索引", "聚簇索引", "非聚簇索引", "二级索引", "回表", "覆盖索引", "最左前缀原则", "最左匹配", "索引失效", "全表扫描",
            # 十、事务与一致性实体
            "ACID", "Atomicity", "Consistency", "Isolation", "Durability", "START TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT",
            "autocommit", "脏读", "不可重复读", "幻读", "READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE",
            "MVCC", "多版本并发控制", "undo log", "redo log", "read view", "读视图", "版本链",
            # 十一、锁机制实体
            "行锁", "表锁", "页锁", "共享锁", "读锁", "排他锁", "写锁", "意向锁", "意向共享锁", "意向排他锁",
            "Gap Lock", "间隙锁", "Next-Key Lock", "临键锁", "死锁", "锁等待超时时间",
            # 十二、执行计划与调优实体
            "慢查询日志", "long_query_time", "EXPLAIN", "执行计划", "id", "select_type", "table", "type", "key", "rows", "Extra",
            "Using index", "Using filesort", "小表驱动大表", "批量插入", "大分页", "主键id分页",
            # 十三、表设计与拆分实体
            "NOT NULL", "默认值", "垂直拆分", "水平拆分", "垂直分库", "垂直分表", "水平分表", "分区表", "大表优化", "热点字段", "非热点字段",
            # 十四、服务器参数与配置实体
            "innodb_buffer_pool_size", "max_connections", "innodb_log_file_size", "物理内存", "查询缓存",
            # 十五、主从复制与高可用实体
            "主从复制", "主库", "从库", "binlog", "二进制日志", "relay log", "IO线程", "SQL线程", "异步复制", "半同步复制", "组复制",
            "读写分离", "故障转移", "负载均衡", "主从延迟",
            # 十六、分库分表与分布式实体
            "分库分表", "范围分片", "哈希分片", "一致性哈希", "跨库分页", "跨库事务", "跨库联表查询", "分布式ID", "UUID", "雪花算法", "Snowflake", "Redis自增", "数据库自增",
            # 十七、中间件与工具实体
            "Sharding-JDBC", "MyCat", "mysqldump", "PreparedStatement", "Druid", "HikariCP", "SpringBoot",
            # 十八、运维与安全实体
            "物理备份", "逻辑备份", "数据恢复", "崩溃恢复", "SQL注入", "预编译SQL",
            # 十九、MySQL 8.0与扩展特性实体
            "MySQL8.0", "窗口函数", "公用表表达式", "CTE", "utf8mb4", "utf8", "emoji", "视图", "存储过程", "触发器", "增强索引",
        ],
        regex_entities=[
            r"\b(?:MySQL|RDBMS|SQL|DDL|DML|DQL|ACID|MVCC|UUID|CTE)\b",
            r"\b(?:InnoDB|MyISAM|Sharding-JDBC|MyCat|mysqldump|PreparedStatement|Druid|HikariCP|SpringBoot)\b",
            r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|EXPLAIN|COMMIT|ROLLBACK|SAVEPOINT)\b",
            r"\b(?:WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|DISTINCT|LIKE)\b",
            r"\b(?:INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN|UNION ALL|UNION)\b",
            r"\b(?:COUNT\(\)|SUM\(\)|AVG\(\)|MAX\(\)|MIN\(\)|CONCAT\(\)|SUBSTRING\(\)|LENGTH\(\)|NOW\(\)|DATE_FORMAT\(\)|DATE_ADD\(\)|ROUND\(\)|CEIL\(\)|FLOOR\(\))\b",
            r"\b(?:READ UNCOMMITTED|READ COMMITTED|REPEATABLE READ|SERIALIZABLE)\b",
            r"\b(?:START TRANSACTION|autocommit|undo log|redo log|read view|Gap Lock|Next-Key Lock)\b",
            r"\b(?:Using index|Using filesort|NOT NULL)\b",
            r"(?:-?XX:[A-Za-z0-9_]+|innodb_buffer_pool_size|max_connections|innodb_log_file_size|long_query_time)",
            r"\b[A-Za-z_][A-Za-z0-9_+\-/.]*(?:\(\))?\b",
        ],
        type_rules=[
            ("领域实体", ["MySQL", "RDBMS", "关系型数据库管理系统", "SQL", "数据库", "存储引擎", "事务", "索引", "锁机制", "主从复制", "读写分离", "分库分表", "SQL优化", "性能调优"], []),
            ("基础架构实体", ["服务层", "存储引擎层", "连接管理", "SQL解析", "优化器", "查询缓存", "执行器", "插件式架构"], []),
            ("存储引擎实体", ["InnoDB", "MyISAM"], []),
            ("数据库设计与规范实体", ["数据库三范式", "1NF", "2NF", "3NF", "反范式", "冗余字段", "联表查询", "主键", "唯一键", "外键", "逻辑外键", "参照完整性"], []),
            ("数据类型实体", ["数值型", "int", "bigint", "double", "decimal", "字符串型", "varchar", "char", "text", "日期型", "datetime", "timestamp", "date"], []),
            ("SQL分类实体", ["DDL", "DML", "DQL", "CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE", "SELECT", "WHERE", "ORDER BY", "LIMIT", "GROUP BY", "HAVING", "DISTINCT", "LIKE"], []),
            ("查询与连接实体", ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "UNION", "UNION ALL", "子查询", "相关子查询", "非相关子查询", "条件查询", "分组查询", "排序", "分页查询"], []),
            ("函数实体", ["聚合函数", "COUNT()", "SUM()", "AVG()", "MAX()", "MIN()", "字符串函数", "CONCAT()", "SUBSTRING()", "LENGTH()", "日期函数", "NOW()", "DATE_FORMAT()", "DATE_ADD()", "数学函数", "ROUND()", "CEIL()", "FLOOR()"], []),
            ("索引相关实体", ["索引", "索引下推", "B+树", "B树", "二叉树", "哈希表", "单列索引", "联合索引", "复合索引", "主键索引", "唯一索引", "普通索引", "全文索引", "聚簇索引", "非聚簇索引", "二级索引", "回表", "覆盖索引", "最左前缀原则", "最左匹配", "索引失效", "全表扫描"], []),
            ("事务与一致性实体", ["ACID", "Atomicity", "Consistency", "Isolation", "Durability", "START TRANSACTION", "COMMIT", "ROLLBACK", "SAVEPOINT", "autocommit", "脏读", "不可重复读", "幻读", "READ UNCOMMITTED", "READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE", "MVCC", "多版本并发控制", "undo log", "redo log", "read view", "读视图", "版本链"], []),
            ("锁机制实体", ["行锁", "表锁", "页锁", "共享锁", "读锁", "排他锁", "写锁", "意向锁", "意向共享锁", "意向排他锁", "Gap Lock", "间隙锁", "Next-Key Lock", "临键锁", "死锁", "锁等待超时时间"], []),
            ("执行计划与调优实体", ["慢查询日志", "long_query_time", "EXPLAIN", "执行计划", "id", "select_type", "table", "type", "key", "rows", "Extra", "Using index", "Using filesort", "小表驱动大表", "批量插入", "大分页", "主键id分页"], []),
            ("表设计与拆分实体", ["NOT NULL", "默认值", "垂直拆分", "水平拆分", "垂直分库", "垂直分表", "水平分表", "分区表", "大表优化", "热点字段", "非热点字段"], []),
            ("服务器参数与配置实体", ["innodb_buffer_pool_size", "max_connections", "innodb_log_file_size", "物理内存", "查询缓存"], []),
            ("主从复制与高可用实体", ["主从复制", "主库", "从库", "binlog", "二进制日志", "relay log", "IO线程", "SQL线程", "异步复制", "半同步复制", "组复制", "读写分离", "故障转移", "负载均衡", "主从延迟"], []),
            ("分库分表与分布式实体", ["分库分表", "范围分片", "哈希分片", "一致性哈希", "跨库分页", "跨库事务", "跨库联表查询", "分布式ID", "UUID", "雪花算法", "Snowflake", "Redis自增", "数据库自增"], []),
            ("中间件与工具实体", ["Sharding-JDBC", "MyCat", "mysqldump", "PreparedStatement", "Druid", "HikariCP", "SpringBoot"], []),
            ("运维与安全实体", ["物理备份", "逻辑备份", "数据恢复", "崩溃恢复", "SQL注入", "预编译SQL"], []),
            ("MySQL8.0与扩展特性实体", ["MySQL8.0", "窗口函数", "公用表表达式", "CTE", "utf8mb4", "utf8", "emoji", "视图", "存储过程", "触发器", "增强索引"], []),
        ],
        ngram_min_freq=2,
        ngram_min_len=2,
        ngram_max_len=8,
        use_ngrams=False,
        max_candidate_terms=2600,
        epochs=3,
        lr=3e-4,
        keep_default_type=False,
        min_mention_count=1,
        min_mention_count_default=3,
        min_entity_len=2,
        max_entity_len=48,
        append_only=False,
        allowed_entity_types=[
            "领域实体",
            "基础架构实体",
            "存储引擎实体",
            "数据库设计与规范实体",
            "数据类型实体",
            "SQL分类实体",
            "查询与连接实体",
            "函数实体",
            "索引相关实体",
            "事务与一致性实体",
            "锁机制实体",
            "执行计划与调优实体",
            "表设计与拆分实体",
            "服务器参数与配置实体",
            "主从复制与高可用实体",
            "分库分表与分布式实体",
            "中间件与工具实体",
            "运维与安全实体",
            "MySQL8.0与扩展特性实体",
        ],
    )


if __name__ == "__main__":
    run_bert_bilstm_crf_ner(build_config())
