from domain_ner_common import DomainConfig, run_bert_bilstm_crf_ner


def build_config() -> DomainConfig:
    return DomainConfig(
        domain_name="java",
        source_file="java.txt",
        output_csv="java.csv",
        default_type="领域实体",
        seed_entities=[
            # 一、领域实体
            "Java", "JVM", "面向对象", "集合框架", "泛型", "反射", "注解", "线程与并发", "异常处理",
            # 二、数据类型实体
            "基本数据类型", "byte", "short", "int", "long", "float", "double", "char", "boolean",
            "引用数据类型", "类", "接口", "数组", "枚举", "注解",
            # 三、变量与成员实体
            "变量", "局部变量", "成员变量", "实例变量", "静态变量", "类变量",
            # 四、运算符实体
            "算术运算符", "关系运算符", "逻辑运算符", "位运算符", "赋值运算符", "三元运算符",
            # 五、流程控制实体
            "条件分支", "if-else", "switch-case", "循环结构", "for", "while", "do-while", "跳转语句", "break", "continue", "return",
            # 六、面向对象核心实体
            "封装", "继承", "多态", "Class", "对象", "Object", "构造方法", "方法重载", "Overloading",
            "方法重写", "Overriding", "父类", "子类", "实例",
            # 七、关键字与修饰符实体
            "访问控制修饰符", "private", "default", "protected", "public",
            "其他关键字", "static", "final", "abstract", "interface", "extends", "implements", "this", "super", "new", "instanceof",
            # 八、继承与类型转换实体
            "单继承", "接口实现", "父类引用", "子类对象", "接口引用", "实现类对象", "向上转型", "向下转型", "强制类型转换",
            # 九、抽象类与接口实体
            "抽象类", "抽象方法", "具体方法", "默认方法", "静态方法",
            # 十、数组与字符串实体
            "多维数组", "int[] arr", "int arr[]", "new int[10]", "{1,2,3}", "int[][] matrix",
            "String", "StringBuilder", "StringBuffer", "immutable", "length()", "charAt()", "substring()", "equals()", "indexOf()", "split()",
            # 十一、异常处理实体
            "Throwable", "Error", "Exception", "Checked Exception", "Unchecked Exception", "try", "catch", "finally", "throw", "throws", "JVM退出",
            # 十二、集合框架实体
            "Collection", "Map", "List", "Set", "Queue",
            "ArrayList", "LinkedList", "Vector", "HashSet", "LinkedHashSet", "TreeSet",
            "HashMap", "Hashtable", "TreeMap", "LinkedHashMap",
            "Collections", "红黑树", "键值对", "null 键值", "插入顺序",
            # 十三、泛型实体
            "泛型类", "泛型方法", "通配符", "类型擦除", "Box<T>",
            # 十四、反射实体
            "Method", "Field", "Constructor", "类名.class", "getClass()", "Class.forName()", "全限定名", "setAccessible(true)",
            # 十五、注解实体
            "Annotation", "@Override", "@Deprecated", "@SuppressWarnings", "@FunctionalInterface",
            "@Retention", "@Target", "@Documented", "@Inherited", "@interface", "自定义注解", "元注解",
            # 十六、线程与并发实体
            "线程", "并发", "Thread", "Runnable", "Callable", "FutureTask", "synchronized", "Lock", "ReentrantLock", "tryLock",
            "lockInterruptibly", "volatile", "可见性", "指令重排序", "原子性", "线程池", "Executor", "ThreadPoolExecutor", "互斥锁",
            # 十七、线程状态实体
            "新建", "就绪", "运行", "阻塞", "等待", "超时等待", "终止",
            # 十八、JVM 与内存管理实体
            "类加载机制", "类加载器", "双亲委派模型", "堆", "栈", "方法区", "程序计数器", "本地方法栈",
            "垃圾回收", "GC", "标记-清除", "复制", "标记-整理", "分代收集",
            "加载", "验证", "准备", "解析", "初始化",
            "Bootstrap ClassLoader", "Extension ClassLoader", "Application ClassLoader",
        ],
        regex_entities=[
            r"\b(?:byte|short|int|long|float|double|char|boolean)\b",
            r"\b(?:if|else|switch|for|while|break|continue|return|try|catch|finally|throw|throws)\b",
            r"\b(?:private|default|protected|public|static|final|abstract|interface|extends|implements|this|super|new|instanceof)\b",
            r"\b(?:Collection|Map|List|Set|Queue|ArrayList|LinkedList|Vector|HashSet|LinkedHashSet|TreeSet|HashMap|Hashtable|TreeMap|LinkedHashMap|Collections)\b",
            r"\b(?:Thread|Runnable|Callable|FutureTask|Lock|ReentrantLock|Executor|ThreadPoolExecutor|volatile|synchronized)\b",
            r"\b(?:Throwable|Error|Exception)\b",
            r"\b(?:JVM|GC)\b",
            r"\b(?:ClassLoader|Method|Field|Constructor)\b",
            r"(?:Class\.forName\(\)|getClass\(\)|setAccessible\(true\)|length\(\)|charAt\(\)|substring\(\)|equals\(\)|indexOf\(\)|split\(\)|tryLock|lockInterruptibly)",
            r"(?:@Override|@Deprecated|@SuppressWarnings|@FunctionalInterface|@Retention|@Target|@Documented|@Inherited|@interface)",
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?:\(\))?\b",
        ],
        type_rules=[
            ("领域实体", ["Java", "JVM", "面向对象", "集合框架", "泛型", "反射", "注解", "线程与并发", "异常处理"], []),
            ("数据类型实体", ["基本数据类型", "byte", "short", "int", "long", "float", "double", "char", "boolean", "引用数据类型", "类", "接口", "数组", "枚举", "注解"], []),
            ("变量与成员实体", ["变量", "局部变量", "成员变量", "实例变量", "静态变量", "类变量"], []),
            ("运算符实体", ["算术运算符", "关系运算符", "逻辑运算符", "位运算符", "赋值运算符", "三元运算符"], []),
            ("流程控制实体", ["条件分支", "if-else", "switch-case", "循环结构", "for", "while", "do-while", "跳转语句", "break", "continue", "return"], []),
            ("面向对象核心实体", ["封装", "继承", "多态", "Class", "对象", "Object", "构造方法", "方法重载", "Overloading", "方法重写", "Overriding", "父类", "子类", "实例"], []),
            ("关键字与修饰符实体", ["访问控制修饰符", "private", "default", "protected", "public", "其他关键字", "static", "final", "abstract", "interface", "extends", "implements", "this", "super", "new", "instanceof"], []),
            ("继承与类型转换实体", ["单继承", "接口实现", "父类引用", "子类对象", "接口引用", "实现类对象", "向上转型", "向下转型", "强制类型转换"], []),
            ("抽象类与接口实体", ["抽象类", "抽象方法", "具体方法", "实例变量", "默认方法", "静态方法"], []),
            ("数组与字符串实体", [
                "数组", "多维数组", "int[] arr", "int arr[]", "new int[10]", "{1,2,3}", "int[][] matrix",
                "String", "StringBuilder", "StringBuffer", "immutable", "length()", "charAt()", "substring()", "equals()", "indexOf()", "split()",
            ], [r"^int\[\]\s*arr$", r"^int\s+arr\[\]$", r"^new int\[10\]$", r"^int\[\]\[\]\s*matrix$"]),
            ("异常处理实体", ["Throwable", "Error", "Exception", "Checked Exception", "Unchecked Exception", "try", "catch", "finally", "throw", "throws", "JVM退出"], []),
            ("集合框架实体", [
                "Collection", "Map", "List", "Set", "Queue", "ArrayList", "LinkedList", "Vector", "HashSet", "LinkedHashSet", "TreeSet",
                "HashMap", "Hashtable", "TreeMap", "LinkedHashMap", "Collections", "红黑树", "键值对", "null 键值", "插入顺序",
            ], []),
            ("泛型实体", ["泛型", "泛型类", "泛型方法", "通配符", "类型擦除", "Box<T>"], [r"^Box<T>$"]),
            ("反射实体", ["反射", "Class", "Method", "Field", "Constructor", "类名.class", "getClass()", "Class.forName()", "全限定名", "setAccessible(true)"], []),
            ("注解实体", [
                "Annotation", "@Override", "@Deprecated", "@SuppressWarnings", "@FunctionalInterface", "@Retention", "@Target",
                "@Documented", "@Inherited", "@interface", "自定义注解", "元注解",
            ], [r"^@"]),
            ("线程与并发实体", [
                "线程", "并发", "Thread", "Runnable", "Callable", "FutureTask", "synchronized", "Lock", "ReentrantLock", "tryLock",
                "lockInterruptibly", "volatile", "可见性", "指令重排序", "原子性", "线程池", "Executor", "ThreadPoolExecutor", "互斥锁",
            ], []),
            ("线程状态实体", ["新建", "就绪", "运行", "阻塞", "等待", "超时等待", "终止"], []),
            ("JVM 与内存管理实体", [
                "JVM", "类加载机制", "类加载器", "双亲委派模型", "堆", "栈", "方法区", "程序计数器", "本地方法栈",
                "垃圾回收", "GC", "标记-清除", "复制", "标记-整理", "分代收集", "加载", "验证", "准备", "解析", "初始化",
                "Bootstrap ClassLoader", "Extension ClassLoader", "Application ClassLoader",
            ], []),
        ],
        ngram_min_freq=2,
        ngram_min_len=2,
        ngram_max_len=7,
        use_ngrams=False,
        max_candidate_terms=2200,
        epochs=3,
        lr=3e-4,
        keep_default_type=False,
        min_mention_count=1,
        min_mention_count_default=3,
        min_entity_len=2,
        max_entity_len=40,
        append_only=False,
        allowed_entity_types=[
            "领域实体",
            "数据类型实体",
            "变量与成员实体",
            "运算符实体",
            "流程控制实体",
            "面向对象核心实体",
            "关键字与修饰符实体",
            "继承与类型转换实体",
            "抽象类与接口实体",
            "数组与字符串实体",
            "异常处理实体",
            "集合框架实体",
            "泛型实体",
            "反射实体",
            "注解实体",
            "线程与并发实体",
            "线程状态实体",
            "JVM 与内存管理实体",
        ],
    )


if __name__ == "__main__":
    run_bert_bilstm_crf_ner(build_config())
