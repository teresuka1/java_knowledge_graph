from domain_ner_common import DomainConfig, run_bert_bilstm_crf_ner


def build_config() -> DomainConfig:
    return DomainConfig(
        domain_name="jvm",
        source_file="JVM.txt",
        output_csv="JVM.csv",
        default_type="领域实体",
        seed_entities=[
            # 一、领域实体
            "JVM", "Java 虚拟机", "Java虚拟机", "JVM 内存模型", "JVM内存模型", "JVM 内存结构", "JVM内存结构",
            "类加载", "双亲委派", "垃圾回收", "GC", "内存泄漏", "内存溢出",
            # 二、内存区域实体
            "程序计数器", "Java 虚拟机栈", "Java虚拟机栈", "虚拟机栈", "本地方法栈", "Java 堆", "Java堆",
            "堆", "方法区", "元空间", "运行时常量池", "直接内存", "堆外内存", "本地内存",
            # 三、栈与栈帧相关实体
            "栈帧", "局部变量表", "操作数栈", "动态链接", "方法出口", "方法调用", "方法返回地址",
            "局部变量", "临时数据", "指针", "对象引用", "基本类型数据",
            # 四、堆结构实体
            "新生代", "老年代", "Eden 区", "Eden区", "Survivor 区", "Survivor区", "From Survivor", "To Survivor",
            "S0", "S1", "Young Generation", "Old Generation", "Tenured Generation", "大对象区", "Humongous Objects",
            # 五、对象与字符串相关实体
            "对象实例", "数组", "大对象", "String", "字符串常量池", "字符串对象", "字面量", "对象头", "GC 分代年龄", "GC分代年龄", "new 指令", "new指令",
            # 六、方法区与元数据相关实体
            "类信息", "类型信息", "常量", "静态变量", "方法字节码", "符号引用", "直接引用", "常量池", "运行时常量池", "常量池缓存",
            "类的元数据信息", "即时编译器编译后的代码缓存",
            # 七、异常与错误实体
            "OutOfMemoryError", "StackOverflowError", "Java heap space", "Metaspace", "Direct buffer memory",
            "ClassFormatError", "VerifyError", "Concurrent Mode Failure",
            # 八、引用类型实体
            "强引用", "软引用", "弱引用", "虚引用", "幻影引用", "SoftReference", "WeakReference", "PhantomReference", "ReferenceQueue",
            # 九、线程与私有性相关实体
            "线程", "时间片", "CPU 调度器", "CPU调度器", "线程私有", "线程共享", "指令地址", "Native 方法", "Native方法",
            # 十、类加载过程实体
            "加载", "连接", "验证", "准备", "解析", "初始化", "使用", "卸载", "类加载检查", "类初始化", "类加载机制", "类加载过程",
            "类的生命周期", "Class 对象", "Class对象", ".class 文件", ".class文件", "二进制字节流", "全限定名", "静态存储结构", "方法区运行时的数据结构",
            # 十一、类加载器实体
            "Bootstrap ClassLoader", "启动类加载器", "Extension ClassLoader", "扩展类加载器",
            "System ClassLoader", "Application ClassLoader", "应用程序类加载器", "自定义类加载器", "ClassLoader", "双亲委派模型",
            # 十二、垃圾回收核心概念实体
            "Garbage Collection", "GC Roots", "可达性分析", "引用计数法", "Reachability Analysis",
            "Minor GC", "Major GC", "Full GC", "Young GC", "Stop The World", "STW", "浮动垃圾", "垃圾对象", "存活对象", "分代回收",
            # 十三、垃圾回收算法实体
            "标记-清除算法", "复制算法", "标记-整理算法", "分代回收算法", "标记阶段", "清除阶段", "转移阶段", "重定位阶段",
            "初始标记", "并发标记", "再标记", "清理阶段", "复制阶段",
            # 十四、垃圾回收器实体
            "Serial", "ParNew", "Parallel Scavenge", "Serial Old", "Parallel Old", "CMS", "Concurrent Mark Sweep", "G1", "Garbage First", "ZGC", "Shenandoah",
            # 十五、工具与排查实体
            "HeapDumpOnOutOfMemoryError", "HeapDumpPath", "heapdump.hprof", "MAT", "Memory Analyzer Tool", "JProfiler",
            "VisualVM", "ByteBuffer.allocateDirect()", "NIO", "JavaNIO", "netty",
            # 十六、JVM 参数与配置实体
            "-XX:+HeapDumpOnOutOfMemoryError", "-XX:HeapDumpPath", "-Xms", "-Xmx", "-Xss", "-XX:MaxTenuringThreshold",
            "System.gc()", "Runtime.getRuntime().gc()",
            # 十七、代码与容器实体
            "HashMap", "ArrayList", "List", "ThreadLocal", "ThreadLocalMap", "Entry", "value", "key", "static", "finally",
            "try-with-resources", "单例", "懒加载",
            # 十八、场景与问题实体
            "堆溢出", "栈溢出", "元空间溢出", "直接内存溢出", "深度递归", "无限递归", "缓存系统", "对象池", "事件监听",
            "静态集合", "批量处理任务", "树形结构遍历", "内存碎片", "大文件处理", "分批处理",
        ],
        regex_entities=[
            r"\b(?:JVM|JDK|JRE|GC|STW|CMS|G1|ZGC|NIO)\b",
            r"\b(?:Minor|Major|Full|Young)\s+GC\b",
            r"\b(?:GC Roots|Garbage Collection|Reachability Analysis|Concurrent Mark Sweep|Garbage First)\b",
            r"\b(?:Serial|ParNew|Parallel Scavenge|Serial Old|Parallel Old|CMS|G1|ZGC|Shenandoah)\b",
            r"\b(?:Bootstrap ClassLoader|Extension ClassLoader|System ClassLoader|Application ClassLoader|ClassLoader)\b",
            r"\b(?:OutOfMemoryError|StackOverflowError|ClassFormatError|VerifyError)\b",
            r"(?:-XX:\+?[A-Za-z0-9_]+(?:=[A-Za-z0-9_./-]+)?|-Xms|-Xmx|-Xss)",
            r"(?:System\.gc\(\)|Runtime\.getRuntime\(\)\.gc\(\)|ByteBuffer\.allocateDirect\(\)|Class\.forName\(\))",
            r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?\b",
        ],
        type_rules=[
            ("领域实体", ["JVM", "Java 虚拟机", "Java虚拟机", "JVM 内存模型", "JVM内存模型", "JVM 内存结构", "JVM内存结构", "类加载", "双亲委派", "垃圾回收", "GC", "内存泄漏", "内存溢出"], []),
            ("内存区域实体", ["程序计数器", "Java 虚拟机栈", "Java虚拟机栈", "虚拟机栈", "本地方法栈", "Java 堆", "Java堆", "堆", "方法区", "元空间", "运行时常量池", "直接内存", "堆外内存", "本地内存"], []),
            ("栈与栈帧相关实体", ["栈帧", "局部变量表", "操作数栈", "动态链接", "方法出口", "方法调用", "方法返回地址", "局部变量", "临时数据", "指针", "对象引用", "基本类型数据"], []),
            ("堆结构实体", ["新生代", "老年代", "Eden 区", "Eden区", "Survivor 区", "Survivor区", "From Survivor", "To Survivor", "S0", "S1", "Young Generation", "Old Generation", "Tenured Generation", "大对象区", "Humongous Objects"], []),
            ("对象与字符串相关实体", ["对象实例", "数组", "大对象", "String", "字符串常量池", "字符串对象", "字面量", "对象头", "GC 分代年龄", "GC分代年龄", "new 指令", "new指令"], []),
            ("方法区与元数据相关实体", ["类信息", "类型信息", "常量", "静态变量", "方法字节码", "符号引用", "直接引用", "常量池", "运行时常量池", "常量池缓存", "类的元数据信息", "即时编译器编译后的代码缓存"], []),
            ("异常与错误实体", ["OutOfMemoryError", "StackOverflowError", "Java heap space", "Metaspace", "Direct buffer memory", "ClassFormatError", "VerifyError", "Concurrent Mode Failure"], []),
            ("引用类型实体", ["强引用", "软引用", "弱引用", "虚引用", "幻影引用", "SoftReference", "WeakReference", "PhantomReference", "ReferenceQueue"], []),
            ("线程与私有性相关实体", ["线程", "时间片", "CPU 调度器", "CPU调度器", "线程私有", "线程共享", "指令地址", "Native 方法", "Native方法"], []),
            ("类加载过程实体", ["加载", "连接", "验证", "准备", "解析", "初始化", "使用", "卸载", "类加载检查", "类初始化", "类加载机制", "类加载过程", "类的生命周期", "Class 对象", "Class对象", ".class 文件", ".class文件", "二进制字节流", "全限定名", "静态存储结构", "方法区运行时的数据结构"], []),
            ("类加载器实体", ["Bootstrap ClassLoader", "启动类加载器", "Extension ClassLoader", "扩展类加载器", "System ClassLoader", "Application ClassLoader", "应用程序类加载器", "自定义类加载器", "ClassLoader", "双亲委派模型"], []),
            ("垃圾回收核心概念实体", ["Garbage Collection", "GC Roots", "可达性分析", "引用计数法", "Reachability Analysis", "Minor GC", "Major GC", "Full GC", "Young GC", "Stop The World", "STW", "浮动垃圾", "垃圾对象", "存活对象", "分代回收"], []),
            ("垃圾回收算法实体", ["标记-清除算法", "复制算法", "标记-整理算法", "分代回收算法", "标记阶段", "清除阶段", "转移阶段", "重定位阶段", "初始标记", "并发标记", "再标记", "清理阶段", "复制阶段"], []),
            ("垃圾回收器实体", ["Serial", "ParNew", "Parallel Scavenge", "Serial Old", "Parallel Old", "CMS", "Concurrent Mark Sweep", "G1", "Garbage First", "ZGC", "Shenandoah"], []),
            ("工具与排查实体", ["HeapDumpOnOutOfMemoryError", "HeapDumpPath", "heapdump.hprof", "MAT", "Memory Analyzer Tool", "JProfiler", "VisualVM", "ByteBuffer.allocateDirect()", "NIO", "JavaNIO", "netty"], []),
            ("JVM参数与配置实体", ["-XX:+HeapDumpOnOutOfMemoryError", "-XX:HeapDumpPath", "-Xms", "-Xmx", "-Xss", "-XX:MaxTenuringThreshold", "System.gc()", "Runtime.getRuntime().gc()"], [r"^-Xms$", r"^-Xmx$", r"^-Xss$", r"^-XX:"]),
            ("代码与容器实体", ["HashMap", "ArrayList", "List", "ThreadLocal", "ThreadLocalMap", "Entry", "value", "key", "static", "finally", "try-with-resources", "单例", "懒加载"], []),
            ("场景与问题实体", ["堆溢出", "栈溢出", "元空间溢出", "直接内存溢出", "深度递归", "无限递归", "缓存系统", "对象池", "事件监听", "静态集合", "批量处理任务", "树形结构遍历", "内存碎片", "大文件处理", "分批处理"], []),
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
            "内存区域实体",
            "栈与栈帧相关实体",
            "堆结构实体",
            "对象与字符串相关实体",
            "方法区与元数据相关实体",
            "异常与错误实体",
            "引用类型实体",
            "线程与私有性相关实体",
            "类加载过程实体",
            "类加载器实体",
            "垃圾回收核心概念实体",
            "垃圾回收算法实体",
            "垃圾回收器实体",
            "工具与排查实体",
            "JVM参数与配置实体",
            "代码与容器实体",
            "场景与问题实体",
        ],
    )


if __name__ == "__main__":
    run_bert_bilstm_crf_ner(build_config())
