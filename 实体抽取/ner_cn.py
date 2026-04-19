from domain_ner_common import DomainConfig, run_bert_bilstm_crf_ner


def build_config() -> DomainConfig:
    return DomainConfig(
        domain_name="cn",
        source_file="cn.txt",
        output_csv="cn.csv",
        default_type="领域实体",
        seed_entities=[
            # 一、领域实体
            "计算机网络", "网络基础", "网络分层模型",
            # 二、模型与体系结构实体
            "OSI 七层模型", "TCP/IP 四层模型", "OSI 七层", "TCP/IP 四层",
            # 三、层级结构实体
            "物理层", "数据链路层", "网络层", "传输层", "会话层", "表示层", "应用层", "网络接口层",
            # 四、协议实体
            "应用层协议", "HTTP", "HTTPS", "FTP", "SMTP", "POP3", "IMAP", "Telnet", "SSH", "DNS", "RTP", "RTCP",
            "DHCP", "SNMP", "NTP", "WebSocket", "SSE", "HTTP/1.0", "HTTP/1.1", "HTTP/2", "HTTP/3",
            "传输层协议", "TCP", "UDP", "SCTP", "QUIC",
            "网络层协议", "IPv4", "IPv6", "ARP", "ICMP", "ICMPv6", "IGMP", "NAT", "OSPF", "RIP", "BGP",
            "网络接口层相关", "以太网", "Wi-Fi", "PPP", "MAC 协议",
            # 五、地址与标识实体
            "IP 地址", "IPv4 地址", "IPv6 地址", "MAC 地址", "私有 IP", "公网 IP", "广播地址", "域名", "URI", "URL", "SessionID",
            # 六、机制与技术实体
            "三次握手", "四次挥手", "序列号", "确认应答 ACK", "超时重传", "流量控制", "拥塞控制", "路由与寻址", "差错纠正",
            "帧封装", "物理寻址", "无状态地址自动配置", "SLAAC", "网络地址转换", "IP 地址过滤", "获取客户端真实 IP",
            "X-Forwarded-For", "TCP Options", "Proxy Protocol", "DSR 模式", "Keep-Alive", "多路复用", "头部压缩",
            "HPACK", "QPACK", "服务器推送", "连接迁移", "TLS1.3", "SSL/TLS", "Cookie", "Session", "Set-Cookie",
            "URL 重写", "Token", "JWT", "DNS 劫持", "DoH", "DoT", "ARP 缓存",
            # 七、报文/字段/请求头/状态相关实体
            "比特流", "帧", "数据包", "字节流", "报文", "首部", "报头",
            "Host", "Connection", "Content-Type", "Cache-Control", "User-Agent", "Range", "ETag",
            "If-Match", "If-None-Match", "If-Modified-Since", "Expires", "SYN", "ACK", "FIN",
            # 八、版本、端口与数值实体
            "0/1 二进制信号", "20 字节", "60 字节", "8 字节", "32 位", "128 位", "48 位", "6 字节",
            "80", "443", "200", "206", "301", "304", "400", "401", "403", "404", "500", "502",
            "0-RTT", "1-RTT", "2MSL", "RTT", "类型 8", "类型 0",
            # 九、概念与特性实体
            "面向连接", "无连接", "可靠传输", "不可靠传输", "有状态", "无状态", "面向字节流", "面向报文", "单播", "多播", "广播",
            "长连接", "短连接", "全双工", "队头阻塞", "幂等性", "缓存机制", "虚拟主机", "断点续传", "明文传输", "加密", "分布式", "微服务",
            # 十、服务器与网络组件实体
            "客户端", "服务器", "主机", "路由器", "浏览器", "根 DNS 服务器", "顶级域服务器", "TLD 服务器", "权威 DNS 服务器",
            "本地 DNS 服务器", "网卡", "自治系统",
            # 十一、场景与应用实体
            "Web 浏览", "文件传输", "邮件收发", "远程登录", "实时音视频", "视频会议", "直播", "在线游戏", "IoT 传感器上报",
            "网络诊断", "故障排查", "页面展示", "协同编辑", "实时聊天", "弹幕", "域名解析", "时间同步", "网络管理",
            # 十二、工具与命令实体
            "PING", "tracert", "nslookup", "arp",
        ],
        regex_entities=[
            r"\b(?:HTTP|HTTPS|FTP|SMTP|POP3|IMAP|TELNET|SSH|DNS|RTP|RTCP|DHCP|SNMP|NTP|TCP|UDP|SCTP|QUIC|IPV4|IPV6|ARP|ICMPV6|ICMP|IGMP|NAT|OSPF|RIP|BGP)\b",
            r"\b(?:Host|Connection|Content-Type|Cache-Control|User-Agent|Range|ETag|If-Match|If-None-Match|If-Modified-Since|Expires)\b",
            r"\b(?:SYN|ACK|FIN|JWT|SLAAC|HPACK|QPACK|RTT)\b",
            r"\b(?:80|443|200|206|301|304|400|401|403|404|500|502)\b",
            r"\b(?:0-RTT|1-RTT|2MSL)\b",
            r"\b(?:PING|tracert|nslookup|arp)\b",
            r"\b[A-Za-z][A-Za-z0-9_+\-/.]{1,40}\b",
            r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        ],
        type_rules=[
            ("领域实体", ["计算机网络", "网络基础", "网络分层模型"], []),
            ("模型与体系结构实体", ["OSI 七层模型", "TCP/IP 四层模型", "OSI 七层", "TCP/IP 四层"], [r"^OSI$", r"^TCP/IP$"]),
            ("层级结构实体", ["物理层", "数据链路层", "网络层", "传输层", "会话层", "表示层", "应用层", "网络接口层"], []),
            ("协议实体", [
                "应用层协议", "HTTP", "HTTPS", "FTP", "SMTP", "POP3", "IMAP", "Telnet", "SSH", "DNS", "RTP", "RTCP", "DHCP", "SNMP", "NTP",
                "WebSocket", "SSE", "HTTP/1.0", "HTTP/1.1", "HTTP/2", "HTTP/3", "传输层协议", "TCP", "UDP", "SCTP", "QUIC", "网络层协议",
                "IPv4", "IPv6", "ARP", "ICMP", "ICMPv6", "IGMP", "NAT", "OSPF", "RIP", "BGP", "网络接口层相关", "以太网", "Wi-Fi", "PPP", "MAC 协议",
            ], [r"^HTTP/1\.0$", r"^HTTP/1\.1$", r"^HTTP/2$", r"^HTTP/3$"]),
            ("地址与标识实体", ["IP 地址", "IPv4 地址", "IPv6 地址", "MAC 地址", "私有 IP", "公网 IP", "广播地址", "域名", "URI", "URL", "SessionID"], [r"\b\d{1,3}(?:\.\d{1,3}){3}\b"]),
            ("机制与技术实体", [
                "三次握手", "四次挥手", "序列号", "确认应答 ACK", "超时重传", "流量控制", "拥塞控制", "路由与寻址", "差错纠正", "帧封装",
                "物理寻址", "无状态地址自动配置", "SLAAC", "网络地址转换", "IP 地址过滤", "获取客户端真实 IP", "X-Forwarded-For", "TCP Options",
                "Proxy Protocol", "DSR 模式", "Keep-Alive", "多路复用", "头部压缩", "HPACK", "QPACK", "服务器推送", "连接迁移",
                "TLS1.3", "SSL/TLS", "Cookie", "Session", "Set-Cookie", "URL 重写", "Token", "JWT", "DNS 劫持", "DoH", "DoT", "ARP 缓存",
            ], [r"^X-Forwarded-For$", r"^TLS1\.3$", r"^SSL/TLS$"]),
            ("报文字段与状态实体", [
                "比特流", "帧", "数据包", "字节流", "报文", "首部", "报头",
                "Host", "Connection", "Content-Type", "Cache-Control", "User-Agent", "Range", "ETag", "If-Match",
                "If-None-Match", "If-Modified-Since", "Expires", "SYN", "ACK", "FIN",
            ], []),
            ("版本端口与数值实体", [
                "0/1 二进制信号", "20 字节", "60 字节", "8 字节", "32 位", "128 位", "48 位", "6 字节",
                "80", "443", "200", "206", "301", "304", "400", "401", "403", "404", "500", "502",
                "0-RTT", "1-RTT", "2MSL", "RTT", "类型 8", "类型 0",
            ], [r"^(80|443|200|206|301|304|400|401|403|404|500|502)$", r"^(0-RTT|1-RTT|2MSL|RTT)$"]),
            ("概念与特性实体", [
                "面向连接", "无连接", "可靠传输", "不可靠传输", "有状态", "无状态", "面向字节流", "面向报文", "单播", "多播", "广播",
                "长连接", "短连接", "全双工", "队头阻塞", "幂等性", "缓存机制", "虚拟主机", "断点续传", "明文传输", "加密", "分布式", "微服务",
            ], []),
            ("服务器与网络组件实体", [
                "客户端", "服务器", "主机", "路由器", "浏览器", "根 DNS 服务器", "顶级域服务器", "TLD 服务器",
                "权威 DNS 服务器", "本地 DNS 服务器", "网卡", "自治系统",
            ], []),
            ("场景与应用实体", [
                "Web 浏览", "文件传输", "邮件收发", "远程登录", "实时音视频", "视频会议", "直播", "在线游戏", "IoT 传感器上报",
                "网络诊断", "故障排查", "页面展示", "协同编辑", "实时聊天", "弹幕", "域名解析", "时间同步", "网络管理",
            ], []),
            ("工具与命令实体", ["PING", "tracert", "nslookup", "arp"], [r"^(PING|tracert|nslookup|arp)$"]),
        ],
        ngram_min_freq=2,
        ngram_min_len=2,
        ngram_max_len=7,
        use_ngrams=False,
        max_candidate_terms=2000,
        epochs=3,
        lr=3e-4,
        keep_default_type=False,
        min_mention_count=2,
        min_mention_count_default=3,
        min_entity_len=2,
        max_entity_len=40,
        append_only=False,
        allowed_entity_types=[
            "领域实体",
            "模型与体系结构实体",
            "层级结构实体",
            "协议实体",
            "地址与标识实体",
            "机制与技术实体",
            "报文字段与状态实体",
            "版本端口与数值实体",
            "概念与特性实体",
            "服务器与网络组件实体",
            "场景与应用实体",
            "工具与命令实体",
        ],
    )


if __name__ == "__main__":
    run_bert_bilstm_crf_ner(build_config())
