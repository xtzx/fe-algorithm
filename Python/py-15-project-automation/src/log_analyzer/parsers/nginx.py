"""
Nginx 访问日志解析器

支持的格式：
- Combined Log Format（默认）
- 自定义格式（带响应时间）
"""

import re
from datetime import datetime

from log_analyzer.models import LogLevel, NginxLogEntry
from log_analyzer.parsers.base import BaseParser

# Combined Log Format
# 192.168.1.1 - - [01/Jan/2024:12:00:00 +0800] "GET /api/users HTTP/1.1" 200 1234 "-" "Mozilla/5.0"
COMBINED_PATTERN = re.compile(
    r'^(?P<remote_addr>\S+)\s+'  # IP
    r'\S+\s+'  # ident
    r'\S+\s+'  # auth user
    r'\[(?P<timestamp>[^\]]+)\]\s+'  # timestamp
    r'"(?P<method>\S+)\s+(?P<uri>\S+)\s+\S+"\s+'  # request
    r'(?P<status>\d+)\s+'  # status code
    r'(?P<bytes>\d+|-)\s*'  # bytes
    r'(?:"(?P<referer>[^"]*)"\s*)?'  # referer
    r'(?:"(?P<agent>[^"]*)")?'  # user agent
    r'(?:\s+(?P<response_time>[\d.]+))?'  # optional response time
)


class NginxLogParser(BaseParser):
    """Nginx 访问日志解析器"""

    def parse_line(self, line: str, line_number: int = 0) -> NginxLogEntry | None:
        match = COMBINED_PATTERN.match(line)
        if not match:
            return None

        groups = match.groupdict()

        # 解析时间戳
        # 01/Jan/2024:12:00:00 +0800
        timestamp_str = groups["timestamp"]
        try:
            # 移除时区部分简化解析
            ts_parts = timestamp_str.rsplit(" ", 1)
            timestamp = datetime.strptime(ts_parts[0], "%d/%b/%Y:%H:%M:%S")
        except ValueError:
            timestamp = datetime.now()

        # 解析状态码
        status_code = int(groups["status"])

        # 确定日志级别
        if status_code >= 500:
            level = LogLevel.ERROR
        elif status_code >= 400:
            level = LogLevel.WARNING
        else:
            level = LogLevel.INFO

        # 解析字节数
        bytes_str = groups["bytes"]
        body_bytes = int(bytes_str) if bytes_str != "-" else 0

        # 解析响应时间
        response_time = 0.0
        if groups.get("response_time"):
            try:
                response_time = float(groups["response_time"])
            except ValueError:
                pass

        return NginxLogEntry(
            timestamp=timestamp,
            level=level,
            message=f"{groups['method']} {groups['uri']} {status_code}",
            remote_addr=groups["remote_addr"],
            request_method=groups["method"],
            request_uri=groups["uri"],
            status_code=status_code,
            body_bytes_sent=body_bytes,
            response_time=response_time,
            http_referer=groups.get("referer") or "",
            http_user_agent=groups.get("agent") or "",
        )

