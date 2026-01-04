"""
终端报告器

支持彩色输出和文本图表
"""

from log_analyzer.models import AnalysisReport


# ANSI 颜色代码
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def colorize(text: str, color: str, bold: bool = False) -> str:
    """给文本添加颜色"""
    prefix = Colors.BOLD if bold else ""
    return f"{prefix}{color}{text}{Colors.RESET}"


class TerminalReporter:
    """终端报告器"""

    def __init__(self, use_colors: bool = True) -> None:
        self.use_colors = use_colors

    def _c(self, text: str, color: str, bold: bool = False) -> str:
        """条件着色"""
        if self.use_colors:
            return colorize(text, color, bold)
        return text

    def generate(self, report: AnalysisReport) -> str:
        """生成终端报告"""
        lines: list[str] = []

        # 标题
        lines.append(self._header())
        lines.append("")

        # 概览
        lines.append(self._c("📋 概览", Colors.CYAN, bold=True))
        lines.append("━" * 60)
        lines.append(f"  分析时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  文件数量: {report.files_analyzed}")
        lines.append(f"  总记录数: {report.total_entries:,}")
        lines.append(f"  有效记录: {report.valid_entries:,}")
        lines.append(f"  无效记录: {report.invalid_entries:,}")
        lines.append("")

        # 错误统计
        if report.error_stats.total_errors > 0 or report.error_stats.total_warnings > 0:
            lines.append(self._c("⚠️ 错误统计", Colors.YELLOW, bold=True))
            lines.append("━" * 60)
            lines.extend(self._error_section(report))
            lines.append("")

        # 请求统计
        if report.request_stats and report.request_stats.total_requests > 0:
            lines.append(self._c("📊 请求统计", Colors.BLUE, bold=True))
            lines.append("━" * 60)
            lines.extend(self._request_section(report))
            lines.append("")

        # 时间分布
        if report.timeline_stats.by_hour:
            lines.append(self._c("📈 时间分布", Colors.MAGENTA, bold=True))
            lines.append("━" * 60)
            lines.extend(self._timeline_section(report))
            lines.append("")

        lines.append(self._footer())

        return "\n".join(lines)

    def _header(self) -> str:
        """生成报告头部"""
        border = "╔" + "═" * 58 + "╗"
        title = "║" + "日志分析报告".center(54) + "║"
        bottom = "╚" + "═" * 58 + "╝"

        if self.use_colors:
            border = self._c(border, Colors.CYAN)
            title = self._c(title, Colors.CYAN, bold=True)
            bottom = self._c(bottom, Colors.CYAN)

        return f"{border}\n{title}\n{bottom}"

    def _footer(self) -> str:
        """生成报告尾部"""
        return "━" * 60

    def _error_section(self, report: AnalysisReport) -> list[str]:
        """错误统计部分"""
        lines = []
        stats = report.error_stats

        # 按级别统计
        total = stats.total_errors + stats.total_warnings + stats.total_critical
        if total > 0:
            for level, count in sorted(
                stats.by_level.items(), key=lambda x: x[1], reverse=True
            ):
                pct = count / total * 100
                bar_len = int(pct / 100 * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)

                color = Colors.RED if level in ("ERROR", "CRITICAL") else Colors.YELLOW
                level_text = self._c(f"{level:10}", color)
                lines.append(f"  {level_text}: {bar} {count:,} ({pct:.1f}%)")

        # Top 错误消息
        if stats.top_messages:
            lines.append("")
            lines.append(self._c("  Top 错误消息:", Colors.YELLOW))
            for msg, count in stats.top_messages[:5]:
                msg_short = msg[:50] + "..." if len(msg) > 50 else msg
                lines.append(f"    [{count:,}] {msg_short}")

        return lines

    def _request_section(self, report: AnalysisReport) -> list[str]:
        """请求统计部分"""
        lines = []
        stats = report.request_stats
        if stats is None:
            return lines

        lines.append(f"  总请求数: {stats.total_requests:,}")
        lines.append(f"  错误率: {stats.error_rate:.2f}%")
        lines.append(f"  平均响应: {stats.avg_response_time:.3f}s")
        lines.append(f"  最大响应: {stats.max_response_time:.3f}s")
        lines.append("")

        # 状态码分布
        lines.append(self._c("  状态码分布:", Colors.BLUE))
        for code, count in sorted(stats.by_status_code.items()):
            color = (
                Colors.RED if code >= 500 else Colors.YELLOW if code >= 400 else Colors.GREEN
            )
            code_text = self._c(str(code), color)
            lines.append(f"    {code_text}: {count:,}")

        # Top URLs
        if stats.top_urls:
            lines.append("")
            lines.append(self._c("  Top URLs:", Colors.BLUE))
            for url, count in stats.top_urls[:5]:
                url_short = url[:40] + "..." if len(url) > 40 else url
                lines.append(f"    {url_short}: {count:,}")

        return lines

    def _timeline_section(self, report: AnalysisReport) -> list[str]:
        """时间分布部分"""
        lines = []
        stats = report.timeline_stats

        if stats.start_time and stats.end_time:
            lines.append(f"  时间范围: {stats.start_time} ~ {stats.end_time}")

        lines.append(f"  高峰时段: {stats.peak_hour:02d}:00 ({stats.peak_count:,} 条)")
        lines.append("")

        # 小时分布图
        lines.append(self._c("  小时分布:", Colors.MAGENTA))
        max_count = max(stats.by_hour.values()) if stats.by_hour else 1

        for hour in range(24):
            count = stats.by_hour.get(hour, 0)
            bar_len = int(count / max_count * 20) if max_count > 0 else 0
            bar = "▓" * bar_len
            lines.append(f"    {hour:02d}:00 │ {bar} {count:,}")

        return lines

    def print(self, report: AnalysisReport) -> None:
        """打印报告到终端"""
        print(self.generate(report))

