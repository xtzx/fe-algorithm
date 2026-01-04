"""
Markdown 报告器
"""

from pathlib import Path

from log_analyzer.models import AnalysisReport


class MarkdownReporter:
    """Markdown 报告器"""

    def generate(self, report: AnalysisReport) -> str:
        """生成 Markdown 报告"""
        lines: list[str] = []

        # 标题
        lines.append("# 日志分析报告")
        lines.append("")
        lines.append(f"> 生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 概览
        lines.append("## 📋 概览")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 分析文件数 | {report.files_analyzed} |")
        lines.append(f"| 总记录数 | {report.total_entries:,} |")
        lines.append(f"| 有效记录 | {report.valid_entries:,} |")
        lines.append(f"| 无效记录 | {report.invalid_entries:,} |")
        lines.append("")

        # 错误统计
        stats = report.error_stats
        if stats.total_errors > 0 or stats.total_warnings > 0:
            lines.append("## ⚠️ 错误统计")
            lines.append("")
            lines.append("### 按级别")
            lines.append("")
            lines.append("| 级别 | 数量 | 占比 |")
            lines.append("|------|------|------|")

            total = sum(stats.by_level.values()) or 1
            for level, count in sorted(
                stats.by_level.items(), key=lambda x: x[1], reverse=True
            ):
                pct = count / total * 100
                lines.append(f"| {level} | {count:,} | {pct:.1f}% |")
            lines.append("")

            # 小时分布
            if stats.by_hour:
                lines.append("### 按小时")
                lines.append("")
                lines.append("```")
                max_count = max(stats.by_hour.values())
                for hour in range(24):
                    count = stats.by_hour.get(hour, 0)
                    bar_len = int(count / max_count * 30) if max_count > 0 else 0
                    bar = "█" * bar_len
                    lines.append(f"{hour:02d}:00 | {bar} {count}")
                lines.append("```")
                lines.append("")

            # Top 错误
            if stats.top_messages:
                lines.append("### Top 错误消息")
                lines.append("")
                lines.append("| 消息 | 次数 |")
                lines.append("|------|------|")
                for msg, count in stats.top_messages[:10]:
                    msg_escaped = msg.replace("|", "\\|")[:80]
                    lines.append(f"| {msg_escaped} | {count:,} |")
                lines.append("")

        # 请求统计
        req_stats = report.request_stats
        if req_stats and req_stats.total_requests > 0:
            lines.append("## 📊 请求统计")
            lines.append("")
            lines.append("| 指标 | 数值 |")
            lines.append("|------|------|")
            lines.append(f"| 总请求数 | {req_stats.total_requests:,} |")
            lines.append(f"| 错误率 | {req_stats.error_rate:.2f}% |")
            lines.append(f"| 平均响应时间 | {req_stats.avg_response_time:.3f}s |")
            lines.append(f"| 最大响应时间 | {req_stats.max_response_time:.3f}s |")
            lines.append("")

            # 状态码
            if req_stats.by_status_code:
                lines.append("### 状态码分布")
                lines.append("")
                lines.append("| 状态码 | 次数 |")
                lines.append("|--------|------|")
                for code, count in sorted(req_stats.by_status_code.items()):
                    lines.append(f"| {code} | {count:,} |")
                lines.append("")

            # Top URLs
            if req_stats.top_urls:
                lines.append("### Top URLs")
                lines.append("")
                lines.append("| URL | 次数 |")
                lines.append("|-----|------|")
                for url, count in req_stats.top_urls[:10]:
                    lines.append(f"| `{url}` | {count:,} |")
                lines.append("")

        # 时间分布
        time_stats = report.timeline_stats
        if time_stats.by_hour:
            lines.append("## 📈 时间分布")
            lines.append("")

            if time_stats.start_time and time_stats.end_time:
                lines.append(f"- **时间范围**: {time_stats.start_time} ~ {time_stats.end_time}")

            lines.append(f"- **高峰时段**: {time_stats.peak_hour:02d}:00 ({time_stats.peak_count:,} 条)")
            lines.append("")

            # 按日分布
            if time_stats.by_day:
                lines.append("### 按日分布")
                lines.append("")
                lines.append("| 日期 | 记录数 |")
                lines.append("|------|--------|")
                for day, count in sorted(time_stats.by_day.items()):
                    lines.append(f"| {day} | {count:,} |")
                lines.append("")

        # 尾部
        lines.append("---")
        lines.append("")
        lines.append("*由 log-analyzer 生成*")

        return "\n".join(lines)

    def save(self, report: AnalysisReport, output_path: Path) -> Path:
        """保存 Markdown 报告到文件"""
        output_path = Path(output_path)
        content = self.generate(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

