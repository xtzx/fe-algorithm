"""
JSON 报告器
"""

import json
from pathlib import Path

from log_analyzer.models import AnalysisReport


class JsonReporter:
    """JSON 报告器"""

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def generate(self, report: AnalysisReport) -> str:
        """生成 JSON 报告"""
        data = report.model_dump(mode="json")
        return json.dumps(data, indent=self.indent, ensure_ascii=False, default=str)

    def save(self, report: AnalysisReport, output_path: Path) -> Path:
        """保存 JSON 报告到文件"""
        output_path = Path(output_path)
        content = self.generate(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

