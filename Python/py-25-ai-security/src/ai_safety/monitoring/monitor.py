"""
监控系统

提供:
- 质量监控
- 成本监控
- 异常告警
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger()


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """指标点"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """告警"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Monitor:
    """
    基础监控器
    
    收集和存储指标
    
    Usage:
        monitor = Monitor()
        
        # 记录指标
        monitor.record("response_time", 150.0, tags={"model": "gpt-4"})
        
        # 获取统计
        stats = monitor.get_stats("response_time")
    """

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self._metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self._last_cleanup = datetime.utcnow()

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """记录指标"""
        point = MetricPoint(
            name=name,
            value=value,
            tags=tags or {},
        )
        self._metrics[name].append(point)
        
        # 定期清理
        self._cleanup_if_needed()

    def get_stats(
        self,
        name: str,
        window_minutes: int = 60,
    ) -> Dict[str, float]:
        """
        获取指标统计
        
        Args:
            name: 指标名称
            window_minutes: 时间窗口（分钟）
        
        Returns:
            统计数据
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        points = [
            p for p in self._metrics.get(name, [])
            if p.timestamp > cutoff
        ]
        
        if not points:
            return {"count": 0}
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
        }

    def get_recent(
        self,
        name: str,
        count: int = 100,
    ) -> List[MetricPoint]:
        """获取最近的指标点"""
        return self._metrics.get(name, [])[-count:]

    def _cleanup_if_needed(self):
        """清理过期数据"""
        now = datetime.utcnow()
        if (now - self._last_cleanup).seconds < 3600:
            return
        
        cutoff = now - timedelta(hours=self.retention_hours)
        
        for name in self._metrics:
            self._metrics[name] = [
                p for p in self._metrics[name]
                if p.timestamp > cutoff
            ]
        
        self._last_cleanup = now


class QualityMonitor(Monitor):
    """
    质量监控器
    
    监控 AI 输出质量
    
    Usage:
        monitor = QualityMonitor()
        
        # 记录响应
        monitor.record_response(
            response="...",
            latency_ms=150,
            model="gpt-4",
        )
        
        # 获取质量报告
        report = monitor.get_quality_report()
    """

    def __init__(self):
        super().__init__()
        self._response_count = 0
        self._error_count = 0
        self._blocked_count = 0

    def record_response(
        self,
        response: str,
        latency_ms: float,
        model: str,
        tokens: int = 0,
        was_blocked: bool = False,
        error: Optional[str] = None,
    ):
        """记录响应"""
        self._response_count += 1
        
        # 延迟
        self.record("latency_ms", latency_ms, tags={"model": model})
        
        # Token
        if tokens:
            self.record("tokens", tokens, tags={"model": model})
        
        # 响应长度
        self.record("response_length", len(response), tags={"model": model})
        
        # 错误
        if error:
            self._error_count += 1
            self.record("error", 1, tags={"model": model, "error": error[:50]})
        
        # 阻止
        if was_blocked:
            self._blocked_count += 1
            self.record("blocked", 1, tags={"model": model})
        
        logger.debug(
            "response_recorded",
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
        )

    def get_quality_report(self, window_minutes: int = 60) -> Dict[str, Any]:
        """获取质量报告"""
        latency_stats = self.get_stats("latency_ms", window_minutes)
        token_stats = self.get_stats("tokens", window_minutes)
        
        return {
            "total_responses": self._response_count,
            "error_count": self._error_count,
            "blocked_count": self._blocked_count,
            "error_rate": self._error_count / self._response_count if self._response_count else 0,
            "block_rate": self._blocked_count / self._response_count if self._response_count else 0,
            "latency": latency_stats,
            "tokens": token_stats,
        }


class CostMonitor(Monitor):
    """
    成本监控器
    
    监控 LLM 使用成本
    
    Usage:
        monitor = CostMonitor(daily_budget=100.0)
        
        # 记录使用
        monitor.record_usage(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
        )
        
        # 检查预算
        if monitor.is_over_budget():
            alert("Budget exceeded!")
    """

    # 模型价格（每 1K token，美元）
    MODEL_PRICES = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, daily_budget: float = 100.0):
        super().__init__()
        self.daily_budget = daily_budget
        self._daily_costs: Dict[str, float] = {}  # date -> cost

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """记录使用"""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # 记录指标
        self.record("input_tokens", input_tokens, tags={"model": model})
        self.record("output_tokens", output_tokens, tags={"model": model})
        self.record("cost", cost, tags={"model": model})
        
        # 累计日成本
        today = datetime.utcnow().date().isoformat()
        self._daily_costs[today] = self._daily_costs.get(today, 0) + cost
        
        logger.debug(
            "usage_recorded",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """计算成本"""
        prices = self.MODEL_PRICES.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * prices["input"]
        output_cost = (output_tokens / 1000) * prices["output"]
        
        return input_cost + output_cost

    def get_today_cost(self) -> float:
        """获取今日成本"""
        today = datetime.utcnow().date().isoformat()
        return self._daily_costs.get(today, 0)

    def is_over_budget(self) -> bool:
        """是否超出预算"""
        return self.get_today_cost() > self.daily_budget

    def get_cost_report(self) -> Dict[str, Any]:
        """获取成本报告"""
        today = datetime.utcnow().date().isoformat()
        today_cost = self._daily_costs.get(today, 0)
        
        cost_stats = self.get_stats("cost", window_minutes=60 * 24)
        token_stats = self.get_stats("input_tokens", window_minutes=60 * 24)
        
        return {
            "today_cost": today_cost,
            "daily_budget": self.daily_budget,
            "budget_used_percent": (today_cost / self.daily_budget * 100) if self.daily_budget else 0,
            "is_over_budget": self.is_over_budget(),
            "cost_stats": cost_stats,
            "token_stats": token_stats,
        }


class AlertManager:
    """
    告警管理器
    
    基于阈值的告警
    
    Usage:
        alert_mgr = AlertManager()
        
        # 添加规则
        alert_mgr.add_rule(
            name="high_latency",
            metric="latency_ms",
            threshold=1000,
            level=AlertLevel.WARNING,
        )
        
        # 检查告警
        alerts = alert_mgr.check(monitor)
    """

    @dataclass
    class AlertRule:
        """告警规则"""
        name: str
        metric: str
        threshold: float
        level: AlertLevel
        comparison: str = "gt"  # gt, lt, eq
        window_minutes: int = 5
        cooldown_minutes: int = 30

    def __init__(self):
        self._rules: List[AlertManager.AlertRule] = []
        self._last_alerts: Dict[str, datetime] = {}
        self._handlers: List[Callable[[Alert], None]] = []

    def add_rule(
        self,
        name: str,
        metric: str,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
        comparison: str = "gt",
        window_minutes: int = 5,
        cooldown_minutes: int = 30,
    ):
        """添加告警规则"""
        rule = self.AlertRule(
            name=name,
            metric=metric,
            threshold=threshold,
            level=level,
            comparison=comparison,
            window_minutes=window_minutes,
            cooldown_minutes=cooldown_minutes,
        )
        self._rules.append(rule)

    def add_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self._handlers.append(handler)

    def check(self, monitor: Monitor) -> List[Alert]:
        """检查所有规则"""
        alerts = []
        now = datetime.utcnow()
        
        for rule in self._rules:
            # 检查冷却
            last_alert = self._last_alerts.get(rule.name)
            if last_alert:
                if (now - last_alert).seconds < rule.cooldown_minutes * 60:
                    continue
            
            # 获取指标
            stats = monitor.get_stats(rule.metric, rule.window_minutes)
            if stats["count"] == 0:
                continue
            
            current_value = stats["avg"]
            
            # 比较
            triggered = False
            if rule.comparison == "gt" and current_value > rule.threshold:
                triggered = True
            elif rule.comparison == "lt" and current_value < rule.threshold:
                triggered = True
            elif rule.comparison == "eq" and current_value == rule.threshold:
                triggered = True
            
            if triggered:
                alert = Alert(
                    level=rule.level,
                    message=f"Rule '{rule.name}' triggered: {current_value:.2f} {rule.comparison} {rule.threshold}",
                    metric_name=rule.metric,
                    current_value=current_value,
                    threshold=rule.threshold,
                )
                alerts.append(alert)
                self._last_alerts[rule.name] = now
                
                # 调用处理器
                for handler in self._handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error("alert_handler_error", error=str(e))
                
                logger.warning(
                    "alert_triggered",
                    rule=rule.name,
                    level=rule.level.value,
                    metric=rule.metric,
                    value=current_value,
                )
        
        return alerts


