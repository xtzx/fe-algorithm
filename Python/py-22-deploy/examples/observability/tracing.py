"""
分布式追踪（OpenTelemetry）

演示:
- Span 创建
- 上下文传播
- 属性添加
- FastAPI 集成
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional

# OpenTelemetry 导入
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("OpenTelemetry not installed. Run: pip install opentelemetry-api opentelemetry-sdk")


# ==================== 追踪配置 ====================


def setup_tracing(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
):
    """
    配置 OpenTelemetry 追踪
    
    Args:
        service_name: 服务名称
        otlp_endpoint: OTLP 导出端点（如 Jaeger）
        console_export: 是否输出到控制台
    """
    if not OTEL_AVAILABLE:
        return
    
    # 创建资源
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })
    
    # 创建 TracerProvider
    provider = TracerProvider(resource=resource)
    
    # 添加导出器
    if console_export:
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    
    if otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
    
    # 设置全局 TracerProvider
    trace.set_tracer_provider(provider)


def get_tracer(name: str = __name__):
    """获取 Tracer"""
    if not OTEL_AVAILABLE:
        return None
    return trace.get_tracer(name)


# ==================== 追踪工具 ====================


@contextmanager
def create_span(
    tracer,
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    创建 Span 上下文管理器
    
    Usage:
        tracer = get_tracer(__name__)
        with create_span(tracer, "process_order", {"order_id": 123}):
            # 处理逻辑
            ...
    """
    if tracer is None:
        yield None
        return
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def add_span_attributes(span, **attributes):
    """添加 Span 属性"""
    if span is None:
        return
    for key, value in attributes.items():
        span.set_attribute(key, value)


def add_span_event(span, name: str, attributes: Optional[Dict[str, Any]] = None):
    """添加 Span 事件"""
    if span is None:
        return
    span.add_event(name, attributes=attributes or {})


# ==================== FastAPI 集成 ====================


def instrument_fastapi(app):
    """
    为 FastAPI 应用添加追踪
    
    Args:
        app: FastAPI 应用实例
    """
    if not OTEL_AVAILABLE:
        return
    
    FastAPIInstrumentor.instrument_app(app)


def uninstrument_fastapi(app):
    """移除 FastAPI 追踪"""
    if not OTEL_AVAILABLE:
        return
    
    FastAPIInstrumentor.uninstrument_app(app)


# ==================== 追踪装饰器 ====================


def traced(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    追踪装饰器
    
    Usage:
        @traced("process_data", {"data_type": "json"})
        async def process_data(data):
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or func.__name__
            
            with create_span(tracer, span_name, attributes) as span:
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or func.__name__
            
            with create_span(tracer, span_name, attributes) as span:
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ==================== 使用示例 ====================


def demo_tracing():
    """演示追踪使用"""
    
    # 配置追踪（输出到控制台）
    setup_tracing("demo-service", console_export=True)
    
    tracer = get_tracer(__name__)
    
    if tracer is None:
        print("Tracing not available")
        return
    
    # 创建根 Span
    with tracer.start_as_current_span("handle_request") as root_span:
        root_span.set_attribute("http.method", "GET")
        root_span.set_attribute("http.url", "/api/users")
        
        # 嵌套 Span
        with tracer.start_as_current_span("validate_request") as validate_span:
            validate_span.add_event("validation_started")
            # 模拟验证
            validate_span.add_event("validation_completed", {"valid": True})
        
        # 数据库查询 Span
        with tracer.start_as_current_span("database_query") as db_span:
            db_span.set_attribute("db.system", "postgresql")
            db_span.set_attribute("db.operation", "SELECT")
            db_span.set_attribute("db.table", "users")
            # 模拟查询
            db_span.add_event("query_executed", {"rows": 10})
        
        # 完成
        root_span.set_status(Status(StatusCode.OK))


if __name__ == "__main__":
    demo_tracing()


