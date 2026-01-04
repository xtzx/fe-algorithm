"""
Worker 实现

执行队列中的任务

Usage:
    # 注册任务处理函数
    worker = Worker(queue)

    @worker.task("send_email")
    def send_email(to: str, subject: str, body: str):
        # 发送邮件逻辑
        ...

    # 启动 worker
    worker.run()
"""

import logging
import signal
import time
import traceback
from typing import Any, Callable, Dict, Optional

from storage_lab.queue.simple import SimpleQueue, Task, TaskStatus

logger = logging.getLogger(__name__)


class Worker:
    """
    任务 Worker

    从队列获取任务并执行
    """

    def __init__(
        self,
        queue: Optional[SimpleQueue] = None,
        queue_name: str = "default",
    ):
        self.queue = queue or SimpleQueue(queue_name)
        self._handlers: Dict[str, Callable] = {}
        self._running = False

    def task(self, name: str) -> Callable:
        """
        任务装饰器

        Usage:
            @worker.task("send_email")
            def send_email(to: str, subject: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            self._handlers[name] = func
            return func
        return decorator

    def register(self, name: str, handler: Callable):
        """
        注册任务处理函数

        Args:
            name: 任务名称
            handler: 处理函数
        """
        self._handlers[name] = handler

    def process_task(self, task: Task) -> Any:
        """
        处理单个任务

        Args:
            task: 任务对象

        Returns:
            任务执行结果
        """
        handler = self._handlers.get(task.name)
        if handler is None:
            raise ValueError(f"No handler registered for task: {task.name}")

        return handler(*task.args, **task.kwargs)

    def run_once(self, timeout: int = 0) -> bool:
        """
        执行一次任务

        Args:
            timeout: 等待任务的超时时间（秒）

        Returns:
            是否执行了任务
        """
        task = self.queue.dequeue(timeout=timeout)
        if task is None:
            return False

        logger.info(f"Processing task: {task.id} ({task.name})")

        try:
            result = self.process_task(task)
            self.queue.complete(task, result)
            logger.info(f"Task completed: {task.id}")
            return True
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Task failed: {task.id} - {error_msg}")
            logger.debug(traceback.format_exc())
            self.queue.fail(task, error_msg)
            return True

    def run(
        self,
        burst: bool = False,
        interval: float = 1.0,
    ):
        """
        启动 Worker

        Args:
            burst: 是否只处理当前队列中的任务后退出
            interval: 空闲时的轮询间隔（秒）
        """
        self._running = True

        # 设置信号处理
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"Worker started, listening on queue: {self.queue.name}")
        logger.info(f"Registered handlers: {list(self._handlers.keys())}")

        while self._running:
            try:
                executed = self.run_once(timeout=int(interval))

                if burst and not executed:
                    # burst 模式下，队列为空时退出
                    break

            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(interval)

        logger.info("Worker stopped")

    def stop(self):
        """停止 Worker"""
        self._running = False


# ==================== 示例任务 ====================


def example_send_email(to: str, subject: str, body: str = "") -> dict:
    """示例：发送邮件任务"""
    logger.info(f"Sending email to {to}: {subject}")
    # 模拟发送
    time.sleep(0.5)
    return {"to": to, "subject": subject, "status": "sent"}


def example_process_image(image_path: str, size: tuple = (100, 100)) -> dict:
    """示例：处理图片任务"""
    logger.info(f"Processing image: {image_path} to size {size}")
    # 模拟处理
    time.sleep(1)
    return {"path": image_path, "new_size": size, "status": "processed"}


def example_generate_report(report_type: str, params: dict) -> dict:
    """示例：生成报告任务"""
    logger.info(f"Generating {report_type} report with params: {params}")
    # 模拟生成
    time.sleep(2)
    return {"type": report_type, "status": "generated"}


def create_example_worker(queue_name: str = "default") -> Worker:
    """创建示例 Worker"""
    worker = Worker(queue_name=queue_name)

    # 注册任务
    worker.register("send_email", example_send_email)
    worker.register("process_image", example_process_image)
    worker.register("generate_report", example_generate_report)

    return worker


if __name__ == "__main__":
    # 运行示例 Worker
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    worker = create_example_worker()
    worker.run()


