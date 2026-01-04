"""
CLI 入口

提供命令行操作:
- 数据库初始化
- 迁移管理
- 缓存操作
- 队列操作
"""

import argparse
import sys


def cmd_db_init(args):
    """初始化数据库"""
    from storage_lab.db.session import init_db
    init_db()


def cmd_db_drop(args):
    """删除所有表"""
    from storage_lab.db.session import drop_db
    if args.yes or input("确认删除所有表? [y/N] ").lower() == "y":
        drop_db()
    else:
        print("已取消")


def cmd_cache_ping(args):
    """检查 Redis 连接"""
    from storage_lab.cache.client import get_cache_client
    client = get_cache_client()
    if client.ping():
        print("✅ Redis 连接正常")
    else:
        print("❌ Redis 连接失败")


def cmd_cache_clear(args):
    """清空缓存"""
    from storage_lab.cache.client import get_cache_client
    client = get_cache_client()
    pattern = args.pattern or "*"
    count = client.delete_pattern(pattern)
    print(f"已删除 {count} 个键")


def cmd_queue_stats(args):
    """查看队列统计"""
    from storage_lab.queue.simple import SimpleQueue
    queue = SimpleQueue(args.name)
    stats = queue.get_stats()
    print(f"队列: {args.name}")
    print(f"  待处理: {stats['pending']}")
    print(f"  处理中: {stats['processing']}")
    print(f"  已完成: {stats['completed']}")
    print(f"  失败: {stats['failed']}")


def cmd_queue_worker(args):
    """启动 Worker"""
    import logging
    from storage_lab.queue.worker import create_example_worker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    worker = create_example_worker(args.name)
    worker.run(burst=args.burst)


def cmd_demo(args):
    """运行演示"""
    from decimal import Decimal
    from storage_lab.db.session import SessionLocal, init_db
    from storage_lab.repositories import UserRepository, ItemRepository

    # 初始化数据库
    init_db()

    # 创建会话
    session = SessionLocal()

    try:
        # 创建 Repository
        user_repo = UserRepository(session)
        item_repo = ItemRepository(session)

        # 创建用户
        print("\n=== 创建用户 ===")
        user = user_repo.create_user(
            username="demo_user",
            email="demo@example.com",
            hashed_password="hashed_password_123",
            full_name="Demo User",
        )
        print(f"创建用户: {user}")

        # 创建商品
        print("\n=== 创建商品 ===")
        item = item_repo.create_item(
            name="Demo Item",
            price=Decimal("99.99"),
            owner_id=user.id,
            description="This is a demo item",
            quantity=10,
        )
        print(f"创建商品: {item}")

        # 添加标签
        print("\n=== 添加标签 ===")
        item_repo.add_tag(item.id, "demo")
        item_repo.add_tag(item.id, "test")
        item_with_tags = item_repo.get_with_tags(item.id)
        print(f"商品标签: {[t.name for t in item_with_tags.tags]}")

        # 查询
        print("\n=== 查询用户及商品 ===")
        user_with_items = user_repo.get_with_items(user.id)
        print(f"用户: {user_with_items.username}")
        print(f"商品数: {len(user_with_items.items)}")

        print("\n✅ 演示完成!")

    finally:
        session.close()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="存储与缓存学习项目 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # db 命令组
    db_parser = subparsers.add_parser("db", help="数据库操作")
    db_subparsers = db_parser.add_subparsers(dest="subcommand")

    # db init
    db_init_parser = db_subparsers.add_parser("init", help="初始化数据库")
    db_init_parser.set_defaults(func=cmd_db_init)

    # db drop
    db_drop_parser = db_subparsers.add_parser("drop", help="删除所有表")
    db_drop_parser.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    db_drop_parser.set_defaults(func=cmd_db_drop)

    # cache 命令组
    cache_parser = subparsers.add_parser("cache", help="缓存操作")
    cache_subparsers = cache_parser.add_subparsers(dest="subcommand")

    # cache ping
    cache_ping_parser = cache_subparsers.add_parser("ping", help="检查连接")
    cache_ping_parser.set_defaults(func=cmd_cache_ping)

    # cache clear
    cache_clear_parser = cache_subparsers.add_parser("clear", help="清空缓存")
    cache_clear_parser.add_argument("--pattern", help="键模式（默认 *）")
    cache_clear_parser.set_defaults(func=cmd_cache_clear)

    # queue 命令组
    queue_parser = subparsers.add_parser("queue", help="队列操作")
    queue_subparsers = queue_parser.add_subparsers(dest="subcommand")

    # queue stats
    queue_stats_parser = queue_subparsers.add_parser("stats", help="队列统计")
    queue_stats_parser.add_argument("--name", default="default", help="队列名称")
    queue_stats_parser.set_defaults(func=cmd_queue_stats)

    # queue worker
    queue_worker_parser = queue_subparsers.add_parser("worker", help="启动 Worker")
    queue_worker_parser.add_argument("--name", default="default", help="队列名称")
    queue_worker_parser.add_argument("--burst", action="store_true", help="处理完当前任务后退出")
    queue_worker_parser.set_defaults(func=cmd_queue_worker)

    # demo 命令
    demo_parser = subparsers.add_parser("demo", help="运行演示")
    demo_parser.set_defaults(func=cmd_demo)

    # 解析参数
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


