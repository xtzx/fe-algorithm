"""
爬虫工程化框架

提供:
- 合规爬取
- URL 去重
- 断点续爬
- 可测试设计
"""

from scraper.dedup import BloomFilter, HashSetDedup, UrlDedup
from scraper.fetcher import Fetcher, RateLimitedFetcher
from scraper.parser import HtmlParser, extract_links, extract_text
from scraper.pipeline import JsonLineWriter, Pipeline
from scraper.robots import RobotsChecker
from scraper.state import FileState, MemoryState, State

__version__ = "0.1.0"

__all__ = [
    # Fetcher
    "Fetcher",
    "RateLimitedFetcher",
    # Parser
    "HtmlParser",
    "extract_links",
    "extract_text",
    # Dedup
    "UrlDedup",
    "HashSetDedup",
    "BloomFilter",
    # State
    "State",
    "MemoryState",
    "FileState",
    # Pipeline
    "Pipeline",
    "JsonLineWriter",
    # Robots
    "RobotsChecker",
]

