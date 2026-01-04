#!/usr/bin/env python3
"""
词频统计器测试
"""

import unittest
from collections import Counter

from main import extract_words, count_words, get_top_n, analyze_text


class TestExtractWords(unittest.TestCase):
    """测试单词提取"""

    def test_simple_text(self):
        text = "Hello World"
        words = extract_words(text)
        self.assertEqual(words, ["hello", "world"])

    def test_with_punctuation(self):
        text = "Hello, World! How are you?"
        words = extract_words(text)
        self.assertEqual(words, ["hello", "world", "how", "are", "you"])

    def test_with_numbers(self):
        text = "Python3 is great in 2024"
        words = extract_words(text)
        self.assertEqual(words, ["python", "is", "great", "in"])

    def test_case_insensitive(self):
        text = "Hello HELLO hello"
        words = extract_words(text)
        self.assertEqual(words, ["hello", "hello", "hello"])

    def test_empty_text(self):
        words = extract_words("")
        self.assertEqual(words, [])


class TestCountWords(unittest.TestCase):
    """测试词频统计"""

    def test_basic_count(self):
        words = ["hello", "world", "hello"]
        counter = count_words(words)
        self.assertEqual(counter["hello"], 2)
        self.assertEqual(counter["world"], 1)

    def test_empty_list(self):
        counter = count_words([])
        self.assertEqual(len(counter), 0)


class TestGetTopN(unittest.TestCase):
    """测试 Top N"""

    def test_top_n(self):
        counter = Counter({"a": 5, "b": 3, "c": 1})
        top = get_top_n(counter, 2)
        self.assertEqual(top, [("a", 5), ("b", 3)])

    def test_n_larger_than_count(self):
        counter = Counter({"a": 5})
        top = get_top_n(counter, 10)
        self.assertEqual(len(top), 1)


class TestAnalyzeText(unittest.TestCase):
    """测试文本分析"""

    def test_analyze(self):
        text = "Hello world. Hello Python."
        result = analyze_text(text, 10)

        self.assertEqual(result["total_words"], 4)
        self.assertEqual(result["unique_words"], 3)
        self.assertEqual(result["top_words"][0], ("hello", 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)

