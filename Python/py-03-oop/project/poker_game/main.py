#!/usr/bin/env python3
"""
扑克牌游戏 - P03 综合项目

实现 Card、Deck 类，支持洗牌、发牌、排序

用法：
    python main.py
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import total_ordering
import random
from typing import Iterator


class Suit(IntEnum):
    """花色"""

    CLUBS = 0  # 梅花 ♣
    DIAMONDS = 1  # 方块 ♦
    HEARTS = 2  # 红桃 ♥
    SPADES = 3  # 黑桃 ♠

    def __str__(self) -> str:
        symbols = ["♣", "♦", "♥", "♠"]
        return symbols[self.value]


class Rank(IntEnum):
    """点数"""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        if self.value <= 10:
            return str(self.value)
        names = {11: "J", 12: "Q", 13: "K", 14: "A"}
        return names[self.value]


@total_ordering
@dataclass(frozen=True)
class Card:
    """扑克牌"""

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.suit}{self.rank}"

    def __repr__(self) -> str:
        return f"Card({self.rank.name}, {self.suit.name})"

    def __lt__(self, other: "Card") -> bool:
        # 先按点数，再按花色
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit


class Deck:
    """牌组"""

    def __init__(self, shuffled: bool = False):
        self._cards: list[Card] = []
        self._build()
        if shuffled:
            self.shuffle()

    def _build(self) -> None:
        """生成一副牌"""
        self._cards = [Card(rank, suit) for suit in Suit for rank in Rank]

    def shuffle(self) -> None:
        """洗牌"""
        random.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        """发牌"""
        if n > len(self._cards):
            raise ValueError(f"Not enough cards. Only {len(self._cards)} left.")
        dealt = self._cards[:n]
        self._cards = self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        """发一张牌"""
        return self.deal(1)[0]

    def reset(self) -> None:
        """重置牌组"""
        self._build()

    def sort(self) -> None:
        """排序"""
        self._cards.sort()

    def __len__(self) -> int:
        return len(self._cards)

    def __getitem__(self, index: int) -> Card:
        return self._cards[index]

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)

    def __repr__(self) -> str:
        return f"Deck({len(self._cards)} cards)"


class Hand:
    """手牌"""

    def __init__(self, cards: list[Card] | None = None):
        self._cards = list(cards) if cards else []

    def add(self, card: Card) -> None:
        """添加一张牌"""
        self._cards.append(card)

    def add_cards(self, cards: list[Card]) -> None:
        """添加多张牌"""
        self._cards.extend(cards)

    def remove(self, card: Card) -> None:
        """移除一张牌"""
        self._cards.remove(card)

    def sort(self) -> None:
        """排序"""
        self._cards.sort()

    def clear(self) -> None:
        """清空"""
        self._cards.clear()

    def __len__(self) -> int:
        return len(self._cards)

    def __getitem__(self, index: int) -> Card:
        return self._cards[index]

    def __iter__(self) -> Iterator[Card]:
        return iter(self._cards)

    def __repr__(self) -> str:
        cards_str = " ".join(str(c) for c in self._cards)
        return f"Hand[{cards_str}]"


class Player:
    """玩家"""

    def __init__(self, name: str):
        self.name = name
        self.hand = Hand()

    def receive_cards(self, cards: list[Card]) -> None:
        """接收牌"""
        self.hand.add_cards(cards)

    def show_hand(self) -> str:
        """展示手牌"""
        return f"{self.name}: {self.hand}"

    def __repr__(self) -> str:
        return f"Player({self.name!r}, {len(self.hand)} cards)"


def demo_basic():
    """基本演示"""
    print("=== 基本演示 ===")

    # 创建单张牌
    card = Card(Rank.ACE, Suit.SPADES)
    print(f"一张牌: {card}")

    # 创建牌组
    deck = Deck()
    print(f"新牌组: {deck}")

    # 洗牌
    deck.shuffle()
    print("洗牌后前 5 张:", [str(c) for c in list(deck)[:5]])

    # 发牌
    cards = deck.deal(5)
    print(f"发 5 张牌: {[str(c) for c in cards]}")
    print(f"剩余: {deck}")


def demo_sorting():
    """排序演示"""
    print("\n=== 排序演示 ===")

    deck = Deck(shuffled=True)
    hand = Hand(deck.deal(10))

    print(f"发牌后: {hand}")
    hand.sort()
    print(f"排序后: {hand}")


def demo_comparison():
    """比较演示"""
    print("\n=== 比较演示 ===")

    c1 = Card(Rank.ACE, Suit.SPADES)
    c2 = Card(Rank.KING, Suit.HEARTS)
    c3 = Card(Rank.ACE, Suit.HEARTS)

    print(f"{c1} > {c2}: {c1 > c2}")
    print(f"{c1} > {c3}: {c1 > c3}")
    print(f"{c1} == {c3}: {c1 == c3}")


def demo_game():
    """模拟游戏"""
    print("\n=== 模拟发牌游戏 ===")

    # 创建玩家
    players = [Player("Alice"), Player("Bob"), Player("Charlie")]

    # 创建并洗牌
    deck = Deck(shuffled=True)

    # 每人发 5 张牌
    for _ in range(5):
        for player in players:
            player.receive_cards(deck.deal(1))

    # 展示手牌
    for player in players:
        player.hand.sort()
        print(player.show_hand())

    print(f"\n剩余牌数: {len(deck)}")


def main():
    """主函数"""
    print("🃏 扑克牌游戏演示")
    print("=" * 40)

    demo_basic()
    demo_sorting()
    demo_comparison()
    demo_game()

    print("\n" + "=" * 40)
    print("✅ 演示完成")


if __name__ == "__main__":
    main()

