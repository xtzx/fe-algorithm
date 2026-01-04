#!/usr/bin/env python3
"""
扑克牌游戏测试
"""

import unittest
from main import Card, Deck, Hand, Player, Rank, Suit


class TestCard(unittest.TestCase):
    """测试 Card 类"""

    def test_create_card(self):
        card = Card(Rank.ACE, Suit.SPADES)
        self.assertEqual(card.rank, Rank.ACE)
        self.assertEqual(card.suit, Suit.SPADES)

    def test_card_str(self):
        card = Card(Rank.ACE, Suit.SPADES)
        self.assertEqual(str(card), "♠A")

    def test_card_comparison(self):
        ace_spades = Card(Rank.ACE, Suit.SPADES)
        king_hearts = Card(Rank.KING, Suit.HEARTS)
        ace_hearts = Card(Rank.ACE, Suit.HEARTS)

        self.assertGreater(ace_spades, king_hearts)
        self.assertGreater(ace_spades, ace_hearts)  # 黑桃 > 红桃
        self.assertNotEqual(ace_spades, ace_hearts)

    def test_card_equality(self):
        c1 = Card(Rank.ACE, Suit.SPADES)
        c2 = Card(Rank.ACE, Suit.SPADES)
        self.assertEqual(c1, c2)

    def test_card_hashable(self):
        """Card 应该可以作为字典键"""
        card = Card(Rank.ACE, Suit.SPADES)
        d = {card: "value"}
        self.assertEqual(d[Card(Rank.ACE, Suit.SPADES)], "value")


class TestDeck(unittest.TestCase):
    """测试 Deck 类"""

    def test_deck_size(self):
        deck = Deck()
        self.assertEqual(len(deck), 52)

    def test_deck_contains_all_cards(self):
        deck = Deck()
        cards = set(deck)
        self.assertEqual(len(cards), 52)

    def test_shuffle(self):
        deck1 = Deck()
        deck2 = Deck()
        deck2.shuffle()
        # 洗牌后顺序应该不同（极小概率相同）
        self.assertNotEqual(list(deck1), list(deck2))

    def test_deal(self):
        deck = Deck()
        cards = deck.deal(5)
        self.assertEqual(len(cards), 5)
        self.assertEqual(len(deck), 47)

    def test_deal_one(self):
        deck = Deck()
        card = deck.deal_one()
        self.assertIsInstance(card, Card)
        self.assertEqual(len(deck), 51)

    def test_deal_too_many(self):
        deck = Deck()
        deck.deal(50)
        with self.assertRaises(ValueError):
            deck.deal(5)

    def test_reset(self):
        deck = Deck()
        deck.deal(10)
        deck.reset()
        self.assertEqual(len(deck), 52)

    def test_sort(self):
        deck = Deck(shuffled=True)
        deck.sort()
        # 排序后第一张应该是梅花 2
        self.assertEqual(deck[0], Card(Rank.TWO, Suit.CLUBS))


class TestHand(unittest.TestCase):
    """测试 Hand 类"""

    def test_empty_hand(self):
        hand = Hand()
        self.assertEqual(len(hand), 0)

    def test_add_card(self):
        hand = Hand()
        card = Card(Rank.ACE, Suit.SPADES)
        hand.add(card)
        self.assertEqual(len(hand), 1)
        self.assertEqual(hand[0], card)

    def test_add_cards(self):
        hand = Hand()
        cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        hand.add_cards(cards)
        self.assertEqual(len(hand), 2)

    def test_remove_card(self):
        card = Card(Rank.ACE, Suit.SPADES)
        hand = Hand([card])
        hand.remove(card)
        self.assertEqual(len(hand), 0)

    def test_sort_hand(self):
        cards = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.KING, Suit.HEARTS),
        ]
        hand = Hand(cards)
        hand.sort()
        self.assertEqual(hand[0], Card(Rank.TWO, Suit.CLUBS))
        self.assertEqual(hand[-1], Card(Rank.ACE, Suit.SPADES))


class TestPlayer(unittest.TestCase):
    """测试 Player 类"""

    def test_create_player(self):
        player = Player("Alice")
        self.assertEqual(player.name, "Alice")
        self.assertEqual(len(player.hand), 0)

    def test_receive_cards(self):
        player = Player("Alice")
        cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
        player.receive_cards(cards)
        self.assertEqual(len(player.hand), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

