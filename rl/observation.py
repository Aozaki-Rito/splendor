from typing import List, Optional

import numpy as np

from game.card import Card, GemColor
from game.game import Game
from game.noble import Noble
from game.player import Player

VISIBLE_COLORS: List[GemColor] = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
]

ALL_GEM_COLORS: List[GemColor] = VISIBLE_COLORS + [GemColor.GOLD]
MAX_DISPLAY_SLOTS_PER_LEVEL = 4
MAX_RESERVED_SLOTS = 3
MAX_NOBLE_SLOTS = 3

MAX_SCORE = 20.0
MAX_GEMS_PER_COLOR = 10.0
MAX_DISCOUNT = 7.0
MAX_CARD_POINTS = 5.0
MAX_CARD_COST = 7.0
MAX_NOBLE_POINTS = 3.0
MAX_NOBLE_REQUIREMENT = 4.0
MAX_RESERVED_COUNT = 3.0
MAX_NOBLES_COUNT = 3.0
MAX_GEM_SLOTS_LEFT = 10.0
MAX_ROUND = 40.0
MAX_PLAYERS = 4.0
MAX_DECK_COUNTS = {1: 36.0, 2: 26.0, 3: 16.0}


def _clip_norm(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _encode_color_one_hot(color: Optional[GemColor]) -> List[float]:
    return [1.0 if color == target else 0.0 for target in VISIBLE_COLORS]


def _encode_level_one_hot(level: Optional[int]) -> List[float]:
    return [1.0 if level == target else 0.0 for target in (1, 2, 3)]


def _missing_after_discount(card: Card, player: Player) -> List[float]:
    discounts = player.get_card_discounts()
    values: List[float] = []
    for color in VISIBLE_COLORS:
        raw_cost = card.cost.get(color, 0)
        missing = max(0, raw_cost - discounts.get(color, 0))
        values.append(_clip_norm(missing, MAX_CARD_COST))
    return values


def _encode_card(card: Optional[Card], player: Player) -> List[float]:
    if card is None:
        return [0.0] * 21

    encoded: List[float] = [1.0]
    encoded.extend(_encode_level_one_hot(card.level))
    encoded.append(_clip_norm(card.points, MAX_CARD_POINTS))
    encoded.extend(_encode_color_one_hot(card.gem_color))
    encoded.extend(_clip_norm(card.cost.get(color, 0), MAX_CARD_COST) for color in VISIBLE_COLORS)
    encoded.append(1.0 if player.can_afford_card(card) else 0.0)
    encoded.extend(_missing_after_discount(card, player))
    return encoded


def _encode_player_features(player: Player) -> List[float]:
    score = _clip_norm(player.get_score(), MAX_SCORE)
    gems = [_clip_norm(player.gems.get(color, 0), MAX_GEMS_PER_COLOR) for color in ALL_GEM_COLORS]
    discounts = [_clip_norm(player.get_card_discount(color), MAX_DISCOUNT) for color in VISIBLE_COLORS]
    reserved_count = _clip_norm(len(player.reserved_cards), MAX_RESERVED_COUNT)
    nobles_count = _clip_norm(len(player.nobles), MAX_NOBLES_COUNT)
    gem_slots_left = _clip_norm(max(0, 10 - player.get_total_gems()), MAX_GEM_SLOTS_LEFT)
    return [score, *gems, *discounts, reserved_count, nobles_count, gem_slots_left]


def _encode_opponent_features(player: Player) -> List[float]:
    score = _clip_norm(player.get_score(), MAX_SCORE)
    gems = [_clip_norm(player.gems.get(color, 0), MAX_GEMS_PER_COLOR) for color in ALL_GEM_COLORS]
    discounts = [_clip_norm(player.get_card_discount(color), MAX_DISCOUNT) for color in VISIBLE_COLORS]
    reserved_count = _clip_norm(len(player.reserved_cards), MAX_RESERVED_COUNT)
    nobles_count = _clip_norm(len(player.nobles), MAX_NOBLES_COUNT)
    return [score, *gems, *discounts, reserved_count, nobles_count]


def _encode_noble(noble: Optional[Noble], self_player: Player, opponent: Player) -> List[float]:
    if noble is None:
        return [0.0] * 17

    self_discounts = self_player.get_card_discounts()
    opp_discounts = opponent.get_card_discounts()
    encoded: List[float] = [1.0, _clip_norm(noble.points, MAX_NOBLE_POINTS)]
    encoded.extend(_clip_norm(noble.requirements.get(color, 0), MAX_NOBLE_REQUIREMENT) for color in VISIBLE_COLORS)
    encoded.extend(
        _clip_norm(max(0, noble.requirements.get(color, 0) - self_discounts.get(color, 0)), MAX_NOBLE_REQUIREMENT)
        for color in VISIBLE_COLORS
    )
    encoded.extend(
        _clip_norm(max(0, noble.requirements.get(color, 0) - opp_discounts.get(color, 0)), MAX_NOBLE_REQUIREMENT)
        for color in VISIBLE_COLORS
    )
    return encoded


def encode_observation(game: Game, player_index: int = 0) -> np.ndarray:
    """将当前局面编码为固定长度 observation。"""
    self_player = game.players[player_index]
    opponents = [player for idx, player in enumerate(game.players) if idx != player_index]
    opponent = opponents[0] if opponents else Player("dummy_opp", "Dummy Opponent")

    features: List[float] = [
        _clip_norm(game.round_number, MAX_ROUND),
        1.0 if game.last_round else 0.0,
        _clip_norm(game.num_players, MAX_PLAYERS),
    ]

    features.extend(_encode_player_features(self_player))
    features.extend(_encode_opponent_features(opponent))

    for level in (1, 2, 3):
        cards = game.board.displayed_cards[level]
        for slot_idx in range(MAX_DISPLAY_SLOTS_PER_LEVEL):
            card = cards[slot_idx] if slot_idx < len(cards) else None
            features.extend(_encode_card(card, self_player))

    for slot_idx in range(MAX_RESERVED_SLOTS):
        card = self_player.reserved_cards[slot_idx] if slot_idx < len(self_player.reserved_cards) else None
        features.extend(_encode_card(card, self_player))

    for slot_idx in range(MAX_RESERVED_SLOTS):
        card = opponent.reserved_cards[slot_idx] if slot_idx < len(opponent.reserved_cards) else None
        features.extend(_encode_card(card, self_player))

    for slot_idx in range(MAX_NOBLE_SLOTS):
        noble = game.board.nobles[slot_idx] if slot_idx < len(game.board.nobles) else None
        features.extend(_encode_noble(noble, self_player, opponent))

    for level in (1, 2, 3):
        features.append(_clip_norm(len(game.board.card_decks[level]), MAX_DECK_COUNTS[level]))

    return np.asarray(features, dtype=np.float32)


OBSERVATION_DIM = len(encode_observation(Game([Player("p1", "P1"), Player("p2", "P2")], seed=0), player_index=0))
