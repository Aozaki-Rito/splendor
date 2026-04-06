from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from game.card import GemColor
from game.game import Action, ActionType, Game

VISIBLE_COLORS: List[GemColor] = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
]

DIFFERENT_GEM_COMBOS: List[Tuple[GemColor, GemColor, GemColor]] = list(combinations(VISIBLE_COLORS, 3))
SAME_GEM_COLORS: List[GemColor] = list(VISIBLE_COLORS)

ACTION_DIM = 45

_DIFFERENT_GEM_TO_ID = {combo: idx for idx, combo in enumerate(DIFFERENT_GEM_COMBOS)}
_SAME_GEM_TO_ID = {color: 10 + idx for idx, color in enumerate(SAME_GEM_COLORS)}


def _get_player(game: Game, player_index: Optional[int] = None):
    if player_index is None:
        return game.get_current_player()
    return game.players[player_index]


def _canonical_different_combo(colors: Sequence[GemColor]) -> Tuple[GemColor, GemColor, GemColor]:
    """
    将 late game 中可能出现的 1-2 色 TAKE_DIFFERENT_GEMS 动作映射到一个稳定的 3 色组合 ID。

    V1 仍保持 45 维固定动作空间，因此这里采用“按固定颜色顺序补齐”的方式做兼容。
    例如：
    - [white, blue] -> (white, blue, green)
    - [red] -> (white, blue, red)
    """
    ordered = list(sorted(colors, key=VISIBLE_COLORS.index))
    for color in VISIBLE_COLORS:
        if color not in ordered:
            ordered.append(color)
        if len(ordered) == 3:
            break
    return tuple(sorted(ordered[:3], key=VISIBLE_COLORS.index))


def encode_action_to_id(game: Game, action: Action, player_index: Optional[int] = None) -> Optional[int]:
    """将当前局面下的 Action 编码为固定动作 ID。"""
    player = _get_player(game, player_index)

    if action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
        colors = action.params.get("colors", [])
        if len(colors) < 3:
            colors = _canonical_different_combo(colors)
        else:
            colors = tuple(sorted(colors, key=VISIBLE_COLORS.index))
        return _DIFFERENT_GEM_TO_ID.get(colors)

    if action.action_type == ActionType.TAKE_SAME_GEMS:
        color = action.params.get("color")
        return _SAME_GEM_TO_ID.get(color)

    if action.action_type == ActionType.RESERVE_CARD:
        level = int(action.params.get("level", 0))
        if action.params.get("from_deck", False):
            return 27 + (level - 1)

        card_id = action.params.get("card_id")
        for slot_idx, card in enumerate(game.board.displayed_cards.get(level, [])):
            if card.card_id == card_id:
                return 15 + (level - 1) * 4 + slot_idx
        return None

    if action.action_type == ActionType.BUY_CARD:
        level = int(action.params.get("level", 0))
        card_id = action.params.get("card_id")
        for slot_idx, card in enumerate(game.board.displayed_cards.get(level, [])):
            if card.card_id == card_id:
                return 30 + (level - 1) * 4 + slot_idx
        return None

    if action.action_type == ActionType.BUY_RESERVED_CARD:
        card_id = action.params.get("card_id")
        for slot_idx, card in enumerate(player.reserved_cards[:3]):
            if card.card_id == card_id:
                return 42 + slot_idx
        return None

    return None


def decode_action_id(game: Game, action_id: int, player_index: Optional[int] = None) -> Optional[Action]:
    """将固定动作 ID 解码为当前局面下的合法 Action。"""
    valid_actions = game.get_valid_actions()
    for action in valid_actions:
        encoded = encode_action_to_id(game, action, player_index=player_index)
        if encoded == action_id:
            return action
    return None


def get_action_mask(game: Game, player_index: Optional[int] = None) -> np.ndarray:
    """获取当前局面下的合法动作掩码。"""
    mask = np.zeros(ACTION_DIM, dtype=np.int8)
    for action in game.get_valid_actions():
        action_id = encode_action_to_id(game, action, player_index=player_index)
        if action_id is not None:
            mask[action_id] = 1
    return mask


def action_id_to_label(action_id: int) -> str:
    """将固定动作 ID 转为可读描述，便于调试。"""
    if 0 <= action_id <= 9:
        combo = DIFFERENT_GEM_COMBOS[action_id]
        return f"take_diff:{','.join(color.value for color in combo)}"
    if 10 <= action_id <= 14:
        color = SAME_GEM_COLORS[action_id - 10]
        return f"take_same:{color.value}"
    if 15 <= action_id <= 26:
        level = (action_id - 15) // 4 + 1
        slot = (action_id - 15) % 4
        return f"reserve_display:l{level}:{slot}"
    if 27 <= action_id <= 29:
        level = action_id - 27 + 1
        return f"reserve_deck:l{level}"
    if 30 <= action_id <= 41:
        level = (action_id - 30) // 4 + 1
        slot = (action_id - 30) % 4
        return f"buy_display:l{level}:{slot}"
    if 42 <= action_id <= 44:
        slot = action_id - 42
        return f"buy_reserved:{slot}"
    return f"unknown:{action_id}"
