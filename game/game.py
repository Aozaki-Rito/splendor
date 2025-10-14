import random
import json
import time
from enum import Enum
from typing import List, Dict, Optional, Union
from game.player import Player
from game.board import Board
from game.card import Card, GemColor
from game.noble import Noble


class ActionType(Enum):
    """玩家可执行的动作类型"""
    TAKE_DIFFERENT_GEMS = "take_different_gems"
    TAKE_SAME_GEMS = "take_same_gems"
    RESERVE_CARD = "reserve_card"
    BUY_CARD = "buy_card"
    BUY_RESERVED_CARD = "buy_reserved_card"


class Action:
    """玩家动作类"""
    
    def __init__(self, action_type: ActionType, **kwargs):
        """初始化动作
        
        Args:
            action_type: 动作类型
            **kwargs: 动作相关的参数
        """
        self.action_type = action_type
        self.params = kwargs
    
    def __str__(self):
        if self.action_type == ActionType.TAKE_DIFFERENT_GEMS:
            colors = [f"{color.value}" for color in self.params.get("colors", [])]
            return f"拿取不同颜色的宝石: {', '.join(colors)}"
        
        elif self.action_type == ActionType.TAKE_SAME_GEMS:
            color = self.params.get("color")
            return f"拿取相同颜色的宝石: {color.value if color else 'Unknown'}"
        
        elif self.action_type == ActionType.RESERVE_CARD:
            card_id = self.params.get("card_id", "Unknown")
            level = self.params.get("level", "Unknown")
            return f"预留{level}级卡牌: {card_id}"
        
        elif self.action_type == ActionType.BUY_CARD:
            card_id = self.params.get("card_id", "Unknown")
            return f"购买卡牌: {card_id}"
        
        elif self.action_type == ActionType.BUY_RESERVED_CARD:
            card_id = self.params.get("card_id", "Unknown")
            return f"购买预留卡牌: {card_id}"
        
        return f"未知动作: {self.action_type}"
        
    def to_dict(self):
        """将动作转换为字典，用于JSON序列化"""
        # 处理参数中的枚举类型
        processed_params = {}
        for k, v in self.params.items():
            if isinstance(v, Enum):
                processed_params[k] = v.value
            elif isinstance(v, list) and all(isinstance(item, Enum) for item in v):
                processed_params[k] = [item.value for item in v]
            else:
                processed_params[k] = v
                
        return {
            "action_type": self.action_type.value,
            "params": processed_params,
            "description": str(self)
        }


class Game:
    """璀璨宝石游戏类"""
    
    def __init__(self, players: List[Player], seed: Optional[int] = None):
        """初始化游戏
        
        Args:
            players: 玩家列表
            seed: 随机数种子，用于复现游戏
        """
        if seed is not None:
            random.seed(seed)
        
        self.players = players
        self.num_players = len(players)
        self.board = Board(self.num_players)
        self.current_player_index = 0
        self.round_number = 1
        self.game_over = False
        self.winner = None
        self.last_round = False
        self.history = []  # 游戏历史记录
    
    def get_current_player(self) -> Player:
        """获取当前行动的玩家"""
        return self.players[self.current_player_index]
    
    def next_player(self):
        """切换到下一个玩家"""
        self.current_player_index = (self.current_player_index + 1) % self.num_players
        if self.current_player_index == 0:
            self.round_number += 1
    
    def get_valid_actions(self) -> List[Action]:
        """获取当前玩家可执行的有效动作列表"""
        actions = []
        player = self.get_current_player()
        
        # 拿取不同颜色的宝石
        different_gems = self._get_different_gems_actions()
        actions.extend(different_gems)
        
        # 拿取相同颜色的宝石
        same_gems = self._get_same_gems_actions()
        actions.extend(same_gems)
        
        # 预留卡牌
        reserve_card = self._get_reserve_card_actions(player)
        actions.extend(reserve_card)
        
        # 购买展示区的卡牌
        buy_card = self._get_buy_card_actions(player)
        actions.extend(buy_card)
        
        # 购买预留的卡牌
        buy_reserved = self._get_buy_reserved_card_actions(player)
        actions.extend(buy_reserved)
        
        return actions
    
    def _get_different_gems_actions(self) -> List[Action]:
        """获取拿取不同颜色宝石的所有可能动作"""
        actions = []
        available_colors = []
        
        for color in [GemColor.WHITE, GemColor.BLUE, GemColor.GREEN, GemColor.RED, GemColor.BLACK]:
            if self.board.gems.get(color, 0) > 0:
                available_colors.append(color)
        
        # 如果可用颜色少于3种，则返回所有可能的组合
        if len(available_colors) <= 3:
            actions.append(Action(ActionType.TAKE_DIFFERENT_GEMS, colors=available_colors))
            return actions
        
        # 否则返回所有3种颜色的组合
        from itertools import combinations
        for combo in combinations(available_colors, 3):
            actions.append(Action(ActionType.TAKE_DIFFERENT_GEMS, colors=list(combo)))
        
        return actions
    
    def _get_same_gems_actions(self) -> List[Action]:
        """获取拿取相同颜色宝石的所有可能动作"""
        actions = []
        
        for color in [GemColor.WHITE, GemColor.BLUE, GemColor.GREEN, GemColor.RED, GemColor.BLACK]:
            if self.board.gems.get(color, 0) >= 4:
                actions.append(Action(ActionType.TAKE_SAME_GEMS, color=color))
        
        return actions
    
    def _get_reserve_card_actions(self, player: Player) -> List[Action]:
        """获取预留卡牌的所有可能动作"""
        actions = []
        
        # 如果玩家已经预留了3张卡，不能再预留
        if len(player.reserved_cards) >= 3:
            return actions
        
        # 预留展示区的卡牌
        for level, cards in self.board.displayed_cards.items():
            for card in cards:
                actions.append(Action(ActionType.RESERVE_CARD, level=level, card_id=card.card_id))
        
        # 也可以预留牌堆顶的卡（如果牌堆不为空）
        for level, deck in self.board.card_decks.items():
            if deck:
                actions.append(Action(ActionType.RESERVE_CARD, level=level, from_deck=True))
        
        return actions
    
    def _get_buy_card_actions(self, player: Player) -> List[Action]:
        """获取购买展示区卡牌的所有可能动作"""
        actions = []
        
        for level, cards in self.board.displayed_cards.items():
            for card in cards:
                if player.can_afford_card(card):
                    actions.append(Action(ActionType.BUY_CARD, level=level, card_id=card.card_id))
        
        return actions
    
    def _get_buy_reserved_card_actions(self, player: Player) -> List[Action]:
        """获取购买预留卡牌的所有可能动作"""
        actions = []
        
        for card in player.reserved_cards:
            if player.can_afford_card(card):
                actions.append(Action(ActionType.BUY_RESERVED_CARD, card_id=card.card_id))
        
        return actions
    
    def execute_action(self, action: Action) -> bool:
        """执行动作
        
        Args:
            action: 要执行的动作
            
        Returns:
            bool: 动作是否执行成功
        """
        player = self.get_current_player()
        
        # 记录动作到历史记录
        self.history.append({
            "round": self.round_number,
            "player": player.player_id,
            "action": str(action),
            "action_data": action.to_dict()
        })
        
        if action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
            return self._execute_take_different_gems(player, action)
        
        elif action.action_type == ActionType.TAKE_SAME_GEMS:
            return self._execute_take_same_gems(player, action)
        
        elif action.action_type == ActionType.RESERVE_CARD:
            return self._execute_reserve_card(player, action)
        
        elif action.action_type == ActionType.BUY_CARD:
            return self._execute_buy_card(player, action)
        
        elif action.action_type == ActionType.BUY_RESERVED_CARD:
            return self._execute_buy_reserved_card(player, action)
        
        return False
    
    def _execute_take_different_gems(self, player: Player, action: Action) -> bool:
        """执行拿取不同颜色宝石的动作"""
        colors = action.params.get("colors", [])
        
        # 验证颜色数量和可用性
        if not colors or len(colors) > 3:
            return False
        
        # 检查所有颜色是否都可用
        gems_to_take = {}
        for color in colors:
            if self.board.gems.get(color, 0) <= 0:
                return False
            gems_to_take[color] = 1
        
        # 执行拿取宝石
        if not self.board.take_gems(gems_to_take):
            return False
        
        # 更新玩家宝石
        for color, count in gems_to_take.items():
            player.gems[color] = player.gems.get(color, 0) + count
        
        # 检查是否需要丢弃宝石
        self._check_and_discard_gems(player)
        
        return True
    
    def _execute_take_same_gems(self, player: Player, action: Action) -> bool:
        """执行拿取相同颜色宝石的动作"""
        color = action.params.get("color")
        
        if not color or self.board.gems.get(color, 0) < 4:
            return False
        
        # 执行拿取宝石
        gems_to_take = {color: 2}
        if not self.board.take_gems(gems_to_take):
            return False
        
        # 更新玩家宝石
        player.gems[color] = player.gems.get(color, 0) + 2
        
        # 检查是否需要丢弃宝石
        self._check_and_discard_gems(player)
        
        return True
    
    def _execute_reserve_card(self, player: Player, action: Action) -> bool:
        """执行预留卡牌的动作"""
        level = action.params.get("level")
        card_id = action.params.get("card_id")
        from_deck = action.params.get("from_deck", False)
        
        # 检查玩家预留卡是否已满
        if len(player.reserved_cards) >= 3:
            return False
        
        card = None
        
        if from_deck:
            # 从牌堆顶预留
            if self.board.card_decks.get(level, []):
                card = self.board.card_decks[level].pop()
        else:
            # 从展示区预留
            for i, c in enumerate(self.board.displayed_cards.get(level, [])):
                if c.card_id == card_id:
                    card = self.board.displayed_cards[level].pop(i)
                    break
        
        if card is None:
            return False
        
        # 添加到玩家的预留卡
        player.reserved_cards.append(card)
        
        # 补充展示区的卡牌
        self.board.replenish_displayed_cards()
        
        # 如果有黄金宝石可用，玩家获得一个黄金宝石
        if self.board.gems.get(GemColor.GOLD, 0) > 0:
            self.board.gems[GemColor.GOLD] -= 1
            player.gems[GemColor.GOLD] = player.gems.get(GemColor.GOLD, 0) + 1
            
            # 检查是否需要丢弃宝石
            self._check_and_discard_gems(player)
        
        return True
    
    def _execute_buy_card(self, player: Player, action: Action) -> bool:
        """执行购买展示区卡牌的动作"""
        level = action.params.get("level")
        card_id = action.params.get("card_id")
        
        # 查找卡牌
        card = None
        for i, c in enumerate(self.board.displayed_cards.get(level, [])):
            if c.card_id == card_id:
                card = c
                break
        
        if card is None or not player.can_afford_card(card):
            return False
        
        # 计算实际支付成本
        actual_cost = player.get_actual_cost(card)
        
        # 支付宝石
        for color, count in actual_cost.items():
            player.gems[color] -= count
            self.board.gems[color] = self.board.gems.get(color, 0) + count
        
        # 从展示区移除卡牌
        self.board.remove_displayed_card(level, card_id)
        
        # 将卡牌添加到玩家的卡牌列表
        player.cards.append(card)
        
        # 检查是否有贵族访问
        self._check_nobles_visit(player)
        
        # 检查游戏是否结束
        self._check_game_end()
        
        return True
    
    def _execute_buy_reserved_card(self, player: Player, action: Action) -> bool:
        """执行购买预留卡牌的动作"""
        card_id = action.params.get("card_id")
        
        # 查找卡牌
        card = None
        card_index = -1
        for i, c in enumerate(player.reserved_cards):
            if c.card_id == card_id:
                card = c
                card_index = i
                break
        
        if card is None or not player.can_afford_card(card):
            return False
        
        # 计算实际支付成本
        actual_cost = player.get_actual_cost(card)
        
        # 支付宝石
        for color, count in actual_cost.items():
            player.gems[color] -= count
            self.board.gems[color] = self.board.gems.get(color, 0) + count
        
        # 从预留卡中移除卡牌
        player.reserved_cards.pop(card_index)
        
        # 将卡牌添加到玩家的卡牌列表
        player.cards.append(card)
        
        # 检查是否有贵族访问
        self._check_nobles_visit(player)
        
        # 检查游戏是否结束
        self._check_game_end()
        
        return True
    
    def _check_and_discard_gems(self, player: Player):
        """检查并处理玩家超过10个宝石的情况"""
        total_gems = player.get_total_gems()
        
        while total_gems > 10:
            # 这里需要由玩家选择丢弃哪些宝石
            # 在AI对战框架中，需要请求代理做出选择
            
            # 临时策略：丢弃随机宝石
            available_colors = [color for color, count in player.gems.items() if count > 0]
            if not available_colors:
                break
                
            color_to_discard = random.choice(available_colors)
            player.gems[color_to_discard] -= 1
            self.board.gems[color_to_discard] += 1
            
            total_gems -= 1
    
    def _check_nobles_visit(self, player: Player):
        """检查并处理贵族访问"""
        visiting_nobles = []
        
        for noble in self.board.nobles:
            if player.can_be_visited_by_noble(noble):
                visiting_nobles.append(noble)
        
        if visiting_nobles:
            # 如果有多个贵族可以访问，玩家可以选择一个
            # 在AI对战框架中，需要请求代理做出选择
            
            # 临时策略：选择第一个贵族
            noble = visiting_nobles[0]
            self.board.remove_noble(noble.noble_id)
            player.nobles.append(noble)
    
    def _check_game_end(self):
        """检查游戏是否结束"""
        # 任何玩家达到15分或以上，触发最后一轮
        for player in self.players:
            if player.get_score() >= 15 and not self.last_round:
                self.last_round = True
                self.last_player_index = self.current_player_index
                break
        
        # 如果已经是最后一轮，并且当前玩家是触发最后一轮的玩家的前一位，则游戏结束
        if self.last_round and (self.current_player_index + 1) % self.num_players == self.last_player_index:
            self.game_over = True
            self._determine_winner()
    
    def _determine_winner(self):
        """决定游戏胜利者"""
        max_score = -1
        min_cards = float('inf')
        potential_winners = []
        
        # 首先找出得分最高的玩家
        for player in self.players:
            score = player.get_score()
            if score > max_score:
                max_score = score
                potential_winners = [player]
                min_cards = len(player.cards)
            elif score == max_score:
                potential_winners.append(player)
                min_cards = min(min_cards, len(player.cards))
        
        # 如果有多个玩家得分相同，则拥有卡牌数量最少的玩家获胜
        final_winners = []
        for player in potential_winners:
            if player.get_score() == max_score and len(player.cards) == min_cards:
                final_winners.append(player)
        
        # 如果仍有多个玩家，则他们共同获胜
        self.winner = final_winners
    
    def play_round(self):
        """玩家完成一个回合的动作"""
        if self.game_over:
            return False
        
        player = self.get_current_player()
        valid_actions = self.get_valid_actions()
        
        if not valid_actions:
            # 如果没有有效动作，跳过当前玩家
            self.next_player()
            return True
        
        # 这里需要由代理选择一个动作
        # 在实际实现中，会调用代理的选择函数
        
        # 临时策略：随机选择一个动作
        action = random.choice(valid_actions)
        success = self.execute_action(action)
        
        if success:
            self.next_player()
            return True
        
        return False
    
    def play_game(self) -> List[Player]:
        """模拟完整游戏，返回胜利者"""
        while not self.game_over:
            self.play_round()
        
        return self.winner
    
    def get_game_state(self) -> dict:
        """获取当前游戏状态，用于AI代理理解"""
        return {
            "round": self.round_number,
            "current_player": self.current_player_index,
            "players": [player.to_dict() for player in self.players],
            "board": self.board.to_dict(),
            "game_over": self.game_over,
            "last_round": self.last_round,
            "winner": [player.player_id for player in self.winner] if self.winner else None
        }
    
    def save_game_history(self, filename: str):
        """保存游戏历史记录到文件"""
        from game.serializers import GameJSONEncoder
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2, cls=GameJSONEncoder) 