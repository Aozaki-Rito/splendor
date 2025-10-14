import random
from typing import Dict, List, Optional, Tuple
from game.card import Card, GemColor, create_standard_cards
from game.noble import Noble, create_standard_nobles


class Board:
    """游戏板类，表示游戏的当前状态"""
    
    def __init__(self, num_players: int = 2):
        """初始化游戏板
        
        Args:
            num_players: 玩家数量，默认为2
        """
        # 初始化宝石代币
        self.gems = self._initialize_gems(num_players)
        
        # 初始化卡牌
        all_cards = create_standard_cards()
        
        # 按等级分类卡牌
        self.card_decks = {
            1: [card for card in all_cards if card.level == 1],
            2: [card for card in all_cards if card.level == 2],
            3: [card for card in all_cards if card.level == 3]
        }
        
        # 洗牌
        for deck in self.card_decks.values():
            random.shuffle(deck)
        
        # 初始化展示的卡牌
        self.displayed_cards = {
            1: [],
            2: [],
            3: []
        }
        
        # 每个等级展示4张卡
        for level in [1, 2, 3]:
            for _ in range(4):
                if self.card_decks[level]:
                    self.displayed_cards[level].append(self.card_decks[level].pop())
        
        # 初始化贵族
        all_nobles = create_standard_nobles()
        random.shuffle(all_nobles)
        self.nobles = all_nobles[:num_players + 1]  # 贵族数量 = 玩家数量 + 1
    
    def _initialize_gems(self, num_players: int) -> Dict[GemColor, int]:
        """根据玩家数量初始化宝石代币"""
        gems = {}
        
        # 2人游戏：每种颜色4个代币
        # 3人游戏：每种颜色5个代币
        # 4人游戏：每种颜色7个代币
        base_gems = {
            2: 4,
            3: 5,
            4: 7
        }
        
        for color in [GemColor.WHITE, GemColor.BLUE, GemColor.GREEN, GemColor.RED, GemColor.BLACK]:
            gems[color] = base_gems.get(num_players, 4)
        
        # 黄金宝石数量固定为5个
        gems[GemColor.GOLD] = 5
        
        return gems
    
    def draw_card(self, level: int) -> Optional[Card]:
        """从指定等级的牌堆中抽取一张卡牌"""
        if not self.card_decks[level]:
            return None
        return self.card_decks[level].pop()
    
    def replenish_displayed_cards(self):
        """补充展示区的卡牌"""
        for level in [1, 2, 3]:
            while len(self.displayed_cards[level]) < 4 and self.card_decks[level]:
                self.displayed_cards[level].append(self.draw_card(level))
    
    def remove_displayed_card(self, level: int, card_id: str) -> Optional[Card]:
        """从展示区移除指定卡牌"""
        for i, card in enumerate(self.displayed_cards[level]):
            if card.card_id == card_id:
                card = self.displayed_cards[level].pop(i)
                self.replenish_displayed_cards()
                return card
        return None
    
    def take_gems(self, gems_to_take: Dict[GemColor, int]) -> bool:
        """尝试从游戏板拿取宝石代币
        
        Args:
            gems_to_take: 要拿取的宝石颜色和数量
            
        Returns:
            bool: 是否成功拿取
        """
        # 检查是否有足够的宝石
        for color, count in gems_to_take.items():
            if self.gems.get(color, 0) < count:
                return False
        
        # 拿取宝石
        for color, count in gems_to_take.items():
            self.gems[color] -= count
        
        return True
    
    def return_gems(self, gems_to_return: Dict[GemColor, int]):
        """将宝石代币归还到游戏板
        
        Args:
            gems_to_return: 要归还的宝石颜色和数量
        """
        for color, count in gems_to_return.items():
            self.gems[color] = self.gems.get(color, 0) + count
    
    def get_card_by_id(self, card_id: str) -> Optional[Tuple[int, Card]]:
        """根据卡牌ID查找卡牌
        
        Returns:
            Tuple[int, Card]: (卡牌等级, 卡牌对象)
        """
        for level, cards in self.displayed_cards.items():
            for card in cards:
                if card.card_id == card_id:
                    return level, card
        return None
    
    def get_noble_by_id(self, noble_id: str) -> Optional[Noble]:
        """根据贵族ID查找贵族"""
        for noble in self.nobles:
            if noble.noble_id == noble_id:
                return noble
        return None
    
    def remove_noble(self, noble_id: str) -> Optional[Noble]:
        """从游戏板移除指定贵族"""
        for i, noble in enumerate(self.nobles):
            if noble.noble_id == noble_id:
                return self.nobles.pop(i)
        return None
    
    def to_dict(self) -> dict:
        """将游戏板状态转换为字典，用于AI代理理解"""
        return {
            "gems": {color.value: count for color, count in self.gems.items()},
            "displayed_cards": {
                level: [
                    {
                        "id": card.card_id,
                        "level": card.level,
                        "points": card.points,
                        "color": card.gem_color.value,
                        "cost": {color.value: count for color, count in card.cost.items()}
                    } 
                    for card in cards
                ]
                for level, cards in self.displayed_cards.items()
            },
            "deck_counts": {level: len(deck) for level, deck in self.card_decks.items()},
            "nobles": [
                {
                    "id": noble.noble_id,
                    "points": noble.points,
                    "requirements": {color.value: count for color, count in noble.requirements.items()}
                }
                for noble in self.nobles
            ]
        } 