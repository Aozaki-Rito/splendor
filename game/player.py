from dataclasses import dataclass, field
from typing import Dict, List, Optional
from game.card import GemColor, Card
from game.noble import Noble


@dataclass
class Player:
    """玩家类，表示游戏中的一位玩家"""
    player_id: str  # 玩家唯一标识符
    name: str  # 玩家名称
    
    # 玩家的宝石代币
    gems: Dict[GemColor, int] = field(default_factory=lambda: {
        GemColor.WHITE: 0,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
        GemColor.GOLD: 0,
    })
    
    # 玩家拥有的发展卡
    cards: List[Card] = field(default_factory=list)
    
    # 玩家预留的卡牌
    reserved_cards: List[Card] = field(default_factory=list)
    
    # 玩家拥有的贵族卡
    nobles: List[Noble] = field(default_factory=list)
    
    def get_gem_count(self, color: GemColor) -> int:
        """获取玩家持有的特定颜色宝石数量"""
        return self.gems.get(color, 0)
    
    def get_total_gems(self) -> int:
        """获取玩家持有的宝石总数"""
        return sum(self.gems.values())
    
    def get_card_discount(self, color: GemColor) -> int:
        """获取特定颜色的卡牌折扣（玩家拥有的该颜色卡牌数量）"""
        return sum(1 for card in self.cards if card.gem_color == color)
    
    def get_card_discounts(self) -> Dict[GemColor, int]:
        """获取所有颜色的卡牌折扣"""
        discounts = {color: 0 for color in GemColor if color != GemColor.GOLD}
        for card in self.cards:
            discounts[card.gem_color] = discounts.get(card.gem_color, 0) + 1
        return discounts
    
    def can_afford_card(self, card: Card) -> bool:
        """判断玩家是否能够购买特定卡牌"""
        discounts = self.get_card_discounts()
        gold_needed = 0
        
        for color, cost in card.cost.items():
            available = self.gems.get(color, 0) + discounts.get(color, 0)
            if available < cost:
                gold_needed += cost - available
        
        return gold_needed <= self.gems.get(GemColor.GOLD, 0)
    
    def get_actual_cost(self, card: Card) -> Dict[GemColor, int]:
        """计算购买卡牌的实际成本（考虑折扣）"""
        discounts = self.get_card_discounts()
        actual_cost = {}
        
        for color, cost in card.cost.items():
            discount = discounts.get(color, 0)
            remaining_cost = max(0, cost - discount)
            if remaining_cost > 0:
                actual_cost[color] = min(remaining_cost, self.gems.get(color, 0))
                remaining_cost -= actual_cost[color]
                if remaining_cost > 0 and self.gems.get(GemColor.GOLD, 0) > 0:
                    gold_needed = min(remaining_cost, self.gems.get(GemColor.GOLD, 0))
                    if gold_needed > 0:
                        actual_cost[GemColor.GOLD] = actual_cost.get(GemColor.GOLD, 0) + gold_needed
        
        return actual_cost
    
    def get_score(self) -> int:
        """计算玩家的当前分数"""
        card_points = sum(card.points for card in self.cards)
        noble_points = sum(noble.points for noble in self.nobles)
        return card_points + noble_points
    
    def can_be_visited_by_noble(self, noble: Noble) -> bool:
        """判断玩家是否满足贵族的访问条件"""
        discounts = self.get_card_discounts()
        for color, requirement in noble.requirements.items():
            if discounts.get(color, 0) < requirement:
                return False
        return True
    
    def __str__(self):
        gems_str = ", ".join([f"{color.value}: {count}" for color, count in self.gems.items() if count > 0])
        cards_count = len(self.cards)
        reserved_count = len(self.reserved_cards)
        nobles_count = len(self.nobles)
        score = self.get_score()
        
        return (f"玩家[{self.name}] - 分数:{score}, 宝石:[{gems_str}], "
                f"卡牌:{cards_count}, 预留卡:{reserved_count}, 贵族:{nobles_count}")
                
    def to_dict(self) -> dict:
        """将玩家状态转换为字典，用于AI代理理解"""
        return {
            "player_id": self.player_id,
            "name": self.name,
            "score": self.get_score(),
            "gems": {color.value: count for color, count in self.gems.items()},
            "cards": [
                {
                    "id": card.card_id,
                    "level": card.level,
                    "points": card.points,
                    "color": card.gem_color.value,
                    "cost": {color.value: count for color, count in card.cost.items()}
                } 
                for card in self.cards
            ],
            "reserved_cards": [
                {
                    "id": card.card_id,
                    "level": card.level,
                    "points": card.points,
                    "color": card.gem_color.value,
                    "cost": {color.value: count for color, count in card.cost.items()}
                } 
                for card in self.reserved_cards
            ],
            "nobles": [
                {
                    "id": noble.noble_id,
                    "points": noble.points,
                    "requirements": {color.value: count for color, count in noble.requirements.items()}
                }
                for noble in self.nobles
            ],
            "card_discounts": {color.value: count for color, count in self.get_card_discounts().items()}
        } 