from enum import Enum
from dataclasses import dataclass
from typing import Dict


class GemColor(Enum):
    """宝石颜色枚举"""
    WHITE = "white"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    BLACK = "black"
    GOLD = "gold"  # 黄金宝石作为通配符


@dataclass
class Card:
    """发展卡类"""
    level: int  # 卡牌等级：1, 2, 3
    points: int  # 胜利点数
    gem_color: GemColor  # 卡牌提供的宝石颜色
    cost: Dict[GemColor, int]  # 购买卡牌所需的宝石成本
    card_id: str  # 卡牌唯一标识符

    def __str__(self):
        cost_str = ", ".join([f"{color.value}: {count}" for color, count in self.cost.items() if count > 0])
        return (f"卡牌[{self.card_id}] - 等级:{self.level}, 点数:{self.points}, "
                f"颜色:{self.gem_color.value}, 成本:[{cost_str}]")


# 预定义的卡牌数据
def create_standard_cards() -> list[Card]:
    """创建标准游戏中的所有发展卡"""
    cards = []
    
    # 一级卡牌
    level1_cards = [
        # 白色卡牌
        Card(1, 0, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.RED: 1, GemColor.BLACK: 1}, "L1_W1"),
        Card(1, 0, GemColor.WHITE, {GemColor.BLUE: 1, GemColor.GREEN: 2, GemColor.RED: 1, GemColor.BLACK: 1}, "L1_W2"),
        Card(1, 0, GemColor.WHITE, {GemColor.BLUE: 2, GemColor.GREEN: 2, GemColor.BLACK: 1}, "L1_W3"),
        Card(1, 0, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.BLUE: 1, GemColor.BLACK: 1}, "L1_W4"),
        Card(1, 0, GemColor.WHITE, {GemColor.BLUE: 2, GemColor.BLACK: 2}, "L1_W5"),
        Card(1, 0, GemColor.WHITE, {GemColor.RED: 2, GemColor.BLACK: 1}, "L1_W6"),
        Card(1, 1, GemColor.WHITE, {GemColor.GREEN: 4}, "L1_W7"),
        Card(1, 0, GemColor.WHITE, {GemColor.BLUE: 3}, "L1_W8"),
        
        # 蓝色卡牌
        Card(1, 0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.GREEN: 1, GemColor.RED: 1, GemColor.BLACK: 1}, "L1_B1"),
        Card(1, 0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.GREEN: 1, GemColor.RED: 2, GemColor.BLACK: 1}, "L1_B2"),
        Card(1, 0, GemColor.BLUE, {GemColor.BLUE: 1, GemColor.GREEN: 3, GemColor.RED: 1}, "L1_B3"),
        Card(1, 0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.RED: 2, GemColor.BLACK: 2}, "L1_B4"),
        Card(1, 0, GemColor.BLUE, {GemColor.BLACK: 3}, "L1_B5"),
        Card(1, 0, GemColor.BLUE, {GemColor.GREEN: 2, GemColor.BLACK: 2}, "L1_B6"),
        Card(1, 0, GemColor.BLUE, {GemColor.WHITE: 1, GemColor.BLACK: 2}, "L1_B7"),
        Card(1, 1, GemColor.BLUE, {GemColor.RED: 4}, "L1_B8"),
        
        # 绿色卡牌
        Card(1, 0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 1, GemColor.BLACK: 2}, "L1_G1"),
        Card(1, 0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 3, GemColor.GREEN: 1}, "L1_G2"),
        Card(1, 0, GemColor.GREEN, {GemColor.BLUE: 1, GemColor.RED: 2, GemColor.BLACK: 2}, "L1_G3"),
        Card(1, 0, GemColor.GREEN, {GemColor.BLUE: 2, GemColor.RED: 2}, "L1_G4"),
        Card(1, 0, GemColor.GREEN, {GemColor.WHITE: 2, GemColor.BLUE: 1}, "L1_G5"),
        Card(1, 0, GemColor.GREEN, {GemColor.RED: 3}, "L1_G6"),
        Card(1, 1, GemColor.GREEN, {GemColor.BLACK: 4}, "L1_G7"),
        Card(1, 0, GemColor.GREEN, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.RED: 1, GemColor.BLACK: 1}, "L1_G8"),
        
        # 红色卡牌
        Card(1, 0, GemColor.RED, {GemColor.WHITE: 2, GemColor.RED: 1}, "L1_R1"),
        Card(1, 0, GemColor.RED, {GemColor.BLUE: 2, GemColor.GREEN: 1}, "L1_R2"),
        Card(1, 1, GemColor.RED, {GemColor.WHITE: 4}, "L1_R3"),
        Card(1, 0, GemColor.RED, {GemColor.WHITE: 3}, "L1_R4"),
        Card(1, 0, GemColor.RED, {GemColor.WHITE: 2, GemColor.GREEN: 1, GemColor.BLACK: 2}, "L1_R5"),
        Card(1, 0, GemColor.RED, {GemColor.WHITE: 2, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 1}, "L1_R6"),
        Card(1, 0, GemColor.RED, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.BLACK: 1}, "L1_R7"),
        Card(1, 0, GemColor.RED, {GemColor.WHITE: 1, GemColor.RED: 1, GemColor.BLACK: 3}, "L1_R8"),
        
        # 黑色卡牌
        Card(1, 0, GemColor.BLACK, {GemColor.GREEN: 1, GemColor.RED: 3, GemColor.BLACK: 1}, "L1_K1"),
        Card(1, 0, GemColor.BLACK, {GemColor.WHITE: 2, GemColor.BLUE: 2, GemColor.RED: 1}, "L1_K2"),
        Card(1, 0, GemColor.BLACK, {GemColor.WHITE: 2, GemColor.GREEN: 2}, "L1_K3"),
        Card(1, 0, GemColor.BLACK, {GemColor.GREEN: 2, GemColor.RED: 1}, "L1_K4"),
        Card(1, 0, GemColor.BLACK, {GemColor.GREEN: 4}, "L1_K5"),
        Card(1, 1, GemColor.BLACK, {GemColor.BLUE: 4}, "L1_K6"),
        Card(1, 0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 2, GemColor.GREEN: 1, GemColor.RED: 1}, "L1_K7"),
        Card(1, 0, GemColor.BLACK, {GemColor.WHITE: 1, GemColor.BLUE: 1, GemColor.GREEN: 1, GemColor.RED: 1}, "L1_K8"),
    ]
    cards.extend(level1_cards)
    
    # 二级卡牌
    level2_cards = [
        # 白色卡牌
        Card(2, 1, GemColor.WHITE, {GemColor.GREEN: 3, GemColor.RED: 2, GemColor.BLACK: 2}, "L2_W1"),
        Card(2, 1, GemColor.WHITE, {GemColor.WHITE: 2, GemColor.BLUE: 3, GemColor.RED: 3}, "L2_W2"),
        Card(2, 2, GemColor.WHITE, {GemColor.GREEN: 1, GemColor.RED: 4, GemColor.BLACK: 2}, "L2_W3"),
        Card(2, 2, GemColor.WHITE, {GemColor.RED: 5, GemColor.BLACK: 3},                     "L2_W4"),
        Card(2, 2, GemColor.WHITE, {GemColor.BLUE: 5},                                         "L2_W5"),
        Card(2, 3, GemColor.WHITE, {GemColor.WHITE: 6},                                         "L2_W6"),

        # 蓝色卡牌
        Card(2, 2, GemColor.BLUE, {GemColor.WHITE: 2, GemColor.RED: 1, GemColor.BLACK: 4}, "L2_B1"),
        Card(2, 3, GemColor.BLUE, {GemColor.BLUE: 6},                                    "L2_B2"),
        Card(2, 2, GemColor.BLUE, {GemColor.BLUE: 5},                                    "L2_B3"),
        Card(2, 1, GemColor.BLUE, {GemColor.BLUE: 2, GemColor.GREEN: 2, GemColor.RED: 3}, "L2_B4"),
        Card(2, 1, GemColor.BLUE, {GemColor.BLUE: 2, GemColor.GREEN: 3, GemColor.BLACK: 3}, "L2_B5"),
        Card(2, 2, GemColor.BLUE, {GemColor.WHITE: 5, GemColor.BLUE: 3}, "L2_B6"),

        # 绿色卡牌
        Card(2, 2, GemColor.GREEN, {GemColor.GREEN: 5}, "L2_G1"),
        Card(2, 2, GemColor.GREEN, {GemColor.BLUE: 5, GemColor.GREEN: 3}, "L2_G2"),
        Card(2, 3, GemColor.GREEN, {GemColor.GREEN: 6}, "L2_G3"),
        Card(2, 1, GemColor.GREEN, {GemColor.WHITE: 2, GemColor.BLUE: 3, GemColor.BLACK: 2}, "L2_G4"),
        Card(2, 1, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.GREEN: 2, GemColor.RED: 3}, "L2_G5"),
        Card(2, 2, GemColor.GREEN, {GemColor.WHITE: 4, GemColor.BLUE: 2, GemColor.BLACK: 1}, "L2_G6"),
        
        # 红色卡牌
        Card(2, 1, GemColor.RED, {GemColor.WHITE: 2, GemColor.RED: 2, GemColor.BLACK: 3}, "L2_R1"),
        Card(2, 1, GemColor.RED, {GemColor.BLUE: 3, GemColor.RED: 2, GemColor.BLACK: 3}, "L2_R2"),
        Card(2, 2, GemColor.RED, {GemColor.WHITE: 1, GemColor.BLUE: 4, GemColor.GREEN: 2}, "L2_R3"),
        Card(2, 2, GemColor.RED, {GemColor.WHITE: 3, GemColor.BLACK: 5}, "L2_R4"),
        Card(2, 2, GemColor.RED, {GemColor.BLACK: 5}, "L2_R5"),
        Card(2, 3, GemColor.RED, {GemColor.RED: 6}, "L2_R6"),
        
        # 黑色卡牌
        Card(2, 1, GemColor.BLACK, {GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.BLACK: 2}, "L2_K1"),
        Card(2, 2, GemColor.BLACK, {GemColor.BLUE: 1, GemColor.GREEN: 4, GemColor.RED: 2}, "L2_K2"),
        Card(2, 1, GemColor.BLACK, {GemColor.WHITE: 3, GemColor.BLUE: 2, GemColor.GREEN: 2}, "L2_K3"),
        Card(2, 2, GemColor.BLACK, {GemColor.GREEN: 5, GemColor.RED: 3}, "L2_K4"),
        Card(2, 2, GemColor.BLACK, {GemColor.WHITE: 5}, "L2_K5"),
        Card(2, 3, GemColor.BLACK, {GemColor.BLACK: 5}, "L2_K6"),
    ]
    cards.extend(level2_cards)
    
    # 三级卡牌
    level3_cards = [
        # 白色卡牌
        Card(3, 5, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.BLACK: 7}, "L3_W1"),
        Card(3, 4, GemColor.WHITE, {GemColor.BLACK: 7}, "L3_W2"),
        Card(3, 4, GemColor.WHITE, {GemColor.WHITE: 3, GemColor.RED: 3, GemColor.BLACK: 6}, "L3_W3"),
        Card(3, 3, GemColor.WHITE, {GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.RED: 5, GemColor.BLACK: 3}, "L3_W4"),
        
        # 蓝色卡牌
        Card(3, 5, GemColor.BLUE, {GemColor.WHITE: 7, GemColor.BLUE: 3}, "L3_B1"),
        Card(3, 4, GemColor.BLUE, {GemColor.WHITE: 6, GemColor.BLUE: 3, GemColor.BLACK: 3}, "L3_B2"),
        Card(3, 3, GemColor.BLUE, {GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.RED: 3, GemColor.BLACK: 5}, "L3_B3"),
        Card(3, 4, GemColor.BLUE, {GemColor.WHITE: 7}, "L3_B4"),
        
        # 绿色卡牌
        Card(3, 4, GemColor.GREEN, {GemColor.WHITE: 3, GemColor.BLUE: 6,  GemColor.GREEN: 3}, "L3_G1"),
        Card(3, 4, GemColor.GREEN, {GemColor.BLUE: 7}, "L3_G2"),
        Card(3, 5, GemColor.GREEN, {GemColor.BLUE: 7, GemColor.GREEN: 5}, "L3_G3"),
        Card(3, 3, GemColor.GREEN, {GemColor.WHITE: 5, GemColor.BLUE: 3, GemColor.RED: 3, GemColor.BLACK: 3}, "L3_G4"),
        
        # 红色卡牌
        Card(3, 4, GemColor.RED, {GemColor.GREEN: 7}, "L3_R1"),
        Card(3, 3, GemColor.RED, {GemColor.WHITE: 3, GemColor.BLUE: 5, GemColor.GREEN: 3, GemColor.BLACK: 3}, "L3_R2"),
        Card(3, 5, GemColor.RED, {GemColor.GREEN: 7, GemColor.RED: 3}, "L3_R3"),
        Card(3, 4, GemColor.RED, {GemColor.BLUE: 3, GemColor.GREEN: 6, GemColor.RED: 3}, "L3_R4"),
        
        # 黑色卡牌
        Card(3, 4, GemColor.BLACK, {GemColor.RED: 7}, "L3_K1"),
        Card(3, 4, GemColor.BLACK, {GemColor.GREEN: 3, GemColor.RED: 6, GemColor.BLACK: 3}, "L3_K2"),
        Card(3, 5, GemColor.BLACK, {GemColor.RED: 7, GemColor.BLACK: 3}, "L3_K3"),
        Card(3, 3, GemColor.BLACK, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.GREEN: 5, GemColor.RED: 3}, "L3_K4"),
    ]
    cards.extend(level3_cards)
    
    return cards 