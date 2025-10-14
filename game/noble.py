from dataclasses import dataclass
from typing import Dict
from game.card import GemColor


@dataclass
class Noble:
    """贵族卡类"""
    points: int  # 贵族提供的胜利点数，通常为3
    requirements: Dict[GemColor, int]  # 获取贵族所需的宝石卡牌数量
    noble_id: str  # 贵族唯一标识符
    
    def __str__(self):
        req_str = ", ".join([f"{color.value}: {count}" for color, count in self.requirements.items() if count > 0])
        return f"贵族[{self.noble_id}] - 点数:{self.points}, 要求:[{req_str}]"


def create_standard_nobles() -> list[Noble]:
    """创建标准游戏中的所有贵族"""
    nobles = [
        Noble(3, {GemColor.WHITE: 4, GemColor.RED: 4}, "N1"),
        Noble(3, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.RED: 3}, "N2"),
        Noble(3, {GemColor.BLUE: 4, GemColor.GREEN: 4}, "N3"),
        Noble(3, {GemColor.BLACK: 4, GemColor.RED: 4}, "N4"),
        Noble(3, {GemColor.WHITE: 3, GemColor.BLUE: 3, GemColor.BLACK: 3}, "N5"),
        Noble(3, {GemColor.GREEN: 4, GemColor.BLACK: 4}, "N6"),
        Noble(3, {GemColor.WHITE: 3, GemColor.GREEN: 3, GemColor.BLACK: 3}, "N7"),
        Noble(3, {GemColor.BLUE: 3, GemColor.GREEN: 3, GemColor.RED: 3}, "N8"),
        Noble(3, {GemColor.WHITE: 4, GemColor.BLUE: 4}, "N9"),
        Noble(3, {GemColor.GREEN: 3, GemColor.RED: 3, GemColor.BLACK: 3}, "N10"),
    ]
    return nobles 