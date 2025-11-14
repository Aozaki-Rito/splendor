from agents.base_agent import BaseAgent
import random
from typing import Dict, List, Any

class RandomAgent(BaseAgent):
    """随机代理，随机选择动作"""
    
    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Any]) -> Any:
        """随机选择一个动作"""
        return random.choice(valid_actions) if valid_actions else None
    
    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        """随机选择要丢弃的宝石"""
        result = {}
        colors = [color for color, count in gems.items() if count > 0]
        
        for _ in range(num_to_discard):
            if not colors:
                break
            color = random.choice(colors)
            result[color] = result.get(color, 0) + 1
            gems[color] -= 1
            if gems[color] <= 0:
                colors.remove(color)
        
        return result
    
    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        """随机选择一个贵族"""
        return random.choice(available_nobles)["id"] if available_nobles else None

