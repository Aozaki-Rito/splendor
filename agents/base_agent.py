from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from game.game import Action, Game


class BaseAgent(ABC):
    """代理基类，定义了所有代理需要实现的接口"""
    
    def __init__(self, player_id: str, name: str):
        """初始化代理
        
        Args:
            player_id: 玩家ID
            name: 代理名称
        """
        self.player_id = player_id
        self.name = name
    
    @abstractmethod
    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> Action:
        """选择一个动作
        
        Args:
            game_state: 游戏状态
            valid_actions: 有效的动作列表
            
        Returns:
            Action: 选择的动作
        """
        pass
    
    @abstractmethod
    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        """选择要丢弃的宝石
        
        Args:
            game_state: 游戏状态
            gems: 当前拥有的宝石
            num_to_discard: 需要丢弃的宝石数量
            
        Returns:
            Dict[str, int]: 要丢弃的宝石及数量
        """
        pass
    
    @abstractmethod
    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        """选择一个贵族
        
        Args:
            game_state: 游戏状态
            available_nobles: 可选的贵族列表
            
        Returns:
            str: 选择的贵族ID
        """
        pass
    
    def on_game_start(self, game_state: Dict[str, Any]):
        """游戏开始时的回调"""
        pass
    
    def on_game_end(self, game_state: Dict[str, Any], winners: List[str]):
        """游戏结束时的回调
        
        Args:
            game_state: 最终游戏状态
            winners: 获胜者列表
        """
        pass
    
    def on_turn_start(self, game_state: Dict[str, Any]):
        """回合开始时的回调"""
        pass
    
    def on_turn_end(self, game_state: Dict[str, Any], action: Action, success: bool):
        """回合结束时的回调
        
        Args:
            game_state: 游戏状态
            action: 执行的动作
            success: 动作是否成功
        """
        pass 