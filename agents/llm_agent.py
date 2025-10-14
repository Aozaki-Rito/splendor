import json
import time
from typing import Dict, List, Optional, Any, Union

from agents.base_agent import BaseAgent
from game.game import Action, ActionType, Game


class LLMAgent(BaseAgent):
    """使用大语言模型作为决策引擎的代理"""
    
    def __init__(self, player_id: str, name: str, llm_client: Any, system_prompt: str = None, temperature: float = 0.5, max_tokens: int = 500):
        """初始化LLM代理
        
        Args:
            player_id: 玩家ID
            name: 代理名称
            llm_client: LLM客户端
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大生成令牌数
        """
        super().__init__(player_id, name)
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 默认系统提示
        if system_prompt is None:
            self.system_prompt = self._get_default_system_prompt()
        else:
            self.system_prompt = system_prompt
            
        # 保存游戏历史，用于LLM上下文
        self.game_history = []
        
    def _get_default_system_prompt(self) -> str:
        """获取默认的系统提示"""
        return """
你是一名璀璨宝石(Splendor)游戏的AI玩家。你的目标是通过策略性地收集宝石、购买卡牌和吸引贵族，尽可能快地获得15分。

游戏规则:
1. 每回合你可以执行以下操作之一:
   - 拿取3个不同颜色的宝石代币
   - 拿取2个相同颜色的宝石代币(该颜色的代币数量至少为4个)
   - 购买一张面朝上的发展卡或预留的卡
   - 预留一张发展卡并获得一个金色宝石(黄金)

2. 你最多持有10个宝石代币，超过需要丢弃
3. 当你的发展卡达到一位贵族的要求时，该贵族会立即访问你，提供额外的胜利点数
4. 游戏在一位玩家达到15分后，完成当前回合结束

策略提示:
- 注意平衡短期与长期利益
- 考虑其他玩家可能的行动
- 关注贵族卡的要求
- 预留对你重要或对对手有价值的卡牌
- 留意游戏板上的卡牌分布

你需要基于游戏状态，从可用动作中选择最佳动作。你的回应应该包含你选择的动作及简短的解释。
"""
    
    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> Action:
        """使用LLM选择一个动作"""
        prompt = self._construct_action_prompt(game_state, valid_actions)
        response = self._query_llm(prompt)
        
        # 解析LLM响应
        selected_action = self._parse_action_response(response, valid_actions)
        
        # 如果无法解析，则随机选择一个动作
        if not selected_action and valid_actions:
            import random
            selected_action = random.choice(valid_actions)
        
        return selected_action
    
    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        """使用LLM选择要丢弃的宝石"""
        prompt = self._construct_discard_prompt(game_state, gems, num_to_discard)
        response = self._query_llm(prompt)
        
        # 解析LLM响应
        discarded_gems = self._parse_discard_response(response, gems, num_to_discard)
        
        # 如果无法解析，则随机丢弃
        if not discarded_gems:
            import random
            discarded_gems = {}
            colors = [color for color, count in gems.items() if count > 0]
            for _ in range(num_to_discard):
                if not colors:
                    break
                color = random.choice(colors)
                discarded_gems[color] = discarded_gems.get(color, 0) + 1
                gems[color] -= 1
                if gems[color] <= 0:
                    colors.remove(color)
        
        return discarded_gems
    
    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        """使用LLM选择一个贵族"""
        prompt = self._construct_noble_prompt(game_state, available_nobles)
        response = self._query_llm(prompt)
        
        # 解析LLM响应
        noble_id = self._parse_noble_response(response, available_nobles)
        
        # 如果无法解析，则选择第一个贵族
        if not noble_id and available_nobles:
            noble_id = available_nobles[0]["id"]
        
        return noble_id
    
    def _construct_action_prompt(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> str:
        """构建选择动作的提示"""
        # 转换游戏状态为易于理解的格式
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        
        # 格式化有效动作
        formatted_actions = []
        for i, action in enumerate(valid_actions):
            formatted_actions.append(f"动作 {i+1}: {str(action)}")
        
        formatted_actions_str = "\n".join(formatted_actions)
        
        # 构建提示
        prompt = f"""
请分析当前的游戏状态，并从以下可用动作中选择最佳动作。

当前游戏状态:
{formatted_state}

可用动作:
{formatted_actions_str}

请选择一个动作并给出简短解释。回复格式:
选择动作: <动作编号>
解释: <你的解释>
"""
        return prompt
    
    def _construct_discard_prompt(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> str:
        """构建丢弃宝石的提示"""
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        formatted_gems = json.dumps(gems, indent=2, ensure_ascii=False)
        
        prompt = f"""
你需要丢弃 {num_to_discard} 个宝石代币，因为你超过了持有上限(10个)。

当前游戏状态:
{formatted_state}

你当前持有的宝石:
{formatted_gems}

请选择要丢弃的宝石。回复格式:
丢弃宝石: {{"<颜色1>": <数量1>, "<颜色2>": <数量2>, ...}}
解释: <你的解释>
"""
        return prompt
    
    def _construct_noble_prompt(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        """构建选择贵族的提示"""
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        formatted_nobles = json.dumps(available_nobles, indent=2, ensure_ascii=False)
        
        prompt = f"""
你满足了多个贵族的要求，现在可以选择一位贵族访问。

当前游戏状态:
{formatted_state}

可选贵族:
{formatted_nobles}

请选择一位贵族。回复格式:
选择贵族: <贵族ID>
解释: <你的解释>
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """调用LLM获取响应"""
        try:
            print(f"正在向LLM发送请求，类型: {type(self.llm_client).__name__}")
            
            # 对于OpenAI客户端
            if hasattr(self.llm_client, "get_completion"):
                print(f"使用get_completion方法")
                response = self.llm_client.get_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                if not response:
                    print("警告: LLM返回了空响应")
                return response
            
            # 示例: 通用完成API
            elif hasattr(self.llm_client, "generate"):
                print(f"使用generate方法")
                response = self.llm_client.generate(
                    system_prompt=self.system_prompt,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                if not response:
                    print("警告: LLM返回了空响应")
                return response
            
            # 无法识别的客户端类型
            else:
                print(f"错误: 未知的LLM客户端类型 {type(self.llm_client)}")
                print(f"可用的方法: {dir(self.llm_client)}")
                raise TypeError(f"不支持的LLM客户端类型: {type(self.llm_client)}")
                
        except Exception as e:
            print(f"LLM调用出错: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
    
    def _parse_action_response(self, response: str, valid_actions: List[Action]) -> Optional[Action]:
        """解析LLM的动作选择响应"""
        try:
            # 查找动作编号
            import re
            
            # 尝试匹配"选择动作: <数字>"格式
            match = re.search(r"选择动作:\s*(\d+)", response)
            if match:
                action_index = int(match.group(1)) - 1
                if 0 <= action_index < len(valid_actions):
                    return valid_actions[action_index]
            
            # 尝试匹配单独的数字
            match = re.search(r"^\s*(\d+)\s*$", response, re.MULTILINE)
            if match:
                action_index = int(match.group(1)) - 1
                if 0 <= action_index < len(valid_actions):
                    return valid_actions[action_index]
            
            # 尝试基于动作描述匹配
            for action in valid_actions:
                action_str = str(action).lower()
                if action_str in response.lower():
                    return action
            
            # 基于动作类型和参数匹配
            for action in valid_actions:
                action_type = action.action_type.value
                if action_type in response.lower():
                    # 进一步匹配参数
                    if action.action_type == ActionType.TAKE_DIFFERENT_GEMS:
                        colors = [color.value for color in action.params.get("colors", [])]
                        if all(color.lower() in response.lower() for color in colors):
                            return action
                    
                    elif action.action_type == ActionType.TAKE_SAME_GEMS:
                        color = action.params.get("color").value
                        if color.lower() in response.lower():
                            return action
                    
                    elif action.action_type in [ActionType.RESERVE_CARD, ActionType.BUY_CARD, ActionType.BUY_RESERVED_CARD]:
                        card_id = action.params.get("card_id", "")
                        if card_id in response:
                            return action
            
            return None
        except Exception as e:
            print(f"解析动作响应出错: {e}")
            return None
    
    def _parse_discard_response(self, response: str, gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        """解析LLM的丢弃宝石响应"""
        try:
            # 提取JSON格式的丢弃宝石信息
            import re
            import json
            
            # 尝试匹配JSON对象
            match = re.search(r"丢弃宝石:\s*({.+?})", response, re.DOTALL)
            if match:
                json_str = match.group(1)
                # 清理可能的非法字符
                json_str = json_str.replace("'", "\"")
                discard_gems = json.loads(json_str)
                
                # 验证丢弃数量
                total_discarded = sum(discard_gems.values())
                if total_discarded != num_to_discard:
                    return None
                
                # 验证玩家是否有足够的宝石可丢弃
                for color, count in discard_gems.items():
                    if gems.get(color, 0) < count:
                        return None
                
                return discard_gems
            
            return None
        except Exception as e:
            print(f"解析丢弃宝石响应出错: {e}")
            return None
    
    def _parse_noble_response(self, response: str, available_nobles: List[Dict[str, Any]]) -> Optional[str]:
        """解析LLM的选择贵族响应"""
        try:
            # 查找贵族ID
            import re
            
            # 尝试匹配"选择贵族: <ID>"格式
            match = re.search(r"选择贵族:\s*(\w+)", response)
            if match:
                noble_id = match.group(1)
                # 验证贵族ID是否有效
                for noble in available_nobles:
                    if noble["id"] == noble_id:
                        return noble_id
            
            # 尝试直接在响应中找到贵族ID
            for noble in available_nobles:
                if noble["id"] in response:
                    return noble["id"]
            
            return None
        except Exception as e:
            print(f"解析选择贵族响应出错: {e}")
            return None
    
    def on_game_start(self, game_state: Dict[str, Any]):
        """游戏开始时的回调"""
        self.game_history = []
        self.game_history.append({
            "event": "game_start",
            "state": game_state
        })
    
    def on_game_end(self, game_state: Dict[str, Any], winners: List[str]):
        """游戏结束时的回调"""
        self.game_history.append({
            "event": "game_end",
            "state": game_state,
            "winners": winners
        })
    
    def on_turn_start(self, game_state: Dict[str, Any]):
        """回合开始时的回调"""
        self.game_history.append({
            "event": "turn_start",
            "state": game_state
        })
    
    def on_turn_end(self, game_state: Dict[str, Any], action: Action, success: bool):
        """回合结束时的回调"""
        self.game_history.append({
            "event": "turn_end",
            "state": game_state,
            "action": str(action),
            "success": success
        }) 