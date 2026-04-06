import json
import os
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from game.game import Action
from utils.log import CustomLogger


class LLMAgent(BaseAgent):
    """原始纯 LLM 策略：直接读取完整状态与合法动作。"""

    def __init__(
        self,
        player_id: str,
        name: str,
        llm_client: Any,
        system_prompt: str = None,
        temperature: float = 0.5,
        max_tokens: int = 500,
        prompt_strategy: str = "legacy",
        run_id: Optional[str] = None,
    ):
        super().__init__(player_id, name)
        if prompt_strategy != "legacy":
            raise ValueError("LLMAgent 当前只支持 legacy 策略。")

        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_strategy = prompt_strategy
        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.game_history: List[Dict[str, Any]] = []

        log_dir = os.path.join(".", "log", "llm_agent_runs")
        os.makedirs(log_dir, exist_ok=True)
        safe_player_id = self.player_id.replace("/", "_").replace(" ", "_")
        self.log_file_path = os.path.join(
            log_dir,
            f"{self.run_id}_{safe_player_id}_{self.prompt_strategy}.log"
        )
        self.logger = CustomLogger(self.log_file_path)
        self.logger.enable_file_logging()
        self.logger.log_info(
            {
                "agent_name": self.name,
                "prompt_strategy": self.prompt_strategy,
            },
            is_print=False
        )

    def _get_default_system_prompt(self) -> str:
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
        self.logger.log_info("\n====== [DEBUG] ENTER select_action ======", is_print=False)
        self.logger.log_info(
            {
                "round": game_state.get("round"),
                "current_player": game_state.get("current_player"),
                "valid_action_count": len(valid_actions),
                "valid_actions": [str(action) for action in valid_actions],
            },
            is_print=False
        )
        prompt = self._construct_action_prompt(game_state, valid_actions)
        self.logger.log_info({"action_prompt_preview": prompt[:1200]}, is_print=False)
        start_time = time.time()
        response = self._query_llm(prompt)
        elapsed = time.time() - start_time
        self.logger.log_info(
            {
                "response_preview": response[:1200] if response else "",
                "response_length": len(response) if response else 0,
                "elapsed_seconds": round(elapsed, 3),
            },
            is_print=False
        )

        selected_action = self._parse_action_response(response, valid_actions)
        if not selected_action and valid_actions:
            import random

            selected_action = random.choice(valid_actions)
            self.logger.log_warning(
                {"fallback": "random_choice", "chosen_action": str(selected_action)},
                is_print=False
            )
        elif selected_action:
            self.logger.log_info({"chosen_action": str(selected_action)}, is_print=False)

        self.logger.log_info("====== [DEBUG] EXIT select_action ======\n", is_print=False)
        return selected_action

    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        prompt = self._construct_discard_prompt(game_state, gems, num_to_discard)
        response = self._query_llm(prompt)
        discarded_gems = self._parse_discard_response(response, gems, num_to_discard)

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
        prompt = self._construct_noble_prompt(game_state, available_nobles)
        response = self._query_llm(prompt)
        noble_id = self._parse_noble_response(response, available_nobles)
        if not noble_id and available_nobles:
            noble_id = available_nobles[0]["id"]
        return noble_id

    def _construct_action_prompt(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> str:
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        formatted_actions = [f"动作 {i+1}: {str(action)}" for i, action in enumerate(valid_actions)]
        formatted_actions_str = "\n".join(formatted_actions)
        return f"""
请分析当前的游戏状态，并从以下可用动作中选择最佳动作。

当前游戏状态:
{formatted_state}

可用动作:
{formatted_actions_str}

请选择一个动作并给出简短解释。回复格式:
选择动作: <动作编号>
解释: <你的解释>
"""

    def _construct_discard_prompt(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> str:
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        formatted_gems = json.dumps(gems, indent=2, ensure_ascii=False)
        return f"""
你需要丢弃 {num_to_discard} 个宝石代币，因为你超过了持有上限(10个)。

当前游戏状态:
{formatted_state}

你当前持有的宝石:
{formatted_gems}

请选择要丢弃的宝石。回复格式:
丢弃宝石: {{"<颜色1>": <数量1>, "<颜色2>": <数量2>, ...}}
解释: <你的解释>
"""

    def _construct_noble_prompt(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        formatted_state = json.dumps(game_state, indent=2, ensure_ascii=False)
        formatted_nobles = json.dumps(available_nobles, indent=2, ensure_ascii=False)
        return f"""
你满足了多个贵族的要求，现在可以选择一位贵族访问。

当前游戏状态:
{formatted_state}

可选贵族:
{formatted_nobles}

请选择一位贵族。回复格式:
选择贵族: <贵族ID>
解释: <你的解释>
"""

    def _query_llm(self, prompt: str) -> str:
        try:
            print(f"正在向LLM发送请求，类型: {type(self.llm_client).__name__}")
            self.logger.log_info(
                {"llm_client_type": type(self.llm_client).__name__, "prompt_length": len(prompt)},
                is_print=False
            )
            if hasattr(self.llm_client, "get_completion"):
                print("使用get_completion方法")
                response = self.llm_client.get_completion(
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if not response:
                    print("警告: LLM返回了空响应")
                return response
            if hasattr(self.llm_client, "generate"):
                print("使用generate方法")
                response = self.llm_client.generate(
                    system_prompt=self.system_prompt,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if not response:
                    print("警告: LLM返回了空响应")
                return response
            raise TypeError(f"不支持的LLM客户端类型: {type(self.llm_client)}")
        except Exception as e:
            print(f"LLM调用出错: {e}")
            import traceback

            print(traceback.format_exc())
            self.logger.log_error(
                {"error": str(e), "traceback": traceback.format_exc()},
                is_print=False
            )
            return ""

    def _parse_action_response(self, response: str, valid_actions: List[Action]) -> Optional[Action]:
        try:
            import re

            match = re.search(r"选择动作:\s*(\d+)", response)
            if match:
                action_index = int(match.group(1)) - 1
                if 0 <= action_index < len(valid_actions):
                    return valid_actions[action_index]

            match = re.search(r"^\s*(\d+)\s*$", response, re.MULTILINE)
            if match:
                action_index = int(match.group(1)) - 1
                if 0 <= action_index < len(valid_actions):
                    return valid_actions[action_index]

            for action in valid_actions:
                action_str = str(action).lower()
                if action_str in response.lower():
                    return action

            return None
        except Exception as e:
            print(f"解析动作响应出错: {e}")
            return None

    def _parse_discard_response(self, response: str, gems: Dict[str, int], num_to_discard: int) -> Optional[Dict[str, int]]:
        try:
            import json as json_lib
            import re

            match = re.search(r"丢弃宝石:\s*({.+?})", response, re.DOTALL)
            if not match:
                return None

            json_str = match.group(1).replace("'", "\"")
            discard_gems = json_lib.loads(json_str)
            if sum(discard_gems.values()) != num_to_discard:
                return None
            for color, count in discard_gems.items():
                if gems.get(color, 0) < count:
                    return None
            return discard_gems
        except Exception as e:
            print(f"解析丢弃宝石响应出错: {e}")
            return None

    def _parse_noble_response(self, response: str, available_nobles: List[Dict[str, Any]]) -> Optional[str]:
        try:
            import re

            match = re.search(r"选择贵族:\s*(\w+)", response)
            if match:
                noble_id = match.group(1)
                for noble in available_nobles:
                    if noble["id"] == noble_id:
                        return noble_id

            for noble in available_nobles:
                if noble["id"] in response:
                    return noble["id"]
            return None
        except Exception as e:
            print(f"解析选择贵族响应出错: {e}")
            return None

    def on_game_start(self, game_state: Dict[str, Any]):
        self.game_history = [{"event": "game_start", "state": game_state}]

    def on_game_end(self, game_state: Dict[str, Any], winners: List[str]):
        self.game_history.append({"event": "game_end", "state": game_state, "winners": winners})

    def on_turn_start(self, game_state: Dict[str, Any]):
        self.game_history.append({"event": "turn_start", "state": game_state})

    def on_turn_end(self, game_state: Dict[str, Any], action: Action, success: bool):
        self.game_history.append(
            {
                "event": "turn_end",
                "state": game_state,
                "action": str(action),
                "success": success,
            }
        )
