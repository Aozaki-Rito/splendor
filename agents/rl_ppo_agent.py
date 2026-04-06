import os
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from game.game import Action, Game
from rl.action_space import decode_action_id, get_action_mask
from rl.observation import encode_observation
from sb3_contrib import MaskablePPO
from utils.log import CustomLogger


class RLPPOAgent(BaseAgent):
    """加载训练后的 PPO 模型进行决策的代理。"""

    def __init__(
        self,
        player_id: str,
        name: str,
        model_path: str,
        deterministic: bool = True,
        device: str = "auto",
        run_id: Optional[str] = None,
    ):
        super().__init__(player_id, name)
        if not model_path:
            raise ValueError("RLPPOAgent 需要提供 model_path。")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RL 模型文件不存在: {model_path}")

        self.model_path = model_path
        self.deterministic = deterministic
        self.device = device
        self.run_id = run_id or f"run_{int(time.time() * 1000)}"
        self.model = MaskablePPO.load(model_path, device=device)

        log_dir = os.path.join(".", "log", "llm_agent_runs")
        os.makedirs(log_dir, exist_ok=True)
        safe_player_id = self.player_id.replace("/", "_").replace(" ", "_")
        self.log_file_path = os.path.join(log_dir, f"{self.run_id}_{safe_player_id}_rl_ppo.log")
        self.logger = CustomLogger(self.log_file_path)
        self.logger.enable_file_logging()
        self.logger.log_info(
            {
                "agent_name": self.name,
                "strategy": "rl_ppo",
                "model_path": self.model_path,
                "deterministic": self.deterministic,
                "device": self.device,
            },
            is_print=False,
        )

    def _get_game(self) -> Game:
        game = getattr(self, "_game", None)
        if game is None:
            raise RuntimeError("RLPPOAgent 缺少游戏引用，请确保在创建 Game 后将 agent._game 指向当前游戏实例。")
        return game

    def _get_player_index(self, game: Game) -> int:
        for idx, player in enumerate(game.players):
            if player.player_id == self.player_id:
                return idx
        raise RuntimeError(f"RLPPOAgent 无法在当前游戏中找到 player_id={self.player_id}")

    def select_action(self, game_state: Dict[str, Any], valid_actions: List[Action]) -> Action:
        game = self._get_game()
        player_index = self._get_player_index(game)
        obs = encode_observation(game, player_index=player_index)
        action_mask = get_action_mask(game, player_index=player_index).astype(bool)

        action_id, _ = self.model.predict(obs, action_masks=action_mask, deterministic=self.deterministic)
        action_id = int(action_id)
        action = decode_action_id(game, action_id, player_index=player_index)
        if action is None:
            self.logger.log_warning(
                {
                    "warning": "decoded_action_none",
                    "action_id": action_id,
                    "fallback": str(valid_actions[0]) if valid_actions else None,
                },
                is_print=False,
            )
            return valid_actions[0]

        self.logger.log_info(
            {
                "round": game_state.get("round"),
                "action_id": action_id,
                "action": str(action),
                "valid_action_count": len(valid_actions),
            },
            is_print=False,
        )
        return action

    def select_gems_to_discard(self, game_state: Dict[str, Any], gems: Dict[str, int], num_to_discard: int) -> Dict[str, int]:
        discarded: Dict[str, int] = {}
        remaining = dict(gems)
        colors = sorted(
            [color for color, count in remaining.items() if count > 0 and color != "gold"],
            key=lambda color: (-remaining[color], color),
        )
        while num_to_discard > 0 and colors:
            color = colors[0]
            discarded[color] = discarded.get(color, 0) + 1
            remaining[color] -= 1
            num_to_discard -= 1
            colors = sorted(
                [c for c, count in remaining.items() if count > 0 and c != "gold"],
                key=lambda c: (-remaining[c], c),
            )
        return discarded

    def select_noble(self, game_state: Dict[str, Any], available_nobles: List[Dict[str, Any]]) -> str:
        if not available_nobles:
            return ""
        return sorted(available_nobles, key=lambda noble: noble.get("id", ""))[0]["id"]
