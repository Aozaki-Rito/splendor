from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from game.game import Game
from game.player import Player
from rl.action_space import ACTION_DIM, decode_action_id, get_action_mask
from rl.observation import OBSERVATION_DIM, encode_observation


class SplendorEnv(gym.Env):
    """单智能体璀璨宝石训练环境。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_type: str = "random",
        num_players: int = 2,
        controlled_player_index: int = 0,
        max_episode_steps: int = 200,
        invalid_action_penalty: float = -1.0,
        win_reward: float = 10.0,
        loss_reward: float = -10.0,
        score_delta_scale: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if num_players != 2:
            raise ValueError("V1 训练环境当前只支持 2 人对局。")
        if controlled_player_index != 0:
            raise ValueError("V1 训练环境当前只支持控制 0 号玩家。")

        self.opponent_type = opponent_type
        self.num_players = num_players
        self.controlled_player_index = controlled_player_index
        self.max_episode_steps = max_episode_steps
        self.invalid_action_penalty = invalid_action_penalty
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.score_delta_scale = score_delta_scale
        self.base_seed = seed

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)

        self.game: Optional[Game] = None
        self.step_count = 0
        self.episode_seed = seed
        self.opponent_agent = None

    def _create_opponent_agent(self):
        if self.opponent_type == "random":
            return RandomAgent(player_id="opponent_1", name="随机代理 1")
        if self.opponent_type == "rule_based":
            return RuleBasedAgent(
                player_id="opponent_1",
                name="规则代理 1",
                candidate_action_limit=6,
                target_limit=4,
                noble_limit=3,
            )
        raise ValueError(f"不支持的对手类型: {self.opponent_type}")

    def _build_game(self, seed: Optional[int]) -> Game:
        players = [
            Player("rl_agent_1", "RL Agent"),
            Player("opponent_1", "Opponent"),
        ]
        game = Game(players, seed=seed)
        self.opponent_agent = self._create_opponent_agent()
        self.opponent_agent._player = players[1]
        return game

    def _get_obs(self) -> np.ndarray:
        return encode_observation(self.game, player_index=self.controlled_player_index)

    def action_masks(self) -> np.ndarray:
        return get_action_mask(self.game, player_index=self.controlled_player_index).astype(bool)

    def _winner_ids(self):
        if not self.game.winner:
            return []
        return [player.player_id for player in self.game.winner]

    def _advance_to_controlled_decision(self):
        """
        将环境推进到“受控玩家且存在合法动作”的状态。

        这和主游戏逻辑保持一致：
        - 如果当前玩家没有合法动作，直接跳过
        - 如果当前是对手且有合法动作，则由对手行动
        - 返回时要么游戏结束，要么轮到受控玩家且其有合法动作
        """
        while not self.game.game_over:
            valid_actions = self.game.get_valid_actions()

            if not valid_actions:
                self.game.next_player()
                continue

            if self.game.current_player_index == self.controlled_player_index:
                return

            opponent_action = self.opponent_agent.select_action(self.game.get_game_state(), valid_actions)
            self.game.execute_action(opponent_action)
            if not self.game.game_over:
                self.game.next_player()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is None:
            seed = self.base_seed
        self.episode_seed = seed
        self.step_count = 0
        self.game = self._build_game(seed=seed)
        self._advance_to_controlled_decision()
        obs = self._get_obs()
        info = {"action_mask": self.action_masks()}
        return obs, info

    def step(self, action: int):
        if self.game is None:
            raise RuntimeError("环境尚未 reset。")
        if self.game.game_over:
            raise RuntimeError("游戏已经结束，请先 reset。")
        if self.game.current_player_index != self.controlled_player_index:
            raise RuntimeError("当前不是受控玩家回合。")

        if not self.game.get_valid_actions():
            raise RuntimeError("环境暴露了一个受控玩家无合法动作的状态，这说明推进逻辑存在问题。")

        self.step_count += 1
        controlled_player = self.game.players[self.controlled_player_index]
        opponent_player = self.game.players[1]
        prev_self_score = controlled_player.get_score()
        prev_opp_score = opponent_player.get_score()

        decoded_action = decode_action_id(self.game, int(action), player_index=self.controlled_player_index)
        if decoded_action is None:
            obs = self._get_obs()
            info = {
                "action_mask": self.action_masks(),
                "invalid_action": True,
            }
            return obs, self.invalid_action_penalty, True, False, info

        success = self.game.execute_action(decoded_action)
        if not success:
            obs = self._get_obs()
            info = {
                "action_mask": self.action_masks(),
                "invalid_action": True,
                "action_description": str(decoded_action),
            }
            return obs, self.invalid_action_penalty, True, False, info

        if not self.game.game_over:
            self.game.next_player()
            self._advance_to_controlled_decision()

        new_self_score = controlled_player.get_score()
        new_opp_score = opponent_player.get_score()
        reward = (new_self_score - prev_self_score) * self.score_delta_scale
        reward -= (new_opp_score - prev_opp_score) * self.score_delta_scale

        terminated = self.game.game_over
        truncated = self.step_count >= self.max_episode_steps and not terminated

        if terminated:
            winner_ids = self._winner_ids()
            if controlled_player.player_id in winner_ids:
                reward += self.win_reward
            elif winner_ids:
                reward += self.loss_reward

        obs = self._get_obs()
        info = {
            "action_mask": self.action_masks() if not terminated and not truncated else np.zeros(ACTION_DIM, dtype=bool),
            "action_description": str(decoded_action),
            "self_score": new_self_score,
            "opponent_score": new_opp_score,
            "winner_ids": self._winner_ids(),
            "invalid_action": False,
        }
        return obs, reward, terminated, truncated, info
