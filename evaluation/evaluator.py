import json
import time
import os
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from game.game import Game
from game.player import Player
from agents.base_agent import BaseAgent


class Evaluator:
    """评估系统，用于评估不同代理的表现"""
    
    def __init__(self, agents: List[BaseAgent], num_games: int = 10, seed: int = None):
        """初始化评估系统
        
        Args:
            agents: 要评估的代理列表
            num_games: 评估游戏数量
            seed: 随机种子
        """
        self.agents = agents
        self.num_games = num_games
        
        if seed is not None:
            random.seed(seed)
        
        self.results = {
            "games": [],
            "summary": {}
        }
    
    def run_evaluation(self, output_dir: str = "results"):
        """运行评估
        
        Args:
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行指定数量的游戏
        for game_idx in range(self.num_games):
            game_seed = random.randint(0, 10000)
            game_results = self._run_game(game_idx, game_seed)
            self.results["games"].append(game_results)
        
        # 生成汇总结果
        self._generate_summary()
        
        # 保存结果
        timestamp = int(time.time())
        result_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        return self.results
    
    def _run_game(self, game_idx: int, seed: int) -> Dict[str, Any]:
        """运行单个游戏并收集结果
        
        Args:
            game_idx: 游戏索引
            seed: 随机种子
            
        Returns:
            Dict[str, Any]: 游戏结果
        """
        print(f"正在运行游戏 {game_idx+1}/{self.num_games}...")
        
        # 随机排序代理，避免先手优势
        shuffled_agents = list(self.agents)
        random.shuffle(shuffled_agents)
        
        # 创建玩家
        players = []
        for i, agent in enumerate(shuffled_agents):
            player = Player(agent.player_id, agent.name)
            players.append(player)
            
            # 保存玩家和代理的映射关系
            agent._player = player
        
        # 创建游戏
        game = Game(players, seed=seed)
        
        # 游戏开始时的回调
        game_state = game.get_game_state()
        for agent in shuffled_agents:
            agent.on_game_start(game_state)
        
        # 运行游戏直到结束
        while not game.game_over:
            current_player = game.get_current_player()
            current_agent = next((a for a in shuffled_agents if a._player == current_player), None)
            
            if current_agent:
                # 回合开始
                game_state = game.get_game_state()
                current_agent.on_turn_start(game_state)
                
                # 获取有效动作
                valid_actions = game.get_valid_actions()
                
                if valid_actions:
                    # 让代理选择动作
                    selected_action = current_agent.select_action(game_state, valid_actions)
                    
                    # 执行动作
                    success = game.execute_action(selected_action)
                    
                    # 回合结束
                    game_state = game.get_game_state()
                    current_agent.on_turn_end(game_state, selected_action, success)
                
            # 进入下一个玩家
            game.next_player()
        
        # 游戏结束
        game_state = game.get_game_state()
        winner_ids = [player.player_id for player in game.winner] if game.winner else []
        
        for agent in shuffled_agents:
            agent.on_game_end(game_state, winner_ids)
        
        # 收集游戏结果
        game_results = {
            "game_id": game_idx,
            "seed": seed,
            "rounds": game.round_number,
            "winners": [{"player_id": player.player_id, "name": player.name, "score": player.get_score()} for player in game.winner] if game.winner else [],
            "players": [{"player_id": player.player_id, "name": player.name, "score": player.get_score(), "agent_type": type(agent).__name__} for player, agent in zip(players, shuffled_agents)],
            "history_length": len(game.history)
        }
        
        # 保存游戏历史
        history_file = os.path.join("results", f"game_{game_idx}_history.json")
        os.makedirs("results", exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(game.history, f, ensure_ascii=False, indent=2)
        
        return game_results
    
    def _generate_summary(self):
        """生成评估汇总结果"""
        # 初始化汇总数据
        summary = {
            "num_games": self.num_games,
            "agent_performance": {},
            "win_rate": {},
            "average_score": {},
            "average_rank": {}
        }
        
        # 统计每个代理的表现
        agent_wins = defaultdict(int)
        agent_scores = defaultdict(list)
        agent_ranks = defaultdict(list)
        
        # 处理每个游戏的结果
        for game in self.results["games"]:
            # 计算玩家排名
            sorted_players = sorted(game["players"], key=lambda p: p["score"], reverse=True)
            ranks = {}
            
            for rank, player in enumerate(sorted_players):
                player_id = player["player_id"]
                agent_type = player["agent_type"]
                score = player["score"]
                
                # 记录排名
                ranks[player_id] = rank + 1  # 排名从1开始
                
                # 记录分数
                agent_scores[agent_type].append(score)
                
                # 记录排名
                agent_ranks[agent_type].append(rank + 1)
            
            # 记录获胜情况
            for winner in game["winners"]:
                player_id = winner["player_id"]
                for player in game["players"]:
                    if player["player_id"] == player_id:
                        agent_type = player["agent_type"]
                        agent_wins[agent_type] += 1
                        break
        
        # 计算胜率
        for agent in self.agents:
            agent_type = type(agent).__name__
            win_rate = agent_wins[agent_type] / self.num_games
            avg_score = sum(agent_scores[agent_type]) / len(agent_scores[agent_type]) if agent_scores[agent_type] else 0
            avg_rank = sum(agent_ranks[agent_type]) / len(agent_ranks[agent_type]) if agent_ranks[agent_type] else 0
            
            summary["agent_performance"][agent_type] = {
                "name": agent.name,
                "player_id": agent.player_id,
                "wins": agent_wins[agent_type],
                "win_rate": win_rate,
                "average_score": avg_score,
                "average_rank": avg_rank,
                "scores": agent_scores[agent_type],
                "ranks": agent_ranks[agent_type]
            }
        
        # 按胜率排序
        summary["win_rate"] = {agent_type: data["win_rate"] for agent_type, data in sorted(summary["agent_performance"].items(), key=lambda x: x[1]["win_rate"], reverse=True)}
        
        # 按平均分数排序
        summary["average_score"] = {agent_type: data["average_score"] for agent_type, data in sorted(summary["agent_performance"].items(), key=lambda x: x[1]["average_score"], reverse=True)}
        
        # 按平均排名排序
        summary["average_rank"] = {agent_type: data["average_rank"] for agent_type, data in sorted(summary["agent_performance"].items(), key=lambda x: x[1]["average_rank"])}
        
        self.results["summary"] = summary 