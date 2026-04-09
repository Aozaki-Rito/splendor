#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.random_agent import RandomAgent
from agents.rl_ppo_agent import RLPPOAgent
from agents.rule_based_agent import RuleBasedAgent
from game.game import Game
from game.player import Player


def parse_args():
    parser = argparse.ArgumentParser(description="在真实主游戏逻辑下评估 RL 模型节奏")
    parser.add_argument("--model-path", type=str, required=True, help="RL 模型路径")
    parser.add_argument("--episodes", type=int, default=20, help="评估局数")
    parser.add_argument("--seed-start", type=int, default=5000, help="起始随机种子")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "rule_based"], help="对手类型")
    parser.add_argument("--rl-seat", type=int, default=2, choices=[1, 2], help="RL 座位号（1-based）")
    parser.add_argument("--output", type=str, default=None, help="可选输出 JSON 路径")
    return parser.parse_args()


def build_opponent(player_id: str, opponent_type: str, run_id: str):
    if opponent_type == "random":
        return RandomAgent(player_id=player_id, name="随机代理")
    return RuleBasedAgent(player_id=player_id, name="规则代理", run_id=run_id)


def play_one(model_path: str, seed: int, opponent_type: str, rl_seat: int):
    rl_index = rl_seat - 1
    opp_index = 1 - rl_index

    rl_agent = RLPPOAgent(
        player_id=f"p{rl_seat}",
        name="RL PPO Agent (Local) 代理",
        model_path=model_path,
        deterministic=True,
        device="auto",
        run_id=f"bench_{opponent_type}_{seed}_{rl_seat}",
    )
    opp_agent = build_opponent(player_id=f"p{opp_index + 1}", opponent_type=opponent_type, run_id=f"bench_{opponent_type}_{seed}_{opp_index + 1}")

    seat_agents = [None, None]
    seat_agents[rl_index] = rl_agent
    seat_agents[opp_index] = opp_agent

    players = []
    for agent in seat_agents:
        player = Player(agent.player_id, agent.name)
        players.append(player)
        agent._player = player

    game = Game(players, seed=seed)
    game.agent_map = {agent.player_id: agent for agent in seat_agents}
    for agent in seat_agents:
        agent._game = game

    rl_player = players[rl_index]
    opp_player = players[opp_index]
    rl_trigger_round = None
    opp_trigger_round = None
    fallback_count = 0

    while not game.game_over:
        current_player = game.get_current_player()
        current_agent = next(agent for agent in seat_agents if agent._player == current_player)
        valid_actions = game.get_valid_actions()
        if valid_actions:
            action = current_agent.select_action(game.get_game_state(), valid_actions)
            success = game.execute_action(action)
            if not success:
                fallback_count += 1
                success = False
                for fallback_action in valid_actions:
                    if game.execute_action(fallback_action):
                        action = fallback_action
                        success = True
                        break
                if not success:
                    raise RuntimeError(f"动作执行失败: {action}")

            if rl_trigger_round is None and rl_player.get_score() >= 15:
                rl_trigger_round = game.round_number
            if opp_trigger_round is None and opp_player.get_score() >= 15:
                opp_trigger_round = game.round_number
        elif game.end_if_stalemated():
            break

        if not game.game_over:
            game.next_player()

    winner_ids = [player.player_id for player in (game.winner or [])]
    return {
        "seed": seed,
        "rl_seat": rl_seat,
        "opponent": opponent_type,
        "final_round": game.round_number,
        "total_actions": len(game.history),
        "rl_trigger_round": rl_trigger_round,
        "opp_trigger_round": opp_trigger_round,
        "rl_score": rl_player.get_score(),
        "opp_score": opp_player.get_score(),
        "rl_win": rl_player.player_id in winner_ids,
        "winner_ids": winner_ids,
        "fallback_count": fallback_count,
    }


def summarize(rows):
    rl_trigger_rounds = [row["rl_trigger_round"] for row in rows if row["rl_trigger_round"] is not None]
    opp_trigger_rounds = [row["opp_trigger_round"] for row in rows if row["opp_trigger_round"] is not None]
    return {
        "episodes": len(rows),
        "win_rate": sum(row["rl_win"] for row in rows) / len(rows) if rows else 0.0,
        "avg_final_round": mean(row["final_round"] for row in rows) if rows else None,
        "min_final_round": min(row["final_round"] for row in rows) if rows else None,
        "max_final_round": max(row["final_round"] for row in rows) if rows else None,
        "avg_total_actions": mean(row["total_actions"] for row in rows) if rows else None,
        "avg_rl_trigger_round": mean(rl_trigger_rounds) if rl_trigger_rounds else None,
        "min_rl_trigger_round": min(rl_trigger_rounds) if rl_trigger_rounds else None,
        "max_rl_trigger_round": max(rl_trigger_rounds) if rl_trigger_rounds else None,
        "avg_opp_trigger_round": mean(opp_trigger_rounds) if opp_trigger_rounds else None,
        "avg_rl_score": mean(row["rl_score"] for row in rows) if rows else None,
        "avg_opp_score": mean(row["opp_score"] for row in rows) if rows else None,
        "fallback_games": sum(1 for row in rows if row["fallback_count"] > 0),
    }


def main():
    args = parse_args()
    rows = [
        play_one(
            model_path=args.model_path,
            seed=args.seed_start + idx,
            opponent_type=args.opponent,
            rl_seat=args.rl_seat,
        )
        for idx in range(args.episodes)
    ]

    result = {
        "model_path": args.model_path,
        "opponent": args.opponent,
        "rl_seat": args.rl_seat,
        "summary": summarize(rows),
        "episodes": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
