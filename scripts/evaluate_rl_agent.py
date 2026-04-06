#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sb3_contrib import MaskablePPO

from rl.env import SplendorEnv


def parse_args():
    parser = argparse.ArgumentParser(description="评估训练后的 PPO Agent")
    parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    parser.add_argument("--episodes", type=int, default=20, help="评估局数")
    parser.add_argument("--seed", type=int, default=7, help="评估随机种子基准")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "rule_based"], help="对手类型")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="单局最大受控步数")
    parser.add_argument("--output", type=str, default=None, help="评估结果输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    model = MaskablePPO.load(args.model_path)

    episode_results = []
    wins = 0
    losses = 0
    draws = 0
    invalid_terminations = 0
    total_reward = 0.0
    total_steps = 0

    for episode_idx in range(args.episodes):
        env = SplendorEnv(
            opponent_type=args.opponent,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed + episode_idx,
        )
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            episode_steps += 1
            done = terminated or truncated

        winner_ids = info.get("winner_ids", [])
        invalid_action = bool(info.get("invalid_action", False))
        if invalid_action:
            outcome = "invalid"
            invalid_terminations += 1
        elif "rl_agent_1" in winner_ids:
            outcome = "win"
            wins += 1
        elif winner_ids:
            outcome = "loss"
            losses += 1
        else:
            outcome = "draw"
            draws += 1

        total_reward += episode_reward
        total_steps += episode_steps
        episode_results.append(
            {
                "episode": episode_idx,
                "seed": args.seed + episode_idx,
                "reward": episode_reward,
                "steps": episode_steps,
                "outcome": outcome,
                "invalid_action": invalid_action,
                "winner_ids": winner_ids,
                "self_score": info.get("self_score"),
                "opponent_score": info.get("opponent_score"),
            }
        )

    summary = {
        "model_path": args.model_path,
        "episodes": args.episodes,
        "opponent": args.opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "invalid_terminations": invalid_terminations,
        "win_rate": wins / args.episodes if args.episodes else 0.0,
        "avg_reward": total_reward / args.episodes if args.episodes else 0.0,
        "avg_steps": total_steps / args.episodes if args.episodes else 0.0,
        "episode_results": episode_results,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
