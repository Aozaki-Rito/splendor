#!/usr/bin/env python

import argparse
import json
import time
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from rl.env import SplendorEnv


def parse_args():
    parser = argparse.ArgumentParser(description="训练璀璨宝石 PPO Agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="训练总步数")
    parser.add_argument("--seed", type=int, default=7, help="训练随机种子")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "rule_based"], help="对手类型")
    parser.add_argument("--run-name", type=str, default=None, help="训练运行名称")
    parser.add_argument("--save-path", type=str, default=None, help="模型保存路径")
    parser.add_argument("--tensorboard-log", type=str, default="runs/tensorboard", help="TensorBoard 输出目录")
    parser.add_argument("--artifact-dir", type=str, default="runs/rl", help="训练产物目录")
    parser.add_argument("--max-episode-steps", type=int, default=200, help="单局最大受控步数")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout 长度")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"ppo_seed{args.seed}_{int(time.time())}"
    artifact_dir = Path(args.artifact_dir) / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "timesteps": args.timesteps,
        "seed": args.seed,
        "opponent": args.opponent,
        "max_episode_steps": args.max_episode_steps,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "run_name": run_name,
    }
    with open(artifact_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    env = Monitor(
        SplendorEnv(
            opponent_type=args.opponent,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        ),
        filename=str(artifact_dir / "monitor.csv"),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True, tb_log_name=run_name)

    save_path = Path(args.save_path) if args.save_path else artifact_dir / "model.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    summary = {
        "run_name": run_name,
        "artifact_dir": str(artifact_dir),
        "model_path": str(save_path),
        "tensorboard_log_root": args.tensorboard_log,
        "timesteps": args.timesteps,
    }
    with open(artifact_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"训练产物目录: {artifact_dir}")
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
