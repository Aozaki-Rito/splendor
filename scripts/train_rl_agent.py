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
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="熵正则系数，鼓励更高探索")
    parser.add_argument("--step-penalty", type=float, default=0.02, help="每个有效动作的惩罚，鼓励更快结束对局")
    parser.add_argument("--round-penalty-scale", type=float, default=0.0, help="按当前回合递增的额外惩罚，鼓励更早结束")
    parser.add_argument("--score-speed-scale", type=float, default=0.0, help="得分越早越值钱的 shaping 强度")
    parser.add_argument("--score-speed-reference-round", type=int, default=25, help="得分速度 shaping 的参考回合数")
    parser.add_argument("--win-speed-scale", type=float, default=0.0, help="胜利时按终局回合给额外 shaping，鼓励更早获胜")
    parser.add_argument("--win-speed-reference-round", type=int, default=40, help="胜利速度 shaping 的参考回合数")
    parser.add_argument("--progress-bar", type=int, choices=[0, 1], default=1, help="是否显示训练进度条")
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
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "step_penalty": args.step_penalty,
        "round_penalty_scale": args.round_penalty_scale,
        "score_speed_scale": args.score_speed_scale,
        "score_speed_reference_round": args.score_speed_reference_round,
        "win_speed_scale": args.win_speed_scale,
        "win_speed_reference_round": args.win_speed_reference_round,
        "progress_bar": bool(args.progress_bar),
        "run_name": run_name,
    }
    with open(artifact_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    env = Monitor(
        SplendorEnv(
            opponent_type=args.opponent,
            max_episode_steps=args.max_episode_steps,
            step_penalty=args.step_penalty,
            round_penalty_scale=args.round_penalty_scale,
            score_speed_scale=args.score_speed_scale,
            score_speed_reference_round=args.score_speed_reference_round,
            win_speed_scale=args.win_speed_scale,
            win_speed_reference_round=args.win_speed_reference_round,
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
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=bool(args.progress_bar), tb_log_name=run_name)

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
