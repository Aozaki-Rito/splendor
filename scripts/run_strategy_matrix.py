#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行策略短测并汇总结果")
    parser.add_argument("--strategies", nargs="+", required=True, help="要测试的 prompt 策略列表")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="固定种子列表")
    parser.add_argument("--max-turns", type=int, default=2, help="每局最多执行的回合数")
    parser.add_argument("--delay", type=float, default=0.0, help="回合间延迟")
    parser.add_argument("--use-pygame", type=int, default=0, choices=[0, 1], help="是否启用 pygame")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="单局最长等待秒数")
    parser.add_argument("--action-ranking", type=str, default="none", help="动作筛选模式")
    parser.add_argument("--candidate-action-limit", type=int, default=10, help="候选动作上限")
    parser.add_argument("--target-limit", type=int, default=5, help="目标卡摘要上限")
    parser.add_argument("--noble-limit", type=int, default=3, help="贵族摘要上限")
    parser.add_argument("--output-dir", type=str, default="", help="输出目录，默认自动生成")
    return parser.parse_args()


def extract_first(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_llm_log(log_path: Path) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    if not log_path.exists():
        return metrics

    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " - " not in line:
            continue
        _, payload = line.split(" - ", 1)
        payload = payload.strip()
        if not payload.startswith("{"):
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if "prompt_length" in data and "prompt_length" not in metrics:
            metrics["prompt_length"] = data["prompt_length"]
        if "elapsed_seconds" in data and "elapsed_seconds" not in metrics:
            metrics["elapsed_seconds"] = data["elapsed_seconds"]
        if "chosen_action" in data and "chosen_action" not in metrics:
            metrics["chosen_action"] = data["chosen_action"]
        if "fallback" in data:
            metrics["fallback"] = data["fallback"]

    return metrics


def run_single_case(
    strategy: str,
    seed: int,
    args: argparse.Namespace,
    experiment_dir: Path,
) -> Dict[str, object]:
    env = os.environ.copy()
    env.update(
        {
            "PROMPT_STRATEGY": strategy,
            "ACTION_RANKING": args.action_ranking,
            "CANDIDATE_ACTION_LIMIT": str(args.candidate_action_limit),
            "TARGET_LIMIT": str(args.target_limit),
            "NOBLE_LIMIT": str(args.noble_limit),
            "SEED": str(seed),
            "MAX_TURNS": str(args.max_turns),
            "USE_PYGAME": str(args.use_pygame),
            "DELAY": str(args.delay),
        }
    )

    cmd = [
        "./scripts/run_doubao.sh",
        "--use_pygame",
        str(args.use_pygame),
        "--delay",
        str(args.delay),
    ]

    started_at = time.time()
    completed = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    finished_at = time.time()

    slug = f"{strategy}_seed{seed}"
    stdout_path = experiment_dir / f"{slug}.stdout.log"
    stderr_path = experiment_dir / f"{slug}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    llm_log_str = extract_first(r"LLM日志文件:\s*(.+)", completed.stdout)
    history_path_str = extract_first(r"游戏历史已保存到:\s*(.+)", completed.stdout)

    metrics = parse_llm_log((ROOT_DIR / llm_log_str).resolve()) if llm_log_str else {}
    first_decision_console = extract_first(r"决策时间:\s*([\d.]+)秒", completed.stdout)

    return {
        "strategy": strategy,
        "action_ranking": args.action_ranking,
        "candidate_action_limit": args.candidate_action_limit,
        "seed": seed,
        "returncode": completed.returncode,
        "wall_time_seconds": round(finished_at - started_at, 3),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "llm_log": str((ROOT_DIR / llm_log_str).resolve()) if llm_log_str else None,
        "history_path": str((ROOT_DIR / history_path_str).resolve()) if history_path_str else None,
        "console_first_decision_seconds": float(first_decision_console) if first_decision_console else None,
        "prompt_length": metrics.get("prompt_length"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "chosen_action": metrics.get("chosen_action"),
        "fallback": metrics.get("fallback"),
    }


def write_summary(entries: List[Dict[str, object]], experiment_dir: Path) -> None:
    json_path = experiment_dir / "summary.json"
    csv_path = experiment_dir / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "strategy",
        "action_ranking",
        "candidate_action_limit",
        "seed",
        "returncode",
        "wall_time_seconds",
        "prompt_length",
        "elapsed_seconds",
        "console_first_decision_seconds",
        "chosen_action",
        "fallback",
        "llm_log",
        "history_path",
        "stdout_log",
        "stderr_log",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def print_brief_summary(entries: List[Dict[str, object]]) -> None:
    by_strategy: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        by_strategy.setdefault(entry["strategy"], []).append(entry)

    print("\n实验汇总:")
    for strategy, items in by_strategy.items():
        valid_elapsed = [item["elapsed_seconds"] for item in items if item.get("elapsed_seconds") is not None]
        avg_elapsed = round(sum(valid_elapsed) / len(valid_elapsed), 3) if valid_elapsed else None
        valid_prompts = [item["prompt_length"] for item in items if item.get("prompt_length") is not None]
        avg_prompt = round(sum(valid_prompts) / len(valid_prompts), 1) if valid_prompts else None
        print(f"- {strategy}: runs={len(items)}, avg_elapsed={avg_elapsed}, avg_prompt={avg_prompt}")


def main() -> int:
    args = parse_args()
    experiment_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    experiment_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "results" / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, object]] = []
    for strategy in args.strategies:
        for seed in args.seeds:
            print(f"[RUN] strategy={strategy} seed={seed}")
            try:
                entry = run_single_case(strategy, seed, args, experiment_dir)
            except subprocess.TimeoutExpired as exc:
                slug = f"{strategy}_seed{seed}"
                stdout_path = experiment_dir / f"{slug}.stdout.log"
                stderr_path = experiment_dir / f"{slug}.stderr.log"
                stdout_path.write_text(exc.stdout or "", encoding="utf-8")
                stderr_path.write_text(exc.stderr or "", encoding="utf-8")
                entry = {
                    "strategy": strategy,
                    "action_ranking": args.action_ranking,
                    "candidate_action_limit": args.candidate_action_limit,
                    "seed": seed,
                    "returncode": "timeout",
                    "wall_time_seconds": args.timeout,
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                    "llm_log": None,
                    "history_path": None,
                    "console_first_decision_seconds": None,
                    "prompt_length": None,
                    "elapsed_seconds": None,
                    "chosen_action": None,
                    "fallback": None,
                }
            entries.append(entry)

    write_summary(entries, experiment_dir)
    print_brief_summary(entries)
    print(f"\n产物目录: {experiment_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
