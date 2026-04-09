#!/usr/bin/env python

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT_DIR / "runs" / "rl"
TARGET_TRIGGER_ROUND = 25.0
POLL_SECONDS = 30


EXPERIMENTS = [
    {
        "run_name": "v20a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip010_step002_speed15_ref25",
        "args": [
            "--timesteps", "100000",
            "--opponent", "random",
            "--learning-rate", "0.0003",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--gamma", "0.95",
            "--gae-lambda", "0.95",
            "--clip-range", "0.10",
            "--ent-coef", "0.0",
            "--step-penalty", "0.02",
            "--win-speed-scale", "1.5",
            "--win-speed-reference-round", "25",
        ],
    },
    {
        "run_name": "v20b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip015_step003_speed15_ref25",
        "args": [
            "--timesteps", "100000",
            "--opponent", "random",
            "--learning-rate", "0.0003",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--gamma", "0.95",
            "--gae-lambda", "0.95",
            "--clip-range", "0.15",
            "--ent-coef", "0.0",
            "--step-penalty", "0.03",
            "--win-speed-scale", "1.5",
            "--win-speed-reference-round", "25",
        ],
    },
    {
        "run_name": "v21a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip010_step003_speed15_ref25",
        "args": [
            "--timesteps", "100000",
            "--opponent", "random",
            "--learning-rate", "0.0003",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--gamma", "0.95",
            "--gae-lambda", "0.95",
            "--clip-range", "0.10",
            "--ent-coef", "0.0",
            "--step-penalty", "0.03",
            "--win-speed-scale", "1.5",
            "--win-speed-reference-round", "25",
        ],
    },
    {
        "run_name": "v21b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip0125_step002_speed15_ref25",
        "args": [
            "--timesteps", "100000",
            "--opponent", "random",
            "--learning-rate", "0.0003",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--gamma", "0.95",
            "--gae-lambda", "0.95",
            "--clip-range", "0.125",
            "--ent-coef", "0.0",
            "--step-penalty", "0.02",
            "--win-speed-scale", "1.5",
            "--win-speed-reference-round", "25",
        ],
    },
    {
        "run_name": "v22a_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip0125_step003_speed15_ref25",
        "args": [
            "--timesteps", "100000",
            "--opponent", "random",
            "--learning-rate", "0.0003",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--gamma", "0.95",
            "--gae-lambda", "0.95",
            "--clip-range", "0.125",
            "--ent-coef", "0.0",
            "--step-penalty", "0.03",
            "--win-speed-scale", "1.5",
            "--win-speed-reference-round", "25",
        ],
    },
    {
        "run_name": "v22b_ppo_random_seed7_t100k_lr3e4_ns1024_bs256_gamma095_clip015_step0025_speed15_ref25",
        "args": [
            "--timesteps", "100000",
            "--opponent", "random",
            "--learning-rate", "0.0003",
            "--n-steps", "1024",
            "--batch-size", "256",
            "--gamma", "0.95",
            "--gae-lambda", "0.95",
            "--clip-range", "0.15",
            "--ent-coef", "0.0",
            "--step-penalty", "0.025",
            "--win-speed-scale", "1.5",
            "--win-speed-reference-round", "25",
        ],
    },
]


def run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def model_path(run_name: str) -> Path:
    return run_dir(run_name) / "model.zip"


def benchmark_path(run_name: str) -> Path:
    return run_dir(run_name) / "benchmark_random_seat2_ep10_seed3000.json"


def is_training_active(run_name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", run_name],
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def wait_for_model(run_name: str) -> None:
    path = model_path(run_name)
    while not path.exists():
        time.sleep(POLL_SECONDS)


def start_training(exp: dict) -> None:
    cmd = [
        sys.executable,
        "scripts/train_rl_agent.py",
        *exp["args"],
        "--run-name",
        exp["run_name"],
        "--progress-bar",
        "0",
    ]
    print(f"[train] start {exp['run_name']}", flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def run_benchmark(run_name: str) -> dict:
    output = benchmark_path(run_name)
    cmd = [
        sys.executable,
        "scripts/benchmark_rl_model.py",
        "--model-path",
        str(model_path(run_name)),
        "--episodes",
        "10",
        "--seed-start",
        "3000",
        "--opponent",
        "random",
        "--rl-seat",
        "2",
        "--output",
        str(output),
    ]
    print(f"[bench] start {run_name}", flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    with open(output, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_benchmark(run_name: str):
    path = benchmark_path(run_name)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    for exp in EXPERIMENTS:
        run_name = exp["run_name"]
        print(f"[queue] processing {run_name}", flush=True)

        result = load_existing_benchmark(run_name)
        if result is None:
            if model_path(run_name).exists():
                result = run_benchmark(run_name)
            elif is_training_active(run_name):
                print(f"[wait] training already running: {run_name}", flush=True)
                wait_for_model(run_name)
                result = run_benchmark(run_name)
            else:
                start_training(exp)
                result = run_benchmark(run_name)

        avg_trigger = result["summary"]["avg_rl_trigger_round"]
        win_rate = result["summary"]["win_rate"]
        print(
            f"[result] {run_name} avg_rl_trigger_round={avg_trigger} win_rate={win_rate}",
            flush=True,
        )

        if avg_trigger is not None and avg_trigger <= TARGET_TRIGGER_ROUND:
            print(f"[done] reached target with {run_name}", flush=True)
            return

    print("[done] queue exhausted without reaching target", flush=True)


if __name__ == "__main__":
    main()
