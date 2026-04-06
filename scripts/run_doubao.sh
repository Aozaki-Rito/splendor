#!/usr/bin/env bash

set -euo pipefail

# =========================================
# 手动配置区
# 直接修改下面这些值即可切换不同厂商/模型
# =========================================
API_KEY="your_api_key_here"
API_KEY_ENV_NAME="ARK_API_KEY"
MODEL_NAME="Doubao Seed 2.0 Pro"
MODEL_TYPE="openai_compatible"
MODEL_ID="doubao-seed-2-0-pro-260215"
BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-rank_v2_auto}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-splendor}"
CANDIDATE_ACTION_LIMIT="${CANDIDATE_ACTION_LIMIT:-6}"
TARGET_LIMIT="${TARGET_LIMIT:-4}"
NOBLE_LIMIT="${NOBLE_LIMIT:-3}"

# 游戏运行参数
NUM_PLAYERS="${NUM_PLAYERS:-2}"
NUM_LLM_AGENTS="${NUM_LLM_AGENTS:-1}"
DELAY="${DELAY:-0.5}"
USE_PYGAME="${USE_PYGAME:-1}"
USE_LANGGRAPH="${USE_LANGGRAPH:-0}"
MAX_TURNS="${MAX_TURNS:-}"
SEED="${SEED:-}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "$API_KEY" || "$API_KEY" == "your_api_key_here" ]]; then
  if [[ "$USE_LANGGRAPH" == "1" ]]; then
    echo "错误: LangGraph 模式需要先在 scripts/run_doubao.sh 中填写 API_KEY。"
    exit 1
  fi
  if [[ "$PROMPT_STRATEGY" != "rank_v2_auto" ]]; then
    echo "错误: legacy 模式需要先在 scripts/run_doubao.sh 中填写 API_KEY。"
    exit 1
  fi
fi

if [[ -n "$API_KEY" && "$API_KEY" != "your_api_key_here" ]]; then
  export "$API_KEY_ENV_NAME=$API_KEY"
fi

TMP_CONFIG="$(mktemp)"
cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

cat > "$TMP_CONFIG" <<EOF
{
  "models": [
    {
      "name": "$MODEL_NAME",
      "type": "$MODEL_TYPE",
      "model_name": "$MODEL_ID",
      "api_key_env": "$API_KEY_ENV_NAME",
      "base_url": "$BASE_URL",
      "prompt_strategy": "$PROMPT_STRATEGY",
      "candidate_action_limit": $CANDIDATE_ACTION_LIMIT,
      "target_limit": $TARGET_LIMIT,
      "noble_limit": $NOBLE_LIMIT,
      "temperature": $TEMPERATURE,
      "max_tokens": $MAX_TOKENS
    }
  ],
  "game_settings": {
    "num_players": $NUM_PLAYERS,
    "seed": null,
    "delay": $DELAY,
    "save_history": true
  },
  "evaluation_settings": {
    "num_games": 10
  }
}
EOF

PYTHON_CMD=(python)
if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python)
fi

CMD=(
  "${PYTHON_CMD[@]}" main.py
  --config "$TMP_CONFIG"
  game
  --model "$MODEL_NAME"
  --num-players "$NUM_PLAYERS"
  --num-llm-agents "$NUM_LLM_AGENTS"
  --delay "$DELAY"
  --use_pygame "$USE_PYGAME"
  --use_langgraph "$USE_LANGGRAPH"
  --temperature "$TEMPERATURE"
)

if [[ -n "$SEED" ]]; then
  CMD+=(--seed "$SEED")
fi

if [[ -n "$MAX_TURNS" ]]; then
  CMD+=(--max-turns "$MAX_TURNS")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "当前模型配置:"
echo "  MODEL_TYPE=$MODEL_TYPE"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  MODEL_ID=$MODEL_ID"
echo "  BASE_URL=$BASE_URL"
echo "  PROMPT_STRATEGY=$PROMPT_STRATEGY"
echo "  CANDIDATE_ACTION_LIMIT=$CANDIDATE_ACTION_LIMIT"
echo "  TARGET_LIMIT=$TARGET_LIMIT"
echo "  NOBLE_LIMIT=$NOBLE_LIMIT"
echo "  CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo
echo "启动命令:"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
