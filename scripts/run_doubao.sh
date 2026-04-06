#!/usr/bin/env bash

set -euo pipefail

# =========================================
# 手动配置区
# 这个脚本只用于“在线模型单局对战”。
# RL 训练 / 评估请直接使用 python scripts/train_rl_agent.py
# 与 python scripts/evaluate_rl_agent.py。
# 直接修改下面这些值即可切换不同厂商/模型。
# =========================================
# 在线模型访问参数
API_KEY="your_api_key_here"                 # 在线模型 API Key
API_KEY_ENV_NAME="ARK_API_KEY"             # 运行时注入到哪个环境变量名
MODEL_NAME="Doubao Seed 2.0 Pro"           # 写入临时 config 的模型显示名
MODEL_TYPE="openai_compatible"             # openai / azure_openai / openai_compatible
MODEL_ID="doubao-seed-2-0-pro-260215"      # 厂商模型 ID
BASE_URL="https://ark.cn-beijing.volces.com/api/v3"  # OpenAI 兼容接口地址

# 代理策略参数
PROMPT_STRATEGY="${PROMPT_STRATEGY:-rank_v2_auto}"   # rank_v2_auto / legacy
TEMPERATURE="${TEMPERATURE:-0.1}"                    # 仅 legacy / LangGraph 有效
MAX_TOKENS="${MAX_TOKENS:-128}"                      # 仅 legacy / LangGraph 有效
CANDIDATE_ACTION_LIMIT="${CANDIDATE_ACTION_LIMIT:-6}"  # 仅 rank_v2_auto 有效
TARGET_LIMIT="${TARGET_LIMIT:-4}"                      # 仅 rank_v2_auto 有效
NOBLE_LIMIT="${NOBLE_LIMIT:-3}"                        # 仅 rank_v2_auto 有效
CONDA_ENV_NAME="${CONDA_ENV_NAME:-splendor}"           # conda 环境名

# 游戏运行参数
NUM_PLAYERS="${NUM_PLAYERS:-2}"              # 总玩家数
NUM_LLM_AGENTS="${NUM_LLM_AGENTS:-1}"        # 使用该模型的代理数量，其余自动补随机
DELAY="${DELAY:-0.5}"                        # 回合间延迟
USE_PYGAME="${USE_PYGAME:-1}"                # 1=pygame, 0=终端渲染
USE_LANGGRAPH="${USE_LANGGRAPH:-0}"          # 1=LangGraph；必须配合 legacy
MAX_TURNS="${MAX_TURNS:-}"                   # 调试用，跑到指定动作数就停
SEED="${SEED:-}"                             # 随机种子

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$MODEL_TYPE" == "rl_ppo" ]]; then
  echo "错误: scripts/run_doubao.sh 只用于在线模型，不用于 rl_ppo 本地模型。"
  exit 1
fi

if [[ "$USE_LANGGRAPH" == "1" && "$PROMPT_STRATEGY" != "legacy" ]]; then
  echo "错误: USE_LANGGRAPH=1 不能与 PROMPT_STRATEGY=$PROMPT_STRATEGY 同时使用。"
  echo "请将 PROMPT_STRATEGY 设为 legacy。"
  exit 1
fi

if (( NUM_LLM_AGENTS > NUM_PLAYERS )); then
  echo "错误: NUM_LLM_AGENTS 不能大于 NUM_PLAYERS。"
  exit 1
fi

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
