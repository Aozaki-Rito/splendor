# Splendor-LLM

本项目是一个“璀璨宝石规则引擎 + 多种代理 + 可视化 + 评测工具”的实验项目。

当前支持的代理类型：

- `RandomAgent`
  - 随机策略
- `legacy`
  - 纯 LLM 策略
- `rank_v2_auto`
  - 纯规则策略
- `LanggraphAgent`
  - LangGraph 多步决策策略
- `RL PPO Agent`
  - 本地 PPO 模型

## 环境准备

推荐使用 conda：

```bash
conda create -n splendor python=3.9 -y
conda activate splendor
pip install -r requirements.txt
```

依赖文件：
- [requirements.txt](/home/aozaki/projects/code/splendor/requirements.txt)

## 统一调用方式

本文档只保留两类主入口：

1. 单局对战 / 批量评测：统一用 `python main.py ...`
2. RL 训练 / RL 评估：统一用 `python scripts/...`

[scripts/run_doubao.sh](/home/aozaki/projects/code/splendor/scripts/run_doubao.sh) 仍然保留，但它只是在线模型的便捷包装脚本，不再作为本文档主流程。

## `main.py` 入口

主入口文件：
- [main.py](/home/aozaki/projects/code/splendor/main.py)

支持三个子命令：

- `game`
  - 运行单局
- `eval`
  - 批量评测
- `list-models`
  - 列出 `config.json` 中可用的模型

先看帮助：

```bash
python main.py game --help
python main.py eval --help
python main.py list-models
```

## 策略是怎么决定的

单局运行时，最终使用哪种策略，由“模型配置 + 命令行参数”共同决定：

1. 如果模型 `type=rl_ppo`
   - 使用 `RL PPO Agent`
2. 否则，如果命令行传了 `--use_langgraph 1`
   - 使用 `LanggraphAgent`
3. 否则，如果模型配置里 `prompt_strategy=rank_v2_auto`
   - 使用纯规则策略
4. 否则
   - 使用 `legacy` 纯 LLM 策略

## 配置文件

默认配置文件：
- [config.json](/home/aozaki/projects/code/splendor/config.json)

示例：

```json
{
  "name": "Doubao Seed 2.0 Pro",
  "type": "openai_compatible",
  "model_name": "doubao-seed-2-0-pro-260215",
  "api_key_env": "ARK_API_KEY",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "prompt_strategy": "legacy",
  "temperature": 0.5,
  "max_tokens": 500
}
```

本地 RL 模型示例：

```json
{
  "name": "RL PPO Agent (Local)",
  "type": "rl_ppo",
  "model_path": "runs/rl/v1_ppo_random_seed7_t50k_post_endfix/model.zip",
  "deterministic": true,
  "device": "auto"
}
```

### 模型字段说明

- `name`
  - 模型显示名
  - 命令行里的 `--model` 就是填这个名字
- `type`
  - 决定模型大类
  - 支持：`openai`、`azure_openai`、`openai_compatible`、`rl_ppo`
- `model_name`
  - 在线模型的真实模型 ID
  - 只对 `openai` / `azure_openai` / `openai_compatible` 有意义
- `model_path`
  - 本地 PPO 模型文件路径
  - 只对 `rl_ppo` 有意义
- `api_key`
  - 直接写在配置里的 API Key
  - 不建议提交到仓库
- `api_key_env`
  - 从环境变量读取 API Key 的变量名
- `base_url`
  - OpenAI 兼容接口地址
  - 只对在线模型有意义
- `api_version`
  - Azure OpenAI 的 API 版本
  - 只对 `azure_openai` 有意义
- `deployment_name`
  - Azure OpenAI 的 deployment 名称
  - 只对 `azure_openai` 有意义
- `prompt_strategy`
  - 在线普通代理的策略模式
  - 当前只正式保留：`legacy`、`rank_v2_auto`
- `temperature`
  - 采样温度
  - 只对 `legacy` / `LanggraphAgent` 有意义
- `max_tokens`
  - LLM 输出 token 上限
  - 只对 `legacy` / `LanggraphAgent` 有意义
- `candidate_action_limit`
  - 规则策略候选动作上限
  - 只对 `rank_v2_auto` 有意义
- `target_limit`
  - 规则策略关注的目标卡数量
  - 只对 `rank_v2_auto` 有意义
- `noble_limit`
  - 规则策略关注的贵族数量
  - 只对 `rank_v2_auto` 有意义
- `deterministic`
  - PPO 推理是否走确定性动作
  - 只对 `rl_ppo` 有意义
- `device`
  - PPO 模型加载设备，如 `auto` / `cpu` / `cuda`
  - 只对 `rl_ppo` 有意义

## `game` 命令参数说明

这些参数最常用：

- `--model`
  - 指定 `config.json` 里的模型名称
- `--num-players`
  - 总玩家数
  - 没有显式创建的玩家会自动补成随机代理
- `--num-llm-agents`
  - 使用配置模型的代理数量
  - 其余玩家自动补成随机代理
- `--use_pygame 1|0`
  - `1` 表示用 pygame 图形界面
  - `0` 表示只用终端渲染
- `--use_langgraph 1|0`
  - `1` 表示强制走 LangGraph
  - `0` 表示不启用 LangGraph
- `--temperature`
  - 覆盖模型配置里的温度
  - 只对 `legacy` / LangGraph 有意义
- `--delay`
  - 每回合之间的显示延迟秒数
- `--seed`
  - 固定随机种子，便于复现
- `--max-turns`
  - 调试用，只跑前若干个动作就停
- `--save-history`
  - 保存游戏历史

## 不能同时开的组合

下面这些组合不要同时开：

- `--use_langgraph 1` 和 `prompt_strategy=rank_v2_auto`
  - 两者互斥
  - 现在代码里会直接报错
- `--use_langgraph 1` 和 `type=rl_ppo`
  - 两者互斥
  - 现在代码里会直接报错

## 会被忽略的变量

有些变量不是报错，而是单纯没作用：

- `temperature`
  - 对 `rank_v2_auto` 没作用
  - 对 `rl_ppo` 没作用
- `max_tokens`
  - 对 `rank_v2_auto` 没作用
  - 对 `rl_ppo` 没作用
- `prompt_strategy`
  - 对 `rl_ppo` 没作用
  - 当 `--use_langgraph 1` 时也不再决定最终代理类型
- `model_path`
  - 对在线模型没作用
- `base_url`
  - 对 `rl_ppo` 没作用

## 最常用命令

### 1. 查看模型列表

```bash
python main.py list-models
```

### 2. 运行纯规则策略

前提：
- 你选的模型配置里 `prompt_strategy=rank_v2_auto`

```bash
python main.py game --model "Doubao Seed 2.0 Pro" --use_pygame 1
```

### 3. 运行纯 LLM 策略

前提：
- 你选的模型配置里 `prompt_strategy=legacy`
- 已经配置好 API Key

```bash
python main.py game --model "Doubao Seed 2.0 Pro" --use_pygame 1
```

### 4. 运行 LangGraph 策略

前提：
- 你选的是在线模型
- 已经配置好 API Key
- 模型配置不要用 `rank_v2_auto`

```bash
python main.py game --model "Doubao Seed 2.0 Pro" --use_langgraph 1 --use_pygame 0
```

### 5. 运行本地 PPO 模型

```bash
python main.py game --model "RL PPO Agent (Local)" --use_pygame 1
```

说明：
- 这条命令当前是 `RL PPO Agent (Local)` 对随机代理的自动对局
- 还不是“人类玩家手动操作 vs RL Agent”

### 6. 只跑前几回合作烟测

```bash
python main.py game --model "RL PPO Agent (Local)" --use_pygame 0 --delay 0 --max-turns 3 --seed 7
```

### 7. 批量评测

```bash
python main.py eval --model "Doubao Seed 2.0 Pro" --num-games 10
```

## RL 训练与评估

RL 设计文档：
- [RL_AGENT_DESIGN.md](/home/aozaki/projects/code/splendor/RL_AGENT_DESIGN.md)

训练：

```bash
python scripts/train_rl_agent.py --timesteps 50000
```

评估：

```bash
python scripts/evaluate_rl_agent.py \
  --model-path runs/rl/v1_ppo_random_seed7_t50k_post_endfix/model.zip \
  --episodes 20 \
  --opponent random
```

当前基线：

- `MaskablePPO + MlpPolicy`
- `50000 timesteps`
- 修复终局判定 bug 后重新训练
- 修复动作编码 bug 后重新评估
- 对随机代理 `20` 局结果为 `18胜 2负 0平`

## 输出产物

常见输出目录：

- `results/runs/...`
  - 单次运行目录
  - 包含参数快照、配置快照、历史和评测结果
- `log/llm_agent_runs/...`
  - 各代理日志
- `runs/rl/...`
  - RL 模型、monitor、评估 JSON
- `runs/tensorboard/...`
  - TensorBoard 日志

## 便捷脚本说明

[scripts/run_doubao.sh](/home/aozaki/projects/code/splendor/scripts/run_doubao.sh) 仍然可用，但它只适合：

- 你明确知道自己在跑在线模型
- 你就是想少打一串命令

这个脚本现在也会检查两类冲突：

- `USE_LANGGRAPH=1` 不能和 `PROMPT_STRATEGY=rank_v2_auto` 同时使用
- `MODEL_TYPE=rl_ppo` 不应该走这个脚本

如果你只是想按项目标准方式运行，请优先看上面的 `python main.py ...` 和 `python scripts/...`。
