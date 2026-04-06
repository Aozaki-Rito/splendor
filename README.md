# Splendor-LLM

本项目基于 [Yokumii/splendor-llm](https://github.com/Yokumii/splendor-llm.git) 的 `Version 1.0.0` 改造而来，当前已经演化成一个“璀璨宝石规则引擎 + 多种代理 + 可视化 + 评测工具”的实验项目。

目前项目里保留并支持三类策略代理：

- `legacy`
  - 纯 LLM 策略
  - 直接把完整游戏状态和全部合法动作发送给模型做决策
- `rank_v2_auto`
  - 纯规则策略
  - 不调用 LLM，由程序根据启发式评分直接选择动作
- `LanggraphAgent`
  - 基于 LangGraph 的多步决策策略
  - 会走 `plan -> think -> reflexion -> replan` 链路
  - 现已支持 OpenAI 兼容接口，包括豆包

此外，项目还保留了：

- `RandomAgent`
  - 随机策略，用于对照测试

## 当前能力

- 完整实现璀璨宝石核心规则
- 支持终端渲染和 `pygame` 图形界面
- 支持单局对战与批量评测
- 支持 OpenAI / Azure OpenAI / OpenAI 兼容接口模型
- 已支持豆包这类 OpenAI 兼容格式模型
- LangGraph 代理已支持豆包这类 OpenAI 兼容接口
- 提供独立运行日志、对局历史和评测结果归档

## 环境准备

推荐使用 conda：

```bash
conda create -n splendor python=3.9 -y
conda activate splendor
pip install -r requirements.txt
```

依赖见 [requirements.txt](/home/aozaki/projects/code/splendor/requirements.txt)。

## 项目入口

主入口是 [main.py](/home/aozaki/projects/code/splendor/main.py)，支持三个子命令：

- `game`
  - 运行单场游戏
- `eval`
  - 运行多场评测
- `list-models`
  - 列出当前配置文件中的可用模型

例如：

```bash
python main.py list-models
python main.py game --model "OpenAI GPT-4o mini"
python main.py eval --model "OpenAI GPT-4o mini" --num-games 10
```

## 配置文件

默认配置文件是 [config.json](/home/aozaki/projects/code/splendor/config.json)。

一个模型配置大概长这样：

```json
{
  "name": "OpenAI GPT-4o mini",
  "type": "openai",
  "model_name": "gpt-4o-mini",
  "api_key_env": "OPENAI_API_KEY",
  "base_url": "https://api.openai.com/v1",
  "prompt_strategy": "legacy",
  "temperature": 0.5,
  "max_tokens": 500
}
```

支持的常见字段：

- `name`
  - 模型显示名，命令行里通过 `--model` 使用
- `type`
  - `openai`、`azure_openai`、`openai_compatible`
- `model_name`
  - 实际调用的模型 ID
- `api_key` / `api_key_env`
  - API Key 可直接写入配置，也可通过环境变量读取
- `base_url`
  - OpenAI 兼容接口地址
- `prompt_strategy`
  - 当前正式支持 `legacy` 和 `rank_v2_auto`
  - LangGraph 不通过 `prompt_strategy` 切换，而是通过 `--use_langgraph 1` 启用
- `temperature`
- `max_tokens`
- `candidate_action_limit`
- `target_limit`
- `noble_limit`

## 三种策略

### `rank_v2_auto`

这是当前默认策略，也是最推荐的运行方式。

特点：

- 纯规则，不依赖 LLM 返回
- 单步决策非常快
- 适合本地演示、UI 测试和与随机代理做基线对战
- 默认脚本就使用这个模式

如果你只想直接启动项目，通常直接运行：

```bash
./scripts/run_doubao.sh
```

即使脚本里没有填写真实 API Key，这个模式也能跑起来。

### `legacy`

这是原始纯 LLM 模式。

特点：

- 直接把完整游戏状态和可用动作交给模型决策
- 更接近“真正让模型自己思考”
- 决策速度明显慢于规则模式
- 需要可用的 API Key

切换方式：

```bash
PROMPT_STRATEGY=legacy ./scripts/run_doubao.sh
```

### `LanggraphAgent`

这是 LangGraph 多步决策模式。

特点：

- 使用 `plan -> think -> reflexion -> replan` 状态机进行决策
- 支持豆包等 OpenAI 兼容接口
- 需要 API Key
- 决策链更长，通常比 `legacy` 和 `rank_v2_auto` 更慢

切换方式：

```bash
USE_LANGGRAPH=1 PROMPT_STRATEGY=legacy ./scripts/run_doubao.sh
```

也可以直接用命令行：

```bash
python main.py game --model "Doubao Seed 2.0 Pro" --use_langgraph 1 --use_pygame 0
```

## 使用豆包 / OpenAI 兼容模型

项目已经支持豆包这类 OpenAI 兼容接口模型。最方便的方式是直接改 [scripts/run_doubao.sh](/home/aozaki/projects/code/splendor/scripts/run_doubao.sh) 顶部配置。

脚本顶部可以修改：

```bash
API_KEY="your_api_key_here"
API_KEY_ENV_NAME="ARK_API_KEY"
MODEL_NAME="Doubao Seed 2.0 Pro"
MODEL_TYPE="openai_compatible"
MODEL_ID="doubao-seed-2-0-pro-260215"
BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-rank_v2_auto}"
```

如果你想换成别的 OpenAI 兼容厂商，通常只需要改：

```bash
MODEL_NAME="你的模型显示名"
MODEL_TYPE="openai_compatible"
MODEL_ID="厂商提供的模型ID"
BASE_URL="厂商提供的兼容接口地址"
API_KEY="你的 key"
```

## 快速开始

### 1. 规则模式启动一局

```bash
./scripts/run_doubao.sh
```

### 2. 纯 LLM 模式启动一局

```bash
PROMPT_STRATEGY=legacy ./scripts/run_doubao.sh
```

### 3. LangGraph 模式启动一局

```bash
USE_LANGGRAPH=1 PROMPT_STRATEGY=legacy ./scripts/run_doubao.sh
```

### 4. 无图形界面测试

```bash
USE_PYGAME=0 ./scripts/run_doubao.sh
```

### 5. 限制只跑前几回合

```bash
MAX_TURNS=3 USE_PYGAME=0 DELAY=0 ./scripts/run_doubao.sh
```

### 6. 指定随机种子

```bash
SEED=7 USE_PYGAME=0 ./scripts/run_doubao.sh
```

## 常用命令

### 终端渲染模式

```bash
python main.py game --model "OpenAI GPT-4o mini" --num-players 2 --use_pygame 0
```

### pygame 图形界面

```bash
python main.py game --model "OpenAI GPT-4o mini" --num-players 2 --use_pygame 1
```

### 两个随机代理对战

```bash
python main.py game --num-llm-agents 0 --num-players 2 --use_pygame 1
```

### 批量评测

```bash
python main.py eval --model "OpenAI GPT-4o mini" --num-games 10
```

### LangGraph 模式

```bash
python main.py game --model "OpenAI GPT-4o mini" --use_langgraph 1 --use_pygame 0
```

### 列出配置中的模型

```bash
python main.py list-models
```

## 输出产物

运行后常见产物有两类：

- `results/runs/...`
  - 单次运行目录
  - 包含运行参数、配置快照、游戏历史、评测结果
- `log/llm_agent_runs/...`
  - 代理日志
  - 包含每次决策的 prompt、响应、耗时或规则摘要

当前这些目录默认都不建议提交到 git。

## 实验脚本

项目里还有一个批量短测脚本 [run_strategy_matrix.py](/home/aozaki/projects/code/splendor/scripts/run_strategy_matrix.py)，用于快速比较不同策略或不同种子下的表现。

示例：

```bash
python scripts/run_strategy_matrix.py \
  --strategies legacy rank_v2_auto \
  --seeds 7 11 19 \
  --max-turns 2
```

它会把汇总结果写到 `results/experiments/...`。

## 已知说明

- `legacy` 模式依赖模型响应速度，单步耗时可能较高
- `rank_v2_auto` 是当前最稳定的默认模式，但它本质上是规则代理，不是纯 LLM 决策
- `LanggraphAgent` 已经可以使用豆包这类 OpenAI 兼容接口运行，但多步链路更长，通常耗时最高
- `LanggraphAgent` 仍保留在仓库中，但不是当前 README 推荐的主流程
- 如果要用 `LanggraphAgent`，建议先确认对应模型接口支持当前 LangChain OpenAI 适配层
- `config.json` 里如果直接写入 API Key，请注意不要提交到公开仓库

## TODO

- 继续优化纯 LLM 模式的状态压缩和决策质量
- 继续整理 `LanggraphAgent` 相关依赖与兼容性
- 为规则代理和 LLM 代理补充更系统的对战基准
