# Splendor-LLM

这是一个由大语言模型驱动的璀璨宝石(Splendor)游戏对战框架。该框架允许不同的LLM作为参赛者进行游戏对抗。

***注意：本项目框架基于 Claude 3.7 生成，作者主要编写了游戏规则和流程部分的代码，并为每个阶段编写了简单的提示词；此项目处于开发之中，现行版本（Version 1.0.0）仅实现了最基本的功能。***

## 项目结构

```
splendor-llm/
├── game/               # 游戏核心逻辑
│   ├── board.py        # 游戏板状态
│   ├── card.py         # 发展卡定义
│   ├── noble.py        # 贵族卡定义
│   ├── player.py       # 玩家状态
│   └── game.py         # 游戏规则和流程
├── agents/             # LLM代理
│   ├── base_agent.py   # 代理基类
│   └── llm_agent.py    # LLM驱动的代理
├── ui/                 # 游戏界面
│   └── renderer.py     # 游戏状态可视化
├── evaluation/         # 评估系统
│   └── evaluator.py    # 评估不同LLM代理的表现
├── utils/              # 工具模块
│   ├── config_loader.py # 配置加载器
│   └── llm_factory.py  # LLM客户端工厂
├── main.py             # 主程序入口
├── config.json         # 配置文件
├── LLM_GUIDE.md        # LLM集成指南
└── requirements.txt    # 项目依赖
```

## 安装

1. 克隆本仓库:
```bash
git clone https://github.com/Yokumii/splendor-llm.git
cd splendor-llm
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 设置API密钥:

新建并编辑`config.json`文件，填入各个模型所需的API密钥，或者设置环境变量:
```bash
export OPENAI_API_KEY=your_api_key_here
export AZURE_OPENAI_API_KEY=your_azure_api_key_here
```

## 配置文件

项目使用`config.json`文件来配置游戏设置和LLM模型。配置文件结构如下:

```json
{
  "models": [
    {
      "name": "OpenAI GPT-3.5",
      "type": "openai",
      "model_name": "gpt-3.5-turbo",
      "api_key": "",
      "base_url": "https://api.openai.com/v1",
      "temperature": 0.5,
      "max_tokens": 500
    },
    ...
  ],
  "game_settings": {
    "num_players": 2,
    "seed": null,
    "delay": 0.5,
    "save_history": true
  },
  "evaluation_settings": {
    "num_games": 10
  }
}
```

### 支持的模型类型

目前支持以下模型类型:

1. `openai` - OpenAI API (GPT-3.5, GPT-4等)
2. `azure_openai` - Azure OpenAI服务

## 使用方法

```bash
python main.py game --model "OpenAI GPT-3.5" --num-players 2 --delay 0.5
```

### 运行多模型对战

让不同的LLM模型相互对战:

```bash
# 两个不同模型对战
python main.py game --num-players 2 --num-llm-agents 2 \
    --model1 "OpenAI GPT-3.5" --model2 "DeepSeek-V3"

# 三个不同模型对战
python main.py game --num-players 3 --num-llm-agents 3 \
    --model1 "OpenAI GPT-3.5" --model2 "OpenAI GPT-4o-mini" --model3 "DeepSeek-V3"
```

### 评估LLM代理性能

```bash
python main.py eval --model "OpenAI GPT-3.5" --num-games 10
```

### 命令行参数

游戏模式 (`game`):
- `--model`: 使用的LLM模型名称（用于所有LLM代理）
- `--model1`, `--model2`, ...: 每个LLM代理使用的模型名称（多模型对战时使用）
- `--num-players`: 玩家数量
- `--num-llm-agents`: LLM代理数量
- `--seed`: 随机种子
- `--delay`: 回合之间的延迟时间(秒)
- `--save-history`: 保存游戏历史记录
- `--temperature`: 覆盖配置中的LLM温度参数

评估模式 (`eval`):
- `--model`: 使用的LLM模型名称
- `--num-games`: 评估时运行的游戏数量
- `--seed`: 随机种子
- `--temperature`: 覆盖配置中的LLM温度参数

## 添加新的LLM支持

请参阅 [LLM集成指南](LLM_GUIDE.md) 了解如何添加和使用不同的LLM作为代理。

## 游戏规则

璀璨宝石是一款资源管理和引擎构建类游戏，玩家通过收集宝石代币购买发展卡，获取永久性宝石加成和胜利点数。第一个获得15分的玩家获胜。

详细规则：
1. 玩家轮流执行以下操作之一：
   - 拿取3个不同颜色的宝石代币
   - 拿取2个相同颜色的宝石代币(该颜色的代币数量至少为4个)
   - 购买一张面朝上的发展卡或预留的卡
   - 预留一张发展卡并获得一个金色宝石(黄金)

2. 玩家最多持有10个宝石代币，超过需要丢弃

3. 当玩家的发展卡达到一位贵族的要求时，该贵族会立即访问该玩家，提供额外的胜利点数

4. 游戏在一位玩家达到15分后，完成当前回合结束

## 正在进行的开发

1. 为代理添加反思和复盘过程；

2. 完善代理的记忆模块管理；

3. 输出更多中间信息，包括思考过程等；