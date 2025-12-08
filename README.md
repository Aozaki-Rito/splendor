# Splendor-LLM

本项目follow自[Yokumii/splendor-llm](https://github.com/Yokumii/splendor-llm.git)的Version 1.0.0版本，感谢原作者的贡献。
在原项目基础上，我们做了如下修改：
- 增加了langgraph_agent.py文件，用于实现基于Plan-and-Execute架构的LLM代理，与随机代理对战胜率提升约50%。
- 使用pygame实现了游戏界面渲染，使用参数--use_pygame True来开启游戏界面渲染。
- 修改了原项目卡牌与游戏本体不相符的bug，当前版本的游戏规则、卡牌种类和数目与游戏本体一致。


## 使用方法

```bash
python main.py game --model "OpenAI GPT-4o mini" --num-players 2 --delay 0.5 --use_pygame True
python main.py game --num-llm-agents 0 --num-players 2 --delay 0.5 --use_pygame True
```

## TODO
- 优化RAG存储对局的关键信息，便于Agent决策。
- 降低Agent执行过程中的token消耗。