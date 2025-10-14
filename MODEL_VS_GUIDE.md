# 璀璨宝石(Splendor) 多模型对战指南

本指南介绍如何配置和运行不同LLM模型之间的璀璨宝石游戏对战。

## 1. 配置模型

在 `config.json` 文件中配置您要使用的模型：

```json
{
  "models": [
    {
      "name": "模型名称",
      "type": "openai",
      "model_name": "gpt-3.5-turbo",
      "api_key": "您的API密钥",
      "base_url": "API基础URL",
      "temperature": 0.7,
      "max_tokens": 1000,
      "http_proxy": null,
      "https_proxy": null
    }
  ]
}
```

主要配置选项：

- `name`: 模型的显示名称，用于在命令行中引用
- `type`: 模型类型，目前支持 "openai" 和 "azure_openai"
- `model_name`: 实际的模型标识符
- `api_key`: API密钥
- `base_url`: API服务的基础URL
- `temperature`: 温度参数
- `max_tokens`: 最大生成标记数
- `http_proxy`/`https_proxy`: 代理服务器配置（新增功能）

## 3. 运行对战

### 自定义对战

使用 `test_vs.sh` 脚本运行任意两个模型之间的对战：

```bash
./test_vs.sh "模型1名称" "模型2名称"
```

例如：

```bash
./test_vs.sh "OpenAI GPT-3.5" "DeepSeek-V3"
```

### 直接使用命令行

您也可以直接使用以下命令运行对战：

```bash
python main.py game --num-players <玩家数量> --num-llm-agents <LLM代理数量> \
    --model1 "模型1" --model2 "模型2" ... \
    --delay <延迟时间> --save-history
```

## 4. 结果分析

游戏结束后，结果将显示在终端，并且如果使用了 `--save-history` 选项，游戏历史将保存在 `results/` 目录中。