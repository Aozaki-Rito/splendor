# 如何添加和使用不同的LLM代理

本文档介绍如何在璀璨宝石(Splendor)对战框架中添加和使用不同的大语言模型(LLM)作为游戏代理。

## 支持的LLM类型

目前，框架原生支持以下LLM:

1. **OpenAI GPT模型** - 通过OpenAI API使用GPT-3.5-Turbo、GPT-4等模型
2. **Azure OpenAI** - 通过Azure OpenAI服务使用部署的模型

要添加更多LLM支持，您需要实现相应的客户端接口。

## 配置文件

框架使用`config.json`配置文件来管理LLM设置。每个模型的配置格式如下:

```json
{
  "models": [
    {
      "name": "模型显示名称",
      "type": "模型类型",
      "model_name": "具体模型标识符",
      "api_key": "API密钥（可选，也可从环境变量获取）",
      "base_url": "API基础URL",
      "temperature": 0.5,
      "max_tokens": 500,
      // 其他特定于模型类型的参数
    }
  ]
}
```

### OpenAI模型配置

```json
{
  "name": "OpenAI GPT-3.5",
  "type": "openai",
  "model_name": "gpt-3.5-turbo",
  "api_key": <请输入你的API_KEY>,
  "base_url": "https://api.openai.com/v1",
  "temperature": 0.5,
  "max_tokens": 500
}
```

### Azure OpenAI配置

```json
{
  "name": "Azure OpenAI",
  "type": "azure_openai",
  "model_name": "gpt-35-turbo",
  "api_key": <请输入你的API_KEY>,
  "base_url": "https://your-resource-name.openai.azure.com",
  "api_version": "2023-05-15",
  "deployment_name": "your-deployment-name",
  "temperature": 0.5,
  "max_tokens": 500
}
```

## 使用配置的模型

### 查看可用模型

```bash
python main.py list-models
```

### 运行对战

```bash
python main.py game --model "OpenAI GPT-3.5"
```

可选参数:
- `--model`: 指定要使用的模型名称（与config.json中的name字段匹配）
- `--temperature`: 覆盖配置中的温度参数
- `--delay`: 回合之间的延迟时间(秒)
- `--save-history`: 保存游戏历史记录

### 运行评估

```bash
python main.py eval --model "OpenAI GPT-3.5" --num-games 10
```

## 添加新的LLM支持

要添加新的LLM支持，您需要:

1. 在`utils/llm_factory.py`中创建新的LLM客户端类
2. 在`config.json`中添加新的模型配置

### 1. 创建LLM客户端类

新的客户端类需要继承`BaseLLMClient`基类并实现`get_completion`方法:

```python
class YourCustomLLMClient(BaseLLMClient):
    """自定义LLM客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化客户端
        
        Args:
            config: 模型配置
        """
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name")
        # 其他特定配置
        
        # 初始化客户端
        self.client = YourLLMLibrary(api_key=self.api_key)
    
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      temperature: Optional[float] = None, 
                      max_tokens: Optional[int] = None) -> str:
        """
        获取LLM的完成结果
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数
            max_tokens: 最大生成令牌数
            
        Returns:
            str: 模型生成的文本
        """
        try:
            # 调用您的LLM API
            response = self.client.generate(
                system_prompt=system_prompt,
                prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response
        except Exception as e:
            print(f"API调用出错: {e}")
            return ""
```

### 2. 更新LLM工厂

在`utils/llm_factory.py`中的`create_llm_client`函数中添加新的类型支持:

```python
def create_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    """创建LLM客户端"""
    client_type = config.get("type", "").lower()
    
    if client_type == "openai":
        return OpenAIClient(config)
    elif client_type == "azure_openai":
        return AzureOpenAIClient(config)
    elif client_type == "your_custom_type":
        return YourCustomLLMClient(config)
    else:
        raise ValueError(f"不支持的LLM类型: {client_type}")
```

### 3. 添加配置示例

在`config.json`中添加新的模型配置:

```json
{
  "models": [
    {
      "name": "您的自定义模型",
      "type": "your_custom_type",
      "model_name": "model-name",
      "api_key": "",
      "your_custom_param1": "value1",
      "your_custom_param2": "value2",
      "temperature": 0.7,
      "max_tokens": 500
    }
  ]
}
```

## 自定义系统提示

您可以修改`agents/llm_agent.py`文件中的`_get_default_system_prompt`方法来自定义系统提示，影响LLM的行为和决策风格。

```python
def _get_default_system_prompt(self) -> str:
    """获取默认的系统提示"""
    return """
    # 修改这里的提示来改变LLM的行为
    你是一名璀璨宝石(Splendor)游戏的AI玩家...
    """
```