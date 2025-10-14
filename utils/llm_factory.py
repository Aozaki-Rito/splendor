from typing import Dict, Any, Optional
import os

import openai
from openai import OpenAI, AzureOpenAI


class BaseLLMClient:
    """LLM客户端基类"""
    
    def get_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.5, max_tokens: int = 500) -> str:
        """获取LLM的完成结果"""
        raise NotImplementedError("子类必须实现此方法")


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化OpenAI客户端
        
        Args:
            config: 模型配置
        """
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("未提供API密钥，请在配置文件中设置api_key或通过环境变量提供")
            
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.base_url = config.get("base_url")
        self.temperature = config.get("temperature", 0.5)
        self.max_tokens = config.get("max_tokens", 500)
        self.http_proxy = config.get("http_proxy") or os.environ.get("HTTP_PROXY")
        self.https_proxy = config.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        
        # 配置OpenAI客户端
        client_kwargs = {
            "api_key": self.api_key,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        # 配置代理
        if self.http_proxy or self.https_proxy:
            proxies = {}
            if self.http_proxy:
                proxies["http"] = self.http_proxy
            if self.https_proxy:
                proxies["https"] = self.https_proxy
            
            print(f"使用代理设置: {proxies}")
            try:
                import httpx
                client_kwargs["http_client"] = httpx.Client(proxies=proxies)
            except (ImportError, TypeError) as e:
                print(f"无法设置代理 (使用httpx): {e}")
                # 不阻止继续运行，只是不使用代理
            
        print(f"初始化OpenAI客户端: model={self.model_name}, base_url={self.base_url}")
        
        try:
            # 创建客户端
            self.client = OpenAI(**client_kwargs)
            
            # 测试连接
            print("测试API连接...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": "你好！"}
                ],
                max_tokens=10
            )
            print(f"API连接成功! 响应: {response.choices[0].message.content}")
        except Exception as e:
            print(f"API连接测试失败: {e}")
            raise ValueError(f"无法连接到OpenAI API: {e}")
    
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      temperature: Optional[float] = None, 
                      max_tokens: Optional[int] = None) -> str:
        """
        获取OpenAI模型的完成结果
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数，如果为None则使用配置值
            max_tokens: 最大生成令牌数，如果为None则使用配置值
            
        Returns:
            str: 模型生成的文本
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API调用出错: {e}")
            return ""


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI API客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Azure OpenAI客户端
        
        Args:
            config: 模型配置
        """
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name", "gpt-35-turbo")
        self.base_url = config.get("base_url")
        self.api_version = config.get("api_version", "2023-05-15")
        self.deployment_name = config.get("deployment_name")
        self.temperature = config.get("temperature", 0.5)
        self.max_tokens = config.get("max_tokens", 500)
        
        # 配置Azure OpenAI客户端
        client_kwargs = {
            "api_version": self.api_version
        }
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["azure_endpoint"] = self.base_url
            
        print(f"初始化Azure OpenAI客户端: model={self.model_name}, endpoint={self.base_url}")
        
        try:
            # 创建客户端
            self.client = AzureOpenAI(**client_kwargs)
            
            # 测试连接
            print("测试Azure API连接...")
            response = self.client.chat.completions.create(
                deployment_name=self.deployment_name,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": "你好！"}
                ],
                max_tokens=10
            )
            print(f"Azure API连接成功! 响应: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Azure API连接测试失败: {e}")
            raise ValueError(f"无法连接到Azure OpenAI API: {e}")
    
    def get_completion(self, system_prompt: str, user_prompt: str, 
                      temperature: Optional[float] = None, 
                      max_tokens: Optional[int] = None) -> str:
        """
        获取Azure OpenAI模型的完成结果
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            temperature: 温度参数，如果为None则使用配置值
            max_tokens: 最大生成令牌数，如果为None则使用配置值
            
        Returns:
            str: 模型生成的文本
        """
        try:
            response = self.client.chat.completions.create(
                deployment_name=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Azure OpenAI API调用出错: {e}")
            return ""


def create_llm_client(config: Dict[str, Any]) -> BaseLLMClient:
    """
    根据配置创建LLM客户端
    
    Args:
        config: 模型配置
        
    Returns:
        BaseLLMClient: LLM客户端实例
    """
    client_type = config.get("type", "").lower()
    
    print(f"创建客户端类型: {client_type}")
    
    if client_type == "openai":
        return OpenAIClient(config)
    elif client_type == "azure_openai":
        return AzureOpenAIClient(config)
    else:
        raise ValueError(f"不支持的LLM类型: {client_type}") 