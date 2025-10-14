import os
import json
from typing import Dict, Any, List, Optional


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置数据
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 处理环境变量中的API密钥
    for model in config.get("models", []):
        env_key = f"{model['type'].upper()}_API_KEY"
        if not model.get("api_key") and env_key in os.environ:
            model["api_key"] = os.environ[env_key]
    
    return config


def get_model_config(config: Dict[str, Any], model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    根据模型名称获取模型配置
    
    Args:
        config: 配置数据
        model_name: 模型名称，如果为None则返回第一个模型配置
        
    Returns:
        Optional[Dict[str, Any]]: 模型配置或None（如果未找到）
    """
    models = config.get("models", [])
    
    if not models:
        return None
    
    if model_name is None:
        return models[0]
    
    for model in models:
        if model.get("name") == model_name or model.get("model_name") == model_name:
            return model
    
    return None


def get_game_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取游戏设置
    
    Args:
        config: 配置数据
        
    Returns:
        Dict[str, Any]: 游戏设置
    """
    return config.get("game_settings", {})


def get_evaluation_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取评估设置
    
    Args:
        config: 配置数据
        
    Returns:
        Dict[str, Any]: 评估设置
    """
    return config.get("evaluation_settings", {})


def get_available_models(config: Dict[str, Any]) -> List[str]:
    """
    获取可用模型列表
    
    Args:
        config: 配置数据
        
    Returns:
        List[str]: 模型名称列表
    """
    return [model.get("name") for model in config.get("models", [])] 