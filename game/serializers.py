import json
from enum import Enum

class GameJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理游戏中的特殊类型"""
    
    def default(self, obj):
        # 处理枚举类型
        if isinstance(obj, Enum):
            return obj.value
            
        # 处理可能的其他自定义类型
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
            
        # 使用基类处理其他情况
        return super().default(obj) 